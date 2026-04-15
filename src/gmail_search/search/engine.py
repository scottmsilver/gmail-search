import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from gmail_search.embed.client import GeminiEmbedder
from gmail_search.index.searcher import ScannSearcher
from gmail_search.store.db import get_connection
from gmail_search.store.queries import search_fts  # noqa: F401

logger = logging.getLogger(__name__)


# Ranking weights — tunable
W_SIMILARITY = 0.40
W_BM25 = 0.15
W_RECENCY = 0.15
W_LABELS = 0.12
W_REPLIED = 0.08
W_MATCH_DENSITY = 0.06
W_THREAD_SIZE = 0.04

# Recency decay: half-life in days. Score = exp(-0.693 * days / half_life)
RECENCY_HALF_LIFE_DAYS = 60

# Label scoring: each label contributes to a 0-1 score for the thread
LABEL_SCORES = {
    "IMPORTANT": 0.35,
    "CATEGORY_PERSONAL": 0.25,
    "CATEGORY_UPDATES": 0.10,
    "CATEGORY_SOCIAL": 0.0,
    "CATEGORY_PROMOTIONS": -0.15,
    "SENT": 0.15,  # you authored it
    "INBOX": 0.10,  # still in inbox = not dismissed
    "STARRED": 0.30,  # explicit user signal
}


def _label_score(labels_per_message: list[list[str]]) -> float:
    """Compute 0-1 label quality score for a thread from all its messages' labels.

    Aggregates across all messages: if any message is IMPORTANT, the thread gets that boost.
    """
    all_labels: set[str] = set()
    for labels in labels_per_message:
        all_labels.update(labels)

    raw = sum(LABEL_SCORES.get(label, 0.0) for label in all_labels)
    # Clamp to 0-1
    return max(0.0, min(1.0, raw))


def _recency_score(date_str: str) -> float:
    """0.0 to 1.0 — exponential decay from now."""
    try:
        dt = datetime.fromisoformat(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        days_ago = max((now - dt).total_seconds() / 86400, 0)
        return math.exp(-0.693 * days_ago / RECENCY_HALF_LIFE_DAYS)
    except (ValueError, TypeError):
        return 0.0


def _match_density_score(match_count: int, thread_message_count: int) -> float:
    """What fraction of the thread matched? Capped at 1.0."""
    if thread_message_count == 0:
        return 0.0
    return min(match_count / thread_message_count, 1.0)


def _thread_size_score(message_count: int) -> float:
    """Log-scaled thread size. 1 msg = 0, 10 msgs ≈ 0.5, 50+ ≈ 1.0."""
    if message_count <= 1:
        return 0.0
    return min(math.log(message_count) / math.log(50), 1.0)


@dataclass
class ParsedQuery:
    """Structured query with filters extracted."""

    text: str  # the freetext portion to embed
    from_filter: str | None = None
    to_filter: str | None = None
    after_filter: str | None = None
    before_filter: str | None = None
    temporal_boost: float = 0.0  # 0.0 = no time intent, up to 0.35 for strong time intent


def parse_query(raw: str) -> ParsedQuery:
    """Extract structured filters and temporal intent from a query string."""
    import re

    text_parts: list[str] = []
    from_filter = None
    to_filter = None
    after_filter = None
    before_filter = None

    tokens = raw.split()
    i = 0
    while i < len(tokens):
        token = tokens[i]
        # from:value or from: value
        for prefix, setter in [("from:", "from"), ("to:", "to"), ("after:", "after"), ("before:", "before")]:
            if token.lower().startswith(prefix):
                val = token[len(prefix) :]
                if not val and i + 1 < len(tokens):
                    i += 1
                    val = tokens[i]
                if setter == "from":
                    from_filter = val
                elif setter == "to":
                    to_filter = val
                elif setter == "after":
                    after_filter = val
                elif setter == "before":
                    before_filter = val
                break
        else:
            text_parts.append(token)
        i += 1

    text = " ".join(text_parts)

    # Detect temporal intent from freetext
    temporal_boost = 0.0
    time_patterns = [
        (r"\b(today|yesterday|this morning|tonight)\b", 0.35),
        (r"\b(this week|last week|few days ago)\b", 0.30),
        (r"\b(this month|last month|recently|recent)\b", 0.25),
        (r"\b(this year|last year|past year)\b", 0.15),
        (r"\b(latest|newest|most recent)\b", 0.30),
    ]
    for pattern, boost in time_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            temporal_boost = max(temporal_boost, boost)

    return ParsedQuery(
        text=text,
        from_filter=from_filter,
        to_filter=to_filter,
        after_filter=after_filter,
        before_filter=before_filter,
        temporal_boost=temporal_boost,
    )


def _contact_frequency_score(from_addrs: list[str], freq_map: dict[str, float]) -> float:
    """Score based on how frequently you interact with thread participants. 0-1."""
    if not from_addrs or not freq_map:
        return 0.0
    best = 0.0
    for addr in from_addrs:
        lower = addr.lower()
        for key, score in freq_map.items():
            if key in lower:
                best = max(best, score)
    return best


@dataclass
class SearchResult:
    """Legacy per-message result, still used by CLI."""

    score: float
    message_id: str
    subject: str
    from_addr: str
    date: str
    snippet: str
    match_type: str
    attachment_filename: str | None = None


@dataclass
class ThreadMatch:
    """A message within a thread that matched the query."""

    message_id: str
    score: float
    from_addr: str
    date: str
    snippet: str
    match_type: str
    attachment_filename: str | None = None


@dataclass
class ThreadResult:
    """A thread-level search result grouping all matching messages."""

    thread_id: str
    score: float  # blended ranking score
    similarity: float  # raw best similarity
    subject: str
    participants: list[str]
    message_count: int
    date_first: str
    date_last: str
    user_replied: bool
    matches: list[ThreadMatch] = field(default_factory=list)


class SearchEngine:
    def __init__(
        self,
        db_path: Path,
        index_dir: Path,
        config: dict[str, Any],
        embedder: GeminiEmbedder | Any = None,
    ):
        self.db_path = db_path
        self.config = config
        dims = config["embedding"]["dimensions"]
        self.searcher = ScannSearcher(index_dir, dimensions=dims)
        self.embedder = embedder or GeminiEmbedder(config)

        # Detect user's email from config or DB
        self.user_emails = set()
        user_email = config.get("user_email")
        if user_email:
            self.user_emails.add(user_email.lower())
        else:
            self._detect_user_email()

        # Load contact frequency map
        self.contact_freq: dict[str, float] = {}
        self._load_contact_frequency()

    def _detect_user_email(self):
        """Find the most frequent sender — that's the user."""
        try:
            conn = get_connection(self.db_path)
            row = conn.execute(
                "SELECT from_addr, COUNT(*) as c FROM messages GROUP BY from_addr ORDER BY c DESC LIMIT 1"
            ).fetchone()
            conn.close()
            if row:
                addr = row["from_addr"].lower()
                # Extract just the email from "Name <email>" format
                if "<" in addr:
                    addr = addr.split("<")[1].rstrip(">")
                self.user_emails.add(addr)
                logger.info(f"Detected user email: {addr}")
        except Exception:
            pass

    def _load_contact_frequency(self):
        try:
            conn = get_connection(self.db_path)
            rows = conn.execute("SELECT email, score FROM contact_frequency").fetchall()
            conn.close()
            self.contact_freq = {r["email"]: r["score"] for r in rows}
        except Exception:
            self.contact_freq = {}

    def _user_in_participants(self, from_addrs: list[str]) -> bool:
        for addr in from_addrs:
            lower = addr.lower()
            for email in self.user_emails:
                if email in lower:
                    return True
        return False

    def _embed_query(self, query: str) -> np.ndarray:
        import time as _time

        for attempt in range(3):
            try:
                return np.array(self.embedder.embed_query(query), dtype=np.float32)
            except Exception as e:
                if attempt < 2 and any(code in str(e) for code in ["503", "429", "UNAVAILABLE"]):
                    _time.sleep(2 ** (attempt + 1))
                    logger.warning(f"Query embed retry {attempt + 1}: {e}")
                else:
                    raise

    def _fetch_embedding_rows(self, embedding_ids: list[int]):
        conn = get_connection(self.db_path)
        placeholders = ",".join("?" * len(embedding_ids))
        rows = conn.execute(
            f"""SELECT e.id, e.message_id, e.attachment_id, e.chunk_type, e.chunk_text,
                       m.thread_id, m.subject, m.from_addr, m.date,
                       a.filename as att_filename
                FROM embeddings e
                JOIN messages m ON e.message_id = m.id
                LEFT JOIN attachments a ON e.attachment_id = a.id
                WHERE e.id IN ({placeholders})""",
            embedding_ids,
        ).fetchall()
        conn.close()
        return {r["id"]: r for r in rows}

    def search(self, query: str, top_k: int = 20) -> list[SearchResult]:
        """Per-message search (used by CLI)."""
        query_vector = self._embed_query(query)

        fetch_k = min(top_k * 3, len(self.searcher.embedding_ids))
        embedding_ids, scores = self.searcher.search(query_vector, top_k=fetch_k)

        if not embedding_ids:
            return []

        row_map = self._fetch_embedding_rows(embedding_ids)
        seen_messages: dict[str, SearchResult] = {}

        for emb_id, score in zip(embedding_ids, scores):
            row = row_map.get(emb_id)
            if row is None:
                continue

            msg_id = row["message_id"]
            if msg_id in seen_messages:
                if score > seen_messages[msg_id].score:
                    seen_messages[msg_id].score = score
                    seen_messages[msg_id].match_type = row["chunk_type"]
                    seen_messages[msg_id].snippet = (row["chunk_text"] or "")[:200]
                continue

            seen_messages[msg_id] = SearchResult(
                score=score,
                message_id=msg_id,
                subject=row["subject"],
                from_addr=row["from_addr"],
                date=row["date"],
                snippet=(row["chunk_text"] or "")[:200],
                match_type=row["chunk_type"],
                attachment_filename=row["att_filename"],
            )

        results = sorted(seen_messages.values(), key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def _clean_query(self, raw_query: str) -> str:
        """Spell-correct and normalize query using Gemini Flash Lite."""
        try:
            import os

            from google import genai  # noqa: F811

            api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            client = genai.Client(api_key=api_key) if api_key else genai.Client()
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite-preview",
                contents=(
                    "Fix any spelling errors in this email search query. "
                    "Keep the same meaning and intent. Keep structured operators like from: and after: intact. "
                    "Return ONLY the corrected query, nothing else.\n\n"
                    f"Query: {raw_query}"
                ),
            )
            cleaned = response.text.strip().strip('"').strip("'")
            if cleaned and len(cleaned) < len(raw_query) * 3:  # sanity check
                if cleaned.lower() != raw_query.lower():
                    logger.info(f"Query corrected: '{raw_query}' -> '{cleaned}'")
                return cleaned
        except Exception as e:
            logger.warning(f"Query cleanup failed: {e}")
        return raw_query

    def search_threads(self, query: str, top_k: int = 20, sort: str = "relevance") -> list[ThreadResult]:
        """Thread-grouped search with multi-signal ranking."""
        cleaned_query = self._clean_query(query)
        pq = parse_query(cleaned_query)
        query_vector = self._embed_query(pq.text or cleaned_query)

        fetch_k = min(top_k * 10, len(self.searcher.embedding_ids))
        embedding_ids, scores = self.searcher.search(query_vector, top_k=fetch_k)

        # If ScaNN has no results, we still continue to pick up BM25-only matches below
        row_map = self._fetch_embedding_rows(embedding_ids) if embedding_ids else {}

        # Normalize similarity scores to 0-1 range
        max_sim = max(scores) if scores else 1.0
        min_sim = min(scores) if scores else 0.0
        sim_range = max_sim - min_sim if max_sim != min_sim else 1.0

        # Group by thread, dedup messages within each thread
        threads: dict[str, dict] = {}

        for emb_id, score in zip(embedding_ids, scores):
            row = row_map.get(emb_id)
            if row is None:
                continue

            thread_id = row["thread_id"]
            msg_id = row["message_id"]
            norm_score = (score - min_sim) / sim_range

            if thread_id not in threads:
                threads[thread_id] = {
                    "best_sim": norm_score,
                    "raw_sim": score,
                    "subject": row["subject"],
                    "matches": {},
                }

            thread = threads[thread_id]
            if norm_score > thread["best_sim"]:
                thread["best_sim"] = norm_score
                thread["raw_sim"] = score

            if msg_id not in thread["matches"]:
                thread["matches"][msg_id] = ThreadMatch(
                    message_id=msg_id,
                    score=score,
                    from_addr=row["from_addr"],
                    date=row["date"],
                    snippet=(row["chunk_text"] or "")[:200],
                    match_type=row["chunk_type"],
                    attachment_filename=row["att_filename"],
                )
            elif score > thread["matches"][msg_id].score:
                m = thread["matches"][msg_id]
                m.score = score
                m.match_type = row["chunk_type"]
                m.snippet = (row["chunk_text"] or "")[:200]

        # BM25 keyword search
        conn = get_connection(self.db_path)
        # BM25 with both corrected and original query (merge best scores)
        bm25_scores = search_fts(conn, pq.text or cleaned_query, limit=200)
        if cleaned_query.lower() != query.lower():
            original_pq = parse_query(query)
            original_bm25 = search_fts(conn, original_pq.text or query, limit=200)
            for mid, score in original_bm25.items():
                if mid not in bm25_scores or score > bm25_scores[mid]:
                    bm25_scores[mid] = score

        # Merge BM25-only results: threads that BM25 found but ScaNN missed
        scann_message_ids = set()
        for tdata in threads.values():
            scann_message_ids.update(tdata["matches"].keys())

        bm25_only_ids = [mid for mid in bm25_scores if mid not in scann_message_ids]
        if bm25_only_ids:
            placeholders = ",".join("?" * len(bm25_only_ids))
            bm25_rows = conn.execute(
                f"""SELECT id, thread_id, subject, from_addr, date, body_text
                    FROM messages WHERE id IN ({placeholders})""",
                bm25_only_ids,
            ).fetchall()
            for r in bm25_rows:
                thread_id = r["thread_id"]
                msg_id = r["id"]
                if thread_id not in threads:
                    threads[thread_id] = {
                        "best_sim": 0.0,  # no embedding similarity
                        "raw_sim": 0.0,
                        "subject": r["subject"],
                        "matches": {},
                    }
                if msg_id not in threads[thread_id]["matches"]:
                    snippet = (r["body_text"] or "")[:200]
                    threads[thread_id]["matches"][msg_id] = ThreadMatch(
                        message_id=msg_id,
                        score=0.0,
                        from_addr=r["from_addr"],
                        date=r["date"],
                        snippet=snippet,
                        match_type="keyword",
                        attachment_filename=None,
                    )

        # Aggregate BM25 to thread level: best message score per thread
        thread_bm25: dict[str, float] = {}
        for thread_id, tdata in threads.items():
            best_bm25 = 0.0
            for msg_id in tdata["matches"]:
                if msg_id in bm25_scores:
                    best_bm25 = max(best_bm25, bm25_scores[msg_id])
            thread_bm25[thread_id] = best_bm25

        # Bulk-fetch precomputed thread metadata (one query instead of N)
        import json as _json

        thread_ids = list(threads.keys())
        placeholders = ",".join("?" * len(thread_ids))
        summary_rows = conn.execute(
            f"SELECT * FROM thread_summary WHERE thread_id IN ({placeholders})",
            thread_ids,
        ).fetchall()
        summary_map = {r["thread_id"]: r for r in summary_rows}

        # Score threads using precomputed metadata
        thread_results: list[ThreadResult] = []

        for thread_id, tdata in threads.items():
            summary = summary_map.get(thread_id)
            matches = sorted(tdata["matches"].values(), key=lambda m: m.score, reverse=True)

            if summary:
                participants = _json.loads(summary["participants"])
                from_addrs = _json.loads(summary["all_from_addrs"])
                all_labels = _json.loads(summary["all_labels"])
                msg_count = summary["message_count"]
                date_first = summary["date_first"]
                date_last = summary["date_last"]
                subject = summary["subject"]
            else:
                # Fallback for threads not yet in summary (shouldn't happen after reindex)
                participants = list({m.from_addr for m in matches})
                from_addrs = [m.from_addr for m in matches]
                all_labels = []
                msg_count = len(matches)
                date_first = matches[-1].date if matches else ""
                date_last = matches[0].date if matches else ""
                subject = tdata["subject"]

            user_replied = self._user_in_participants(from_addrs)

            # Apply structured filters — skip threads that don't match
            if pq.from_filter:
                if not any(pq.from_filter.lower() in a.lower() for a in from_addrs):
                    continue
            if pq.to_filter:
                # Check to_addr on matching messages
                skip = True
                for m in matches:
                    # We don't have to_addr on ThreadMatch, check via participants
                    if pq.to_filter.lower() in " ".join(participants).lower():
                        skip = False
                        break
                if skip:
                    continue
            if pq.after_filter:
                try:
                    from dateutil.parser import parse as _dateparse

                    after_dt = _dateparse(pq.after_filter)
                    thread_dt = datetime.fromisoformat(date_last) if date_last else None
                    if thread_dt and thread_dt < after_dt:
                        continue
                except Exception:
                    pass
            if pq.before_filter:
                try:
                    from dateutil.parser import parse as _dateparse

                    before_dt = _dateparse(pq.before_filter)
                    thread_dt = datetime.fromisoformat(date_first) if date_first else None
                    if thread_dt and thread_dt > before_dt:
                        continue
                except Exception:
                    pass

            # Compute blended score
            sim = tdata["best_sim"]
            bm25 = thread_bm25.get(thread_id, 0.0)
            recency = _recency_score(date_last)
            labels = _label_score([all_labels])  # already aggregated
            replied = 1.0 if user_replied else 0.0
            density = _match_density_score(len(matches), msg_count)
            tsize = _thread_size_score(msg_count)
            contact = _contact_frequency_score(from_addrs, self.contact_freq)

            # Dynamic recency weight: boost if query has temporal intent
            w_recency = W_RECENCY + pq.temporal_boost
            # Redistribute the extra weight from similarity
            w_similarity = W_SIMILARITY - pq.temporal_boost

            blended = (
                w_similarity * sim
                + W_BM25 * bm25
                + w_recency * recency
                + W_LABELS * labels
                + W_REPLIED * replied
                + W_MATCH_DENSITY * density
                + W_THREAD_SIZE * tsize
                + 0.08 * contact  # contact frequency bonus (on top of 1.0 total)
            )

            thread_results.append(
                ThreadResult(
                    thread_id=thread_id,
                    score=blended,
                    similarity=tdata["raw_sim"],
                    subject=subject,
                    participants=participants,
                    message_count=msg_count,
                    date_first=date_first,
                    date_last=date_last,
                    user_replied=user_replied,
                    matches=matches,
                )
            )

        conn.close()

        if sort == "recent":
            thread_results.sort(key=lambda t: t.date_last, reverse=True)
        else:
            thread_results.sort(key=lambda t: t.score, reverse=True)

        # Collapse repeat senders
        thread_results = self._collapse_repeat_senders(thread_results, top_k * 2)

        # LLM reranker on top candidates (if enabled)
        rerank = self.config.get("search", {}).get("rerank", True)
        if rerank and sort == "relevance" and len(thread_results) > 3:
            thread_results = self._llm_rerank(pq.text or query, thread_results, top_k)

        return thread_results[:top_k]

    @staticmethod
    def _normalize_subject(subject: str) -> str:
        """Strip Re:/Fwd:/numbers/dates to compare subject similarity."""
        import re

        s = subject.lower().strip()
        s = re.sub(r"^(re|fwd|fw):\s*", "", s)
        s = re.sub(r"\d+", "#", s)  # normalize numbers
        return s.strip()

    @staticmethod
    def _collapse_repeat_senders(results: list[ThreadResult], top_k: int) -> list[ThreadResult]:
        """Collapse repeated single-message threads from the same sender with similar subjects.

        If sender X has 3+ single-message threads with similar subjects, keep only
        the top-scoring one. Threads with distinct subjects are preserved.
        """
        from collections import defaultdict

        # Group single-message threads by (sender, normalized_subject)
        group_counts: dict[tuple[str, str], int] = defaultdict(int)
        for t in results:
            if t.message_count == 1 and t.participants:
                key = (t.participants[0].lower(), SearchEngine._normalize_subject(t.subject))
                group_counts[key] += 1

        # Groups with 3+ hits get collapsed
        repeat_groups = {k for k, c in group_counts.items() if c >= 3}
        if not repeat_groups:
            return results[:top_k]

        seen_groups: set[tuple[str, str]] = set()
        collapsed: list[ThreadResult] = []
        suppressed = 0

        for t in results:
            if t.message_count == 1 and t.participants:
                key = (t.participants[0].lower(), SearchEngine._normalize_subject(t.subject))
                if key in repeat_groups:
                    if key in seen_groups:
                        suppressed += 1
                        continue
                    seen_groups.add(key)

            collapsed.append(t)

        if suppressed:
            logger.info(f"Collapsed {suppressed} repeat single-message threads")

        return collapsed[:top_k]

    def _llm_rerank(self, query: str, results: list[ThreadResult], top_k: int) -> list[ThreadResult]:
        """Use Gemini Flash to rerank the top candidates by relevance."""
        import json as _json

        # Only rerank the top 30 to keep cost/latency low
        candidates = results[:30]

        # Build a concise summary for each candidate
        items = []
        for i, t in enumerate(candidates):
            snippet = t.matches[0].snippet[:100] if t.matches else ""
            people = ", ".join(p.split("<")[0].strip().strip('"') for p in t.participants[:3])
            items.append(f"{i}: [{t.subject}] from {people} ({t.message_count} msgs) - {snippet}")

        prompt = (
            f"Query: {query}\n\n"
            "Rank these email threads by relevance to the query. "
            "Return ONLY a JSON array of the indices in order of relevance, most relevant first. "
            "Example: [3, 0, 7, 1, ...]. Return the top 20 most relevant.\n\n" + "\n".join(items)
        )

        try:
            import os

            from google import genai  # noqa: F811

            api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            client = genai.Client(api_key=api_key) if api_key else genai.Client()
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite-preview",
                contents=prompt,
            )
            text = response.text.strip()
            # Parse the JSON array from response (handle markdown code blocks)
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            ranked_indices = _json.loads(text)

            if not isinstance(ranked_indices, list):
                return results

            # Build reranked list
            reranked: list[ThreadResult] = []
            seen = set()
            for idx in ranked_indices:
                if isinstance(idx, int) and 0 <= idx < len(candidates) and idx not in seen:
                    reranked.append(candidates[idx])
                    seen.add(idx)

            # Append any candidates the LLM didn't include
            for i, t in enumerate(candidates):
                if i not in seen:
                    reranked.append(t)

            # Append remaining results beyond top 30
            reranked.extend(results[30:])

            logger.info(f"LLM reranked {len(ranked_indices)} results")
            return reranked

        except Exception as e:
            logger.warning(f"LLM rerank failed, using default ranking: {e}")
            return results
