import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from gmail_search.embed.client import GeminiEmbedder
from gmail_search.index.searcher import ScannSearcher
from gmail_search.search.parser import ParsedQuery, parse_query  # noqa: F401  (ParsedQuery re-exported)
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

# When structured filters (from:/to:/subject:/date/has_attachment) pre-restrict
# the corpus, we run vector similarity directly over the restricted set if
# it's small enough — brute-force cosine over a few thousand embeddings is
# faster + strictly more accurate than ScaNN-with-overfetch-then-filter and
# guarantees recall. Above this threshold we fall back to ScaNN with heavy
# overfetch and filter the result to `candidate_ids`.
VECTOR_BRUTEFORCE_THRESHOLD = 20_000

# Cap on the number of message IDs returned by the structured-filter pre-pass.
# Past this point we'd need a different retrieval strategy (e.g. per-shard
# filtered ANN). Practical mail corpora don't hit this unless the filter is
# nearly a no-op (`from:.com`), in which case we may as well search the whole
# index — but we prefer to fail loud so the caller notices.
CANDIDATE_ID_CAP = 100_000

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

        # Load spell corrector from corpus dictionary
        self._spell = None
        self._load_spell_dictionary()

        # Load term aliases for query expansion
        self._aliases: dict[str, list[str]] = {}
        self._load_term_aliases()

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

    def _load_spell_dictionary(self):
        """Load SymSpell dictionary built from email corpus."""
        try:
            from symspellpy import SymSpell, Verbosity  # noqa: F401

            data_dir = Path(self.config.get("data_dir", "data"))
            dict_path = data_dir / "spell_dictionary.txt"
            if dict_path.exists():
                self._spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
                self._spell.load_dictionary(str(dict_path), term_index=0, count_index=1)
                logger.info(f"Loaded spell dictionary: {dict_path}")
            else:
                logger.info("No spell dictionary found, spell correction disabled")
        except ImportError:
            logger.info("symspellpy not installed, spell correction disabled")
        except Exception as e:
            logger.warning(f"Failed to load spell dictionary: {e}")

    def _load_term_aliases(self):
        """Load precomputed term aliases from embedding nearest neighbors."""
        import json

        try:
            conn = get_connection(self.db_path)
            rows = conn.execute("SELECT term, expansions FROM term_aliases").fetchall()
            conn.close()
            self._aliases = {r["term"]: json.loads(r["expansions"]) for r in rows}
            if self._aliases:
                logger.info(f"Loaded {len(self._aliases)} term aliases")
        except Exception:
            self._aliases = {}

    def _expand_query_with_aliases(self, query: str) -> str:
        """Expand short terms in the query using discovered aliases.

        'ke board' → 'ke kol emeth board' — adds the expansions alongside
        the original term so both the alias and original match.
        """
        if not self._aliases:
            return query

        words = query.split()
        expanded = []
        for word in words:
            expanded.append(word)
            aliases = self._aliases.get(word.lower())
            if aliases:
                # Add top 2 expansions alongside the original
                for alias in aliases[:2]:
                    expanded.append(alias)
                logger.info(f"Alias expansion: '{word}' → +{aliases[:2]}")

        result = " ".join(expanded)
        if result != query:
            logger.info(f"Expanded query: '{query}' → '{result}'")
        return result

    def _user_in_participants(self, from_addrs: list[str]) -> bool:
        for addr in from_addrs:
            lower = addr.lower()
            for email in self.user_emails:
                if email in lower:
                    return True
        return False

    def _get_cached_embedding(self, cache_key: str) -> np.ndarray | None:
        """Look up a query embedding in the persistent cache."""
        import struct

        dims = self.config["embedding"]["dimensions"]
        try:
            conn = get_connection(self.db_path)
            row = conn.execute(
                "SELECT embedding FROM query_cache WHERE query_text = %s AND model = %s",
                (cache_key, self.config["embedding"]["model"]),
            ).fetchone()
            conn.close()
            if row:
                return np.array(struct.unpack(f"{dims}f", row["embedding"]), dtype=np.float32)
        except Exception:
            pass
        return None

    def _store_cached_embedding(self, cache_key: str, vector: list[float]) -> None:
        """Persist a query embedding to the cache."""
        import struct
        from datetime import datetime, timezone

        try:
            blob = struct.pack(f"{len(vector)}f", *vector)
            conn = get_connection(self.db_path)
            # Portable upsert: `INSERT OR REPLACE` is SQLite-only; the
            # ON CONFLICT form here works on SQLite and Postgres.
            # Conflict target is (query_text, model) — the composite PK
            # of query_cache.
            conn.execute(
                """INSERT INTO query_cache (query_text, model, embedding, created_at)
                   VALUES (%s, %s, %s, %s)
                   ON CONFLICT(query_text, model) DO UPDATE SET
                     embedding = excluded.embedding,
                     created_at = excluded.created_at""",
                (cache_key, self.config["embedding"]["model"], blob, datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

    def _call_embedding_api_with_retry(self, query: str, max_retries: int = 3) -> list[float]:
        """Call the Gemini embedding API with exponential backoff on transient errors."""
        import time as _time

        for attempt in range(max_retries):
            try:
                return self.embedder.embed_query(query)
            except Exception as e:
                if attempt < max_retries - 1 and any(code in str(e) for code in ["503", "429", "UNAVAILABLE"]):
                    _time.sleep(2 ** (attempt + 1))
                    logger.warning(f"Query embed retry {attempt + 1}: {e}")
                else:
                    raise

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a query, checking persistent cache first to avoid repeat
        API calls. Records a cost row on cache MISS so the ledger
        reflects search-as-you-type spend — previously silent, which
        made the cost badge look artificially low. A ~10-token query
        at $0.20/1M tokens ≈ $0.000002, so the recorded values are
        tiny but the sum-over-time is now visible.
        """
        from gmail_search.store.cost import record_cost

        cache_key = query.lower().strip()

        cached = self._get_cached_embedding(cache_key)
        if cached is not None:
            logger.info(f"Query cache hit: '{query}'")
            return cached

        vector = self._call_embedding_api_with_retry(query)
        self._store_cached_embedding(cache_key, vector)

        try:
            # Token estimate: Gemini's embedding tokenizer sits around
            # 4 chars/token for English prose — good enough for ledger
            # accuracy. We'd only need the real count if per-call cost
            # mattered, and at fractions of a cent each it doesn't.
            from gmail_search.store.cost import estimate_cost

            est_tokens = max(1, len(query) // 4)
            conn = get_connection(self.db_path)
            try:
                record_cost(
                    conn,
                    operation="embed_query",
                    model=self.config["embedding"]["model"],
                    input_tokens=est_tokens,
                    image_count=0,
                    estimated_cost_usd=estimate_cost(input_tokens=est_tokens),
                    message_id="",
                )
            finally:
                conn.close()
        except Exception:
            # Cost-ledger failures must never block a search.
            pass

        return np.array(vector, dtype=np.float32)

    def _resolve_candidate_msg_ids(
        self,
        pq: ParsedQuery,
        conn,
        date_from: str | None,
        date_to: str | None,
    ) -> list[str] | None:
        """Return the set of message IDs that pass every structured filter
        (``from:``, ``to:``, ``subject:``, ``has:attachment``, date range).

        - ``None`` → no structured filter is active; caller should search the
          full corpus.
        - ``[]``   → filters are active but matched zero rows; caller should
          short-circuit to an empty result.
        - otherwise a list of message IDs (preserving DB order) ready to be
          fed into BM25 / vector search.

        ``date_from`` / ``date_to`` are the effective dates (endpoint
        overrides merged with operator-parsed values). Passing them in lets
        this resolver be the single source of truth for corpus restriction,
        so the downstream date post-filter becomes redundant.
        """
        has_any_filter = bool(
            pq.from_filter or pq.to_filter or pq.subject_filter or pq.has_attachment or date_from or date_to
        )
        if not has_any_filter:
            return None

        where: list[str] = []
        params: list = []

        if pq.from_filter:
            where.append("m.from_addr ILIKE %s")
            params.append(f"%{pq.from_filter}%")
        if pq.to_filter:
            where.append("m.to_addr ILIKE %s")
            params.append(f"%{pq.to_filter}%")
        if pq.subject_filter:
            where.append("m.subject ILIKE %s")
            params.append(f"%{pq.subject_filter}%")
        if date_from:
            where.append("m.date >= %s")
            params.append(f"{date_from}T00:00:00")
        if date_to:
            where.append("m.date <= %s")
            params.append(f"{date_to}T23:59:59")
        if pq.has_attachment is True:
            where.append("EXISTS (SELECT 1 FROM attachments a WHERE a.message_id = m.id)")

        where_sql = " AND ".join(where) if where else "TRUE"
        # Cap +1 so we can detect overflow and warn.
        sql = f"SELECT m.id FROM messages m WHERE {where_sql} LIMIT {CANDIDATE_ID_CAP + 1}"
        rows = conn.execute(sql, params).fetchall()

        ids = [r["id"] for r in rows]
        if len(ids) > CANDIDATE_ID_CAP:
            logger.warning(
                "Structured filter matched > %d messages; truncating to cap. "
                "Recall may suffer — consider narrowing the filter.",
                CANDIDATE_ID_CAP,
            )
            ids = ids[:CANDIDATE_ID_CAP]
        return ids

    def _decode_embedding_blob(self, blob: bytes) -> np.ndarray:
        """Little-endian float32 blob → numpy vector. Mirrors the
        zero-copy ``np.frombuffer`` path used by
        ``index/builder.py::_load_embeddings_matrix`` so vector geometry
        stays identical between indexed and on-the-fly lookups.
        """
        return np.frombuffer(blob, dtype=np.float32)

    def _bruteforce_vector_search(
        self,
        query_vector: np.ndarray,
        candidate_msg_ids: list[str],
        fetch_k: int,
    ) -> tuple[list[int], list[float]]:
        """Cosine similarity (dot product on L2-normalized vectors) over
        the embeddings of a restricted message set. Returns the same
        ``(embedding_ids, scores)`` interface as ``ScannSearcher.search``
        so the downstream merge code is backend-agnostic.
        """
        model = self.config["embedding"]["model"]
        conn = get_connection(self.db_path)
        try:
            placeholders = ",".join(["%s"] * len(candidate_msg_ids))
            rows = conn.execute(
                f"""SELECT id, message_id, embedding
                    FROM embeddings
                    WHERE message_id IN ({placeholders}) AND model = %s""",
                list(candidate_msg_ids) + [model],
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            return [], []

        dims = self.config["embedding"]["dimensions"]
        mat = np.empty((len(rows), dims), dtype=np.float32)
        ids: list[int] = []
        for i, r in enumerate(rows):
            ids.append(int(r["id"]))
            mat[i] = self._decode_embedding_blob(r["embedding"])

        # ScaNN indexes are built with "dot_product" similarity on the raw
        # stored vectors (builder.py:62). We mirror that exactly — no
        # additional normalization — so scores here are directly comparable
        # to ScaNN's.
        scores = mat @ query_vector.astype(np.float32)

        k = min(fetch_k, len(ids))
        if k <= 0:
            return [], []
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [ids[i] for i in top_idx], [float(scores[i]) for i in top_idx]

    def _restricted_vector_search(
        self,
        query_vector: np.ndarray,
        candidate_msg_ids: list[str],
        fetch_k: int,
    ) -> tuple[list[int], list[float]]:
        """Vector search constrained to a pre-filtered candidate set.

        Small set → brute-force cosine (exact recall, same scale as
        ScaNN's dot-product). Large set → ScaNN with heavy overfetch,
        then drop embeddings whose message is outside the candidate set.
        """
        n = len(candidate_msg_ids)
        if n == 0:
            return [], []

        if n < VECTOR_BRUTEFORCE_THRESHOLD:
            return self._bruteforce_vector_search(query_vector, candidate_msg_ids, fetch_k)

        # Large restricted set — fall back to ScaNN with overfetch, then
        # filter. We want enough candidates that the restricted subset
        # still covers fetch_k items.
        overfetch = max(fetch_k, 10_000)
        overfetch = min(overfetch, len(self.searcher.embedding_ids))
        emb_ids, scores = self.searcher.search(query_vector, top_k=overfetch)
        if not emb_ids:
            return [], []

        # Map each returned embedding_id back to its message_id, drop the
        # ones outside the candidate set.
        candidate_set = set(candidate_msg_ids)
        conn = get_connection(self.db_path)
        try:
            placeholders = ",".join(["%s"] * len(emb_ids))
            rows = conn.execute(
                f"SELECT id, message_id FROM embeddings WHERE id IN ({placeholders})",
                emb_ids,
            ).fetchall()
        finally:
            conn.close()
        emb_to_msg = {int(r["id"]): r["message_id"] for r in rows}

        kept_ids: list[int] = []
        kept_scores: list[float] = []
        for eid, sc in zip(emb_ids, scores):
            mid = emb_to_msg.get(int(eid))
            if mid is not None and mid in candidate_set:
                kept_ids.append(int(eid))
                kept_scores.append(float(sc))
                if len(kept_ids) >= fetch_k:
                    break
        return kept_ids, kept_scores

    def _fetch_embedding_rows(self, embedding_ids: list[int]):
        conn = get_connection(self.db_path)
        placeholders = ",".join(["%s"] * len(embedding_ids))
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
        """Spell-correct query using local corpus dictionary. Sub-millisecond."""
        if not self._spell:
            return raw_query

        from symspellpy import Verbosity

        words = raw_query.split()
        corrected = []
        changed = False

        for word in words:
            # Don't correct structured operators or very short words
            if ":" in word or len(word) <= 2:
                corrected.append(word)
                continue

            suggestions = self._spell.lookup(word.lower(), Verbosity.CLOSEST, max_edit_distance=2)
            if suggestions and suggestions[0].term != word.lower():
                corrected.append(suggestions[0].term)
                changed = True
            else:
                corrected.append(word)

        result = " ".join(corrected)
        if changed:
            logger.info(f"Query corrected: '{raw_query}' -> '{result}'")
        return result

    def search_threads(
        self,
        query: str,
        top_k: int = 20,
        sort: str = "relevance",
        filter_offtopic: bool = True,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[ThreadResult]:
        """Thread-grouped search with multi-signal ranking.

        Optional date filters (ISO YYYY-MM-DD) restrict results to
        threads with at least one matching message in the range —
        computed AFTER ScaNN + BM25 ranking, so you get top relevance
        WITHIN the window rather than raw chronological results.
        """
        cleaned_query = self._clean_query(query)
        expanded_query = self._expand_query_with_aliases(cleaned_query)
        pq = parse_query(expanded_query)
        query_vector = self._embed_query(pq.text or expanded_query)

        # Merge: explicit endpoint-level date_from/date_to win over values
        # derived from operators in the query string (`after:`/`newer_than:`/…).
        date_from = date_from or pq.date_from
        date_to = date_to or pq.date_to

        # Pre-restrict the candidate corpus using structured filters
        # (from:/to:/subject:/date/has_attachment). This guarantees recall:
        # if the user asks for "board notes from:david", we search *only
        # within David's mail*, instead of hoping David's 3 messages survive
        # a top-200 ScaNN pass over the whole index. `None` = no filter
        # active (full corpus); `[]` = filter matched zero rows (short
        # circuit to an empty result).
        conn_pre = get_connection(self.db_path)
        try:
            candidate_ids = self._resolve_candidate_msg_ids(pq, conn_pre, date_from, date_to)
        finally:
            conn_pre.close()

        if candidate_ids is not None and len(candidate_ids) == 0:
            logger.info("Structured filter matched zero messages; returning []")
            return []

        # When a date filter is present, ScaNN's top-K might all fall
        # outside the window, leaving us zero results even though there
        # are relevant matches further down the similarity ranking.
        # Grab a much bigger candidate pool so the post-filter has
        # something to work with. ScaNN is cheap enough that fetching
        # several thousand candidates adds tens of ms at most.
        has_date_filter = bool(date_from or date_to)
        fetch_k_target = max(top_k * 10, 2000) if has_date_filter else top_k * 10
        fetch_k = min(fetch_k_target, len(self.searcher.embedding_ids))

        if candidate_ids is None:
            embedding_ids, scores = self.searcher.search(query_vector, top_k=fetch_k)
        else:
            # Structured filter active — vector search runs only over the
            # pre-filtered candidate set. Small N → brute-force cosine;
            # large N → ScaNN with overfetch + filter.
            embedding_ids, scores = self._restricted_vector_search(query_vector, candidate_ids, fetch_k)

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
        # BM25 with expanded query (includes alias terms) + original query.
        # Bump the FTS limit when a date filter is present — just like
        # ScaNN, the top-200 matches across all time might all fall
        # outside a tight window. When `candidate_ids` is set, BM25 runs
        # only within that restricted corpus (SQL-level `id = ANY(...)`),
        # so the date-window SQL post-filter below becomes redundant.
        bm25_limit = 2000 if has_date_filter else 200
        bm25_scores = search_fts(conn, pq.text or expanded_query, limit=bm25_limit, candidate_ids=candidate_ids)
        if expanded_query.lower() != query.lower():
            original_pq = parse_query(query)
            original_bm25 = search_fts(conn, original_pq.text or query, limit=bm25_limit, candidate_ids=candidate_ids)
            for mid, score in original_bm25.items():
                if mid not in bm25_scores or score > bm25_scores[mid]:
                    bm25_scores[mid] = score

        # Merge BM25-only results: threads that BM25 found but ScaNN missed
        scann_message_ids = set()
        for tdata in threads.values():
            scann_message_ids.update(tdata["matches"].keys())

        bm25_only_ids = [mid for mid in bm25_scores if mid not in scann_message_ids]
        if bm25_only_ids:
            placeholders = ",".join(["%s"] * len(bm25_only_ids))
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
        placeholders = ",".join(["%s"] * len(thread_ids))
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

            # Structured filters (from:/to:/subject:/date/has_attachment)
            # are now enforced by the candidate pre-filter above, not a
            # thread-level post-filter. Every match this thread contains
            # is guaranteed to satisfy every structured predicate.

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

        # Date range + has:attachment post-filters are retired: the
        # candidate pre-filter restricts the corpus at the message level,
        # so every surviving match is already in-window / has an attachment.
        # Previously this was a post-filter on top of ScaNN+BM25 results.

        if sort == "recent":
            thread_results.sort(key=lambda t: t.date_last, reverse=True)
        else:
            thread_results.sort(key=lambda t: t.score, reverse=True)

        # Collapse repeat senders
        thread_results = self._collapse_repeat_senders(thread_results, top_k * 2)

        # LLM reranker — only when ranking is uncertain (top scores are close)
        rerank = self.config.get("search", {}).get("rerank", True)
        if rerank and sort == "relevance" and len(thread_results) > 3:
            top_scores = [t.score for t in thread_results[:5]]
            score_spread = max(top_scores) - min(top_scores) if len(top_scores) >= 2 else 1.0
            if score_spread < 0.05:
                # Top results are tightly clustered — reranker will help
                logger.info(f"Reranking: top-5 spread={score_spread:.3f} (tight)")
                thread_results = self._llm_rerank(pq.text or query, thread_results, top_k)
            else:
                logger.info(f"Skipping rerank: top-5 spread={score_spread:.3f} (clear winner)")

        # Off-topic filter: drop results with large score gap from #1
        if filter_offtopic and len(thread_results) >= 2:
            thread_results = self._filter_offtopic(thread_results)

        return thread_results[:top_k]

    @staticmethod
    def _filter_offtopic(results: list[ThreadResult]) -> list[ThreadResult]:
        """Drop results that are clearly off-topic based on score gap from best result.

        Uses adaptive threshold: keeps results within 60% of the best score,
        but always keeps at least 3 results.
        """
        if not results:
            return results

        best_score = results[0].score
        if best_score <= 0:
            return results

        threshold = best_score * 0.4  # drop anything below 40% of best
        filtered = [r for r in results if r.score >= threshold]

        # Always return at least 3 results
        if len(filtered) < 3:
            return results[: max(3, len(filtered))]

        logger.info(f"Off-topic filter: {len(results)} -> {len(filtered)} (threshold={threshold:.3f})")
        return filtered

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
