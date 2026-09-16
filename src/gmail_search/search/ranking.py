"""Pure shared ranking signals and result types; no database or provider access."""
import math
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone


logger = logging.getLogger(__name__)

# Ranking weights — tunable
W_SIMILARITY = 0.40
W_BM25 = 0.15
W_RECENCY = 0.15
W_LABELS = 0.12
W_REPLIED = 0.08
W_MATCH_DENSITY = 0.06
W_THREAD_SIZE = 0.04

# Freshness-of-match bonus, added ON TOP of the normalized 1.0 blend (like the
# contact bonus). Rewards results that are BOTH recent AND a strong match:
# bonus = W_FRESH_MATCH * match_strength * recency, where match_strength is the
# stronger of semantic similarity and an exact subject-phrase hit. Old exact
# matches earn almost nothing (recency has decayed); a freshly-arrived, on-point
# message (e.g. this month's "Draw Request") floats up past deep but stale
# threads instead of being buried by their thread-size / reply signals.
W_FRESH_MATCH = 0.18

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


def _exact_subject_phrase(query: str, subject: str) -> float:
    """1.0 when the full query phrase appears verbatim in the subject, else 0.0.

    This is the one match signal BM25 (`ts_rank`) does NOT specially reward — a
    plain tsquery treats the words as independent lexemes, so a contiguous
    subject-line hit ("draw request" in "Silver May 2026 Draw Request") scores no
    higher than scattered body tokens. Token-level / TF-IDF relevance is left to
    BM25; semantic closeness to similarity. We only add the literal-phrase lift.
    """
    if not query or not subject:
        return 0.0
    q = query.strip().lower()
    return 1.0 if q and q in subject.lower() else 0.0


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



def filter_offtopic(results: list[ThreadResult]) -> list[ThreadResult]:
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

def normalize_subject(subject: str) -> str:
    """Strip Re:/Fwd:/numbers/dates to compare subject similarity."""
    import re

    s = subject.lower().strip()
    s = re.sub(r"^(re|fwd|fw):\s*", "", s)
    s = re.sub(r"\d+", "#", s)  # normalize numbers
    return s.strip()

def collapse_repeat_senders(results: list[ThreadResult], top_k: int) -> list[ThreadResult]:
    """Collapse repeated single-message threads from the same sender with similar subjects.

    If sender X has 3+ single-message threads with similar subjects, keep only
    the top-scoring one. Threads with distinct subjects are preserved.
    """
    from collections import defaultdict

    # Group single-message threads by (sender, normalized_subject)
    group_counts: dict[tuple[str, str], int] = defaultdict(int)
    for t in results:
        if t.message_count == 1 and t.participants:
            key = (t.participants[0].lower(), normalize_subject(t.subject))
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
            key = (t.participants[0].lower(), normalize_subject(t.subject))
            if key in repeat_groups:
                if key in seen_groups:
                    suppressed += 1
                    continue
                seen_groups.add(key)

        collapsed.append(t)

    if suppressed:
        logger.info(f"Collapsed {suppressed} repeat single-message threads")

    return collapsed[:top_k]

