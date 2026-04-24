import re
from pathlib import Path

# The canonical schema now lives in `pg_schema.sql`. The SQLite `SCHEMA`
# string constant — and its inline DDL branch below — was removed during
# the 2026-04-20 Stage 2 cleanup when the SQLite backend was retired.
_PG_SCHEMA_PATH = Path(__file__).parent / "pg_schema.sql"


def _read_pg_schema() -> str:
    """Load the Postgres DDL as text for table-name introspection."""
    return _PG_SCHEMA_PATH.read_text()


# ─────────────────────────────────────────────────────────────────────
# TABLE_DOCS — human-readable descriptions for every table the LLM may
# query. Kept next to the schema so they don't drift. Surfaced to the
# model via /api/sql_schema and inlined into the system prompt for the
# sql_query tool. When you add or change a table in pg_schema.sql, add
# or update its entry below.
# ─────────────────────────────────────────────────────────────────────
TABLE_DOCS: dict[str, str] = {
    "messages": (
        "One row per email message. Primary content store.\n"
        "- id (TEXT PK): Gmail message ID. Use as cite_ref source via thread_id.\n"
        "- thread_id (TEXT): Gmail thread this message belongs to.\n"
        "- from_addr (TEXT): 'Name <email>' format — use LIKE '%@domain%' to match domains.\n"
        "- to_addr (TEXT): comma-separated 'Name <email>' recipients.\n"
        "- subject (TEXT): often starts with 'Re:' or 'Fwd:'.\n"
        "- body_text (TEXT): plain-text body.\n"
        "- body_html (TEXT): HTML body (rarely needed).\n"
        "- date (TEXT): ISO 8601 UTC, e.g. '2026-04-17T14:32:00+00:00'. Sort and compare as strings.\n"
        '- labels (TEXT): JSON array of Gmail labels, e.g. \'["INBOX","IMPORTANT"]\'. Use LIKE \'%"UNREAD"%\'.\n'
        "- history_id (INT): Gmail incremental-sync watermark."
    ),
    "thread_summary": (
        "Precomputed per-thread metadata — fastest way to enumerate threads without scanning messages.\n"
        "- thread_id (TEXT PK).\n"
        "- subject (TEXT): subject of the first message in the thread.\n"
        "- participants (TEXT): JSON array of all 'Name <email>' strings ever on the thread.\n"
        "- all_from_addrs (TEXT): JSON array of distinct senders.\n"
        "- all_labels (TEXT): JSON array of every label any message in the thread has.\n"
        "- message_count (INT): how many messages in the thread.\n"
        "- date_first / date_last (TEXT): ISO UTC of earliest/latest message."
    ),
    "attachments": (
        "One row per attachment. Linked via message_id.\n"
        "- id (INT PK).\n"
        "- message_id (TEXT FK -> messages.id).\n"
        "- filename (TEXT).\n"
        "- mime_type (TEXT): e.g. 'application/pdf', 'image/png'.\n"
        "- size_bytes (INT).\n"
        "- extracted_text (TEXT): parsed text (PDF/DOCX/TXT). NULL if image-only or extraction failed.\n"
        "- image_path (TEXT): on-disk path to extracted image, NULL if non-image.\n"
        "- raw_path (TEXT): on-disk path to original raw bytes."
    ),
    "topics": (
        "Hierarchical clustering of messages by embedding similarity.\n"
        "- topic_id (TEXT PK): dotted path like 'root.0.1.0.1' — depth = number of dots.\n"
        "- parent_id (TEXT): NULL for root.\n"
        "- label (TEXT): short human label (auto-generated).\n"
        "- depth (INT).\n"
        "- message_count (INT): how many messages fall under this topic (including descendants).\n"
        "- top_senders, sample_subjects (TEXT): JSON arrays."
    ),
    "message_topics": (
        "Many-to-many between messages and leaf topics.\n" "- message_id (TEXT FK), topic_id (TEXT FK)."
    ),
    "contact_frequency": (
        "Per-email-address rollup of how often a contact appears as sender.\n"
        "- email (TEXT PK): bare email address (no display name).\n"
        "- message_count (INT): number of messages from this address.\n"
        "- score (REAL): popularity score for ranking auto-suggest results."
    ),
    "message_summaries": (
        "Local-LLM-generated 1-3 sentence summary per message. Backfilled by `gmail-search summarize`.\n"
        "- message_id (TEXT PK FK).\n"
        "- summary (TEXT): the actual summary.\n"
        "- model (TEXT): which model produced it (e.g. 'qwen2.5:7b').\n"
        "- created_at (TEXT)."
    ),
    "summary_failures": (
        "Messages where the summarizer currently fails. Row written on each failure,\n"
        "deleted on the next successful attempt — so this is a live 'what's broken now' list,\n"
        "not an append-only log. Use to triage backend outages, prompt bugs, or pathological inputs.\n"
        "- message_id (TEXT PK FK).\n"
        "- model (TEXT): model key at time of last failure.\n"
        "- error (TEXT): short error message (first 400 chars).\n"
        "- attempts (INT): cumulative failure count.\n"
        "- first_seen / last_seen (TIMESTAMPTZ)."
    ),
    "model_battles": (
        "User vote results from A/B model comparisons in chat.\n"
        "- id (INT PK), question (TEXT), variant_a/variant_b (TEXT JSON: model + thinking level).\n"
        "- winner: 'a' | 'b' | 'tie' | 'both_bad'.\n"
        "- request_id_a / request_id_b (TEXT): chat-log IDs for audit trail.\n"
        "- created_at (TEXT)."
    ),
    "conversations": (
        "User-facing chat conversations (sidebar list).\n"
        "- id (TEXT PK), title (TEXT, may be NULL until auto-titled), created_at, updated_at."
    ),
    "conversation_messages": (
        "Persisted chat history per conversation.\n"
        "- id (INT PK), conversation_id (FK), seq (INT, ordered), role ('user'|'assistant'),\n"
        "  parts (TEXT JSON: AI SDK message parts), created_at."
    ),
    "costs": (
        "Per-API-call cost log for embeddings + summarization.\n"
        "- id, timestamp, operation, model, input_tokens, image_count, output_tokens,\n"
        "  estimated_cost_usd, message_id."
    ),
    "term_aliases": (
        "Spell correction + personal-abbreviation expansion dictionary.\n"
        "- term (TEXT PK): user's typed term.\n"
        "- expansions (TEXT JSON): list of corrected/expanded terms.\n"
        "- similarity (REAL): how confident the expansion is."
    ),
    "job_progress": (
        "Live status of long-running jobs (sync, watch, update). Used by /api/status.\n"
        "- job_id (TEXT PK), stage, status, total, completed, detail, started_at, updated_at, pid."
    ),
    # Tables intentionally NOT documented for the LLM:
    # - embeddings: huge BLOB column, never query directly (use search_emails).
    # - sync_state: internal kv store.
    # - query_cache: internal embedding cache.
}


# Tables that exist in pg_schema.sql but should not appear in the
# LLM-facing schema. Either huge BLOB stores or internal kv/cache tables.
# Keep this in sync with the "intentionally NOT documented" comment in
# TABLE_DOCS above.
_INTERNAL_TABLES = {
    "embeddings",
    "sync_state",
    "query_cache",
    # scann_index_pointer is a single-row KV tracking the active ScaNN
    # index directory. Infrastructure, not something an LLM should query.
    "scann_index_pointer",
    # Deep-analysis agent state. The chat LLM has no business querying
    # its own session log; if anything it'd invite recursive confusion.
    # See docs/DEEP_ANALYSIS_AGENT.md.
    "agent_sessions",
    "agent_events",
    "agent_artifacts",
}

_SCHEMA_TABLE_RE = re.compile(
    r"CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+(\w+)",
    re.IGNORECASE,
)


def _schema_table_names() -> set[str]:
    return {m.group(1) for m in _SCHEMA_TABLE_RE.finditer(_read_pg_schema())}


def assert_table_docs_cover_schema() -> None:
    """Fail fast if a developer adds a table to pg_schema.sql without docs.

    Called once at server startup so a missing entry in TABLE_DOCS becomes a
    visible error, not a silent omission from the LLM-facing schema.
    """
    schema_tables = _schema_table_names() - _INTERNAL_TABLES
    documented = set(TABLE_DOCS.keys())
    missing = schema_tables - documented
    extra = documented - schema_tables
    problems = []
    if missing:
        problems.append(f"missing TABLE_DOCS entries for: {sorted(missing)}")
    if extra:
        problems.append(f"TABLE_DOCS has stale entries (no matching CREATE TABLE): {sorted(extra)}")
    if problems:
        raise RuntimeError("schema/docs drift in db.py — " + "; ".join(problems))


def describe_schema_for_llm() -> str:
    """Return a Markdown summary of every table the LLM may query."""
    parts = []
    for name, doc in TABLE_DOCS.items():
        parts.append(f"### {name}\n{doc}")
    return "\n\n".join(parts)


def init_db(db_path: Path) -> None:
    """Create the PG DB schema if it doesn't exist.

    Loads `pg_schema.sql` — the PG DDL is idempotent via `IF NOT EXISTS`,
    so calling `init_db` against an already-initialized PG is a no-op.
    The `db_path` argument is ignored (kept for signature compatibility
    with the ~60 existing call sites); the PG DSN is resolved via
    `_pg_dsn()`.
    """
    _init_db_pg()


def _init_db_pg() -> None:
    """Load the hand-translated PG schema shipped at
    `gmail_search/store/pg_schema.sql`. Uses a fresh psycopg
    connection (no wrapper) because the schema script contains
    `$$`-quoted function bodies and other raw-SQL idioms.

    Serialized via a Postgres transaction-scoped advisory lock so
    concurrent CLI starts (supervisor + daemons + server all call
    `init_db` at boot) don't deadlock on `ALTER TABLE … ADD COLUMN
    IF NOT EXISTS` while other sessions hold row-level locks on the
    same tables for normal writes (heartbeats, upserts). The lock
    key is a fixed sentinel — no risk of collision with anything
    else in the app.
    """
    import psycopg

    _INIT_DB_LOCK_KEY = 0x676D_7373_6368_6D61  # "gmsschma" (gmail-search schema)

    # The advisory lock serializes concurrent `init_db` calls (supervisor
    # + daemons + server all hit this at boot) — but once inside, the
    # schema DDL takes ACCESS EXCLUSIVE on tables that are constantly
    # being written by the running daemons (heartbeats, upserts, etc).
    # Without a lock_timeout, the DDL parks forever and a concurrent
    # write → deadlock. Prefer "schema init aborted, try again" over
    # "daemon crashes on start-up". The schema is idempotent, so a
    # transient skip is safe.
    with psycopg.connect(_pg_dsn()) as raw:
        with raw.cursor() as cur:
            cur.execute("SET lock_timeout = '3s'")
            cur.execute("SELECT pg_advisory_xact_lock(%s)", (_INIT_DB_LOCK_KEY,))
            try:
                cur.execute(_PG_SCHEMA_PATH.read_text())
                raw.commit()
            except (psycopg.errors.LockNotAvailable, psycopg.errors.DeadlockDetected):
                raw.rollback()
                # Schema is idempotent — if we can't grab the locks now,
                # the schema is presumably already up to date (or another
                # caller will finish it). Skip silently so the daemon
                # comes up instead of crash-looping on start.
                return


def rebuild_thread_summary(db_path: Path) -> int:
    """Precompute thread metadata for fast search ranking. Returns thread count."""
    import json

    conn = get_connection(db_path)

    conn.execute("DELETE FROM thread_summary")

    rows = conn.execute(
        """SELECT thread_id, from_addr, date, labels, subject
           FROM messages ORDER BY date"""
    ).fetchall()

    threads: dict[str, dict] = {}
    for r in rows:
        tid = r["thread_id"]
        if tid not in threads:
            threads[tid] = {
                "subject": r["subject"],
                "from_addrs": [],
                "all_labels": set(),
                "dates": [],
            }
        t = threads[tid]
        t["from_addrs"].append(r["from_addr"])
        t["dates"].append(r["date"])
        for label in json.loads(r["labels"]):
            t["all_labels"].add(label)

    for tid, t in threads.items():
        participants = list(dict.fromkeys(t["from_addrs"]))  # ordered unique
        conn.execute(
            """INSERT INTO thread_summary
               (thread_id, subject, participants, all_from_addrs, all_labels,
                message_count, date_first, date_last)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                tid,
                t["subject"],
                json.dumps(participants),
                json.dumps(t["from_addrs"]),
                json.dumps(sorted(t["all_labels"])),
                len(t["dates"]),
                t["dates"][0],
                t["dates"][-1],
            ),
        )

    conn.commit()
    count = len(threads)
    conn.close()
    return count


def _load_message_embeddings(conn, limit=50000):
    """Load message embeddings with metadata for clustering."""
    import struct

    import numpy as np

    rows = conn.execute(
        """SELECT e.message_id, e.embedding, m.subject, m.from_addr
           FROM embeddings e JOIN messages m ON e.message_id = m.id
           WHERE e.chunk_type = 'message'
           ORDER BY m.date DESC LIMIT %s""",
        (limit,),
    ).fetchall()

    return {
        "msg_ids": [r["message_id"] for r in rows],
        "subjects": [r["subject"] for r in rows],
        "senders": [r["from_addr"] for r in rows],
        "vectors": np.array(
            [list(struct.unpack("3072f", r["embedding"])) for r in rows],
            dtype=np.float32,
        ),
    }


def _bisect(vectors, indices, n_iterations=15, seed=42):
    """Split a set of vectors into 2 clusters. Returns (left_indices, right_indices)."""
    import numpy as np

    if len(indices) < 4:
        return indices, np.array([], dtype=int)

    subset = vectors[indices]
    rng = np.random.RandomState(seed)

    # Pick two initial centroids far apart (k-means++ for k=2)
    c0 = subset[rng.randint(len(subset))]
    dists = np.sum((subset - c0) ** 2, axis=1)
    c1 = subset[np.argmax(dists)]
    centroids = np.array([c0, c1])

    for _ in range(n_iterations):
        sims = subset @ centroids.T
        labels = np.argmax(sims, axis=1)
        for k in range(2):
            mask = labels == k
            if mask.sum() > 0:
                centroids[k] = subset[mask].mean(axis=0)

    sims = subset @ centroids.T
    labels = np.argmax(sims, axis=1)
    return indices[labels == 0], indices[labels == 1]


def _cluster_coherence(vectors, indices):
    """Measure how tight a cluster is (0=scattered, 1=identical). Uses avg cosine sim to centroid."""
    import numpy as np

    if len(indices) < 2:
        return 1.0
    subset = vectors[indices]
    centroid = subset.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm < 1e-10:
        return 0.0
    centroid = centroid / norm
    norms = np.linalg.norm(subset, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    sims = (subset / norms) @ centroid
    return float(sims.mean())


def _extract_sender_name(from_addr: str) -> str:
    """Extract display name from 'Name <email>' format."""
    return from_addr.split("<")[0].strip().strip('"')[:30]


def _summarize_cluster(indices, subjects, senders):
    """Compute top senders and sample subjects for a cluster."""
    from collections import Counter

    sender_names = [_extract_sender_name(senders[i]) for i in indices]
    top_senders = [name for name, _ in Counter(sender_names).most_common(5)]
    sample_subjects = [subjects[i][:80] for i in indices[:8]]
    return top_senders, sample_subjects


def _build_topic_tree(vectors, indices, subjects, senders, topic_id, parent_id, depth, max_depth, min_cluster_size):
    """Recursively bisect clusters to build a topic hierarchy.

    Returns a list of (topic_id, parent_id, depth, indices) tuples.
    Each message appears in its leaf node AND all ancestor nodes.
    """
    nodes = [(topic_id, parent_id, depth, indices)]

    # Stop splitting if: too small, too deep, or too coherent
    if len(indices) < min_cluster_size * 2 or depth >= max_depth:
        return nodes

    coherence = _cluster_coherence(vectors, indices)
    if coherence > 0.85:
        return nodes

    left, right = _bisect(vectors, indices, seed=42 + depth * 7)

    if len(left) < min_cluster_size or len(right) < min_cluster_size:
        return nodes  # split too uneven, keep as leaf

    left_id = f"{topic_id}.0"
    right_id = f"{topic_id}.1"

    left_nodes = _build_topic_tree(
        vectors, left, subjects, senders, left_id, topic_id, depth + 1, max_depth, min_cluster_size
    )
    right_nodes = _build_topic_tree(
        vectors, right, subjects, senders, right_id, topic_id, depth + 1, max_depth, min_cluster_size
    )

    return nodes + left_nodes + right_nodes


def _auto_label_topics(conn):
    """Use Gemini Flash Lite to generate short topic labels from cluster summaries."""
    import json
    import logging

    logger = logging.getLogger(__name__)
    topic_rows = conn.execute(
        "SELECT topic_id, parent_id, depth, top_senders, sample_subjects, message_count FROM topics ORDER BY depth, message_count DESC"
    ).fetchall()

    summaries = []
    for t in topic_rows:
        sndrs = json.loads(t["top_senders"])
        subjs = json.loads(t["sample_subjects"])
        parent_info = f" (child of {t['parent_id']})" if t["parent_id"] else ""
        summaries.append(
            f"Node {t['topic_id']}{parent_info} (depth={t['depth']}, {t['message_count']} msgs): "
            f"Senders: {', '.join(sndrs[:3])}. "
            f"Subjects: {'; '.join(subjs[:4])}"
        )

    try:
        import os

        from google import genai

        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key) if api_key else genai.Client()

        response = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=(
                "Label each node in this email topic hierarchy with a short 2-4 word name. "
                "Parent nodes should have broader names, children should be more specific. "
                "Return ONLY a JSON object mapping node ID to label. "
                'Example: {"root": "All Email", "root.0": "Work", "root.0.0": "Projects"}\n\n' + "\n".join(summaries)
            ),
        )
        text = response.text.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        label_map = json.loads(text)

        for tid, label in label_map.items():
            conn.execute("UPDATE topics SET label = %s WHERE topic_id = %s", (label, tid))
        conn.commit()
        logger.info(f"Auto-labeled {len(label_map)} topic nodes")

    except Exception as e:
        logger.warning(f"Auto-labeling failed, using top sender as label: {e}")
        for t in topic_rows:
            sndrs = json.loads(t["top_senders"])
            label = sndrs[0] if sndrs else f"Topic {t['topic_id']}"
            conn.execute("UPDATE topics SET label = %s WHERE topic_id = %s", (label, t["topic_id"]))
        conn.commit()


def rebuild_topics(db_path: Path, max_depth: int = 4, min_cluster_size: int = 50) -> int:
    """Build hierarchical topic tree using recursive bisecting k-means. Returns node count."""
    import json
    import logging

    import numpy as np

    logger = logging.getLogger(__name__)

    conn = get_connection(db_path)

    data = _load_message_embeddings(conn)
    n_msgs = len(data["msg_ids"])
    if n_msgs < min_cluster_size * 2:
        logger.warning(f"Not enough messages ({n_msgs}) for topic hierarchy")
        conn.close()
        return 0

    logger.info(f"Building topic hierarchy from {n_msgs} messages (max_depth={max_depth})")

    all_indices = np.arange(n_msgs)
    nodes = _build_topic_tree(
        data["vectors"],
        all_indices,
        data["subjects"],
        data["senders"],
        topic_id="root",
        parent_id=None,
        depth=0,
        max_depth=max_depth,
        min_cluster_size=min_cluster_size,
    )

    conn.execute("DELETE FROM message_topics")
    conn.execute("DELETE FROM topics")

    for topic_id, parent_id, depth, indices in nodes:
        top_senders, sample_subjects = _summarize_cluster(indices, data["subjects"], data["senders"])
        conn.execute(
            "INSERT INTO topics (topic_id, parent_id, label, depth, message_count, top_senders, sample_subjects) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (topic_id, parent_id, "", depth, len(indices), json.dumps(top_senders), json.dumps(sample_subjects)),
        )
        # Messages belong to their leaf node and all ancestors
        for idx in indices:
            conn.execute(
                "INSERT INTO message_topics (message_id, topic_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (data["msg_ids"][idx], topic_id),
            )

    conn.commit()
    logger.info(f"Built {len(nodes)} topic nodes")

    _auto_label_topics(conn)

    count = conn.execute("SELECT COUNT(*) FROM topics").fetchone()[0]
    conn.close()
    return count


def _extract_terms_from_messages(conn):
    """Build a reverse index: lowercase term → set of message indices."""
    import re

    rows = conn.execute("SELECT id, subject, body_text, from_addr FROM messages").fetchall()

    term_to_msgs: dict[str, set[int]] = {}
    abbrev_to_msgs: dict[str, set[int]] = {}
    msg_id_to_idx: dict[str, int] = {}

    for idx, r in enumerate(rows):
        msg_id_to_idx[r["id"]] = idx
        text = f"{r['subject']} {r['body_text']} {r['from_addr']}"

        # Single words (lowercased)
        word_list = re.findall(r"[a-zA-Z]{2,}", text.lower())
        for w in set(word_list):
            if w not in term_to_msgs:
                term_to_msgs[w] = set()
            term_to_msgs[w].add(idx)

        # Bigrams and trigrams (multi-word phrases like "kol emeth", "frank rimerman")
        for ngram_size in (2, 3):
            for i in range(len(word_list) - ngram_size + 1):
                ngram = " ".join(word_list[i : i + ngram_size])
                if len(ngram) > 5:  # only useful as expansion if longer than abbreviation
                    if ngram not in term_to_msgs:
                        term_to_msgs[ngram] = set()
                    term_to_msgs[ngram].add(idx)

        # Uppercase abbreviations (2-5 chars, all caps) as alias candidates
        raw_words = re.findall(r"\b[A-Z]{2,5}\b", text)
        for w in raw_words:
            key = w.lower()
            if key not in abbrev_to_msgs:
                abbrev_to_msgs[key] = set()
            abbrev_to_msgs[key].add(idx)

    return term_to_msgs, abbrev_to_msgs, msg_id_to_idx, [r["id"] for r in rows]


def _compute_term_centroid(vectors, indices):
    """Average the embedding vectors at the given indices."""
    import numpy as np

    subset = vectors[list(indices)]
    centroid = subset.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 1e-10:
        centroid = centroid / norm
    return centroid


def _find_nearest_terms(query_centroid, candidate_centroids, candidate_terms, top_k=5, min_similarity=0.7):
    """Find the most similar terms to a query centroid by cosine similarity."""
    import numpy as np

    if not candidate_terms:
        return []

    centroids_matrix = np.array(candidate_centroids, dtype=np.float32)
    sims = centroids_matrix @ query_centroid
    top_indices = np.argsort(sims)[::-1][:top_k]

    results = []
    for i in top_indices:
        if sims[i] >= min_similarity:
            results.append((candidate_terms[i], float(sims[i])))
    return results


def rebuild_term_aliases(db_path: Path, min_term_len=2, max_term_len=5, min_occurrences=3, min_similarity=0.75) -> int:
    """Discover term aliases using embedding nearest neighbors.

    For short terms (likely abbreviations), finds longer terms whose message
    embeddings cluster nearby — indicating they refer to the same concept.
    """
    import json
    import logging
    import struct

    import numpy as np

    logger = logging.getLogger(__name__)

    conn = get_connection(db_path)

    # Load embeddings
    emb_rows = conn.execute("SELECT message_id, embedding FROM embeddings WHERE chunk_type = 'message'").fetchall()
    if not emb_rows:
        conn.close()
        return 0

    emb_by_msg: dict[str, int] = {}
    vectors_list = []
    for r in emb_rows:
        emb_by_msg[r["message_id"]] = len(vectors_list)
        vectors_list.append(list(struct.unpack("3072f", r["embedding"])))

    vectors = np.array(vectors_list, dtype=np.float32)
    logger.info(f"Loaded {len(vectors)} embeddings for alias discovery")

    # Build term → message index mapping (all terms + uppercase abbreviations)
    term_to_msgs, abbrev_to_msgs, _, msg_ids = _extract_terms_from_messages(conn)

    def _remap_to_emb_indices(term_map):
        result = {}
        for term, msg_indices in term_map.items():
            emb_indices = [emb_by_msg[msg_ids[mi]] for mi in msg_indices if msg_ids[mi] in emb_by_msg]
            if len(emb_indices) >= min_occurrences:
                result[term] = emb_indices
        return result

    # Short terms: only uppercase abbreviations (KE, HOA, IRS — not "the", "and")
    short_terms = _remap_to_emb_indices(abbrev_to_msgs)
    # Long terms: all words > 5 chars
    long_terms = {t: idxs for t, idxs in _remap_to_emb_indices(term_to_msgs).items() if len(t) > max_term_len}

    logger.info(f"Abbreviations (alias candidates): {len(short_terms)}")
    logger.info(f"Long terms (expansion candidates): {len(long_terms)}")

    if not short_terms or not long_terms:
        conn.close()
        return 0

    # Build reverse index: message_index → set of long terms in that message
    # This lets us find co-occurring long terms without checking all 36k x 36k pairs
    logger.info("Building reverse index for co-occurrence...")
    msg_to_long_terms: dict[int, set[str]] = {}
    for long_term, indices in long_terms.items():
        for idx in indices:
            if idx not in msg_to_long_terms:
                msg_to_long_terms[idx] = set()
            msg_to_long_terms[idx].add(long_term)

    # For each short term, find long terms that co-occur via the reverse index
    logger.info("Discovering aliases via co-occurrence + embedding similarity...")
    conn.execute("DELETE FROM term_aliases")
    alias_count = 0
    from collections import Counter

    for term, short_emb_indices in short_terms.items():
        short_set = set(short_emb_indices)

        # Count co-occurring long terms via reverse index (fast)
        cooccur_counts: Counter = Counter()
        for idx in short_emb_indices:
            for long_term in msg_to_long_terms.get(idx, set()):
                if term not in long_term and long_term not in term:
                    cooccur_counts[long_term] += 1

        # Keep long terms with 5+ co-occurrences and strong Jaccard
        candidates = []
        for long_term, overlap_count in cooccur_counts.most_common(20):
            if overlap_count < 5:
                break

            long_set = set(long_terms[long_term])
            jaccard = overlap_count / len(short_set | long_set)
            if jaccard < 0.2:
                continue

            candidates.append((long_term, jaccard))

        # Keep top 3 by co-occurrence strength
        expansions = candidates[:3]
        if expansions:
            conn.execute(
                "INSERT INTO term_aliases (term, expansions, similarity) VALUES (%s, %s, %s)",
                (term, json.dumps([t for t, _ in expansions]), expansions[0][1]),
            )
            alias_count += 1

    conn.commit()
    logger.info(f"Discovered {alias_count} candidate aliases, validating with Gemini...")

    _validate_aliases_with_llm(conn)

    final_count = conn.execute("SELECT COUNT(*) FROM term_aliases").fetchone()[0]
    conn.close()
    logger.info(f"Validated {final_count} term aliases")
    return final_count


def _validate_aliases_with_llm(conn):
    """Use Gemini to filter out noise from candidate aliases.

    Sends all candidates in batches and removes ones Gemini marks as bad.
    """
    import json
    import logging
    import os

    logger = logging.getLogger(__name__)

    rows = conn.execute("SELECT term, expansions, similarity FROM term_aliases").fetchall()
    if not rows:
        return

    # Build the sample for Gemini with co-occurrence strength
    candidates = []
    for r in rows:
        exps = json.loads(r["expansions"])
        candidates.append(f"{r['term']} → {', '.join(exps[:2])} (co-occurrence: {r['similarity']:.0%})")

    # Process in chunks of 100 (Gemini can handle this in one call)
    try:
        from google import genai

        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key) if api_key else genai.Client()

        bad_terms = set()
        for i in range(0, len(candidates), 100):
            chunk = candidates[i : i + 100]
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite-preview",
                contents=(
                    "These are candidate abbreviation→expansion pairs mined from a personal email corpus. "
                    "The co-occurrence percentage shows how often they appear in the same emails. "
                    "Score each from 1 to 5 on how likely this is a REAL abbreviation or alias:\n"
                    "  5 = definitely real (e.g., NY→New York, GCP→Google Cloud)\n"
                    "  4 = very likely real (e.g., personal/org abbreviation)\n"
                    "  3 = possibly real, unclear\n"
                    "  2 = probably noise\n"
                    "  1 = definitely noise (encoding artifacts, random words)\n"
                    'Return ONLY a JSON object: {"abbreviation": score}\n\n' + "\n".join(chunk)
                ),
            )
            text = response.text.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            scores = json.loads(text)
            for term, score in scores.items():
                try:
                    if int(score) < 2:
                        bad_terms.add(term.lower())
                except (ValueError, TypeError):
                    pass

        # Delete low-scoring aliases, but keep high-Jaccard ones regardless
        # (strong co-occurrence is a reliable signal even if LLM is unsure)
        if bad_terms:
            protected = set()
            for term in bad_terms:
                row = conn.execute("SELECT similarity FROM term_aliases WHERE term = %s", (term,)).fetchone()
                if row and row["similarity"] >= 0.25:
                    protected.add(term)

            removable = bad_terms - protected
            for term in removable:
                conn.execute("DELETE FROM term_aliases WHERE term = %s", (term,))
            conn.commit()
            logger.info(
                f"Removed {len(removable)} noise aliases via LLM validation"
                f" (protected {len(protected)} high-Jaccard aliases)"
            )

    except Exception as e:
        logger.warning(f"LLM alias validation failed, keeping all candidates: {e}")


def rebuild_spell_dictionary(db_path: Path, data_dir: Path) -> int:
    """Build a word frequency dictionary from the email corpus for spell correction."""
    import re
    from collections import Counter

    conn = get_connection(db_path)

    word_counts: Counter = Counter()

    # Extract words from subjects and body text
    rows = conn.execute("SELECT subject, body_text FROM messages").fetchall()
    for r in rows:
        for field in [r["subject"], r["body_text"]]:
            if field:
                words = re.findall(r"[a-zA-Z]+", field.lower())
                word_counts.update(w for w in words if len(w) >= 2)

    # Also extract from sender names
    rows = conn.execute("SELECT DISTINCT from_addr FROM messages").fetchall()
    for r in rows:
        addr = r["from_addr"]
        # Extract name part from "Name <email>"
        if "<" in addr:
            name = addr.split("<")[0].strip().strip('"')
        else:
            name = addr.split("@")[0]
        words = re.findall(r"[a-zA-Z]+", name.lower())
        # Boost names heavily so they're preferred corrections
        word_counts.update({w: 50 for w in words if len(w) >= 2})

    conn.close()

    # Write dictionary file (word\tfrequency format for SymSpell)
    dict_path = data_dir / "spell_dictionary.txt"
    with open(dict_path, "w") as f:
        for word, count in word_counts.most_common():
            f.write(f"{word} {count}\n")

    return len(word_counts)


def rebuild_contact_frequency(db_path: Path) -> int:
    """Precompute contact frequency scores. Returns contact count."""
    conn = get_connection(db_path)

    conn.execute("DELETE FROM contact_frequency")

    # Count messages per sender email
    rows = conn.execute("SELECT from_addr, COUNT(*) as c FROM messages GROUP BY from_addr ORDER BY c DESC").fetchall()

    if not rows:
        conn.close()
        return 0

    # Log-scale normalize: top sender = 1.0
    import math

    max_count = rows[0]["c"]
    log_max = math.log(max_count + 1)

    for r in rows:
        addr = r["from_addr"].lower()
        # Extract just the email from "Name <email>" format
        if "<" in addr:
            addr = addr.split("<")[1].rstrip(">")
        score = math.log(r["c"] + 1) / log_max if log_max > 0 else 0.0
        conn.execute(
            """INSERT INTO contact_frequency (email, message_count, score)
               VALUES (%s, %s, %s)
               ON CONFLICT(email) DO UPDATE SET
                 message_count = excluded.message_count,
                 score = excluded.score""",
            (addr, r["c"], score),
        )

    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM contact_frequency").fetchone()[0]
    conn.close()
    return count


class JobProgress:
    """Track progress of a long-running job via the DB (queryable from other processes)."""

    def __init__(self, db_path: Path, job_id: str, start_completed: int = 0):
        """`start_completed` is the baseline used for rate/ETA math:
        how many units the job had already completed before this run
        began. For backfill it's the existing corpus size; for sync-style
        jobs it's 0.

        Also records `os.getpid()` into `job_progress.pid` so the
        supervisor and HTTP stop endpoints can signal the right process
        without a pid file on disk.
        """
        self.db_path = db_path
        self.job_id = job_id
        import os as _os
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        my_pid = _os.getpid()
        conn = get_connection(db_path)
        conn.execute(
            """INSERT INTO job_progress
                 (job_id, stage, status, total, completed, start_completed, detail, started_at, updated_at, pid)
               VALUES (%s, '', 'running', 0, %s, %s, '', %s, %s, %s)
               ON CONFLICT(job_id) DO UPDATE SET
                 stage = excluded.stage,
                 status = excluded.status,
                 total = excluded.total,
                 completed = excluded.completed,
                 start_completed = excluded.start_completed,
                 detail = excluded.detail,
                 started_at = excluded.started_at,
                 updated_at = excluded.updated_at,
                 pid = excluded.pid""",
            (job_id, start_completed, start_completed, now, now, my_pid),
        )
        conn.commit()
        conn.close()

    def update(self, stage: str, completed: int, total: int, detail: str = ""):
        from datetime import datetime, timezone

        conn = get_connection(self.db_path)
        conn.execute(
            "UPDATE job_progress SET stage=%s, completed=%s, total=%s, detail=%s, updated_at=%s WHERE job_id=%s",
            (stage, completed, total, detail, datetime.now(timezone.utc).isoformat(), self.job_id),
        )
        conn.commit()
        conn.close()

    def heartbeat(self):
        """Bump `updated_at = now()` without touching any other column.

        Used by the watch daemon's idle sleep so an idle frontfill
        doesn't look dead to the supervisor / heartbeat-based status
        endpoints. The other two daemons (update, summarize) already
        call `.update()` frequently enough during work to serve the
        same purpose.
        """
        from datetime import datetime, timezone

        conn = get_connection(self.db_path)
        conn.execute(
            "UPDATE job_progress SET updated_at=%s WHERE job_id=%s",
            (datetime.now(timezone.utc).isoformat(), self.job_id),
        )
        conn.commit()
        conn.close()

    def finish(self, status: str = "done", detail: str = ""):
        from datetime import datetime, timezone

        conn = get_connection(self.db_path)
        conn.execute(
            "UPDATE job_progress SET status=%s, detail=%s, updated_at=%s WHERE job_id=%s",
            (status, detail, datetime.now(timezone.utc).isoformat(), self.job_id),
        )
        conn.commit()
        conn.close()

    @staticmethod
    def get(db_path: Path, job_id: str = None) -> dict | list | None:
        conn = get_connection(db_path)
        if job_id:
            row = conn.execute("SELECT * FROM job_progress WHERE job_id=%s", (job_id,)).fetchone()
            conn.close()
            return dict(row) if row else None
        rows = conn.execute("SELECT * FROM job_progress ORDER BY updated_at DESC LIMIT 10").fetchall()
        conn.close()
        return [dict(r) for r in rows]


def reap_stale_jobs(conn, staleness_seconds: int = 600) -> int:
    """Mark any `running` job whose updated_at is older than the
    threshold as `stopped`.

    Used to clean up zombie rows left behind when a worker process gets
    killed (OOM, SIGKILL, …) before it can mark itself done. Without
    this, the UI keeps showing a phantom "syncing…" banner indefinitely.

    Returns the number of rows reaped. Only touches rows with
    status='running' — historical done/stopped/error rows are never
    rewritten.
    """
    from datetime import datetime, timedelta, timezone

    cutoff = (datetime.now(timezone.utc) - timedelta(seconds=staleness_seconds)).isoformat()
    cur = conn.execute(
        "UPDATE job_progress SET status='stopped', detail='stale — process gone' "
        "WHERE status='running' AND updated_at < %s",
        (cutoff,),
    )
    conn.commit()
    return cur.rowcount


def clear_query_cache(db_path: Path) -> int:
    """Clear the query embedding cache. Call after re-embedding or reindexing."""
    conn = get_connection(db_path)
    count = conn.execute("DELETE FROM query_cache").rowcount
    conn.commit()
    conn.close()
    return count


def rebuild_fts(db_path: Path) -> int:
    """Rebuild FTS indexes. On Postgres the BM25 indexes are maintained
    incrementally by pg_search — there's nothing to rebuild, so this is
    effectively a no-op kept for CLI/API compatibility. Returns the
    total number of messages + attachments currently covered.
    """
    conn = get_connection(db_path)
    msg_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    att_count = conn.execute(
        "SELECT COUNT(*) FROM attachments WHERE extracted_text IS NOT NULL AND extracted_text <> ''"
    ).fetchone()[0]
    conn.close()
    return int(msg_count) + int(att_count)


def get_connection(db_path):
    """Return a dict-row Postgres connection via the compatibility shim.

    The function signature stays `(db_path)` for back-compat with the
    ~60 existing call sites. `db_path` is ignored; the PG DSN is read
    from `DB_DSN` (or the local docker-compose default).

    Callers can assume:
        - `conn.execute(sql, params)` — runs the statement. Parameter
          placeholders are psycopg-native `%s`.
        - `conn.commit()` / `conn.close()`.
        - Row access via `row["col"]` or `row.keys()` via psycopg's
          `dict_row`, wrapped in `_CompatRow` so `row[0]` also works.

    The PG wrapper is `_PgConnWrapper` below — a thin shim that adds
    a `Cursor` façade and sqlite3-style conveniences.
    """
    return _connect_pg()


# ── Postgres connection layer ──────────────────────────────────────────────
#
# Thin wrapper around psycopg that preserves the sqlite3-style
# `conn.execute(...)` ergonomics our call sites were written against.
# Placeholders in SQL are psycopg-native `%s`.


def _pg_dsn() -> str:
    """DSN for the PG backend. Reads `DB_DSN` if set; otherwise the
    local docker-compose default from `docker-compose.yml`.
    """
    import os as _os

    return _os.environ.get("DB_DSN") or "postgresql://gmail_search:gmail_search@127.0.0.1:5544/gmail_search"


class _CompatRow:
    """Row type that mimics `sqlite3.Row` semantics — retained so call
    sites can mix dict-style (`row["name"]`) and positional (`row[0]`)
    access. psycopg's `dict_row` only does the former; this shim adds
    positional access via the column list from `cursor.description`.

    Kept intentionally minimal: indexing, `keys()`, iteration over
    values, `len`, `dict(row)` conversion, and a `.get()` helper.
    """

    __slots__ = ("_cols", "_values", "_map")

    def __init__(self, cols, values):
        self._cols = cols
        # Normalize to a tuple so indexing is cheap + immutable.
        self._values = tuple(values)
        self._map = dict(zip(cols, self._values))

    def __getitem__(self, k):
        if isinstance(k, int):
            return self._values[k]
        return self._map[k]

    def __contains__(self, k):
        return k in self._map

    def keys(self):
        return list(self._cols)

    def values(self):
        return list(self._values)

    def items(self):
        return list(self._map.items())

    def get(self, k, default=None):
        return self._map.get(k, default)

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __repr__(self):
        return f"_CompatRow({self._map!r})"


def _compat_row_factory(cursor):
    """psycopg row_factory that produces `_CompatRow` instances.
    Called once per cursor; returns a `make_row(values)` closure.
    """
    desc = cursor.description
    cols = [d.name for d in desc] if desc is not None else []
    return lambda values: _CompatRow(cols, values)


def _connect_pg():
    """Return a `_PgConnWrapper` around a fresh psycopg connection.
    We don't pool here because the existing call sites open/close
    connections freely.
    """
    import psycopg

    raw = psycopg.connect(_pg_dsn(), row_factory=_compat_row_factory)
    return _PgConnWrapper(raw)


class _PgCursorWrapper:
    """Cursor shim: proxies psycopg cursor with a few sqlite3-style
    conveniences (`lastrowid`, chainable `execute`).
    """

    def __init__(self, raw):
        self._raw = raw
        self.lastrowid = None

    def execute(self, sql, params=None):
        if params is None:
            self._raw.execute(sql)
        else:
            self._raw.execute(sql, params)
        return self

    def executemany(self, sql, params_seq):
        self._raw.executemany(sql, params_seq)
        return self

    def fetchone(self):
        return self._raw.fetchone()

    def fetchall(self):
        return self._raw.fetchall()

    def fetchmany(self, size=None):
        return self._raw.fetchmany(size) if size is not None else self._raw.fetchmany()

    def close(self):
        self._raw.close()

    def __iter__(self):
        return iter(self._raw)

    @property
    def rowcount(self):
        return self._raw.rowcount

    @property
    def description(self):
        return self._raw.description


class _PgConnWrapper:
    """Connection shim. Exposes `row_factory` as a no-op (PG uses
    `dict_row` at the connection level) and proxies the rest.
    """

    def __init__(self, raw):
        self._raw = raw
        self.row_factory = None  # no-op; kept for API compat

    def execute(self, sql, params=None):
        cur = self._raw.cursor()
        if params is None:
            cur.execute(sql)
        else:
            cur.execute(sql, params)
        return _PgCursorWrapper(cur)

    def executemany(self, sql, params_seq):
        cur = self._raw.cursor()
        cur.executemany(sql, params_seq)
        return _PgCursorWrapper(cur)

    def cursor(self):
        return _PgCursorWrapper(self._raw.cursor())

    def commit(self):
        self._raw.commit()

    def rollback(self):
        self._raw.rollback()

    def close(self):
        self._raw.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if exc[0] is None:
            self._raw.commit()
        else:
            self._raw.rollback()
        self.close()
        return False
