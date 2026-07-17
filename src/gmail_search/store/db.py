import re
from pathlib import Path
from typing import Optional

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
        "One row per email message. Primary content store. ~410k rows.\n"
        "- id (TEXT PK): Gmail message ID. Use as cite_ref source via thread_id.\n"
        "- thread_id (TEXT): Gmail thread this message belongs to.\n"
        "- from_addr (TEXT): 'Name <email>' format. For sender or domain matching, prefer BM25 (below) over LIKE.\n"
        "- to_addr (TEXT): comma-separated 'Name <email>' recipients.\n"
        "- subject (TEXT): often starts with 'Re:' or 'Fwd:'.\n"
        "- body_text (TEXT): plain-text body.\n"
        "- body_html (TEXT): HTML body (rarely needed).\n"
        "- date (TEXT): ISO 8601 UTC, e.g. '2026-04-17T14:32:00+00:00'. Sort and compare as strings.\n"
        '- labels (TEXT): JSON array of Gmail labels, e.g. \'["INBOX","IMPORTANT"]\'. Use LIKE \'%"UNREAD"%\'.\n'
        "- history_id (INT): Gmail incremental-sync watermark.\n"
        "\n"
        "FAST FREE-TEXT SEARCH — use BM25, NOT `LIKE '%...%'`:\n"
        "  `messages_bm25_idx` covers (subject, body_text, from_addr, to_addr).\n"
        "  Syntax: `WHERE id @@@ '<tantivy-query>'`. Tantivy uses `field:term`,\n"
        "  combinable with AND / OR / NOT and parentheses. Order by\n"
        "  `paradedb.score(id) DESC` for relevance ranking.\n"
        "  Examples:\n"
        "    -- Sender + subject keyword:\n"
        "    SELECT id, subject FROM messages\n"
        "    WHERE id @@@ 'from_addr:delta AND subject:cancel'\n"
        "    ORDER BY paradedb.score(id) DESC LIMIT 50;\n"
        "    -- Multi-field OR (any of these fields contains 'invoice'):\n"
        "    SELECT id FROM messages\n"
        "    WHERE id @@@ 'subject:invoice OR body_text:invoice';\n"
        "    -- Exact phrase in body_text:\n"
        "    SELECT id FROM messages WHERE id @@@ 'body_text:\"refund issued\"';\n"
        "  `LIKE '%term%'` cannot use this index and forces a full seq scan\n"
        "  (~1s/query on 410k rows). Use LIKE only when BM25 can't express\n"
        "  the predicate (e.g. structural patterns inside `labels` JSON)."
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
        "One row per attachment. Linked via message_id. ~600k rows.\n"
        "- id (INT PK).\n"
        "- message_id (TEXT FK -> messages.id).\n"
        "- filename (TEXT).\n"
        "- mime_type (TEXT): e.g. 'application/pdf', 'image/png'.\n"
        "- size_bytes (INT).\n"
        "- extracted_text (TEXT): parsed text (PDF/DOCX/TXT). NULL if image-only or extraction failed.\n"
        "- image_path (TEXT): on-disk path to extracted image, NULL if non-image.\n"
        "- raw_path (TEXT): on-disk path to original raw bytes.\n"
        "\n"
        "FAST FREE-TEXT SEARCH — use BM25, NOT `LIKE '%...%'`:\n"
        "  `attachments_bm25_idx` covers (filename, extracted_text).\n"
        "  Syntax: `WHERE id @@@ '<tantivy-query>'`, same as messages.\n"
        "  Example:\n"
        "    SELECT id, message_id, filename FROM attachments\n"
        "    WHERE id @@@ 'filename:invoice OR extracted_text:\"total due\"'\n"
        "    ORDER BY paradedb.score(id) DESC LIMIT 50;"
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
    # Internal mapping that pins one Claude session UUID per
    # conversation so deep-mode turns can `--resume` into the same
    # JSONL transcript. See pg_schema.sql for rationale.
    "conversation_claude_session",
    # Multi-tenant identity tables (Phase 1 of PER_USER_LOGIN). Off-limits
    # to the LLM-facing sql_query tool — querying a user's email or invite
    # status is an auth concern, not a corpus concern.
    "users",
    "invited_emails",
    # MCP OAuth provider state (hashed token/code rows + client regs).
    # Auth infrastructure — the LLM must never query it.
    "mcp_oauth_state",
    # Propositions (find_facts): LLM-extracted fact rows + idempotency
    # marker. Served through the dedicated find_facts tool; the analyst
    # reader role has no grant on them.
    "propositions",
    "prop_processed",
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


def rebuild_thread_summary(db_path: Path, *, user_id: Optional[str] = None) -> int:
    """Precompute thread metadata for fast search ranking. Returns thread count.

    Per-user: rebuilds only the given user's thread_summary rows. Daemon
    callers pass `user_id=None` and the bootstrap user is resolved.
    Phase 3c per-user sync passes the syncing user's id explicitly.
    """
    import json

    from gmail_search.auth.write_user import resolve_write_user_id

    conn = get_connection(db_path)
    uid = resolve_write_user_id(conn, user_id=user_id)

    # NOTE: we deliberately do NOT `DELETE FROM thread_summary WHERE user_id`
    # here. This function runs on every (frequent) reindex pass, and a
    # wipe-and-reinsert churned ~308K dead tuples per cycle — 294M ins/del
    # over time, bloating the table to 25 GB and seq-scanning search to ~10s.
    # Instead we UPSERT with a no-op guard (unchanged rows write nothing, so
    # zero dead tuples) and delete only threads that truly no longer exist.

    # Pull user_id along with the thread fields — every thread is
    # owned by the user whose messages compose it. The (user_id,
    # thread_id) tuple is what makes the rebuilt rows multi-tenant
    # safe: if two users somehow share a thread_id (shouldn't happen
    # — Gmail thread_ids are per-account), they'd land in distinct
    # thread_summary rows once the PK gets promoted in a follow-up.
    rows = conn.execute(
        """SELECT thread_id, from_addr, date, labels, subject, user_id
           FROM messages WHERE user_id = %s ORDER BY date""",
        (uid,),
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
                "user_id": r["user_id"],
            }
        t = threads[tid]
        t["from_addrs"].append(r["from_addr"])
        t["dates"].append(r["date"])
        for label in json.loads(r["labels"]):
            t["all_labels"].add(label)

    from gmail_search.store.queries import upsert_thread_summary

    for tid, t in threads.items():
        participants = list(dict.fromkeys(t["from_addrs"]))  # ordered unique
        # Computed values are identical to the previous wipe-and-reinsert,
        # so search ranking is unaffected.
        upsert_thread_summary(
            conn,
            thread_id=tid,
            subject=t["subject"],
            participants_json=json.dumps(participants),
            all_from_addrs_json=json.dumps(t["from_addrs"]),
            all_labels_json=json.dumps(sorted(t["all_labels"])),
            message_count=len(t["dates"]),
            date_first=t["dates"][0],
            date_last=t["dates"][-1],
            user_id=t["user_id"],
        )

    # Drop summaries for threads that no longer have any messages (deleted /
    # purged). This is the only DELETE — it touches just the genuinely-stale
    # rows (normally zero), not the whole table, so it adds no meaningful churn.
    conn.execute(
        """DELETE FROM thread_summary ts
           WHERE ts.user_id = %s
             AND NOT EXISTS (
               SELECT 1 FROM messages m
               WHERE m.thread_id = ts.thread_id AND m.user_id = %s)""",
        (uid, uid),
    )

    conn.commit()
    count = len(threads)
    conn.close()
    return count


def _load_message_embeddings(conn, limit=50000, *, user_id: Optional[str] = None):
    """Load message embeddings with metadata for clustering. Per-user
    when `user_id` is given (the normal call path); otherwise loads
    everything across users (legacy / dev-only)."""

    import numpy as np

    # Smart sampling: UNIFORM across the whole corpus via a stable hash of
    # message_id (not most-recent-N, which would skew the topic tree toward
    # recent themes). Representative across all years + deterministic.
    # k = ceil(total/limit); k=1 keeps everything for small corpora.
    if user_id is None:
        total = conn.execute("SELECT COUNT(*) FROM embeddings WHERE chunk_type = 'message'").fetchone()[0]
        k = max(1, -(-total // limit))
        rows = conn.execute(
            """SELECT e.message_id, e.embedding, m.subject, m.from_addr
               FROM embeddings e JOIN messages m ON e.message_id = m.id
               WHERE e.chunk_type = 'message' AND abs(hashtext(e.message_id)) %% %s = 0
               LIMIT %s""",
            (k, limit),
        ).fetchall()
    else:
        total = conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE chunk_type = 'message' AND user_id = %s",
            (user_id,),
        ).fetchone()[0]
        k = max(1, -(-total // limit))
        rows = conn.execute(
            """SELECT e.message_id, e.embedding, m.subject, m.from_addr
               FROM embeddings e JOIN messages m ON e.message_id = m.id
               WHERE e.chunk_type = 'message' AND e.user_id = %s AND abs(hashtext(e.message_id)) %% %s = 0
               LIMIT %s""",
            (user_id, k, limit),
        ).fetchall()

    # Build the matrix via np.frombuffer into a preallocated array.
    # `list(struct.unpack("3072f", ...))` materialized 3072 Python float
    # objects PER row (~97 KiB/row of object overhead) — ~5 GB for a 50k
    # cap, and the dominant cost on the heavy reindex path. frombuffer is
    # a zero-copy view; the assignment copies just the float32s.
    n = len(rows)
    vectors = np.empty((n, 3072), dtype=np.float32)
    for i, r in enumerate(rows):
        vectors[i] = np.frombuffer(r["embedding"], dtype=np.float32)
    return {
        "msg_ids": [r["message_id"] for r in rows],
        "subjects": [r["subject"] for r in rows],
        "senders": [r["from_addr"] for r in rows],
        "vectors": vectors,
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


def _auto_label_topics(conn, user_id):
    """Use Gemini Flash Lite to generate short topic labels from cluster summaries.

    Scoped to one user: topic_id is generated per-user (sequential "0"/"0.0"…) so
    it is NOT globally unique — reading/updating across users would relabel another
    tenant's topics. Filter every read and write by user_id."""
    import json
    import logging

    logger = logging.getLogger(__name__)
    topic_rows = conn.execute(
        "SELECT topic_id, parent_id, depth, top_senders, sample_subjects, message_count "
        "FROM topics WHERE user_id = %s ORDER BY depth, message_count DESC",
        (user_id,),
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
            conn.execute("UPDATE topics SET label = %s WHERE topic_id = %s AND user_id = %s", (label, tid, user_id))
        conn.commit()
        logger.info(f"Auto-labeled {len(label_map)} topic nodes")

    except Exception as e:
        logger.warning(f"Auto-labeling failed, using top sender as label: {e}")
        for t in topic_rows:
            sndrs = json.loads(t["top_senders"])
            label = sndrs[0] if sndrs else f"Topic {t['topic_id']}"
            conn.execute(
                "UPDATE topics SET label = %s WHERE topic_id = %s AND user_id = %s",
                (label, t["topic_id"], user_id),
            )
        conn.commit()


def rebuild_topics(
    db_path: Path,
    max_depth: int = 4,
    min_cluster_size: int = 50,
    *,
    user_id: Optional[str] = None,
) -> int:
    """Build hierarchical topic tree using recursive bisecting k-means.
    Per-user: clusters only the given user's messages, replaces only
    that user's `topics` + `message_topics` rows. Returns node count."""
    import json
    import logging

    import numpy as np
    from gmail_search.auth.write_user import resolve_write_user_id

    logger = logging.getLogger(__name__)

    conn = get_connection(db_path)
    uid = resolve_write_user_id(conn, user_id=user_id)

    data = _load_message_embeddings(conn, user_id=uid)
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

    conn.execute("DELETE FROM message_topics WHERE user_id = %s", (uid,))
    conn.execute("DELETE FROM topics WHERE user_id = %s", (uid,))

    for topic_id, parent_id, depth, indices in nodes:
        top_senders, sample_subjects = _summarize_cluster(indices, data["subjects"], data["senders"])
        conn.execute(
            "INSERT INTO topics (topic_id, parent_id, label, depth, message_count, top_senders, sample_subjects, user_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (topic_id, parent_id, "", depth, len(indices), json.dumps(top_senders), json.dumps(sample_subjects), uid),
        )
        # Messages belong to their leaf node and all ancestors
        for idx in indices:
            conn.execute(
                "INSERT INTO message_topics (message_id, topic_id, user_id) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                (data["msg_ids"][idx], topic_id, uid),
            )

    conn.commit()
    logger.info(f"Built {len(nodes)} topic nodes")

    _auto_label_topics(conn, uid)

    count = conn.execute("SELECT COUNT(*) FROM topics WHERE user_id = %s", (uid,)).fetchone()[0]
    conn.close()
    return count


def _extract_terms_from_messages(conn, *, user_id: Optional[str] = None, limit: int = 50000):
    """Build a reverse index: lowercase term → set of message indices.
    Per-user when `user_id` is given (the normal call path).

    Capped at `limit` most-recent messages. The unigram+bigram+trigram
    index is by far the heaviest structure in the whole reindex — over the
    full corpus it grows to tens of millions of keys (~15 GB) and OOM-kills
    the build. 50k recent messages surface the same common aliases at a
    bounded few-GB footprint.
    """
    import re

    # Smart sampling: a UNIFORM sample across the whole corpus, not the
    # most-recent N (which would bias aliases toward recent mail and miss
    # the older history). Take ~every k-th message by a stable hash of its
    # id — representative across all years, deterministic (reproducible
    # builds), and index-cheap (no global sort). k = ceil(total/limit), so
    # k=1 (small corpus) keeps everything.
    if user_id is None:
        total = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        k = max(1, -(-total // limit))
        rows = conn.execute(
            "SELECT id, subject, body_text, from_addr FROM messages " "WHERE abs(hashtext(id)) %% %s = 0 LIMIT %s",
            (k, limit),
        ).fetchall()
    else:
        total = conn.execute("SELECT COUNT(*) FROM messages WHERE user_id = %s", (user_id,)).fetchone()[0]
        k = max(1, -(-total // limit))
        rows = conn.execute(
            "SELECT id, subject, body_text, from_addr FROM messages "
            "WHERE user_id = %s AND abs(hashtext(id)) %% %s = 0 LIMIT %s",
            (user_id, k, limit),
        ).fetchall()

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


def rebuild_term_aliases(
    db_path: Path,
    min_term_len=2,
    max_term_len=5,
    min_occurrences=3,
    min_similarity=0.75,
    *,
    data_dir: Optional[Path] = None,
    user_id: Optional[str] = None,
    validate: bool = True,
    max_embedding_id: Optional[int] = None,
    full_rebuild: bool = False,
    rebuild_after_days: int = 30,
) -> int:
    """Discover term aliases (abbreviation → expansion) via co-occurrence.

    INCREMENTAL + out-of-core. Co-occurrence counts and long-term doc-
    frequencies are additive, so they're kept as append-only sort-merge run
    files under data/users/<uid>/aliases/. Each call ingests ONLY messages
    embedded since the last run (a watermark on the embeddings bigint id,
    which advances for both fresh and backfilled mail), appends new runs,
    then derives aliases by k-way merging all runs (heapq.merge) — bounded
    RAM regardless of corpus size. The expensive cross-product pass is paid
    once at bootstrap; ongoing updates are cheap deltas.

    Uses bigram (not trigram) phrases — the eval showed bigram-only is
    iso-quality (≈945 vs 948 aliases) at ~10x fewer co-occurrence records.

    Falls back to a one-shot temp build when `data_dir` is None (tests).
    Per-user: replaces only that user's `term_aliases` rows.
    """
    import json
    import logging
    import re as _re
    from collections import Counter

    from gmail_search.auth.write_user import resolve_write_user_id

    logger = logging.getLogger(__name__)
    conn = get_connection(db_path)
    uid = resolve_write_user_id(conn, user_id=user_id)

    _WORD = _re.compile(r"[a-zA-Z]{2,}")
    _ABBR = _re.compile(r"\b[A-Z]{2,5}\b")

    def _terms_in(subject, body, from_addr):
        """(abbreviations, long_terms) for one message. Long terms = words and
        BIGRAMS longer than max_term_len chars. Trigrams were dropped: the
        eval showed they don't change the discovered aliases but ~10x the
        co-occurrence records (and the runtime / OOM risk)."""
        raw = f"{subject or ''} {body or ''} {from_addr or ''}"
        words = _WORD.findall(raw.lower())
        long_terms = {w for w in words if len(w) > max_term_len}
        for i in range(len(words) - 1):
            ng = words[i] + " " + words[i + 1]
            if len(ng) > max_term_len:
                long_terms.add(ng)
        abbrevs = {w.lower() for w in _ABBR.findall(raw)}
        return abbrevs, long_terms

    # ---- persistent append-only run state (cooc + long-term doc-freq) ----
    import heapq as _heapq
    import marshal as _marshal
    import tempfile as _tempfile
    import uuid as _uuid
    from itertools import groupby as _groupby

    _tmp = None
    if data_dir is not None:
        _state = Path(data_dir) / "users" / uid / "aliases"
    else:
        _tmp = _tempfile.TemporaryDirectory(prefix="gms_alias_")
        _state = Path(_tmp.name)
    _cooc_dir = _state / "cooc"
    _ldf_dir = _state / "ldf"
    _cooc_dir.mkdir(parents=True, exist_ok=True)
    _ldf_dir.mkdir(parents=True, exist_ok=True)
    _meta_path = _state / "meta.json"

    watermark = 0
    abbrev_df: Counter = Counter()
    _bootstrap_at = None
    if _meta_path.exists():
        try:
            _m = json.loads(_meta_path.read_text())
            watermark = int(_m.get("watermark", 0))
            abbrev_df = Counter(_m.get("abbrev_df", {}))
            _bootstrap_at = _m.get("bootstrap_at")
        except Exception:
            watermark, abbrev_df, _bootstrap_at = 0, Counter(), None

    # Periodic full-rebuild safety net: incremental runs only ADD counts, so
    # deletions / re-embeds drift over time. On explicit full_rebuild, or every
    # `rebuild_after_days`, wipe the runs and re-ingest from scratch (watermark
    # 0). Self-healing — no external scheduler needed.
    import datetime as _dt
    import shutil as _shutil

    _now = _dt.datetime.now(_dt.timezone.utc)
    _stale = False
    if _bootstrap_at:
        try:
            _stale = (_now - _dt.datetime.fromisoformat(_bootstrap_at)).days >= rebuild_after_days
        except Exception:
            _stale = True
    if full_rebuild or _bootstrap_at is None or _stale:
        _shutil.rmtree(_cooc_dir, ignore_errors=True)
        _shutil.rmtree(_ldf_dir, ignore_errors=True)
        _cooc_dir.mkdir(parents=True, exist_ok=True)
        _ldf_dir.mkdir(parents=True, exist_ok=True)
        watermark, abbrev_df = 0, Counter()
        _bootstrap_at = _now.isoformat()
        logger.info("alias: full rebuild (bootstrap or stale > %dd)", rebuild_after_days)

    def _read_run(p):
        with open(p, "rb") as f:
            while True:
                try:
                    yield _marshal.load(f)
                except EOFError:
                    return

    def _write_run(records, d):
        with open(d / f"{_uuid.uuid4().hex}.run", "wb") as f:
            for rec in records:
                _marshal.dump(rec, f)

    # ---- INGEST: only messages embedded since `watermark` (new + backfill) ----
    # The embeddings bigint id advances whenever a message is embedded — for
    # fresh frontfill mail AND backfilled old mail — so it's one monotonic
    # cursor capturing everything not yet folded into the runs. Co-occurrence
    # and doc-freq are additive, so each call just appends new sorted runs.
    _CAP = 2_000_000
    cacc: dict = {}
    dacc: dict = {}
    ingested = 0
    last_eid = watermark
    _cap_sql = " AND e.id <= %s" if max_embedding_id is not None else ""
    while True:
        _params = (uid, last_eid) + ((max_embedding_id, 5000) if max_embedding_id is not None else (5000,))
        rows = conn.execute(
            "SELECT e.id AS eid, m.subject, m.body_text, m.from_addr "
            "FROM embeddings e JOIN messages m ON e.message_id = m.id "
            "WHERE e.chunk_type = 'message' AND e.user_id = %s AND e.id > %s" + _cap_sql + " "
            "ORDER BY e.id LIMIT %s",
            _params,
        ).fetchall()
        if not rows:
            break
        last_eid = rows[-1]["eid"]
        for r in rows:
            abbrevs, long_terms = _terms_in(r["subject"], r["body_text"], r["from_addr"])
            for a in abbrevs:
                abbrev_df[a] += 1
            for lt in long_terms:
                dacc[lt] = dacc.get(lt, 0) + 1
            for a in abbrevs:
                for lt in long_terms:
                    if a not in lt:
                        cacc[(a, lt)] = cacc.get((a, lt), 0) + 1
            ingested += 1
            if len(cacc) >= _CAP:
                _write_run(((k[0], k[1], v) for k, v in sorted(cacc.items())), _cooc_dir)
                cacc = {}
            if len(dacc) >= _CAP:
                _write_run(((k, v) for k, v in sorted(dacc.items())), _ldf_dir)
                dacc = {}
    if cacc:
        _write_run(((k[0], k[1], v) for k, v in sorted(cacc.items())), _cooc_dir)
    if dacc:
        _write_run(((k, v) for k, v in sorted(dacc.items())), _ldf_dir)
    logger.info("alias: ingested %d newly-embedded message(s); watermark %d -> %d", ingested, watermark, last_eid)
    _meta_path.write_text(
        json.dumps({"watermark": last_eid, "abbrev_df": dict(abbrev_df), "bootstrap_at": _bootstrap_at})
    )

    # ---- COMPACTION: merge-sum runs so derive cost stays bounded as deltas
    # accumulate. Re-summing equal keys is lossless. ----
    def _compact(d, is_pair, threshold=40):
        runs = sorted(d.glob("*.run"))
        if len(runs) <= threshold:
            return
        merged = _heapq.merge(*(_read_run(p) for p in runs))
        out = d / "_compacted.tmp"
        with open(out, "wb") as f:
            if is_pair:
                for (a, lt), grp in _groupby(merged, key=lambda r: (r[0], r[1])):
                    _marshal.dump((a, lt, sum(r[2] for r in grp)), f)
            else:
                for lt, grp in _groupby(merged, key=lambda r: r[0]):
                    _marshal.dump((lt, sum(r[1] for r in grp)), f)
        for p in runs:
            p.unlink()
        out.rename(d / f"{_uuid.uuid4().hex}.run")

    _compact(_cooc_dir, True)
    _compact(_ldf_dir, False)

    # ---- DERIVE: k-way sort-merge all runs → overlap>=5 → score ----
    short_terms = {a for a, c in abbrev_df.items() if c >= min_occurrences}
    cooc_runs = sorted(_cooc_dir.glob("*.run"))
    ldf_runs = sorted(_ldf_dir.glob("*.run"))
    cooccur: dict[str, dict] = {}
    cooccur_terms: set[str] = set()
    if cooc_runs and short_terms:
        for (_a, _l), _grp in _groupby(_heapq.merge(*(_read_run(p) for p in cooc_runs)), key=lambda r: (r[0], r[1])):
            _ov = sum(r[2] for r in _grp)
            if _ov >= 5 and _a in short_terms:
                cooccur.setdefault(_a, {})[_l] = _ov
                cooccur_terms.add(_l)
    logger.info("alias: %d abbrev(s) with candidates · %d surviving long terms", len(cooccur), len(cooccur_terms))

    if not cooccur or not cooccur_terms:
        conn.close()
        return 0

    long_df: Counter = Counter()
    if ldf_runs:
        for _l, _grp in _groupby(_heapq.merge(*(_read_run(p) for p in ldf_runs)), key=lambda r: r[0]):
            _s = sum(r[1] for r in _grp)
            if _l in cooccur_terms:
                long_df[_l] = _s

    # Score: overlap >= 5, Jaccard >= 0.2, top 3 (deterministic tie-break by
    # term). |A∪B| = df_a + df_l − overlap.
    conn.execute("DELETE FROM term_aliases WHERE user_id = %s", (uid,))
    alias_count = 0
    for term in cooccur:
        df_a = abbrev_df[term]
        candidates = []
        for long_term, overlap in sorted(cooccur[term].items(), key=lambda kv: (-kv[1], kv[0]))[:20]:
            union = df_a + long_df.get(long_term, 0) - overlap
            if union <= 0 or overlap / union < 0.2:
                continue
            candidates.append((long_term, overlap / union))
        if candidates[:3]:
            conn.execute(
                "INSERT INTO term_aliases (term, expansions, similarity, user_id) VALUES (%s, %s, %s, %s)",
                (term, json.dumps([t for t, _ in candidates[:3]]), candidates[0][1], uid),
            )
            alias_count += 1

    conn.commit()
    if validate:
        logger.info(f"Discovered {alias_count} candidate aliases, validating with Gemini...")
        _validate_aliases_with_llm(conn, user_id=uid)

    final_count = conn.execute("SELECT COUNT(*) FROM term_aliases WHERE user_id = %s", (uid,)).fetchone()[0]
    conn.close()
    logger.info(f"Validated {final_count} term aliases")
    return final_count


def _validate_aliases_with_llm(conn, *, user_id: Optional[str] = None):
    """Use Gemini to filter out noise from candidate aliases.

    Sends all candidates in batches and removes ones Gemini marks as bad.
    """
    import json
    import logging
    import os

    logger = logging.getLogger(__name__)

    if user_id is None:
        rows = conn.execute("SELECT term, expansions, similarity FROM term_aliases").fetchall()
    else:
        rows = conn.execute(
            "SELECT term, expansions, similarity FROM term_aliases WHERE user_id = %s",
            (user_id,),
        ).fetchall()
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
                if user_id is None:
                    row = conn.execute("SELECT similarity FROM term_aliases WHERE term = %s", (term,)).fetchone()
                else:
                    row = conn.execute(
                        "SELECT similarity FROM term_aliases WHERE user_id = %s AND term = %s",
                        (user_id, term),
                    ).fetchone()
                if row and row["similarity"] >= 0.25:
                    protected.add(term)

            removable = bad_terms - protected
            for term in removable:
                if user_id is None:
                    conn.execute("DELETE FROM term_aliases WHERE term = %s", (term,))
                else:
                    conn.execute(
                        "DELETE FROM term_aliases WHERE user_id = %s AND term = %s",
                        (user_id, term),
                    )
            conn.commit()
            logger.info(
                f"Removed {len(removable)} noise aliases via LLM validation"
                f" (protected {len(protected)} high-Jaccard aliases)"
            )

    except Exception as e:
        logger.warning(f"LLM alias validation failed, keeping all candidates: {e}")


def rebuild_spell_dictionary(
    db_path: Path,
    data_dir: Path,
    *,
    user_id: Optional[str] = None,
    full_rebuild: bool = False,
    rebuild_after_days: int = 30,
    max_embedding_id: Optional[int] = None,
) -> int:
    """Build a word-frequency dictionary from the corpus for spell correction.

    INCREMENTAL + out-of-core, like rebuild_term_aliases: subject/body word
    counts are additive, so they're kept as append-only sort-merge run files
    under data/users/<uid>/spell/. Each call ingests only messages embedded
    since the last watermark (embeddings bigint id), appends a run, then
    derives the dictionary by k-way merging all runs. Sender-name boosts
    (DISTINCT from_addr) are cheap and recomputed fresh at derive time.
    Self-healing full rebuild every `rebuild_after_days`.

    Per-user output: data/users/<uid>/spell_dictionary.txt (SymSpell format).
    """
    import datetime as _dt
    import heapq as _heapq
    import json
    import logging
    import marshal as _marshal
    import re
    import shutil as _shutil
    import uuid as _uuid
    from collections import Counter
    from itertools import groupby as _groupby

    from gmail_search.auth.write_user import resolve_write_user_id

    logger = logging.getLogger(__name__)
    conn = get_connection(db_path)
    uid = resolve_write_user_id(conn, user_id=user_id)

    _word = re.compile(r"[a-zA-Z]+")
    state = data_dir / "users" / uid / "spell"
    run_dir = state / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    meta_path = state / "meta.json"

    watermark = 0
    bootstrap_at = None
    if meta_path.exists():
        try:
            m = json.loads(meta_path.read_text())
            watermark = int(m.get("watermark", 0))
            bootstrap_at = m.get("bootstrap_at")
        except Exception:
            watermark, bootstrap_at = 0, None

    now = _dt.datetime.now(_dt.timezone.utc)
    stale = False
    if bootstrap_at:
        try:
            stale = (now - _dt.datetime.fromisoformat(bootstrap_at)).days >= rebuild_after_days
        except Exception:
            stale = True
    if full_rebuild or bootstrap_at is None or stale:
        _shutil.rmtree(run_dir, ignore_errors=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        watermark = 0
        bootstrap_at = now.isoformat()
        logger.info("spell: full rebuild (bootstrap or stale > %dd)", rebuild_after_days)

    def _flush(acc):
        if not acc:
            return
        with open(run_dir / f"{_uuid.uuid4().hex}.run", "wb") as f:
            for w, c in sorted(acc.items()):
                _marshal.dump((w, c), f)

    # ---- INGEST: subject+body word TF for messages embedded since watermark ----
    acc: dict = {}
    last_eid = watermark
    cap_sql = " AND e.id <= %s" if max_embedding_id is not None else ""
    while True:
        params = (uid, last_eid) + ((max_embedding_id, 5000) if max_embedding_id is not None else (5000,))
        rows = conn.execute(
            "SELECT e.id AS eid, m.subject, m.body_text "
            "FROM embeddings e JOIN messages m ON e.message_id = m.id "
            "WHERE e.chunk_type = 'message' AND e.user_id = %s AND e.id > %s" + cap_sql + " "
            "ORDER BY e.id LIMIT %s",
            params,
        ).fetchall()
        if not rows:
            break
        last_eid = rows[-1]["eid"]
        for r in rows:
            for field in (r["subject"], r["body_text"]):
                if field:
                    for w in _word.findall(field.lower()):
                        if len(w) >= 2:
                            acc[w] = acc.get(w, 0) + 1
        if len(acc) >= 2_000_000:
            _flush(acc)
            acc = {}
    _flush(acc)
    meta_path.write_text(json.dumps({"watermark": last_eid, "bootstrap_at": bootstrap_at}))

    def _read(p):
        with open(p, "rb") as f:
            while True:
                try:
                    yield _marshal.load(f)
                except EOFError:
                    return

    # ---- COMPACTION: merge-sum runs so derive cost stays bounded ----
    runs = sorted(run_dir.glob("*.run"))
    if len(runs) > 40:
        out = run_dir / "_compacted.tmp"
        with open(out, "wb") as f:
            for w, grp in _groupby(_heapq.merge(*(_read(p) for p in runs)), key=lambda r: r[0]):
                _marshal.dump((w, sum(r[1] for r in grp)), f)
        for p in runs:
            p.unlink()
        out.rename(run_dir / f"{_uuid.uuid4().hex}.run")
        runs = sorted(run_dir.glob("*.run"))

    # ---- DERIVE: merge word runs + fresh sender-name boost → dictionary ----
    word_counts: Counter = Counter()
    for w, grp in _groupby(_heapq.merge(*(_read(p) for p in runs)), key=lambda r: r[0]):
        word_counts[w] = sum(r[1] for r in grp)

    # Sender names (DISTINCT, cheap) boosted so they're preferred corrections.
    for r in conn.execute("SELECT DISTINCT from_addr FROM messages WHERE user_id = %s", (uid,)).fetchall():
        addr = r["from_addr"]
        name = addr.split("<")[0].strip().strip('"') if "<" in addr else addr.split("@")[0]
        for w in _word.findall(name.lower()):
            if len(w) >= 2:
                word_counts[w] += 50
    conn.close()

    # Filter at WRITE time (run files keep full counts, so a word that
    # crosses the floor later still accumulates correctly). Without a floor
    # the dictionary was ~5.4M terms, 70% count==1 — base64/message-id
    # shrapnel that cost SymSpell ~6.6 GB heap and 112 s of load, and worse,
    # stored the corpus's own typos ("reciept", "mortage") as count-1
    # "correct" words, which blocked their correction. Floor 20 + length cap:
    # ~180k terms, ~0.4 GB, 3 s load, and those corrections actually fire.
    # Sender names are boosted +50 above, so real contacts always survive.
    min_count, max_len = 20, 24
    user_dir = data_dir / "users" / uid
    user_dir.mkdir(parents=True, exist_ok=True)
    dict_path = user_dir / "spell_dictionary.txt"
    written = 0
    with open(dict_path, "w") as f:
        for word, count in word_counts.most_common():
            if count < min_count or len(word) > max_len:
                continue
            f.write(f"{word} {count}\n")
            written += 1
    return written


def rebuild_contact_frequency(db_path: Path, *, user_id: Optional[str] = None) -> int:
    """Precompute contact frequency scores. Returns contact count.

    Per-user: scans only the given user's messages and replaces only
    that user's contact_frequency rows. The composite PK
    (user_id, email) makes ON CONFLICT a clean per-user upsert."""
    from gmail_search.auth.write_user import resolve_write_user_id

    conn = get_connection(db_path)
    uid = resolve_write_user_id(conn, user_id=user_id)

    conn.execute("DELETE FROM contact_frequency WHERE user_id = %s", (uid,))

    # Count messages per sender email (per-user)
    rows = conn.execute(
        "SELECT from_addr, COUNT(*) as c FROM messages WHERE user_id = %s GROUP BY from_addr ORDER BY c DESC",
        (uid,),
    ).fetchall()

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
            """INSERT INTO contact_frequency (user_id, email, message_count, score)
               VALUES (%s, %s, %s, %s)
               ON CONFLICT(user_id, email) DO UPDATE SET
                 message_count = excluded.message_count,
                 score = excluded.score""",
            (uid, addr, r["c"], score),
        )

    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM contact_frequency WHERE user_id = %s", (uid,)).fetchone()[0]
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

    Optionally applies a default `statement_timeout`, gated by the
    `GMS_DEFAULT_STATEMENT_TIMEOUT_MS` env var so it only binds in
    processes that opt in. The serve (user-facing) process sets it, so a
    runaway query — e.g. a future seq-scan regression like the
    thread_summary bloat — can't tie up a request indefinitely. Daemons
    (separate processes: reindex/backfill/summarize/vacuum) leave it
    unset and stay unbounded; they are the "marked specially"
    long-running work. An empty/0/invalid value disables the cap.
    """
    import os as _os

    import psycopg

    raw = psycopg.connect(_pg_dsn(), row_factory=_compat_row_factory)
    raw_timeout = _os.environ.get("GMS_DEFAULT_STATEMENT_TIMEOUT_MS")
    if raw_timeout:
        try:
            ms = int(raw_timeout)
        except ValueError:
            ms = 0
        if ms > 0:
            # autocommit so the SET persists at session scope (a SET inside a
            # rolled-back txn would revert); restore the txn mode callers expect.
            raw.autocommit = True
            with raw.cursor() as cur:
                cur.execute(f"SET statement_timeout = {ms}")
            raw.autocommit = False
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
