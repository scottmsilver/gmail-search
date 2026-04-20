import json
import sqlite3
from datetime import datetime

from gmail_search.store.models import Attachment, EmbeddingRecord, Message


def upsert_message(conn: sqlite3.Connection, msg: Message) -> None:
    conn.execute(
        """INSERT INTO messages (id, thread_id, from_addr, to_addr, subject,
           body_text, body_html, date, labels, history_id, raw_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
             thread_id=excluded.thread_id, from_addr=excluded.from_addr,
             to_addr=excluded.to_addr, subject=excluded.subject,
             body_text=excluded.body_text, body_html=excluded.body_html,
             date=excluded.date, labels=excluded.labels,
             history_id=excluded.history_id, raw_json=excluded.raw_json""",
        (
            msg.id,
            msg.thread_id,
            msg.from_addr,
            msg.to_addr,
            msg.subject,
            msg.body_text,
            msg.body_html,
            msg.date.isoformat(),
            json.dumps(msg.labels),
            msg.history_id,
            msg.raw_json,
        ),
    )
    conn.commit()


def get_message(conn: sqlite3.Connection, message_id: str) -> Message | None:
    row = conn.execute("SELECT * FROM messages WHERE id = ?", (message_id,)).fetchone()
    if row is None:
        return None
    return Message(
        id=row["id"],
        thread_id=row["thread_id"],
        from_addr=row["from_addr"],
        to_addr=row["to_addr"],
        subject=row["subject"],
        body_text=row["body_text"],
        body_html=row["body_html"],
        date=datetime.fromisoformat(row["date"]),
        labels=json.loads(row["labels"]),
        history_id=row["history_id"],
        raw_json=row["raw_json"],
    )


def get_messages_without_embeddings(conn: sqlite3.Connection, model: str) -> list[Message]:
    rows = conn.execute(
        """SELECT m.* FROM messages m
           WHERE m.id NOT IN (
             SELECT DISTINCT message_id FROM embeddings
             WHERE chunk_type = 'message' AND model = ?
           )""",
        (model,),
    ).fetchall()
    return [
        Message(
            id=r["id"],
            thread_id=r["thread_id"],
            from_addr=r["from_addr"],
            to_addr=r["to_addr"],
            subject=r["subject"],
            body_text=r["body_text"],
            body_html=r["body_html"],
            date=datetime.fromisoformat(r["date"]),
            labels=json.loads(r["labels"]),
            history_id=r["history_id"],
            raw_json=r["raw_json"],
        )
        for r in rows
    ]


def upsert_attachment(conn: sqlite3.Connection, att: Attachment) -> int:
    # `RETURNING id` works on both SQLite >= 3.35 and Postgres; it's the
    # portable replacement for the old `cursor.lastrowid` (which returned
    # None on psycopg under our dict_row factory). We fetch the scalar
    # right after execute so callers get the same int back unchanged.
    cursor = conn.execute(
        """INSERT INTO attachments (message_id, filename, mime_type, size_bytes,
           extracted_text, image_path, raw_path)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(message_id, filename) DO UPDATE SET
             mime_type=excluded.mime_type, size_bytes=excluded.size_bytes,
             raw_path=excluded.raw_path
           RETURNING id""",
        (att.message_id, att.filename, att.mime_type, att.size_bytes, att.extracted_text, att.image_path, att.raw_path),
    )
    row = cursor.fetchone()
    conn.commit()
    # row can be a sqlite3.Row (tuple-like, row[0]) or a dict (psycopg dict_row, row["id"]).
    if row is None:
        return 0
    try:
        return int(row["id"])
    except (KeyError, TypeError):
        return int(row[0])


def get_attachments_for_message(conn: sqlite3.Connection, message_id: str) -> list[Attachment]:
    rows = conn.execute("SELECT * FROM attachments WHERE message_id = ?", (message_id,)).fetchall()
    return [
        Attachment(
            id=r["id"],
            message_id=r["message_id"],
            filename=r["filename"],
            mime_type=r["mime_type"],
            size_bytes=r["size_bytes"],
            extracted_text=r["extracted_text"],
            image_path=r["image_path"],
            raw_path=r["raw_path"],
        )
        for r in rows
    ]


# ─── Drive stubs ───────────────────────────────────────────────────────────
# Drive-linked docs don't come from Gmail attachments; they're referenced
# by URL from the message body. We represent them as attachment rows with
# `mime_type = application/vnd.google-apps.*`, no `raw_path` (content is
# remote), and a `filename` that encodes the drive_id so the fetch step
# can round-trip to it: "Drive: [<drive_id>]" as a stub, then
# "Drive: <title> [<drive_id>]" once fetched.
#
# Reason the SQL lives here (not in gmail/drive.py): everything that
# touches the attachments table routes through this module so the
# schema is editable in one place. gmail/drive.py stays a pure API
# client (no sqlite3 import).


def upsert_drive_stub(
    conn: sqlite3.Connection,
    *,
    message_id: str,
    drive_id: str,
    mime_type: str,
) -> int:
    """Insert a stub row for a Drive-linked doc. Returns the number
    of new rows inserted (0 if the stub already existed; the
    `(message_id, filename)` dedup index enforces idempotency).
    """
    filename = f"Drive: [{drive_id}]"
    # Use `ON CONFLICT DO NOTHING` (portable SQLite >= 3.24 + Postgres)
    # instead of `INSERT OR IGNORE` (SQLite-only). Both produce
    # rowcount=0 on skip and rowcount=1 on insert, which is what the
    # caller relies on for idempotency reporting.
    cursor = conn.execute(
        """INSERT INTO attachments
           (message_id, filename, mime_type, size_bytes)
           VALUES (?, ?, ?, 0)
           ON CONFLICT (message_id, filename) DO NOTHING""",
        (message_id, filename, mime_type),
    )
    return cursor.rowcount


def set_active_index_dir(conn: sqlite3.Connection, path: str) -> None:
    """Flip the one-row `scann_index_pointer` to a new on-disk path.
    Readers resolve the active index through this row so a reindex
    swap is atomic at the DB layer — no reliance on filesystem rename
    semantics. See `build_index_sharded` for the writer side.
    """
    conn.execute(
        """INSERT INTO scann_index_pointer (id, current_dir, updated_at)
           VALUES (1, ?, CURRENT_TIMESTAMP)
           ON CONFLICT(id) DO UPDATE SET
             current_dir = excluded.current_dir,
             updated_at = CURRENT_TIMESTAMP""",
        (path,),
    )


def get_active_index_dir(conn: sqlite3.Connection) -> str | None:
    row = conn.execute("SELECT current_dir FROM scann_index_pointer WHERE id = 1").fetchone()
    return row["current_dir"] if row else None


def fill_drive_attachment(
    conn: sqlite3.Connection,
    *,
    attachment_id: int,
    title: str,
    text: str,
    drive_id: str,
) -> None:
    """Populate a previously-stubbed Drive row with fetched content.
    Renames the filename to include the title so the UI shows it.
    """
    new_filename = f"Drive: {title} [{drive_id}]"
    conn.execute(
        "UPDATE attachments SET extracted_text = ?, filename = ?, size_bytes = ? WHERE id = ?",
        (text, new_filename, len(text), attachment_id),
    )


def insert_embedding(conn: sqlite3.Connection, rec: EmbeddingRecord) -> int:
    # `RETURNING id` — portable across SQLite 3.35+ and Postgres, and
    # the only reliable way to get the newly-generated BIGSERIAL from
    # psycopg.
    cursor = conn.execute(
        """INSERT INTO embeddings (message_id, attachment_id, chunk_type,
           chunk_text, embedding, model)
           VALUES (?, ?, ?, ?, ?, ?)
           RETURNING id""",
        (rec.message_id, rec.attachment_id, rec.chunk_type, rec.chunk_text, rec.embedding, rec.model),
    )
    row = cursor.fetchone()
    conn.commit()
    if row is None:
        return 0
    try:
        return int(row["id"])
    except (KeyError, TypeError):
        return int(row[0])


def embedding_exists(
    conn: sqlite3.Connection, message_id: str, attachment_id: int | None, chunk_type: str, model: str
) -> bool:
    if attachment_id is None:
        row = conn.execute(
            """SELECT 1 FROM embeddings
               WHERE message_id = ? AND attachment_id IS NULL
               AND chunk_type = ? AND model = ?""",
            (message_id, chunk_type, model),
        ).fetchone()
    else:
        row = conn.execute(
            """SELECT 1 FROM embeddings
               WHERE message_id = ? AND attachment_id = ?
               AND chunk_type = ? AND model = ?""",
            (message_id, attachment_id, chunk_type, model),
        ).fetchone()
    return row is not None


def load_all_embeddings(conn: sqlite3.Connection, model: str) -> tuple[list[int], list[bytes]]:
    rows = conn.execute("SELECT id, embedding FROM embeddings WHERE model = ?", (model,)).fetchall()
    ids = [r["id"] for r in rows]
    blobs = [r["embedding"] for r in rows]
    return ids, blobs


def set_sync_state(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO sync_state (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )
    conn.commit()


def get_sync_state(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM sync_state WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


# Hardened FTS5 input pipeline. The model can hand us anything — quoted
# phrases, parenthesized expressions, column filters, control chars,
# enormous strings. We allow-list a tiny character set and cap sizes.
import re as _re

# Keep word chars (letters/digits/underscore), apostrophe (don't), hyphen
# (well-known), and whitespace. Everything else is collapsed to a space.
_FTS_KEEP = _re.compile(r"[^\w'\-\s]+", _re.UNICODE)
_MAX_TOKEN_LEN = 64
_MAX_TOKENS = 32
_MAX_QUERY_LEN = 2000


def _sanitize_fts_tokens(raw: str) -> list[str]:
    """Convert arbitrary user/model input into a safe list of FTS5 tokens.

    - Allow-lists a tiny character set (word chars, ', -, whitespace).
    - Drops empty tokens, leading/trailing punctuation, and FTS5 operator
      keywords (AND, OR, NOT, NEAR) so they cannot be injected.
    - Caps both per-token length and total token count.
    """
    if not raw:
        return []
    truncated = raw[:_MAX_QUERY_LEN]
    cleaned = _FTS_KEEP.sub(" ", truncated)
    tokens: list[str] = []
    for tok in cleaned.split():
        # Trim leading/trailing hyphens and apostrophes (they aren't useful
        # alone and could confuse FTS5 tokenization).
        stripped = tok.strip("-'")
        if not stripped:
            continue
        if stripped.upper() in {"AND", "OR", "NOT", "NEAR"}:
            continue
        tokens.append(stripped[:_MAX_TOKEN_LEN])
        if len(tokens) >= _MAX_TOKENS:
            break
    return tokens


def _escape_fts_query(query: str) -> str:
    """Escape a user query for safe use in FTS5 MATCH.

    Wraps each sanitized token in double quotes so it's treated as a
    literal phrase and FTS5 cannot interpret any character inside as an
    operator.
    """
    return " ".join(f'"{t}"' for t in _sanitize_fts_tokens(query))


def _fts_search_table(
    conn: sqlite3.Connection,
    table: str,
    query: str,
    limit: int,
    logger,
) -> dict[str, float]:
    """Search a single FTS table, return {message_id: score}.

    Any FTS5 syntax error is logged loudly with the offending query so it
    can be diagnosed. We never let an FTS failure crash the request.
    """
    scores: dict[str, float] = {}
    try:
        rows = conn.execute(
            f"SELECT message_id, rank FROM {table} WHERE {table} MATCH ? ORDER BY rank LIMIT ?",
            (query, limit),
        ).fetchall()
        for r in rows:
            mid = r["message_id"]
            raw = -r["rank"]
            if mid not in scores or raw > scores[mid]:
                scores[mid] = raw
    except sqlite3.OperationalError as e:
        logger.error(f"FTS5 syntax error on {table}: {e!s} | query={query!r}")
    except Exception as e:  # paranoid catch — FTS must never fail the request
        logger.exception(f"Unexpected FTS error on {table}: {e!s} | query={query!r}")
    return scores


def search_fts(conn, query: str, limit: int = 200) -> dict[str, float]:
    """Search messages + attachments via full-text search, return {message_id: score}.

    Backend-aware entry point:
      - SQLite path routes to `_search_fts_sqlite` (FTS5 MATCH + bm25).
      - Postgres path routes to `_search_fts_postgres`
        (`tsvector @@ websearch_to_tsquery` + `ts_rank_cd`).

    Both backends search with a phrase-match pass (higher weight) and a
    disjunction pass (broader recall), then return normalized scores in
    [0, 1] so downstream blenders can mix FTS with vector similarity.
    """
    # Late import to avoid a top-of-module cycle with db.py.
    from gmail_search.store.db import _is_postgres_backend

    if _is_postgres_backend():
        return _search_fts_postgres(conn, query, limit)
    return _search_fts_sqlite(conn, query, limit)


def _search_fts_sqlite(conn: sqlite3.Connection, query: str, limit: int) -> dict[str, float]:
    """SQLite FTS5 path. Unchanged from the original `search_fts` body."""
    import logging

    logger = logging.getLogger(__name__)
    tokens = _sanitize_fts_tokens(query)
    if not tokens:
        return {}

    scores: dict[str, float] = {}

    # Strategy 1: Phrase match (all words must appear near each other)
    # FTS5 NEAR groups words that appear within 10 tokens of each other
    if len(tokens) > 1:
        escaped_tokens = [f'"{t}"' for t in tokens]
        phrase_query = " ".join(escaped_tokens)
        for table in ["messages_fts", "attachments_fts"]:
            for mid, raw in _fts_search_table(conn, table, phrase_query, limit, logger).items():
                # Boost phrase matches by 1.5x
                boosted = raw * 1.5
                if mid not in scores or boosted > scores[mid]:
                    scores[mid] = boosted

    # Strategy 2: Individual terms (any word matches — broader recall)
    individual_query = " OR ".join(f'"{t}"' for t in tokens)
    for table in ["messages_fts", "attachments_fts"]:
        for mid, raw in _fts_search_table(conn, table, individual_query, limit, logger).items():
            if mid not in scores or raw > scores[mid]:
                scores[mid] = raw

    # Normalize to 0-1 (single result gets 1.0)
    if scores:
        max_score = max(scores.values())
        min_score = min(scores.values())
        if max_score == min_score:
            scores = {mid: 1.0 for mid in scores}
        else:
            score_range = max_score - min_score
            scores = {mid: (s - min_score) / score_range for mid, s in scores.items()}

    return scores


# Weights passed to `ts_rank_cd`: {D, C, B, A}. Mirrors the `setweight`
# assignments in pg_schema.sql — subject is A (1.0), from/to are B (0.6),
# body is C (0.3), and the D slot (0.1) is unused by our documents but
# required by the function signature.
_PG_TS_RANK_WEIGHTS = "{0.1, 0.3, 0.6, 1.0}"
# Normalization bits for `ts_rank_cd`: 32 divides the rank by
# (1 + log(unique-words-in-doc)), which keeps scores in (0, 1) regardless
# of document length. We still min-max below for blend compatibility with
# the SQLite path.
_PG_TS_RANK_NORM = 32


def _pg_rank_table(
    conn,
    table: str,
    pg_tsquery_fn: str,
    query: str,
    limit: int,
    logger,
) -> dict[str, float]:
    """Run one FTS pass against a single PG table.

    `pg_tsquery_fn` is either `phraseto_tsquery` (phrase pass) or
    `websearch_to_tsquery` (disjunction pass). Both accept the raw
    sanitized query string; we never build tsquery fragments by hand,
    so there's no SQL-injection surface beyond what psycopg's parameter
    binding already handles.
    """
    scores: dict[str, float] = {}
    try:
        rows = conn.execute(
            f"SELECT id AS message_id, "
            f"ts_rank_cd('{_PG_TS_RANK_WEIGHTS}'::float4[], fts, "
            f"{pg_tsquery_fn}('english', ?), {_PG_TS_RANK_NORM}) AS rank "
            f"FROM {table} "
            f"WHERE fts @@ {pg_tsquery_fn}('english', ?) "
            f"ORDER BY rank DESC LIMIT ?",
            (query, query, limit),
        ).fetchall()
        for r in rows:
            mid = r["message_id"]
            raw = float(r["rank"] or 0.0)
            if mid not in scores or raw > scores[mid]:
                scores[mid] = raw
    except Exception as e:  # paranoid catch — FTS must never fail the request
        # `websearch_to_tsquery` raises on pathological input in rare cases;
        # a malformed query must not crash the whole search pipeline.
        logger.exception(f"PG FTS error on {table} via {pg_tsquery_fn}: {e!s} | query={query!r}")
    return scores


def _pg_rank_table_attachments(
    conn,
    pg_tsquery_fn: str,
    query: str,
    limit: int,
    logger,
) -> dict[str, float]:
    """Attachments variant: the rank row pulls `message_id` (not `id`)
    so the caller can blend scores by message like the SQLite path does.
    """
    scores: dict[str, float] = {}
    try:
        rows = conn.execute(
            f"SELECT message_id, "
            f"ts_rank_cd('{_PG_TS_RANK_WEIGHTS}'::float4[], fts, "
            f"{pg_tsquery_fn}('english', ?), {_PG_TS_RANK_NORM}) AS rank "
            f"FROM attachments "
            f"WHERE fts @@ {pg_tsquery_fn}('english', ?) "
            f"ORDER BY rank DESC LIMIT ?",
            (query, query, limit),
        ).fetchall()
        for r in rows:
            mid = r["message_id"]
            raw = float(r["rank"] or 0.0)
            if mid not in scores or raw > scores[mid]:
                scores[mid] = raw
    except Exception as e:
        logger.exception(f"PG FTS error on attachments via {pg_tsquery_fn}: {e!s} | query={query!r}")
    return scores


def _search_fts_postgres(conn, query: str, limit: int) -> dict[str, float]:
    """Postgres `tsvector @@ tsquery` path.

    Two passes mirror the SQLite strategy:
      1. Phrase pass via `phraseto_tsquery` — words must appear in order,
         adjacent. Results get the same ×1.5 boost the SQLite path gives
         FTS5 phrase matches.
      2. Disjunction pass via `websearch_to_tsquery` — any sanitized term
         can match. `websearch_to_tsquery` natively accepts quotes, `-`,
         and `OR`, but we still sanitize because pathological input can
         make the query parser slow enough to matter.

    Scores from `ts_rank_cd(..., 32)` are already in (0, 1); we still
    min-max across the result set so downstream blend code sees the same
    shape as the SQLite path.
    """
    import logging

    logger = logging.getLogger(__name__)
    tokens = _sanitize_fts_tokens(query)
    if not tokens:
        return {}

    # Hand the sanitized tokens to PG as plain whitespace-joined text —
    # `phraseto_tsquery` / `websearch_to_tsquery` do their own parsing.
    sanitized_query = " ".join(tokens)
    scores: dict[str, float] = {}

    # Strategy 1: Phrase match (words must appear in order, adjacent).
    if len(tokens) > 1:
        for tbl_scores in (
            _pg_rank_table(conn, "messages", "phraseto_tsquery", sanitized_query, limit, logger),
            _pg_rank_table_attachments(conn, "phraseto_tsquery", sanitized_query, limit, logger),
        ):
            for mid, raw in tbl_scores.items():
                # Boost phrase matches by 1.5x (mirrors SQLite path).
                boosted = raw * 1.5
                if mid not in scores or boosted > scores[mid]:
                    scores[mid] = boosted

    # Strategy 2: Individual terms (any word matches — broader recall).
    for tbl_scores in (
        _pg_rank_table(conn, "messages", "websearch_to_tsquery", sanitized_query, limit, logger),
        _pg_rank_table_attachments(conn, "websearch_to_tsquery", sanitized_query, limit, logger),
    ):
        for mid, raw in tbl_scores.items():
            if mid not in scores or raw > scores[mid]:
                scores[mid] = raw

    # Normalize to 0-1 (single result gets 1.0) — same as SQLite path so
    # downstream blend math is backend-agnostic.
    if scores:
        max_score = max(scores.values())
        min_score = min(scores.values())
        if max_score == min_score:
            scores = {mid: 1.0 for mid in scores}
        else:
            score_range = max_score - min_score
            scores = {mid: (s - min_score) / score_range for mid, s in scores.items()}

    return scores
