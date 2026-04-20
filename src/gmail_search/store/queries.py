import json
from datetime import datetime

from gmail_search.store.models import Attachment, EmbeddingRecord, Message


def upsert_message(conn, msg: Message) -> None:
    conn.execute(
        """INSERT INTO messages (id, thread_id, from_addr, to_addr, subject,
           body_text, body_html, date, labels, history_id, raw_json)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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


def get_message(conn, message_id: str) -> Message | None:
    row = conn.execute("SELECT * FROM messages WHERE id = %s", (message_id,)).fetchone()
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


def get_messages_without_embeddings(conn, model: str) -> list[Message]:
    rows = conn.execute(
        """SELECT m.* FROM messages m
           WHERE m.id NOT IN (
             SELECT DISTINCT message_id FROM embeddings
             WHERE chunk_type = 'message' AND model = %s
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


def upsert_attachment(conn, att: Attachment) -> int:
    # `RETURNING id` — portable across SQLite 3.35+ and Postgres, and the
    # only reliable way to get the newly-generated BIGSERIAL from psycopg.
    cursor = conn.execute(
        """INSERT INTO attachments (message_id, filename, mime_type, size_bytes,
           extracted_text, image_path, raw_path)
           VALUES (%s, %s, %s, %s, %s, %s, %s)
           ON CONFLICT(message_id, filename) DO UPDATE SET
             mime_type=excluded.mime_type, size_bytes=excluded.size_bytes,
             raw_path=excluded.raw_path
           RETURNING id""",
        (att.message_id, att.filename, att.mime_type, att.size_bytes, att.extracted_text, att.image_path, att.raw_path),
    )
    row = cursor.fetchone()
    conn.commit()
    if row is None:
        return 0
    try:
        return int(row["id"])
    except (KeyError, TypeError):
        return int(row[0])


def get_attachments_for_message(conn, message_id: str) -> list[Attachment]:
    rows = conn.execute("SELECT * FROM attachments WHERE message_id = %s", (message_id,)).fetchall()
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


def upsert_drive_stub(
    conn,
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
    cursor = conn.execute(
        """INSERT INTO attachments
           (message_id, filename, mime_type, size_bytes)
           VALUES (%s, %s, %s, 0)
           ON CONFLICT (message_id, filename) DO NOTHING""",
        (message_id, filename, mime_type),
    )
    return cursor.rowcount


def set_active_index_dir(conn, path: str) -> None:
    """Flip the one-row `scann_index_pointer` to a new on-disk path.
    Readers resolve the active index through this row so a reindex
    swap is atomic at the DB layer — no reliance on filesystem rename
    semantics. See `build_index_sharded` for the writer side.
    """
    conn.execute(
        """INSERT INTO scann_index_pointer (id, current_dir, updated_at)
           VALUES (1, %s, CURRENT_TIMESTAMP)
           ON CONFLICT(id) DO UPDATE SET
             current_dir = excluded.current_dir,
             updated_at = CURRENT_TIMESTAMP""",
        (path,),
    )


def get_active_index_dir(conn) -> str | None:
    row = conn.execute("SELECT current_dir FROM scann_index_pointer WHERE id = 1").fetchone()
    return row["current_dir"] if row else None


def fill_drive_attachment(
    conn,
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
        "UPDATE attachments SET extracted_text = %s, filename = %s, size_bytes = %s WHERE id = %s",
        (text, new_filename, len(text), attachment_id),
    )


def insert_embedding(conn, rec: EmbeddingRecord) -> int:
    # `RETURNING id` — portable across SQLite 3.35+ and Postgres, and
    # the only reliable way to get the newly-generated BIGSERIAL from
    # psycopg.
    cursor = conn.execute(
        """INSERT INTO embeddings (message_id, attachment_id, chunk_type,
           chunk_text, embedding, model)
           VALUES (%s, %s, %s, %s, %s, %s)
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


def embedding_exists(conn, message_id: str, attachment_id: int | None, chunk_type: str, model: str) -> bool:
    if attachment_id is None:
        row = conn.execute(
            """SELECT 1 FROM embeddings
               WHERE message_id = %s AND attachment_id IS NULL
               AND chunk_type = %s AND model = %s""",
            (message_id, chunk_type, model),
        ).fetchone()
    else:
        row = conn.execute(
            """SELECT 1 FROM embeddings
               WHERE message_id = %s AND attachment_id = %s
               AND chunk_type = %s AND model = %s""",
            (message_id, attachment_id, chunk_type, model),
        ).fetchone()
    return row is not None


def load_all_embeddings(conn, model: str) -> tuple[list[int], list[bytes]]:
    rows = conn.execute("SELECT id, embedding FROM embeddings WHERE model = %s", (model,)).fetchall()
    ids = [r["id"] for r in rows]
    blobs = [r["embedding"] for r in rows]
    return ids, blobs


def set_sync_state(conn, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO sync_state (key, value) VALUES (%s, %s) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )
    conn.commit()


def get_sync_state(conn, key: str) -> str | None:
    row = conn.execute("SELECT value FROM sync_state WHERE key = %s", (key,)).fetchone()
    return row["value"] if row else None


# Hardened FTS input pipeline. The model can hand us anything — quoted
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
    """Convert arbitrary user/model input into a safe list of FTS tokens.

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
        # alone and could confuse FTS tokenization).
        stripped = tok.strip("-'")
        if not stripped:
            continue
        if stripped.upper() in {"AND", "OR", "NOT", "NEAR"}:
            continue
        tokens.append(stripped[:_MAX_TOKEN_LEN])
        if len(tokens) >= _MAX_TOKENS:
            break
    return tokens


def search_fts(conn, query: str, limit: int = 200) -> dict[str, float]:
    """Search messages + attachments via pg_search BM25, return {message_id: score}.

    Two-pass strategy mirroring the historic SQLite FTS5 behaviour:
      1. Phrase pass (higher weight).
      2. Disjunction pass (broader recall).
    Scores are min-max normalized to [0, 1] so downstream blenders can mix
    FTS with vector similarity.
    """
    return _search_fts_postgres(conn, query, limit)


def _pg_bm25_messages(
    conn,
    bm25_query: str,
    limit: int,
    logger,
) -> dict[str, float]:
    """BM25 pass against the `messages` table via pg_search.

    Uses the `@@@` operator (Tantivy BM25 match) and `paradedb.score(id)`
    for a real BM25 score. The query string is interpreted by Tantivy's
    query parser, which natively handles phrase quoting (`"foo bar"`),
    boolean operators, and per-field targeting.
    """
    scores: dict[str, float] = {}
    try:
        rows = conn.execute(
            "SELECT id AS message_id, paradedb.score(id) AS rank "
            "FROM messages "
            "WHERE messages @@@ %s "
            "ORDER BY rank DESC LIMIT %s",
            (bm25_query, limit),
        ).fetchall()
        for r in rows:
            mid = r["message_id"]
            raw = float(r["rank"] or 0.0)
            if mid not in scores or raw > scores[mid]:
                scores[mid] = raw
    except Exception as e:  # paranoid catch — FTS must never fail the request
        logger.exception(f"PG BM25 error on messages: {e!s} | query={bm25_query!r}")
    return scores


def _pg_bm25_attachments(
    conn,
    bm25_query: str,
    limit: int,
    logger,
) -> dict[str, float]:
    """BM25 pass against `attachments`. Returns `{message_id: score}` by
    joining the attachment row's `message_id`. `paradedb.score(id)` keys
    on the attachment PK (the BM25 `key_field`), so the score is per
    attachment; multiple attachment hits on the same message collapse to
    the max score in the caller.
    """
    scores: dict[str, float] = {}
    try:
        rows = conn.execute(
            "SELECT message_id, paradedb.score(id) AS rank "
            "FROM attachments "
            "WHERE attachments @@@ %s "
            "ORDER BY rank DESC LIMIT %s",
            (bm25_query, limit),
        ).fetchall()
        for r in rows:
            mid = r["message_id"]
            raw = float(r["rank"] or 0.0)
            if mid not in scores or raw > scores[mid]:
                scores[mid] = raw
    except Exception as e:
        logger.exception(f"PG BM25 error on attachments: {e!s} | query={bm25_query!r}")
    return scores


# Field lists for the BM25 multi-field query builder. pg_search's Tantivy
# parser requires explicit `field:term` syntax — an unqualified `invoice`
# searches nothing. We mirror the SQLite FTS column set: the `messages`
# FTS5 virtual table indexed {subject, from, to, body}; `attachments`
# indexed {filename, extracted_text}. Keep these in sync with the
# `USING bm25 (...)` column list in pg_schema.sql.
_BM25_MESSAGE_FIELDS = ("subject", "from_addr", "to_addr", "body_text")
_BM25_ATTACHMENT_FIELDS = ("filename", "extracted_text")


def _build_bm25_query(tokens: list[str], fields: tuple[str, ...]) -> tuple[str, str | None]:
    """Build Tantivy query strings targeting every FTS field.

    Returns `(disjunction_query, phrase_query_or_none)`:
      * Disjunction pass: `(f1:t1 f1:t2 ... f2:t1 f2:t2 ...)` — any
        token in any field matches. Tantivy's default combinator is OR.
      * Phrase pass: `(f1:"t1 t2 ..." f2:"t1 t2 ..." ...)` when there
        are ≥2 tokens — ordered-adjacent match.

    Tokens come from `_sanitize_fts_tokens()` so they are safe to
    interpolate (no quotes, backslashes, or Tantivy operators).
    """
    disjunction_terms = [f"{f}:{t}" for f in fields for t in tokens]
    disjunction = " ".join(disjunction_terms)

    phrase: str | None = None
    if len(tokens) > 1:
        phrase_body = " ".join(tokens)
        phrase = " ".join(f'{f}:"{phrase_body}"' for f in fields)
    return disjunction, phrase


def _search_fts_postgres(conn, query: str, limit: int) -> dict[str, float]:
    """Postgres BM25 path via pg_search (paradedb / Tantivy).

    Two passes:
      1. Phrase pass — `"t1 t2 t3"` — tokens must appear in order,
         adjacent. Results get a ×1.5 boost.
      2. Disjunction pass — `t1 t2 t3` (Tantivy default = OR) — any
         sanitized term can match.

    Scores come from `paradedb.score(id)` (real BM25). We min-max
    normalize across the result set so downstream blend code sees
    values in [0, 1].
    """
    import logging

    logger = logging.getLogger(__name__)
    tokens = _sanitize_fts_tokens(query)
    if not tokens:
        return {}

    msg_disj, msg_phrase = _build_bm25_query(tokens, _BM25_MESSAGE_FIELDS)
    att_disj, att_phrase = _build_bm25_query(tokens, _BM25_ATTACHMENT_FIELDS)
    scores: dict[str, float] = {}

    # Strategy 1: Phrase match (words must appear in order, adjacent).
    if msg_phrase is not None:
        for tbl_scores in (
            _pg_bm25_messages(conn, msg_phrase, limit, logger),
            _pg_bm25_attachments(conn, att_phrase, limit, logger),
        ):
            for mid, raw in tbl_scores.items():
                boosted = raw * 1.5
                if mid not in scores or boosted > scores[mid]:
                    scores[mid] = boosted

    # Strategy 2: Individual terms (any word matches — broader recall).
    for tbl_scores in (
        _pg_bm25_messages(conn, msg_disj, limit, logger),
        _pg_bm25_attachments(conn, att_disj, limit, logger),
    ):
        for mid, raw in tbl_scores.items():
            if mid not in scores or raw > scores[mid]:
                scores[mid] = raw

    # Normalize to 0-1 (single result gets 1.0).
    if scores:
        max_score = max(scores.values())
        min_score = min(scores.values())
        if max_score == min_score:
            scores = {mid: 1.0 for mid in scores}
        else:
            score_range = max_score - min_score
            scores = {mid: (s - min_score) / score_range for mid, s in scores.items()}

    return scores
