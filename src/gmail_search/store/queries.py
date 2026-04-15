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
    cursor = conn.execute(
        """INSERT INTO attachments (message_id, filename, mime_type, size_bytes,
           extracted_text, image_path, raw_path)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(message_id, filename) DO UPDATE SET
             mime_type=excluded.mime_type, size_bytes=excluded.size_bytes,
             raw_path=excluded.raw_path""",
        (att.message_id, att.filename, att.mime_type, att.size_bytes, att.extracted_text, att.image_path, att.raw_path),
    )
    conn.commit()
    return cursor.lastrowid


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


def insert_embedding(conn: sqlite3.Connection, rec: EmbeddingRecord) -> int:
    cursor = conn.execute(
        """INSERT INTO embeddings (message_id, attachment_id, chunk_type,
           chunk_text, embedding, model)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (rec.message_id, rec.attachment_id, rec.chunk_type, rec.chunk_text, rec.embedding, rec.model),
    )
    conn.commit()
    return cursor.lastrowid


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


def _escape_fts_query(query: str) -> str:
    """Escape a user query for safe use in FTS5 MATCH.

    Wraps each token in double quotes to prevent FTS5 operator injection.
    """
    tokens = query.split()
    return " ".join(f'"{t}"' for t in tokens if t)


def search_fts(conn: sqlite3.Connection, query: str, limit: int = 200) -> dict[str, float]:
    """Search messages + attachments via FTS5, return {message_id: bm25_score}.

    BM25 scores from FTS5 are negative (lower = better match), so we negate them.
    Returns normalized scores in 0-1 range.
    """
    import logging

    logger = logging.getLogger(__name__)
    safe_query = _escape_fts_query(query)
    if not safe_query:
        return {}

    scores: dict[str, float] = {}

    # Search messages
    try:
        rows = conn.execute(
            """SELECT message_id, rank FROM messages_fts
               WHERE messages_fts MATCH ? ORDER BY rank LIMIT ?""",
            (safe_query, limit),
        ).fetchall()
        for r in rows:
            mid = r["message_id"]
            raw = -r["rank"]  # negate: FTS5 rank is negative, lower=better
            if mid not in scores or raw > scores[mid]:
                scores[mid] = raw
    except sqlite3.OperationalError as e:
        logger.warning(f"FTS message search failed: {e}")

    # Search attachments
    try:
        rows = conn.execute(
            """SELECT message_id, rank FROM attachments_fts
               WHERE attachments_fts MATCH ? ORDER BY rank LIMIT ?""",
            (safe_query, limit),
        ).fetchall()
        for r in rows:
            mid = r["message_id"]
            raw = -r["rank"]
            if mid not in scores or raw > scores[mid]:
                scores[mid] = raw
    except sqlite3.OperationalError as e:
        logger.warning(f"FTS attachment search failed: {e}")

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
