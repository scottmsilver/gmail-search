import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    from_addr TEXT NOT NULL,
    to_addr TEXT NOT NULL,
    subject TEXT NOT NULL DEFAULT '',
    body_text TEXT NOT NULL DEFAULT '',
    body_html TEXT NOT NULL DEFAULT '',
    date TEXT NOT NULL,
    labels TEXT NOT NULL DEFAULT '[]',
    history_id INTEGER NOT NULL DEFAULT 0,
    raw_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS attachments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT NOT NULL REFERENCES messages(id),
    filename TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    size_bytes INTEGER NOT NULL DEFAULT 0,
    extracted_text TEXT,
    image_path TEXT,
    raw_path TEXT,
    UNIQUE(message_id, filename)
);

CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT NOT NULL REFERENCES messages(id),
    attachment_id INTEGER REFERENCES attachments(id),
    chunk_type TEXT NOT NULL,
    chunk_text TEXT,
    embedding BLOB NOT NULL,
    model TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS costs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    operation TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    image_count INTEGER NOT NULL DEFAULT 0,
    estimated_cost_usd REAL NOT NULL DEFAULT 0.0,
    message_id TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sync_state (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS thread_summary (
    thread_id TEXT PRIMARY KEY,
    subject TEXT NOT NULL DEFAULT '',
    participants TEXT NOT NULL DEFAULT '[]',
    all_from_addrs TEXT NOT NULL DEFAULT '[]',
    all_labels TEXT NOT NULL DEFAULT '[]',
    message_count INTEGER NOT NULL DEFAULT 0,
    date_first TEXT NOT NULL DEFAULT '',
    date_last TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_attachments_message_id ON attachments(message_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_message_id ON embeddings(message_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_lookup ON embeddings(message_id, attachment_id, chunk_type, model);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    message_id UNINDEXED,
    subject,
    body_text,
    from_addr,
    to_addr,
    tokenize='porter unicode61'
);

CREATE VIRTUAL TABLE IF NOT EXISTS attachments_fts USING fts5(
    message_id UNINDEXED,
    attachment_id UNINDEXED,
    filename,
    extracted_text,
    tokenize='porter unicode61'
);
"""


def init_db(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


def rebuild_thread_summary(db_path: Path) -> int:
    """Precompute thread metadata for fast search ranking. Returns thread count."""
    import json

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row

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
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
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


def rebuild_fts(db_path: Path) -> int:
    """Rebuild FTS index from current messages and attachments. Returns count indexed."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")

    # Clear and repopulate messages FTS
    conn.execute("DELETE FROM messages_fts")
    conn.execute(
        """INSERT INTO messages_fts (message_id, subject, body_text, from_addr, to_addr)
           SELECT id, subject, body_text, from_addr, to_addr FROM messages"""
    )

    # Clear and repopulate attachments FTS
    conn.execute("DELETE FROM attachments_fts")
    conn.execute(
        """INSERT INTO attachments_fts (message_id, attachment_id, filename, extracted_text)
           SELECT message_id, id, filename, COALESCE(extracted_text, '') FROM attachments
           WHERE extracted_text IS NOT NULL AND extracted_text != ''"""
    )

    count = conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
    att_count = conn.execute("SELECT COUNT(*) FROM attachments_fts").fetchone()[0]

    conn.commit()
    conn.close()
    return count + att_count


def get_connection(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn
