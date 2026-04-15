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

CREATE TABLE IF NOT EXISTS topics (
    topic_id INTEGER PRIMARY KEY,
    label TEXT NOT NULL DEFAULT '',
    message_count INTEGER NOT NULL DEFAULT 0,
    top_senders TEXT NOT NULL DEFAULT '[]',
    sample_subjects TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS message_topics (
    message_id TEXT PRIMARY KEY REFERENCES messages(id),
    topic_id INTEGER NOT NULL REFERENCES topics(topic_id)
);

CREATE INDEX IF NOT EXISTS idx_message_topics_topic ON message_topics(topic_id);

CREATE TABLE IF NOT EXISTS contact_frequency (
    email TEXT PRIMARY KEY,
    message_count INTEGER NOT NULL DEFAULT 0,
    score REAL NOT NULL DEFAULT 0.0
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


def _load_message_embeddings(conn, limit=50000):
    """Load message embeddings with metadata for clustering."""
    import struct

    import numpy as np

    rows = conn.execute(
        """SELECT e.message_id, e.embedding, m.subject, m.from_addr
           FROM embeddings e JOIN messages m ON e.message_id = m.id
           WHERE e.chunk_type = 'message'
           ORDER BY m.date DESC LIMIT ?""",
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


def _kmeans_cluster(vectors, n_clusters, n_iterations=15, seed=42):
    """Run k-means++ on vectors. Returns cluster labels (numpy array)."""
    import numpy as np

    rng = np.random.RandomState(seed)

    # K-means++ initialization: pick first centroid randomly, then pick
    # subsequent centroids with probability proportional to distance
    centroids = [vectors[rng.randint(len(vectors))]]
    for _ in range(n_clusters - 1):
        dists = np.min([np.sum((vectors - c) ** 2, axis=1) for c in centroids], axis=0)
        probs = dists / (dists.sum() + 1e-10)
        centroids.append(vectors[rng.choice(len(vectors), p=probs)])
    centroids = np.array(centroids)

    # Iterate: assign each vector to nearest centroid, then update centroids
    for _ in range(n_iterations):
        sims = vectors @ centroids.T
        labels = np.argmax(sims, axis=1)
        for k in range(n_clusters):
            mask = labels == k
            if mask.sum() > 0:
                centroids[k] = vectors[mask].mean(axis=0)

    return labels


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


def _store_cluster(conn, cluster_id, msg_ids, indices, top_senders, sample_subjects):
    """Write one cluster's topic row and message assignments to the DB."""
    import json

    conn.execute(
        "INSERT INTO topics (topic_id, label, message_count, top_senders, sample_subjects) VALUES (?, ?, ?, ?, ?)",
        (cluster_id, "", len(indices), json.dumps(top_senders), json.dumps(sample_subjects)),
    )
    for idx in indices:
        conn.execute(
            "INSERT OR REPLACE INTO message_topics (message_id, topic_id) VALUES (?, ?)",
            (msg_ids[idx], cluster_id),
        )


def _auto_label_topics(conn):
    """Use Gemini Flash Lite to generate short topic labels from cluster summaries."""
    import json
    import logging

    logger = logging.getLogger(__name__)
    topic_rows = conn.execute(
        "SELECT topic_id, top_senders, sample_subjects, message_count FROM topics ORDER BY message_count DESC"
    ).fetchall()

    # Build a summary of each cluster for the LLM
    summaries = []
    for t in topic_rows:
        sndrs = json.loads(t["top_senders"])
        subjs = json.loads(t["sample_subjects"])
        summaries.append(
            f"Cluster {t['topic_id']} ({t['message_count']} msgs): "
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
                "Label each email cluster with a short 2-4 word topic name. "
                "Return ONLY a JSON object mapping cluster number to label. "
                'Example: {"0": "Shopping Deals", "3": "Travel Plans"}\n\n' + "\n".join(summaries)
            ),
        )
        text = response.text.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        label_map = json.loads(text)

        for tid_str, label in label_map.items():
            conn.execute("UPDATE topics SET label = ? WHERE topic_id = ?", (label, int(tid_str)))
        conn.commit()
        logger.info(f"Auto-labeled {len(label_map)} topics")

    except Exception as e:
        logger.warning(f"Auto-labeling failed, using top sender as label: {e}")
        for t in topic_rows:
            sndrs = json.loads(t["top_senders"])
            label = sndrs[0] if sndrs else f"Topic {t['topic_id']}"
            conn.execute("UPDATE topics SET label = ? WHERE topic_id = ?", (label, t["topic_id"]))
        conn.commit()


def rebuild_topics(db_path: Path, n_clusters: int = 25) -> int:
    """Cluster messages into topics using k-means on embeddings. Returns topic count."""
    import logging

    import numpy as np

    logger = logging.getLogger(__name__)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    data = _load_message_embeddings(conn)
    if len(data["msg_ids"]) < n_clusters * 5:
        logger.warning(f"Not enough messages ({len(data['msg_ids'])}) for {n_clusters} clusters")
        conn.close()
        return 0

    logger.info(f"Clustering {len(data['msg_ids'])} messages into {n_clusters} topics")
    labels = _kmeans_cluster(data["vectors"], n_clusters)

    conn.execute("DELETE FROM message_topics")
    conn.execute("DELETE FROM topics")

    for cluster_id in range(n_clusters):
        indices = np.where(labels == cluster_id)[0]
        if len(indices) == 0:
            continue
        top_senders, sample_subjects = _summarize_cluster(indices, data["subjects"], data["senders"])
        _store_cluster(conn, cluster_id, data["msg_ids"], indices, top_senders, sample_subjects)

    conn.commit()
    _auto_label_topics(conn)

    count = conn.execute("SELECT COUNT(*) FROM topics WHERE message_count > 0").fetchone()[0]
    conn.close()
    return count


def rebuild_spell_dictionary(db_path: Path, data_dir: Path) -> int:
    """Build a word frequency dictionary from the email corpus for spell correction."""
    import re
    from collections import Counter

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

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
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row

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
            "INSERT OR REPLACE INTO contact_frequency (email, message_count, score) VALUES (?, ?, ?)",
            (addr, r["c"], score),
        )

    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM contact_frequency").fetchone()[0]
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
