"""Behavioral tests for rebuild_spell_dictionary's noise filter.

The corpus dictionary used to include every alpha token seen even once —
5.4M terms, 70% count==1 base64/message-id shrapnel — which cost SymSpell
~6.6 GB and (worse) stored the corpus's own typos as "correct" words,
blocking their correction. The builder now floors counts at 20 and caps
term length at 24 when writing the dictionary.
"""

import struct
from datetime import datetime

from gmail_search.store.db import get_connection, init_db, rebuild_spell_dictionary
from gmail_search.store.models import EmbeddingRecord, Message
from gmail_search.store.queries import insert_embedding, upsert_message


def _add_message(conn, i: int, body: str, from_addr: str = "a@b.com") -> str:
    mid = f"sp{i:05d}"
    upsert_message(
        conn,
        Message(
            id=mid,
            thread_id="t1",
            from_addr=from_addr,
            to_addr="c@d.com",
            subject="hello",
            body_text=body,
            body_html="",
            date=datetime(2025, 1, 1),
            labels=[],
            history_id=1,
            raw_json="{}",
        ),
    )
    insert_embedding(
        conn,
        EmbeddingRecord(
            id=None,
            message_id=mid,
            attachment_id=None,
            chunk_type="message",
            chunk_text=body,
            embedding=struct.pack("4f", 0.1, 0.2, 0.3, 0.4),
            model="test-model",
        ),
    )
    return mid


def test_spell_dictionary_floors_counts_and_caps_length(tmp_path):
    db = tmp_path / "t.db"
    init_db(db)
    conn = get_connection(db)
    long_token = "x" * 30  # base64-ish shrapnel: frequent but absurdly long
    for i in range(25):
        # per-message unique junk must be pure alpha — the tokenizer is
        # [a-zA-Z]+, so a digit suffix would collapse back to one word
        junk = f"zqxjunk{chr(97 + i)}"
        _add_message(conn, i, f"mortgage insurance {long_token} {junk}")
    conn.commit()
    conn.close()

    n = rebuild_spell_dictionary(db, tmp_path, full_rebuild=True)

    uid_dirs = list((tmp_path / "users").iterdir())
    assert len(uid_dirs) == 1
    entries = dict(line.rsplit(" ", 1) for line in (uid_dirs[0] / "spell_dictionary.txt").read_text().splitlines())
    # Repeated real words survive the floor.
    assert "mortgage" in entries and int(entries["mortgage"]) >= 25
    assert "insurance" in entries
    # Count-1 noise is filtered (each zqxjunk<i> appears exactly once).
    assert not any(t.startswith("zqxjunk") for t in entries)
    # Long tokens are filtered no matter how frequent.
    assert long_token not in entries
    assert n == len(entries)


def test_spell_dictionary_sender_boost_survives_floor(tmp_path):
    """A contact name seen only once still lands in the dictionary: the
    +50 sender boost lifts it past the count floor, so rare-but-real
    people remain correction targets."""
    db = tmp_path / "t.db"
    init_db(db)
    conn = get_connection(db)
    for i in range(25):
        _add_message(conn, i, "regular body words appearing often")
    _add_message(conn, 99, "one-off note", from_addr='"Georgina Kalitzki" <gk@x.com>')
    conn.commit()
    conn.close()

    rebuild_spell_dictionary(db, tmp_path, full_rebuild=True)

    uid_dirs = list((tmp_path / "users").iterdir())
    text = (uid_dirs[0] / "spell_dictionary.txt").read_text()
    assert "georgina" in text and "kalitzki" in text
