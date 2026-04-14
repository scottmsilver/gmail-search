import struct
from datetime import datetime

from gmail_search.index.builder import build_index, load_index_metadata
from gmail_search.store.db import get_connection, init_db
from gmail_search.store.models import EmbeddingRecord, Message
from gmail_search.store.queries import insert_embedding, upsert_message


def _make_embedding(dims=16):
    vec = [float(i) / dims for i in range(dims)]
    return struct.pack(f"{dims}f", *vec)


def test_build_index(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = get_connection(db_path)

    for i in range(50):
        msg = Message(
            id=f"msg{i}",
            thread_id="t1",
            from_addr="a@b.com",
            to_addr="c@d.com",
            subject="Test",
            body_text="Hello",
            body_html="",
            date=datetime(2025, 1, 1),
            labels=[],
            history_id=1,
            raw_json="{}",
        )
        upsert_message(conn, msg)
        insert_embedding(
            conn,
            EmbeddingRecord(
                id=None,
                message_id=f"msg{i}",
                attachment_id=None,
                chunk_type="message",
                chunk_text="test",
                embedding=_make_embedding(16),
                model="test-model",
            ),
        )
    conn.close()

    index_dir = tmp_path / "scann_index"
    build_index(db_path, index_dir, model="test-model", dimensions=16)
    assert index_dir.exists()

    ids = load_index_metadata(index_dir)
    assert len(ids) == 50


def test_build_index_empty_db(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    index_dir = tmp_path / "scann_index"
    build_index(db_path, index_dir, model="test-model", dimensions=16)
