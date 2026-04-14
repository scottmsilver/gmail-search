import struct
from datetime import datetime

import numpy as np

from gmail_search.index.builder import build_index
from gmail_search.index.searcher import ScannSearcher
from gmail_search.store.db import get_connection, init_db
from gmail_search.store.models import EmbeddingRecord, Message
from gmail_search.store.queries import insert_embedding, upsert_message


def _make_vec(dims, value):
    return struct.pack(f"{dims}f", *([value] * dims))


def test_searcher_returns_nearest(tmp_path):
    dims = 16
    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = get_connection(db_path)

    for i in range(50):
        msg = Message(
            id=f"msg{i}",
            thread_id="t1",
            from_addr="a@b.com",
            to_addr="c@d.com",
            subject=f"Message {i}",
            body_text="test",
            body_html="",
            date=datetime(2025, 1, 1),
            labels=[],
            history_id=1,
            raw_json="{}",
        )
        upsert_message(conn, msg)
        val = float(i) / 50
        insert_embedding(
            conn,
            EmbeddingRecord(
                id=None,
                message_id=f"msg{i}",
                attachment_id=None,
                chunk_type="message",
                chunk_text=f"Message {i}",
                embedding=_make_vec(dims, val),
                model="test-model",
            ),
        )
    conn.close()

    index_dir = tmp_path / "scann_index"
    build_index(db_path, index_dir, model="test-model", dimensions=dims)

    searcher = ScannSearcher(index_dir, dimensions=dims)

    query = np.array([0.98] * dims, dtype=np.float32)
    embedding_ids, scores = searcher.search(query, top_k=5)

    assert len(embedding_ids) == 5
    assert len(scores) == 5
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1]
