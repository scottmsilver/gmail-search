import random
import struct
from datetime import datetime

import numpy as np

from gmail_search.index.builder import _load_embeddings_matrix, build_index, load_index_metadata
from gmail_search.store.db import get_connection, init_db
from gmail_search.store.models import EmbeddingRecord, Message
from gmail_search.store.queries import insert_embedding, upsert_message


def _make_embedding(dims=16):
    vec = [float(i) / dims for i in range(dims)]
    return struct.pack(f"{dims}f", *vec)


def _make_random_embedding(dims: int, seed: int) -> bytes:
    rng = random.Random(seed)
    vec = [rng.uniform(-1.0, 1.0) for _ in range(dims)]
    return struct.pack(f"{dims}f", *vec)


def _legacy_load_matrix(conn, model: str, dimensions: int):
    """The pre-fix implementation — kept here only as a parity oracle."""
    rows = conn.execute("SELECT id, embedding FROM embeddings WHERE model = ? ORDER BY id", (model,)).fetchall()
    ids = [r["id"] for r in rows]
    vectors = np.array(
        [list(struct.unpack(f"{dimensions}f", r["embedding"])) for r in rows],
        dtype=np.float32,
    )
    return ids, vectors


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


def test_load_embeddings_matrix_matches_legacy(tmp_path):
    """The streaming loader must produce byte-identical output to the old
    list-of-lists-of-Python-floats path. This is the before/after parity
    check for the OOM fix in index/builder.py.
    """
    dims = 128
    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = get_connection(db_path)

    for i in range(75):
        msg = Message(
            id=f"msg{i:03d}",
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
                message_id=f"msg{i:03d}",
                attachment_id=None,
                chunk_type="message",
                chunk_text="test",
                embedding=_make_random_embedding(dims, seed=i),
                model="test-model",
            ),
        )
    # An embedding from a different model must be ignored.
    insert_embedding(
        conn,
        EmbeddingRecord(
            id=None,
            message_id="msg000",
            attachment_id=None,
            chunk_type="message",
            chunk_text="other",
            embedding=_make_random_embedding(dims, seed=999),
            model="other-model",
        ),
    )

    legacy_ids, legacy_vectors = _legacy_load_matrix(conn, "test-model", dims)
    new_ids, new_vectors = _load_embeddings_matrix(conn, "test-model", dims)
    conn.close()

    assert new_ids == legacy_ids
    assert new_vectors.shape == legacy_vectors.shape == (75, dims)
    assert new_vectors.dtype == np.float32
    assert np.array_equal(new_vectors, legacy_vectors)


def test_load_embeddings_matrix_empty(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = get_connection(db_path)
    ids, vectors = _load_embeddings_matrix(conn, "test-model", 16)
    conn.close()
    assert ids == []
    assert vectors.shape == (0, 16)
    assert vectors.dtype == np.float32
