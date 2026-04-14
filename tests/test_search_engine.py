import struct
from datetime import datetime
from unittest.mock import MagicMock

from gmail_search.config import load_config
from gmail_search.index.builder import build_index
from gmail_search.search.engine import SearchEngine, SearchResult
from gmail_search.store.db import get_connection, init_db
from gmail_search.store.models import Attachment, EmbeddingRecord, Message
from gmail_search.store.queries import insert_embedding, upsert_attachment, upsert_message


def _make_vec(dims, value):
    return struct.pack(f"{dims}f", *([value] * dims))


def _setup_db_with_index(tmp_path, dims=16, n=50):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = get_connection(db_path)

    for i in range(n):
        msg = Message(
            id=f"msg{i}",
            thread_id=f"t{i}",
            from_addr=f"user{i}@test.com",
            to_addr="me@test.com",
            subject=f"Subject {i}",
            body_text=f"Body of message {i}",
            body_html="",
            date=datetime(2025, 1, 1 + i % 28),
            labels=["INBOX"],
            history_id=i,
            raw_json="{}",
        )
        upsert_message(conn, msg)
        val = float(i) / n
        insert_embedding(
            conn,
            EmbeddingRecord(
                id=None,
                message_id=f"msg{i}",
                attachment_id=None,
                chunk_type="message",
                chunk_text=f"Body of message {i}",
                embedding=_make_vec(dims, val),
                model="test-model",
            ),
        )
    conn.close()

    index_dir = tmp_path / "scann_index"
    build_index(db_path, index_dir, model="test-model", dimensions=dims)
    return db_path, index_dir


def test_search_engine_returns_results(tmp_path):
    dims = 16
    db_path, index_dir = _setup_db_with_index(tmp_path, dims=dims)

    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.9] * dims
    mock_embedder.model = "test-model"
    mock_embedder.dimensions = dims

    cfg = load_config(data_dir=tmp_path / "data")
    cfg["embedding"]["model"] = "test-model"
    cfg["embedding"]["dimensions"] = dims

    engine = SearchEngine(db_path, index_dir, cfg, embedder=mock_embedder)
    results = engine.search("test query", top_k=5)

    assert len(results) == 5
    assert all(isinstance(r, SearchResult) for r in results)
    assert all(r.message_id.startswith("msg") for r in results)
    assert all(r.subject.startswith("Subject") for r in results)
    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score


def test_search_engine_deduplicates_by_message(tmp_path):
    dims = 16
    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = get_connection(db_path)

    msg = Message(
        id="msg1",
        thread_id="t1",
        from_addr="a@b.com",
        to_addr="c@d.com",
        subject="Duped",
        body_text="Body",
        body_html="",
        date=datetime(2025, 1, 1),
        labels=[],
        history_id=1,
        raw_json="{}",
    )
    upsert_message(conn, msg)

    att_id = upsert_attachment(
        conn,
        Attachment(
            id=None,
            message_id="msg1",
            filename="doc.pdf",
            mime_type="application/pdf",
            size_bytes=1024,
        ),
    )

    vec = _make_vec(dims, 0.9)
    insert_embedding(
        conn,
        EmbeddingRecord(
            id=None,
            message_id="msg1",
            attachment_id=None,
            chunk_type="message",
            chunk_text="Body",
            embedding=vec,
            model="test-model",
        ),
    )
    insert_embedding(
        conn,
        EmbeddingRecord(
            id=None,
            message_id="msg1",
            attachment_id=att_id,
            chunk_type="attachment_text",
            chunk_text="Attachment",
            embedding=vec,
            model="test-model",
        ),
    )

    for i in range(48):
        m = Message(
            id=f"pad{i}",
            thread_id="t",
            from_addr="x@x.com",
            to_addr="y@y.com",
            subject="Pad",
            body_text="pad",
            body_html="",
            date=datetime(2025, 1, 1),
            labels=[],
            history_id=1,
            raw_json="{}",
        )
        upsert_message(conn, m)
        insert_embedding(
            conn,
            EmbeddingRecord(
                id=None,
                message_id=f"pad{i}",
                attachment_id=None,
                chunk_type="message",
                chunk_text="pad",
                embedding=_make_vec(dims, 0.01),
                model="test-model",
            ),
        )
    conn.close()

    index_dir = tmp_path / "scann_index"
    build_index(db_path, index_dir, model="test-model", dimensions=dims)

    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.9] * dims
    mock_embedder.model = "test-model"
    mock_embedder.dimensions = dims

    cfg = load_config(data_dir=tmp_path / "data")
    cfg["embedding"]["model"] = "test-model"
    cfg["embedding"]["dimensions"] = dims

    engine = SearchEngine(db_path, index_dir, cfg, embedder=mock_embedder)
    results = engine.search("test", top_k=10)

    msg1_results = [r for r in results if r.message_id == "msg1"]
    assert len(msg1_results) == 1
