from unittest.mock import MagicMock

from gmail_search.config import load_config
from gmail_search.embed.pipeline import run_embedding_pipeline
from gmail_search.store.cost import get_total_spend
from gmail_search.store.db import get_connection, init_db
from gmail_search.store.models import Message
from gmail_search.store.queries import embedding_exists, upsert_message


def _fake_vector(dims=3072):
    return [0.1] * dims


def _make_msg(id="msg1"):
    from datetime import datetime

    return Message(
        id=id,
        thread_id="t1",
        from_addr="a@b.com",
        to_addr="c@d.com",
        subject="Test",
        body_text="Hello world",
        body_html="",
        date=datetime(2025, 1, 1),
        labels=[],
        history_id=1,
        raw_json="{}",
    )


def test_pipeline_embeds_messages(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = get_connection(db_path)
    upsert_message(conn, _make_msg("msg1"))
    upsert_message(conn, _make_msg("msg2"))
    conn.close()

    cfg = load_config(data_dir=tmp_path / "data")
    cfg["embedding"]["model"] = "test-model"

    mock_embedder = MagicMock()
    mock_embedder.model = "test-model"
    mock_embedder.dimensions = 3072
    mock_embedder.embed_texts_batch.return_value = [_fake_vector(), _fake_vector()]

    count = run_embedding_pipeline(db_path, cfg, embedder=mock_embedder)
    assert count == 2

    conn = get_connection(db_path)
    assert embedding_exists(conn, "msg1", None, "message", "test-model")
    assert embedding_exists(conn, "msg2", None, "message", "test-model")
    conn.close()


def test_pipeline_skips_already_embedded(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = get_connection(db_path)
    upsert_message(conn, _make_msg("msg1"))
    conn.close()

    cfg = load_config(data_dir=tmp_path / "data")
    cfg["embedding"]["model"] = "test-model"

    mock_embedder = MagicMock()
    mock_embedder.model = "test-model"
    mock_embedder.dimensions = 3072
    mock_embedder.embed_texts_batch.return_value = [_fake_vector()]

    run_embedding_pipeline(db_path, cfg, embedder=mock_embedder)
    mock_embedder.reset_mock()
    count = run_embedding_pipeline(db_path, cfg, embedder=mock_embedder)
    assert count == 0
    mock_embedder.embed_texts_batch.assert_not_called()


def test_pipeline_tracks_cost(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = get_connection(db_path)
    upsert_message(conn, _make_msg("msg1"))
    conn.close()

    cfg = load_config(data_dir=tmp_path / "data")
    cfg["embedding"]["model"] = "test-model"

    mock_embedder = MagicMock()
    mock_embedder.model = "test-model"
    mock_embedder.dimensions = 3072
    mock_embedder.embed_texts_batch.return_value = [_fake_vector()]

    run_embedding_pipeline(db_path, cfg, embedder=mock_embedder)

    conn = get_connection(db_path)
    total = get_total_spend(conn)
    assert total > 0
    conn.close()
