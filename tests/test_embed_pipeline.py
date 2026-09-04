from unittest.mock import MagicMock

from gmail_search.config import load_config
from gmail_search.embed.pipeline import run_embedding_pipeline
from gmail_search.store.cost import get_total_spend
from gmail_search.store.db import get_connection, init_db
from gmail_search.store.models import Message
from gmail_search.store.queries import embedding_exists, upsert_message


def _fake_vector(dims=3072):
    return [0.1] * dims


def _mock_embedder() -> MagicMock:
    emb = MagicMock()
    emb.model = "test-model"
    emb.dimensions = 3072
    emb.embed_texts_batch.return_value = [_fake_vector()]
    return emb


def _test_config(tmp_path):
    cfg = load_config(data_dir=tmp_path / "data")
    cfg["embedding"]["model"] = "test-model"
    return cfg


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

    cfg = _test_config(tmp_path)

    mock_embedder = _mock_embedder()
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

    cfg = _test_config(tmp_path)

    mock_embedder = _mock_embedder()
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

    cfg = _test_config(tmp_path)

    mock_embedder = _mock_embedder()
    mock_embedder.embed_texts_batch.return_value = [_fake_vector()]

    run_embedding_pipeline(db_path, cfg, embedder=mock_embedder)

    conn = get_connection(db_path)
    total = get_total_spend(conn)
    assert total > 0
    conn.close()


# ─── Image embed failures are recorded, not retried forever (issue #12) ──


def _png(path, size=(8, 8)):
    from PIL import Image

    Image.new("RGB", size).save(path, format="PNG")
    return path


def _add_attachment(
    conn, msg_id: str, *, filename: str, mime_type: str, image_path
) -> int:
    from gmail_search.store.models import Attachment
    from gmail_search.store.queries import upsert_attachment

    return upsert_attachment(
        conn,
        Attachment(
            id=None,
            message_id=msg_id,
            filename=filename,
            mime_type=mime_type,
            size_bytes=1,
            image_path=str(image_path),
        ),
    )


def _single_image_attachment(conn, msg_id, tmp_path) -> int:
    return _add_attachment(
        conn,
        msg_id,
        filename="pic.png",
        mime_type="image/png",
        image_path=_png(tmp_path / "pic.png"),
    )


def _page_dir_attachment(conn, msg_id, tmp_path, pages=2) -> int:
    """PDF-style attachment: image_path is a directory of page_NNNN.png."""
    d = tmp_path / "doc_pages"
    d.mkdir()
    for i in range(pages):
        _png(d / f"page_{i + 1:04d}.png")
    return _add_attachment(
        conn, msg_id, filename="doc.pdf", mime_type="application/pdf", image_path=d
    )


def _pipeline_setup(tmp_path, make_attachment=_single_image_attachment):
    """DB with one message + one attachment; returns (db_path, cfg, embedder, att_id)."""
    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = get_connection(db_path)
    upsert_message(conn, _make_msg("msg1"))
    att_id = make_attachment(conn, "msg1", tmp_path)
    conn.commit()
    conn.close()
    return db_path, _test_config(tmp_path), _mock_embedder(), att_id


def _embed_row(conn, att_id: int):
    return conn.execute(
        "SELECT embed_status, embed_error, embed_attempts FROM attachments WHERE id = %s",
        (att_id,),
    ).fetchone()


def _read_embed_row(db_path, att_id: int):
    conn = get_connection(db_path)
    try:
        return _embed_row(conn, att_id)
    finally:
        conn.close()


def _api_error(code: int, status: str):
    """A google-genai error of the class the SDK would raise for `code`."""
    from google.genai import errors

    cls = errors.ClientError if code < 500 else errors.ServerError
    return cls(
        code,
        {
            "error": {
                "code": code,
                "message": "Provided image is not valid.",
                "status": status,
            }
        },
    )


_BAD_REQUEST = lambda: _api_error(400, "INVALID_ARGUMENT")  # noqa: E731
_UNAVAILABLE = lambda: _api_error(503, "UNAVAILABLE")  # noqa: E731


def _fail_page_1_with(exc, others=_fake_vector):
    """embed_image side effect: page_0001.png raises `exc`, other pages get `others()`."""

    def side_effect(path):
        if path.name == "page_0001.png":
            raise exc
        return others()

    return side_effect


def _raise(exc):
    def side_effect(path):
        raise exc

    return side_effect


def test_image_400_marks_attachment_failed_permanent(tmp_path):
    db_path, cfg, emb, att_id = _pipeline_setup(tmp_path)
    emb.embed_image.side_effect = _BAD_REQUEST()

    run_embedding_pipeline(db_path, cfg, embedder=emb)

    row = _read_embed_row(db_path, att_id)
    assert row["embed_status"] == "failed_permanent"
    assert row["embed_attempts"] == 1
    assert "INVALID_ARGUMENT" in row["embed_error"]
    assert emb.embed_image.call_count == 1


def test_failed_permanent_attachment_is_not_selected_on_next_pass(tmp_path):
    db_path, cfg, emb, att_id = _pipeline_setup(tmp_path)
    emb.embed_image.side_effect = _BAD_REQUEST()
    run_embedding_pipeline(db_path, cfg, embedder=emb)

    emb.embed_image.reset_mock()
    emb.embed_image.side_effect = None
    emb.embed_image.return_value = _fake_vector()
    run_embedding_pipeline(db_path, cfg, embedder=emb)

    emb.embed_image.assert_not_called()
    conn = get_connection(db_path)
    assert not embedding_exists(
        conn, "msg1", att_id, "attachment_image_0", "test-model"
    )
    conn.close()


def test_invalid_image_is_permanent_without_api_call(tmp_path):
    from gmail_search.embed.client import InvalidImage

    db_path, cfg, emb, att_id = _pipeline_setup(tmp_path)
    emb.embed_image.side_effect = InvalidImage(
        "pic.png: UnidentifiedImageError: cannot identify image file"
    )

    run_embedding_pipeline(db_path, cfg, embedder=emb)

    row = _read_embed_row(db_path, att_id)
    assert row["embed_status"] == "failed_permanent"
    assert "UnidentifiedImageError" in row["embed_error"]


def test_image_503_is_retried_and_not_marked_permanent(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    db_path, cfg, emb, att_id = _pipeline_setup(tmp_path)
    emb.embed_image.side_effect = _UNAVAILABLE()

    run_embedding_pipeline(db_path, cfg, embedder=emb)

    row = _read_embed_row(db_path, att_id)
    assert row["embed_status"] is None
    assert row["embed_attempts"] == 1
    assert "503" in row["embed_error"]
    assert emb.embed_image.call_count > 1  # in-pass backoff still happens

    emb.embed_image.reset_mock()
    emb.embed_image.side_effect = None
    emb.embed_image.return_value = _fake_vector()
    run_embedding_pipeline(db_path, cfg, embedder=emb)
    emb.embed_image.assert_called_once()
    conn = get_connection(db_path)
    assert embedding_exists(conn, "msg1", att_id, "attachment_image_0", "test-model")
    conn.close()


def test_transient_failures_go_permanent_after_attempt_cap(tmp_path, monkeypatch):
    from gmail_search.embed.pipeline import MAX_IMAGE_EMBED_ATTEMPTS

    monkeypatch.setattr("time.sleep", lambda s: None)
    db_path, cfg, emb, att_id = _pipeline_setup(tmp_path)
    emb.embed_image.side_effect = _UNAVAILABLE()

    for _ in range(MAX_IMAGE_EMBED_ATTEMPTS - 1):
        run_embedding_pipeline(db_path, cfg, embedder=emb)
    assert _read_embed_row(db_path, att_id)["embed_status"] is None

    run_embedding_pipeline(db_path, cfg, embedder=emb)
    row = _read_embed_row(db_path, att_id)
    assert row["embed_status"] == "failed_permanent"
    assert row["embed_attempts"] == MAX_IMAGE_EMBED_ATTEMPTS


def test_embed_error_is_sanitized_and_capped(tmp_path):
    from gmail_search.embed.client import InvalidImage

    db_path, cfg, emb, att_id = _pipeline_setup(tmp_path)
    emb.embed_image.side_effect = InvalidImage("bad\x00\x1bbytes\n" + "x" * 5000)

    run_embedding_pipeline(db_path, cfg, embedder=emb)

    err = _read_embed_row(db_path, att_id)["embed_error"]
    assert "\x00" not in err and "\x1b" not in err
    assert err.startswith("pic.png: bad")  # which image, then why
    assert len(err) <= 500


def test_one_bad_page_does_not_block_the_good_pages(tmp_path):
    db_path, cfg, emb, att_id = _pipeline_setup(tmp_path, _page_dir_attachment)
    emb.embed_image.side_effect = _fail_page_1_with(_BAD_REQUEST())

    run_embedding_pipeline(db_path, cfg, embedder=emb)

    conn = get_connection(db_path)
    assert not embedding_exists(
        conn, "msg1", att_id, "attachment_image_0", "test-model"
    )
    assert embedding_exists(conn, "msg1", att_id, "attachment_image_1", "test-model")
    conn.close()
    row = _read_embed_row(db_path, att_id)
    # Nothing left to do for this attachment, so it is now permanent.
    assert row["embed_status"] == "failed_permanent"
    assert row["embed_attempts"] == 1


def test_permanent_page_plus_transient_page_keeps_attachment_retryable(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("time.sleep", lambda s: None)
    db_path, cfg, emb, att_id = _pipeline_setup(tmp_path, _page_dir_attachment)
    emb.embed_image.side_effect = _fail_page_1_with(
        _BAD_REQUEST(), others=_raise(_UNAVAILABLE())
    )
    run_embedding_pipeline(db_path, cfg, embedder=emb)

    row = _read_embed_row(db_path, att_id)
    # Page 2 might still succeed later, so the attachment must stay selectable;
    # one failed pass = one attempt regardless of how many pages failed.
    assert row["embed_status"] is None
    assert row["embed_attempts"] == 1

    emb.embed_image.reset_mock()
    emb.embed_image.side_effect = _fail_page_1_with(_BAD_REQUEST())
    run_embedding_pipeline(db_path, cfg, embedder=emb)
    conn = get_connection(db_path)
    assert embedding_exists(conn, "msg1", att_id, "attachment_image_1", "test-model")
    conn.close()
    assert _read_embed_row(db_path, att_id)["embed_status"] == "failed_permanent"


def test_permanent_error_classification_unwraps_retry_wrapper():
    from gmail_search.embed.pipeline import _is_permanent_image_error

    wrapped = RuntimeError("API call failed after 5 retries: 400 ...")
    wrapped.__cause__ = _BAD_REQUEST()
    assert _is_permanent_image_error(wrapped) is True
    assert _is_permanent_image_error(_api_error(429, "RESOURCE_EXHAUSTED")) is False
    assert _is_permanent_image_error(_api_error(408, "REQUEST_TIMEOUT")) is False
    assert _is_permanent_image_error(_UNAVAILABLE()) is False
    assert _is_permanent_image_error(RuntimeError("no cause")) is False


def test_embed_error_does_not_double_the_filename(tmp_path):
    from gmail_search.embed.client import InvalidImage

    db_path, cfg, emb, att_id = _pipeline_setup(tmp_path)
    # InvalidImage messages already start with the file name.
    emb.embed_image.side_effect = InvalidImage(
        "pic.png: UnidentifiedImageError: cannot identify image file"
    )

    run_embedding_pipeline(db_path, cfg, embedder=emb)

    err = _read_embed_row(db_path, att_id)["embed_error"]
    assert err.startswith("pic.png: UnidentifiedImageError")
    assert err.count("pic.png") == 1
