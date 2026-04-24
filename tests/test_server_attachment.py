"""Tests for the `/api/attachment/<id>/text` fallback extraction path.

The chat agent hits this endpoint to read attachment text. If the
`update --loop` daemon hasn't run its extract step yet, the DB column
`extracted_text` will be NULL even though the raw bytes are already
on disk. These tests cover the on-the-fly extraction fallback that
fills that gap.
"""

from __future__ import annotations

from pathlib import Path

from gmail_search.config import load_config
from gmail_search.store.db import get_connection, init_db
from gmail_search.store.models import Attachment, Message
from gmail_search.store.queries import (
    extract_attachment_on_demand,
    get_attachments_for_message,
    upsert_attachment,
    upsert_message,
)


def _make_message(msg_id: str = "msg-att-1") -> Message:
    from datetime import datetime

    return Message(
        id=msg_id,
        thread_id="thr-att-1",
        from_addr="sender@example.com",
        to_addr="me@example.com",
        subject="Has attachment",
        body_text="See attached.",
        body_html="<p>See attached.</p>",
        date=datetime(2026, 4, 21, 12, 0),
        labels=["INBOX"],
        history_id=1,
        raw_json='{"id": "msg-att-1"}',
    )


def _seed_attachment(conn, raw_path: Path, mime_type: str = "text/plain") -> int:
    """Insert a message + one NULL-text attachment pointing at raw_path."""
    upsert_message(conn, _make_message())
    att = Attachment(
        id=None,
        message_id="msg-att-1",
        filename=raw_path.name,
        mime_type=mime_type,
        size_bytes=raw_path.stat().st_size,
        extracted_text=None,
        image_path=None,
        raw_path=str(raw_path),
    )
    return upsert_attachment(conn, att)


# ─── extract_attachment_on_demand helper ────────────────────────────────


def test_extract_on_demand_fills_null_text(db_backend, tmp_path):
    """Happy path: NULL extracted_text + file on disk → helper extracts,
    returns the text, and persists it to the row.
    """
    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)

    raw = tmp_path / "notes.txt"
    raw.write_text("hello fallback extract path")
    att_id = _seed_attachment(conn, raw)

    result = extract_attachment_on_demand(conn, att_id, config=load_config())
    assert result is not None
    assert result.text == "hello fallback extract path"

    atts = get_attachments_for_message(conn, "msg-att-1")
    assert atts[0].extracted_text == "hello fallback extract path"
    conn.close()


def test_extract_on_demand_noop_when_already_extracted(db_backend, tmp_path):
    """Row already has text → helper returns None without touching the
    file or the DB. Guards the "zero regression on the warm path"
    requirement.
    """
    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)

    raw = tmp_path / "already.txt"
    raw.write_text("fresh disk bytes")
    upsert_message(conn, _make_message())
    att = Attachment(
        id=None,
        message_id="msg-att-1",
        filename="already.txt",
        mime_type="text/plain",
        size_bytes=raw.stat().st_size,
        extracted_text="CACHED TEXT FROM DAEMON",
        image_path=None,
        raw_path=str(raw),
    )
    att_id = upsert_attachment(conn, att)

    result = extract_attachment_on_demand(conn, att_id, config=load_config())
    assert result is None

    # DB unchanged — cached text still there, file bytes ignored.
    atts = get_attachments_for_message(conn, "msg-att-1")
    assert atts[0].extracted_text == "CACHED TEXT FROM DAEMON"
    conn.close()


def test_extract_on_demand_missing_file_returns_none(db_backend, tmp_path):
    """raw_path points at a file that no longer exists → None, no
    DB write. The endpoint will surface an empty-text response, which
    matches existing behaviour for this case.
    """
    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)

    raw = tmp_path / "gone.txt"
    raw.write_text("temp")
    att_id = _seed_attachment(conn, raw)
    raw.unlink()

    result = extract_attachment_on_demand(conn, att_id, config=load_config())
    assert result is None
    conn.close()


def test_extract_on_demand_unknown_mime_returns_none(db_backend, tmp_path):
    """Unsupported mime → dispatch returns None → helper returns None
    and doesn't write anything. Keeps unsupported attachments empty
    rather than writing garbage.
    """
    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)

    raw = tmp_path / "weird.bin"
    raw.write_bytes(b"\x00\x01\x02")
    att_id = _seed_attachment(conn, raw, mime_type="application/x-unknown")

    result = extract_attachment_on_demand(conn, att_id, config=load_config())
    assert result is None

    atts = get_attachments_for_message(conn, "msg-att-1")
    assert atts[0].extracted_text is None
    conn.close()


# ─── /api/attachment/<id>/text endpoint ─────────────────────────────────


def test_api_attachment_text_fallback_extraction(db_backend, tmp_path):
    """End-to-end: NULL extracted_text + file on disk → endpoint
    dispatches to the extractor, returns the fresh text, and persists
    it. Exercises the full FastAPI wiring through TestClient.
    """
    from fastapi.testclient import TestClient

    from gmail_search.server import create_app

    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    raw = data_dir / "memo.txt"
    raw.write_text("body content that extractor should surface")
    att_id = _seed_attachment(conn, raw)
    conn.close()

    app = create_app(db_path=db_path, data_dir=data_dir, config=load_config(data_dir=data_dir))
    client = TestClient(app)

    resp = client.get(f"/api/attachment/{att_id}/text")
    assert resp.status_code == 200
    body = resp.json()
    assert body["attachment_id"] == att_id
    assert body["extracted_text"] == "body content that extractor should surface"

    # Verify the row was actually updated so subsequent calls are instant.
    conn = get_connection(db_path)
    atts = get_attachments_for_message(conn, "msg-att-1")
    conn.close()
    assert atts[0].extracted_text == "body content that extractor should surface"


def test_api_attachment_text_returns_cached_text_unchanged(db_backend, tmp_path):
    """Zero-regression check: when extracted_text is already stored,
    the endpoint returns it as-is and does NOT re-run the extractor
    (even if the on-disk file would give different text).
    """
    from fastapi.testclient import TestClient

    from gmail_search.server import create_app

    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    raw = data_dir / "memo.txt"
    raw.write_text("NEW DISK CONTENT")

    upsert_message(conn, _make_message())
    att = Attachment(
        id=None,
        message_id="msg-att-1",
        filename="memo.txt",
        mime_type="text/plain",
        size_bytes=raw.stat().st_size,
        extracted_text="OLD CACHED TEXT",
        image_path=None,
        raw_path=str(raw),
    )
    att_id = upsert_attachment(conn, att)
    conn.close()

    app = create_app(db_path=db_path, data_dir=data_dir, config=load_config(data_dir=data_dir))
    client = TestClient(app)

    resp = client.get(f"/api/attachment/{att_id}/text")
    assert resp.status_code == 200
    assert resp.json()["extracted_text"] == "OLD CACHED TEXT"


def test_api_attachment_text_missing_returns_404(db_backend, tmp_path):
    """Nonexistent attachment id still surfaces 404, not a 500."""
    from fastapi.testclient import TestClient

    from gmail_search.server import create_app

    db_path = db_backend["db_path"]
    init_db(db_path)

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    app = create_app(db_path=db_path, data_dir=data_dir, config=load_config(data_dir=data_dir))
    client = TestClient(app)

    resp = client.get("/api/attachment/99999/text")
    assert resp.status_code == 404


# ─── /api/attachment/<id>/render_pages ──────────────────────────────────


def _tiny_pdf_bytes() -> bytes:
    """Build a minimal one-page PDF. 8-line hand-written PDF is the
    shortest valid document — enough for fitz to parse, not enough to
    test text extraction quality. Good for "can the rasterizer run at
    all" and path-check tests.
    """
    return (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 72 72]/Contents 4 0 R>>endobj\n"
        b"4 0 obj<</Length 8>>stream\n"
        b"BT ET\n"
        b"endstream endobj\n"
        b"xref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000054 00000 n \n0000000102 00000 n \n0000000163 00000 n \n"
        b"trailer<</Size 5/Root 1 0 R>>\n"
        b"startxref\n211\n"
        b"%%EOF\n"
    )


def _seed_pdf_attachment(conn, raw_path: Path) -> int:
    raw_path.write_bytes(_tiny_pdf_bytes())
    upsert_message(conn, _make_message())
    att = Attachment(
        id=None,
        message_id="msg-att-1",
        filename=raw_path.name,
        mime_type="application/pdf",
        size_bytes=raw_path.stat().st_size,
        extracted_text=None,
        image_path=None,
        raw_path=str(raw_path),
    )
    return upsert_attachment(conn, att)


def test_render_pages_non_pdf_returns_400(db_backend, tmp_path):
    """Non-PDF attachments must be rejected — the endpoint is
    rasterization-only. Keeps the surface area narrow so clients can
    rely on getting PNGs back or a clean error.
    """
    from fastapi.testclient import TestClient

    from gmail_search.server import create_app

    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    raw = data_dir / "notes.txt"
    raw.write_text("not a pdf")
    att_id = _seed_attachment(conn, raw, mime_type="text/plain")
    conn.close()

    app = create_app(db_path=db_path, data_dir=data_dir, config=load_config(data_dir=data_dir))
    client = TestClient(app)

    resp = client.get(f"/api/attachment/{att_id}/render_pages")
    assert resp.status_code == 400
    assert "pdf" in resp.json()["error"].lower()


def test_render_pages_missing_file_returns_404(db_backend, tmp_path):
    """raw_path pointing at a gone-from-disk file → 404, not a 500.
    Race condition: user deletes data/attachments/<msg>/foo.pdf while
    the chat agent is mid-turn.
    """
    from fastapi.testclient import TestClient

    from gmail_search.server import create_app

    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    raw = data_dir / "ghost.pdf"
    att_id = _seed_pdf_attachment(conn, raw)
    raw.unlink()
    conn.close()

    app = create_app(db_path=db_path, data_dir=data_dir, config=load_config(data_dir=data_dir))
    client = TestClient(app)

    resp = client.get(f"/api/attachment/{att_id}/render_pages")
    assert resp.status_code == 404


def test_render_pages_bad_pages_arg_returns_400(db_backend, tmp_path):
    """Malformed `pages` query string → 400 BEFORE we touch the DB or
    open the PDF. Cheaper to fail early; also serves as a wafer-thin
    input-validation layer for an endpoint that feeds pymupdf.
    """
    from fastapi.testclient import TestClient

    from gmail_search.server import create_app

    db_path = db_backend["db_path"]
    init_db(db_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    app = create_app(db_path=db_path, data_dir=data_dir, config=load_config(data_dir=data_dir))
    client = TestClient(app)

    resp = client.get("/api/attachment/1/render_pages?pages=not,ints")
    assert resp.status_code == 400


def test_render_pages_path_traversal_is_rejected(db_backend, tmp_path):
    """A poisoned raw_path outside data_dir must yield 403 — defense in
    depth against a poisoned `attachments.raw_path` row. Uses
    `is_relative_to` (not string-prefix) so `/data-evil/...` can't sneak
    past when data_dir is `/data`.
    """
    from fastapi.testclient import TestClient

    from gmail_search.server import create_app

    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # Raw file lives OUTSIDE data_dir.
    evil = tmp_path / "evil.pdf"
    evil.write_bytes(_tiny_pdf_bytes())
    upsert_message(conn, _make_message())
    att = Attachment(
        id=None,
        message_id="msg-att-1",
        filename="evil.pdf",
        mime_type="application/pdf",
        size_bytes=evil.stat().st_size,
        extracted_text=None,
        image_path=None,
        raw_path=str(evil),
    )
    att_id = upsert_attachment(conn, att)
    conn.close()

    app = create_app(db_path=db_path, data_dir=data_dir, config=load_config(data_dir=data_dir))
    client = TestClient(app)

    resp = client.get(f"/api/attachment/{att_id}/render_pages")
    assert resp.status_code == 403
