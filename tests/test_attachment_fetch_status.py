"""Attachment ingest must never silently drop a file.

Regression coverage for the 2026-09-02 bug: attachments larger than
`max_file_size_mb` (and any attachment whose download raised) were
skipped with `continue` and never got an `attachments` row, so a thread
manifest that omitted a file was indistinguishable from a thread that
never had one. Every part Gmail lists now gets a row; `fetch_status`
says whether the bytes are on disk.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from gmail_search.gmail.client import ingest_attachment
from gmail_search.gmail.parser import attachment_metas_from_raw
from gmail_search.store.db import get_connection, init_db
from gmail_search.store.models import Attachment, Message
from gmail_search.store.queries import (
    find_unfetched_attachments,
    get_attachments_for_message,
    iter_missing_attachment_metas,
    upsert_attachment,
    upsert_message,
)

MB = 1024 * 1024


# ─── fakes ─────────────────────────────────────────────────────────────


class _FakeGmailService:
    """Mimics service.users().messages().attachments().get(...).execute()."""

    def __init__(self, payload: bytes | None = None, error: Exception | None = None):
        import base64

        self._data = base64.urlsafe_b64encode(payload or b"").decode().rstrip("=")
        self._error = error
        self.calls: list[tuple[str, str]] = []

    def users(self):
        return self

    def messages(self):
        return self

    def attachments(self):
        return self

    def get(self, *, userId, messageId, id):
        self.calls.append((messageId, id))
        return self

    def execute(self):
        if self._error:
            raise self._error
        return {"data": self._data}


def _raw_with_parts(parts: list[tuple[str, int]]) -> dict:
    return {
        "id": "msg-big-1",
        "sizeEstimate": sum(s for _, s in parts) + 1000,
        "payload": {
            "mimeType": "multipart/mixed",
            "parts": [
                {"mimeType": "text/plain", "body": {"data": "aGk"}},
                {
                    "mimeType": "multipart/related",
                    "parts": [
                        {
                            "filename": fn,
                            "mimeType": "application/pdf",
                            "body": {"attachmentId": f"att-{i}", "size": size},
                        }
                        for i, (fn, size) in enumerate(parts)
                    ],
                },
            ],
        },
    }


def _message(msg_id: str = "msg-big-1", raw: dict | None = None) -> Message:
    return Message(
        id=msg_id,
        thread_id="thr-big-1",
        from_addr="lw@example.com",
        to_addr="me@example.com",
        subject="Draw Request",
        body_text="see attached",
        body_html="",
        date=datetime(2026, 1, 15, 12, 0),
        labels=["INBOX"],
        history_id=1,
        raw_json=json.dumps(raw or {"id": msg_id}),
    )


def _meta(filename="Big Draw.pdf", size=18 * MB, attachment_id="att-0") -> dict:
    return {
        "filename": filename,
        "mime_type": "application/pdf",
        "attachment_id": attachment_id,
        "size": size,
    }


@pytest.fixture
def conn(db_backend):
    init_db(db_backend["db_path"])
    c = get_connection(db_backend["db_path"])
    upsert_message(c, _message())
    yield c
    c.close()


# ─── ingest_attachment ─────────────────────────────────────────────────


def test_oversize_attachment_records_manifest_row_without_download(conn, tmp_path):
    svc = _FakeGmailService(payload=b"x")
    status = ingest_attachment(
        conn,
        svc,
        "msg-big-1",
        _meta(size=18 * MB),
        attachments_dir=tmp_path,
        max_attachment_size=10 * MB,
    )
    assert status == "skipped_too_large"
    assert svc.calls == []  # never hit the API for a file we won't keep
    (row,) = get_attachments_for_message(conn, "msg-big-1")
    assert row.filename == "Big Draw.pdf"
    assert row.raw_path is None
    assert row.size_bytes == 18 * MB  # Gmail-declared size survives
    assert row.fetch_status == "skipped_too_large"


def test_download_failure_records_fetch_failed(conn, tmp_path):
    svc = _FakeGmailService(error=RuntimeError("quota"))
    status = ingest_attachment(
        conn,
        svc,
        "msg-big-1",
        _meta(size=2 * MB),
        attachments_dir=tmp_path,
        max_attachment_size=10 * MB,
    )
    assert status == "fetch_failed"
    (row,) = get_attachments_for_message(conn, "msg-big-1")
    assert row.raw_path is None
    assert row.size_bytes == 2 * MB
    assert row.fetch_status == "fetch_failed"


def test_successful_fetch_is_ok_and_writes_file(conn, tmp_path):
    svc = _FakeGmailService(payload=b"%PDF-1.4 hello")
    status = ingest_attachment(
        conn,
        svc,
        "msg-big-1",
        _meta(size=14),
        attachments_dir=tmp_path,
        max_attachment_size=10 * MB,
    )
    assert status == "ok"
    (row,) = get_attachments_for_message(conn, "msg-big-1")
    assert row.fetch_status == "ok"
    assert row.size_bytes == 14
    assert row.raw_path == str(tmp_path / "msg-big-1" / "Big Draw.pdf")
    assert (tmp_path / "msg-big-1" / "Big Draw.pdf").read_bytes() == b"%PDF-1.4 hello"


def test_later_failure_never_clobbers_a_fetched_row(conn, tmp_path):
    ok_svc = _FakeGmailService(payload=b"bytes")
    ingest_attachment(
        conn,
        ok_svc,
        "msg-big-1",
        _meta(size=5),
        attachments_dir=tmp_path,
        max_attachment_size=10 * MB,
    )
    bad_svc = _FakeGmailService(error=RuntimeError("timeout"))
    ingest_attachment(
        conn,
        bad_svc,
        "msg-big-1",
        _meta(size=5),
        attachments_dir=tmp_path,
        max_attachment_size=10 * MB,
    )
    (row,) = get_attachments_for_message(conn, "msg-big-1")
    assert row.fetch_status == "ok"
    assert row.raw_path is not None
    assert row.size_bytes == 5


def test_retry_after_skip_upgrades_row_to_ok(conn, tmp_path):
    svc = _FakeGmailService(payload=b"p" * 12)
    ingest_attachment(
        conn,
        svc,
        "msg-big-1",
        _meta(size=12),
        attachments_dir=tmp_path,
        max_attachment_size=10,
    )
    (row,) = get_attachments_for_message(conn, "msg-big-1")
    assert row.fetch_status == "skipped_too_large"
    ingest_attachment(
        conn,
        svc,
        "msg-big-1",
        _meta(size=12),
        attachments_dir=tmp_path,
        max_attachment_size=10 * MB,
    )
    (row,) = get_attachments_for_message(conn, "msg-big-1")
    assert row.fetch_status == "ok"
    assert row.raw_path is not None


# ─── backfill discovery ────────────────────────────────────────────────


def test_find_unfetched_attachments_returns_only_non_ok_rows(conn, tmp_path):
    ingest_attachment(
        conn,
        _FakeGmailService(payload=b"ok"),
        "msg-big-1",
        _meta("small.pdf", 2, "att-s"),
        attachments_dir=tmp_path,
        max_attachment_size=10 * MB,
    )
    ingest_attachment(
        conn,
        _FakeGmailService(),
        "msg-big-1",
        _meta("huge.pdf", 18 * MB, "att-h"),
        attachments_dir=tmp_path,
        max_attachment_size=10 * MB,
    )
    rows = find_unfetched_attachments(conn)
    assert [r["filename"] for r in rows] == ["huge.pdf"]
    assert rows[0]["message_id"] == "msg-big-1"


def test_attachment_metas_from_raw_walks_nested_parts():
    raw = _raw_with_parts([("a.pdf", 100), ("b.pdf", 18 * MB)])
    metas = attachment_metas_from_raw(raw)
    assert [(m["filename"], m["size"], m["attachment_id"]) for m in metas] == [
        ("a.pdf", 100, "att-0"),
        ("b.pdf", 18 * MB, "att-1"),
    ]


def test_iter_missing_attachment_metas_finds_parts_without_rows(conn):
    """Legacy backfill: messages ingested before fetch_status existed have
    NO row for the dropped file — only raw_json knows about it."""
    raw = _raw_with_parts([("a.pdf", 100), ("b.pdf", 18 * MB)])
    upsert_message(conn, _message(raw=raw))
    upsert_attachment(
        conn,
        Attachment(
            id=None,
            message_id="msg-big-1",
            filename="a.pdf",
            mime_type="application/pdf",
            size_bytes=100,
            raw_path="/x/a.pdf",
        ),
    )
    missing = list(iter_missing_attachment_metas(conn, min_size_estimate=10 * MB))
    assert [(mid, m["filename"]) for mid, m in missing] == [("msg-big-1", "b.pdf")]


def test_iter_missing_attachment_metas_skips_small_messages(conn):
    raw = _raw_with_parts([("a.pdf", 100)])
    upsert_message(conn, _message(raw=raw))
    assert list(iter_missing_attachment_metas(conn, min_size_estimate=10 * MB)) == []


# ─── manifest ──────────────────────────────────────────────────────────


def test_manifest_marks_unfetched_attachment():
    from gmail_search.server import _attachment_manifest_dict

    a = Attachment(
        id=7,
        message_id="m",
        filename="Big.pdf",
        mime_type="application/pdf",
        size_bytes=18 * MB,
        raw_path=None,
        fetch_status="skipped_too_large",
    )
    d = _attachment_manifest_dict(a)
    assert d["fetch_status"] == "skipped_too_large"
    assert d["suggested_as"] == "unfetched"
    assert d["can_inline_pdf"] is False
    assert d["can_render_pages"] is False


def test_manifest_default_is_ok():
    from gmail_search.server import _attachment_manifest_dict

    a = Attachment(
        id=1,
        message_id="m",
        filename="s.pdf",
        mime_type="application/pdf",
        size_bytes=10,
        raw_path="/x",
    )
    assert _attachment_manifest_dict(a)["fetch_status"] == "ok"


def test_iter_missing_attachment_metas_message_ids_skips_size_filter(conn):
    raw = _raw_with_parts([("tiny.pdf", 100)])
    upsert_message(conn, _message(raw=raw))
    missing = list(
        iter_missing_attachment_metas(
            conn, min_size_estimate=10 * MB, message_ids=["msg-big-1"]
        )
    )
    assert [(mid, m["filename"]) for mid, m in missing] == [("msg-big-1", "tiny.pdf")]
    assert (
        list(
            iter_missing_attachment_metas(
                conn, min_size_estimate=10 * MB, message_ids=["other"]
            )
        )
        == []
    )


# ─── hardening (codex review 2026-09-02) ──────────────────────────────


def test_actual_payload_over_cap_is_skipped_even_if_declared_small(conn, tmp_path):
    svc = _FakeGmailService(payload=b"x" * 64)
    status = ingest_attachment(
        conn,
        svc,
        "msg-big-1",
        _meta(size=1),
        attachments_dir=tmp_path,
        max_attachment_size=32,
    )
    assert status == "skipped_too_large"
    assert not (tmp_path / "msg-big-1" / "Big Draw.pdf").exists()
    (row,) = get_attachments_for_message(conn, "msg-big-1")
    assert row.fetch_status == "skipped_too_large" and row.raw_path is None


def test_negative_declared_size_does_not_bypass_cap(conn, tmp_path):
    svc = _FakeGmailService(payload=b"x" * 64)
    status = ingest_attachment(
        conn,
        svc,
        "msg-big-1",
        _meta(size=-5),
        attachments_dir=tmp_path,
        max_attachment_size=32,
    )
    assert status == "skipped_too_large"


def test_credential_error_is_recorded_then_reraised(conn, tmp_path):
    svc = _FakeGmailService(error=PermissionError("insufficient scope"))
    with pytest.raises(PermissionError):
        ingest_attachment(
            conn,
            svc,
            "msg-big-1",
            _meta(size=5),
            attachments_dir=tmp_path,
            max_attachment_size=10 * MB,
        )
    (row,) = get_attachments_for_message(conn, "msg-big-1")
    assert row.fetch_status == "fetch_failed"


def test_sanitize_strips_control_characters():
    from gmail_search.gmail.client import _sanitize_filename

    assert _sanitize_filename("evil\r\nname\x1b[31m.pdf") == "evilname[31m.pdf"


def test_manifest_stub_row_is_ok_but_not_byte_backed():
    from gmail_search.server import _attachment_manifest_dict

    stub = Attachment(
        id=3,
        message_id="m",
        filename="https://x.test/page",
        mime_type="text/html",
        size_bytes=0,
        extracted_text="crawled " * 100,
        raw_path=None,
    )
    d = _attachment_manifest_dict(stub)
    assert d["fetch_status"] == "ok"
    assert d["suggested_as"] == "text"
    assert d["can_inline_pdf"] is False and d["can_inline_image"] is False
