"""Regression tests for the `update --loop` daemon's extract step.

The daemon was letting fresh arrivals sit unextracted for days because
its pending-attachments query had no ORDER BY — Postgres returned rows
in physical-layout order, which for us means oldest-first (backfill).
The multi-year backfill perpetually ground in front of today's PDFs.

These tests lock in the frontfill-first ordering guarantee on
`get_pending_extraction_message_ids`: anything received in the last
24h comes first, then everything else by date DESC.
"""

from datetime import UTC, datetime, timedelta

from gmail_search.store.db import get_connection, init_db
from gmail_search.store.models import Attachment, Message
from gmail_search.store.queries import get_pending_extraction_message_ids, upsert_attachment, upsert_message


def _make_message(msg_id: str, date: datetime) -> Message:
    return Message(
        id=msg_id,
        thread_id=f"thread_{msg_id}",
        from_addr="sender@example.com",
        to_addr="me@example.com",
        subject=f"Subject {msg_id}",
        body_text="body",
        body_html="<p>body</p>",
        date=date,
        labels=["INBOX"],
        history_id=1,
        raw_json="{}",
    )


def _make_pending_attachment(message_id: str, filename: str = "doc.pdf") -> Attachment:
    """An attachment with raw bytes on disk but no extracted_text yet —
    i.e. the exact state that the extract step is supposed to drain."""
    return Attachment(
        id=None,
        message_id=message_id,
        filename=filename,
        mime_type="application/pdf",
        size_bytes=1024,
        extracted_text=None,
        image_path=None,
        raw_path=f"/tmp/does-not-need-to-exist-for-this-test/{message_id}.pdf",
    )


def test_pending_extraction_returns_fresh_arrivals_before_backfill(db_backend):
    """The `update` daemon's extract step must process today's PDFs
    before it finishes grinding through a 30-day-old backlog. Without
    this ordering, fresh mail waits for the entire backfill to drain.
    """
    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)

    now = datetime.now(UTC).replace(tzinfo=None)
    old_date = now - timedelta(days=30)
    today_date = now - timedelta(hours=1)  # well inside the 24h fresh window

    # Insert the OLD one first so physical layout favours it — this is
    # what was breaking us in production (no ORDER BY → insertion
    # order).
    upsert_message(conn, _make_message("old_msg", old_date))
    upsert_attachment(conn, _make_pending_attachment("old_msg"))

    upsert_message(conn, _make_message("today_msg", today_date))
    upsert_attachment(conn, _make_pending_attachment("today_msg"))

    ids = get_pending_extraction_message_ids(conn)
    assert ids == ["today_msg", "old_msg"], f"fresh arrival must come first, got {ids}"
    conn.close()


def test_pending_extraction_orders_non_fresh_by_date_desc(db_backend):
    """Outside the 24h fresh bucket, ties break on date DESC — so a
    message from last week still beats one from last year.
    """
    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)

    now = datetime.now(UTC).replace(tzinfo=None)
    year_old = now - timedelta(days=365)
    week_old = now - timedelta(days=7)

    upsert_message(conn, _make_message("year_old", year_old))
    upsert_attachment(conn, _make_pending_attachment("year_old"))

    upsert_message(conn, _make_message("week_old", week_old))
    upsert_attachment(conn, _make_pending_attachment("week_old"))

    ids = get_pending_extraction_message_ids(conn)
    assert ids == ["week_old", "year_old"], ids
    conn.close()


def test_pending_extraction_skips_already_extracted_rows(db_backend):
    """Rows with extracted_text already populated must not appear —
    otherwise the extract step would reprocess them on every cycle.
    """
    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)

    now = datetime.now(UTC).replace(tzinfo=None)
    upsert_message(conn, _make_message("done_msg", now))
    done = _make_pending_attachment("done_msg")
    done.extracted_text = "already extracted"
    upsert_attachment(conn, done)

    upsert_message(conn, _make_message("pending_msg", now - timedelta(days=2)))
    upsert_attachment(conn, _make_pending_attachment("pending_msg"))

    ids = get_pending_extraction_message_ids(conn)
    assert ids == ["pending_msg"], ids
    conn.close()


def test_pending_extraction_skips_rows_without_raw_path(db_backend):
    """Drive stubs / URL stubs (no local raw_path) use different code
    paths; the local-file extractor must not pick them up.
    """
    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)

    now = datetime.now(UTC).replace(tzinfo=None)
    # Drive stub — no raw_path.
    upsert_message(conn, _make_message("drive_msg", now))
    stub = Attachment(
        id=None,
        message_id="drive_msg",
        filename="Drive: [abc123]",
        mime_type="application/vnd.google-apps.document",
        size_bytes=0,
        extracted_text=None,
        image_path=None,
        raw_path=None,
    )
    upsert_attachment(conn, stub)

    # Regular local PDF pending extraction.
    upsert_message(conn, _make_message("local_msg", now - timedelta(hours=2)))
    upsert_attachment(conn, _make_pending_attachment("local_msg"))

    ids = get_pending_extraction_message_ids(conn)
    assert ids == ["local_msg"], ids
    conn.close()


def test_extract_pending_attachments_runs_when_no_new_downloads(db_backend, tmp_path, monkeypatch):
    """Regression for the real shipped bug — not just the ordering fix.

    Before the drain_only change, the `update --loop` inner loop did
    `if dl_count == 0: break` right after `download_messages`, which
    meant the extract step was skipped entirely when frontfill had
    already pulled the day's mail. Fresh PDFs then sat unextracted.

    This drives `_extract_pending_attachments` directly (the new helper
    the inner loop calls) and proves it processes a pending row even
    when zero new downloads happened. The helper used to live inlined
    after a `break` that guarded on `dl_count`; now it runs
    unconditionally on every cycle.
    """
    from gmail_search.cli import _extract_pending_attachments

    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)

    # Create an attachment on disk that fitz / text extractor can read.
    raw = tmp_path / "today.txt"
    raw.write_text("late-arrival content the update loop should pick up")

    now = datetime.now(UTC).replace(tzinfo=None)
    upsert_message(conn, _make_message("today_msg", now))
    att = Attachment(
        id=None,
        message_id="today_msg",
        filename="today.txt",
        mime_type="text/plain",
        size_bytes=raw.stat().st_size,
        extracted_text=None,
        image_path=None,
        raw_path=str(raw),
    )
    upsert_attachment(conn, att)

    # `dispatch` accepts a config dict; the text extractor only cares
    # about max_text_chars, but we pass an empty dict to keep the test
    # close to the real call site.
    n = _extract_pending_attachments(conn, att_config={})
    assert n == 1

    # Verify the text landed, i.e. the extract step ran even though
    # no downloads happened this "cycle".
    from gmail_search.store.queries import get_attachments_for_message

    atts = get_attachments_for_message(conn, "today_msg")
    assert atts[0].extracted_text is not None
    assert "late-arrival content" in atts[0].extracted_text
    conn.close()
