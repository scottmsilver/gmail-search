"""Priority Inbox endpoint tests.

Covers the three-section classifier used by `/api/priority-inbox`:

  1. Important and unread — IMPORTANT + UNREAD + INBOX
  2. Starred             — STARRED (regardless of INBOX / read state)
  3. Everything else     — INBOX minus (sections 1 ∪ 2)

We drive the logic via the `_inbox_rows` helper that powers the endpoint,
so each test asserts on real Postgres results without booting FastAPI.
"""

from datetime import datetime

from gmail_search.server import _inbox_rows
from gmail_search.store.db import get_connection, init_db
from gmail_search.store.models import Message
from gmail_search.store.queries import upsert_message


def _make_message(
    id: str,
    thread_id: str,
    labels: list[str],
    subject: str = "Subj",
    date: datetime | None = None,
) -> Message:
    return Message(
        id=id,
        thread_id=thread_id,
        from_addr="alice@example.com",
        to_addr="bob@example.com",
        subject=subject,
        body_text="body",
        body_html="<p>body</p>",
        date=date or datetime(2026, 4, 20, 10, 0),
        labels=labels,
        history_id=1,
        raw_json="{}",
    )


# Predicates match exactly what `api_priority_inbox` passes to the helper.
_IMPORTANT_UNREAD_SQL = "m.labels LIKE %s AND m.labels LIKE %s AND m.labels LIKE %s"
_IMPORTANT_UNREAD_PARAMS = ('%"IMPORTANT"%', '%"UNREAD"%', '%"INBOX"%')

_STARRED_SQL = "m.labels LIKE %s"
_STARRED_PARAMS = ('%"STARRED"%',)

_EVERYTHING_ELSE_SQL = (
    "m.labels LIKE %s"
    " AND m.thread_id NOT IN ("
    "   SELECT m2.thread_id FROM messages m2"
    "   WHERE (m2.labels LIKE %s AND m2.labels LIKE %s AND m2.labels LIKE %s)"
    "      OR (m2.labels LIKE %s)"
    " )"
)
_EVERYTHING_ELSE_PARAMS = (
    '%"INBOX"%',
    '%"IMPORTANT"%',
    '%"UNREAD"%',
    '%"INBOX"%',
    '%"STARRED"%',
)


def _fetch_all_sections(conn) -> tuple[list[dict], list[dict], list[dict]]:
    """Helper: run all three predicates with limit=50, offset=0."""
    important_unread = _inbox_rows(conn, _IMPORTANT_UNREAD_SQL, _IMPORTANT_UNREAD_PARAMS, 50, 0)
    starred = _inbox_rows(conn, _STARRED_SQL, _STARRED_PARAMS, 50, 0)
    everything_else = _inbox_rows(conn, _EVERYTHING_ELSE_SQL, _EVERYTHING_ELSE_PARAMS, 50, 0)
    return important_unread, starred, everything_else


def test_priority_inbox_empty_when_no_matching_messages(db_backend):
    """A message that isn't INBOX / IMPORTANT / STARRED lands in no section."""
    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)
    # A plain SENT-only message: no INBOX, IMPORTANT, or STARRED label.
    upsert_message(conn, _make_message("m1", "t1", ["SENT"]))

    important_unread, starred, everything_else = _fetch_all_sections(conn)
    assert important_unread == []
    assert starred == []
    assert everything_else == []
    conn.close()


def test_important_unread_message_lands_in_section_one_only(db_backend):
    """IMPORTANT + UNREAD + INBOX → section 1, NOT section 3."""
    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)
    upsert_message(
        conn,
        _make_message("m1", "t1", ["IMPORTANT", "UNREAD", "INBOX"], subject="Important unread"),
    )

    important_unread, starred, everything_else = _fetch_all_sections(conn)
    assert [r["thread_id"] for r in important_unread] == ["t1"]
    assert starred == []
    # Section 3 MUST exclude threads already in section 1.
    assert [r["thread_id"] for r in everything_else] == []
    conn.close()


def test_starred_message_appears_in_section_two_regardless_of_inbox_or_read(db_backend):
    """STARRED label alone puts a thread in section 2 even when it's
    archived (no INBOX) and read (no UNREAD). Section 3 excludes it."""
    db_path = db_backend["db_path"]
    init_db(db_backend["db_path"])  # idempotent
    conn = get_connection(db_path)
    # Archived + read, but STARRED. No INBOX, no UNREAD.
    upsert_message(
        conn,
        _make_message("m1", "t1", ["STARRED"], subject="Starred archived"),
    )
    # Also exercise the "starred-and-inbox" variant to confirm starred
    # matches regardless of other labels — it should still appear here
    # (and should NOT appear in "everything else").
    upsert_message(
        conn,
        _make_message("m2", "t2", ["STARRED", "INBOX"], subject="Starred inbox"),
    )

    important_unread, starred, everything_else = _fetch_all_sections(conn)
    assert important_unread == []
    starred_ids = sorted(r["thread_id"] for r in starred)
    assert starred_ids == ["t1", "t2"]
    # The STARRED+INBOX thread must NOT duplicate into section 3.
    assert [r["thread_id"] for r in everything_else] == []
    conn.close()


def test_plain_inbox_message_lands_in_everything_else_only(db_backend):
    """INBOX message without IMPORTANT or STARRED → section 3 only."""
    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)
    upsert_message(
        conn,
        _make_message("m1", "t1", ["INBOX"], subject="Plain inbox"),
    )
    # Also an UNREAD-but-not-IMPORTANT inbox message — should still
    # fall through to "everything else".
    upsert_message(
        conn,
        _make_message("m2", "t2", ["INBOX", "UNREAD"], subject="Unread inbox"),
    )

    important_unread, starred, everything_else = _fetch_all_sections(conn)
    assert important_unread == []
    assert starred == []
    else_ids = sorted(r["thread_id"] for r in everything_else)
    assert else_ids == ["t1", "t2"]
    conn.close()
