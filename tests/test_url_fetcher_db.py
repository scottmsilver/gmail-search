"""DB semantics of the URL crawler's attempt/fill bookkeeping.

Pins down two bugs found 2026-07-08:

1. Deadlock storm — the multi-row UPDATEs in `_mark_attempt_sync` /
   `_abandon_sync` / `_write_result_sync` locked rows in arbitrary
   physical order, and concurrent workers stamping the same filename
   deadlocked ~5.5k times/day.
2. Cross-user contamination — `_write_result_sync` filled the fetched
   page into ONE user's row and permanently capped every other user's
   unfilled copies, so only one user's index ever saw the page.
"""

from __future__ import annotations

import threading

import pytest
from gmail_search.gmail import url_fetcher as uf
from gmail_search.store.db import get_connection, init_db
from gmail_search.store.queries import _MAX_CRAWL_ATTEMPTS

STUB = "URL: https://ex.example.com/page"
URL = "https://ex.example.com/page"


@pytest.fixture
def db(db_backend):
    init_db(db_backend["db_path"])
    return db_backend["db_path"]


def _seed(db_path, spec):
    """spec: list of (user_id, message_id, filename) — creates users,
    messages, and one text/html attachment stub per row."""
    conn = get_connection(db_path)
    try:
        for user in sorted({u for u, _, _ in spec}):
            conn.execute(
                "INSERT INTO users (id, email) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING",
                (user, f"{user}@test.local"),
            )
        for user, mid, fn in spec:
            conn.execute(
                "INSERT INTO messages (id, thread_id, from_addr, to_addr, date, user_id)"
                " VALUES (%s, %s, 'a@b.c', 'd@e.f', '2026-01-01', %s)",
                (mid, f"t-{mid}", user),
            )
            conn.execute(
                "INSERT INTO attachments (message_id, filename, mime_type, user_id)"
                " VALUES (%s, %s, 'text/html', %s)",
                (mid, fn, user),
            )
        conn.commit()
    finally:
        conn.close()


def _rows(db_path):
    conn = get_connection(db_path)
    try:
        return conn.execute(
            "SELECT id, user_id, extracted_text, crawl_attempts FROM attachments ORDER BY id"
        ).fetchall()
    finally:
        conn.close()


class TestWriteResultUserSeparation:
    def test_fills_one_representative_per_user(self, db):
        _seed(
            db,
            [
                ("uA", "m1", STUB),
                ("uA", "m2", STUB),
                ("uB", "m3", STUB),
                ("uB", "m4", STUB),
                ("uB", "m5", STUB),
            ],
        )
        rep_id = next(r["id"] for r in _rows(db) if r["user_id"] == "uA")
        uf._write_result_sync(db, {"id": rep_id, "url": URL, "filename": STUB}, "Title", "PAGE BODY")

        rows = _rows(db)
        filled = {u: [r for r in rows if r["user_id"] == u and r["extracted_text"]] for u in ("uA", "uB")}
        # The fetched stub's owner keeps their fetched row as representative…
        assert [r["id"] for r in filled["uA"]] == [rep_id]
        # …and EVERY other user with unfilled copies gets exactly one copy of
        # the same content (pre-fix: uB got nothing, permanently).
        assert len(filled["uB"]) == 1
        assert filled["uB"][0]["extracted_text"] == "PAGE BODY"
        # True duplicates (non-representatives) are resolved out of the queue.
        for r in rows:
            if r["extracted_text"] is None:
                assert r["crawl_attempts"] == _MAX_CRAWL_ATTEMPTS

    def test_single_user_dedup_unchanged(self, db):
        _seed(db, [("uA", "m1", STUB), ("uA", "m2", STUB), ("uA", "m3", STUB)])
        rep_id = _rows(db)[0]["id"]
        uf._write_result_sync(db, {"id": rep_id, "url": URL, "filename": STUB}, "T", "BODY")
        rows = _rows(db)
        assert sum(1 for r in rows if r["extracted_text"]) == 1
        assert all(r["crawl_attempts"] == _MAX_CRAWL_ATTEMPTS for r in rows if not r["extracted_text"])


class TestAttemptStamping:
    def test_mark_attempt_stamps_all_unfilled_copies(self, db):
        # Shared backoff across users is DELIBERATE: URL reachability is a
        # property of the URL, not the mailbox that linked it.
        _seed(db, [("uA", "m1", STUB), ("uB", "m2", STUB)])
        uf._mark_attempt_sync(db, STUB)
        assert [r["crawl_attempts"] for r in _rows(db)] == [1, 1]

    def test_abandon_caps_all_unfilled_copies(self, db):
        _seed(db, [("uA", "m1", STUB), ("uB", "m2", STUB)])
        uf._abandon_sync(db, STUB)
        assert all(r["crawl_attempts"] == _MAX_CRAWL_ATTEMPTS for r in _rows(db))

    def test_concurrent_stamps_are_deadlock_free_and_exact(self, db):
        # 4 workers × 5 stamps over the same 40-copy filename. Pre-fix this
        # was the production deadlock storm shape; post-fix the id-ordered
        # FOR UPDATE serializes lock acquisition, so every row ends at
        # exactly 20 and no worker raises.
        _seed(db, [("uA", f"m{i}", STUB) for i in range(40)])
        errors: list[Exception] = []

        def stamp():
            try:
                for _ in range(5):
                    uf._mark_attempt_sync(db, STUB)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=stamp) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert all(r["crawl_attempts"] == 20 for r in _rows(db))
