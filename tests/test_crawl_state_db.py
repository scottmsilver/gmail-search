"""Crawler memory: dead-URL state and the per-host circuit breaker.

Pins down two waste patterns found 2026-09-01 while chasing a GCP bill:

1. Dead URLs came back to life. `pending_url_stubs` capped retries per
   ROW, but every new email carrying the same link inserted a fresh
   0-attempt copy, which went straight into the fast lane and re-fetched
   a URL that had already failed its full retry budget. Rows had
   1,000+ attempts. `crawl_url_state` now remembers a dead URL by its
   stub filename, and new copies are never selected.

2. Anti-bot walls were retried per URL, per attempt, through Chromium
   and the paid egress proxy. 7,140 browser failures in two weeks came
   from 345 URLs on a few dozen hosts. `crawl_host_state` records a
   strike per DISTINCT failing URL and blocks the host after three, so
   the rest of its links are abandoned without a fetch.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from gmail_search.gmail import url_fetcher as uf
from gmail_search.store import queries as q
from gmail_search.store.db import get_connection, init_db

HOST = "wall.example.com"
URL_A = f"https://{HOST}/a"
URL_B = f"https://{HOST}/b"
URL_C = f"https://{HOST}/c"
STUB_A = f"URL: {URL_A}"


@pytest.fixture
def db(db_backend):
    init_db(db_backend["db_path"])
    return db_backend["db_path"]


def _seed(db_path, spec):
    """spec: list of (user_id, message_id, filename)."""
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
            "SELECT id, message_id, user_id, extracted_text, crawl_attempts FROM attachments ORDER BY id"
        ).fetchall()
    finally:
        conn.close()


def _pending(db_path, limit=50):
    conn = get_connection(db_path)
    try:
        return q.pending_url_stubs(conn, limit)
    finally:
        conn.close()


def _with_conn(db_path, fn):
    conn = get_connection(db_path)
    try:
        out = fn(conn)
        conn.commit()
        return out
    finally:
        conn.close()


# ─── dead-URL memory ───────────────────────────────────────────────────


class TestDeadUrlMemory:
    def test_fresh_copy_of_abandoned_url_is_not_selected(self, db):
        _seed(db, [("uA", "m1", STUB_A), ("uA", "m2", STUB_A)])
        uf._abandon_sync(db, STUB_A)
        # A new email arrives carrying the same link → new 0-attempt copy.
        _seed(db, [("uB", "m3", STUB_A)])
        assert [r["crawl_attempts"] for r in _rows(db)][-1] == 0
        assert _pending(db) == []

    def test_reaching_retry_cap_records_url_dead(self, db):
        _seed(db, [("uA", "m1", STUB_A)])
        for _ in range(q._MAX_CRAWL_ATTEMPTS - 1):
            uf._mark_attempt_sync(db, STUB_A)
        assert _with_conn(db, lambda c: q.is_url_dead(c, STUB_A)) is False
        uf._mark_attempt_sync(db, STUB_A)  # the 10th and final try
        assert _with_conn(db, lambda c: q.is_url_dead(c, STUB_A)) is True
        _seed(db, [("uA", "m2", STUB_A)])
        assert _pending(db) == []

    def test_live_url_is_still_selected(self, db):
        _seed(db, [("uA", "m1", STUB_A)])
        uf._mark_attempt_sync(db, STUB_A)
        # One failure, backoff window not yet elapsed → not selected now…
        assert _pending(db) == []
        # …but it is NOT dead: a fresh copy from new mail is fair game.
        _seed(db, [("uA", "m2", STUB_A)])
        assert [s["url"] for s in _pending(db)] == [URL_A]


# ─── host circuit breaker ──────────────────────────────────────────────


class TestHostCircuitBreaker:
    def test_three_distinct_url_strikes_block_the_host(self, db):
        outcomes = [
            _with_conn(db, lambda c, u=u: q.record_host_strike(c, HOST, u, "DataDome captcha"))
            for u in (URL_A, URL_B, URL_C)
        ]
        assert outcomes == [False, False, True]
        assert HOST in _with_conn(db, q.blocked_hosts)

    def test_repeated_strikes_on_one_url_do_not_block(self, db):
        for _ in range(5):
            assert _with_conn(db, lambda c: q.record_host_strike(c, HOST, URL_A, "captcha")) is False
        assert HOST not in _with_conn(db, q.blocked_hosts)

    def test_success_clears_strikes(self, db):
        for u in (URL_A, URL_B):
            _with_conn(db, lambda c, u=u: q.record_host_strike(c, HOST, u, "captcha"))
        _with_conn(db, lambda c: q.clear_host_strikes(c, HOST))
        assert _with_conn(db, lambda c: q.record_host_strike(c, HOST, URL_C, "captcha")) is False
        assert HOST not in _with_conn(db, q.blocked_hosts)

    def test_block_expires(self, db):
        for u in (URL_A, URL_B, URL_C):
            _with_conn(db, lambda c, u=u: q.record_host_strike(c, HOST, u, "captcha"))
        conn = get_connection(db)
        try:
            conn.execute(
                "UPDATE crawl_host_state SET blocked_until = now() - interval '1 second' WHERE host = %s",
                (HOST,),
            )
            conn.commit()
        finally:
            conn.close()
        assert HOST not in _with_conn(db, q.blocked_hosts)

    def test_second_block_lasts_longer(self, db):
        def block_once():
            for u in (URL_A, URL_B, URL_C):
                _with_conn(db, lambda c, u=u: q.record_host_strike(c, HOST, u, "captcha"))

        block_once()
        first = _with_conn(db, lambda c: q.host_state(c, HOST))
        _with_conn(db, lambda c: q.unblock_host(c, HOST))
        block_once()
        second = _with_conn(db, lambda c: q.host_state(c, HOST))
        assert second["block_count"] == 2
        assert second["blocked_until"] - first["blocked_until"] > timedelta(days=q._HOST_BLOCK_BASE_DAYS - 1)

    def test_unblock_host(self, db):
        for u in (URL_A, URL_B, URL_C):
            _with_conn(db, lambda c, u=u: q.record_host_strike(c, HOST, u, "captcha"))
        _with_conn(db, lambda c: q.unblock_host(c, HOST))
        assert HOST not in _with_conn(db, q.blocked_hosts)

    def test_pending_abandons_stubs_on_blocked_host(self, db):
        _seed(db, [("uA", "m1", STUB_A), ("uA", "m2", "URL: https://fine.example.com/x")])
        for u in (URL_A, URL_B, URL_C):
            _with_conn(db, lambda c, u=u: q.record_host_strike(c, HOST, u, "captcha"))
        got = _pending(db)
        assert [s["url"] for s in got] == ["https://fine.example.com/x"]
        blocked_row = next(r for r in _rows(db) if r["message_id"] == "m1")
        assert blocked_row["crawl_attempts"] == q._MAX_CRAWL_ATTEMPTS
        assert _with_conn(db, lambda c: q.is_url_dead(c, STUB_A)) is True


# ─── fetcher glue ──────────────────────────────────────────────────────


class TestFetcherGlue:
    def test_host_strike_sync_blocks_after_three_urls(self, db):
        for u in (URL_A, URL_B, URL_C):
            uf._host_strike_sync(db, u, "Blocked by anti-bot protection: DataDome captcha")
        assert HOST in _with_conn(db, q.blocked_hosts)

    def test_host_ok_sync_clears(self, db):
        uf._host_strike_sync(db, URL_A, "captcha")
        uf._host_strike_sync(db, URL_B, "captcha")
        uf._host_ok_sync(db, URL_C)
        uf._host_strike_sync(db, URL_C, "captcha")
        assert HOST not in _with_conn(db, q.blocked_hosts)


class TestCodexFindings:
    """Regressions for the 2026-09-01 codex review of the circuit breaker."""

    def test_alternating_two_urls_do_not_block(self, db):
        # Strikes count DISTINCT urls: A, B, A, B is two strikes, not four.
        for u in (URL_A, URL_B, URL_A, URL_B):
            assert _with_conn(db, lambda c, u=u: q.record_host_strike(c, HOST, u, "captcha")) is False
        assert HOST not in _with_conn(db, q.blocked_hosts)
        assert _with_conn(db, lambda c: q.host_state(c, HOST))["strikes"] == 2
        assert _with_conn(db, lambda c: q.record_host_strike(c, HOST, URL_C, "captcha")) is True

    def test_success_on_final_retry_lifts_tombstone(self, db):
        # Attempt N is stamped BEFORE it runs, so the 10th stamp tombstones
        # the URL; if that fetch then succeeds the tombstone must go.
        _seed(db, [("uA", "m1", STUB_A)])
        for _ in range(q._MAX_CRAWL_ATTEMPTS):
            uf._mark_attempt_sync(db, STUB_A)
        assert _with_conn(db, lambda c: q.is_url_dead(c, STUB_A)) is True
        rep_id = _rows(db)[0]["id"]
        uf._write_result_sync(db, {"id": rep_id, "url": URL_A, "filename": STUB_A}, "T", "BODY")
        assert _with_conn(db, lambda c: q.is_url_dead(c, STUB_A)) is False
        # …and a new user's copy is crawlable again.
        _seed(db, [("uB", "m2", STUB_A)])
        assert [s["url"] for s in _pending(db)] == [URL_A]

    def test_all_blocked_slice_does_not_look_drained(self, db):
        # 6 stubs on a blocked host sort ahead of one live stub (newest id
        # first); the live one must still come back in the same call.
        _seed(db, [("uA", "m0", "URL: https://fine.example.com/live")])
        _seed(db, [("uA", f"m{i}", f"URL: https://{HOST}/p{i}") for i in range(1, 7)])
        for u in (URL_A, URL_B, URL_C):
            _with_conn(db, lambda c, u=u: q.record_host_strike(c, HOST, u, "captcha"))
        got = _pending(db, limit=1)
        assert [s["url"] for s in got] == ["https://fine.example.com/live"]

    def test_reason_control_chars_are_stripped(self, db):
        uf._host_strike_sync(db, URL_A, "captcha\n\x1b[31mFAKE LOG LINE\x1b[0m")
        st = _with_conn(db, lambda c: q.host_state(c, HOST))
        assert "\n" not in st["last_reason"] and "\x1b" not in st["last_reason"]

    def test_query_and_fragment_variants_count_as_one_strike(self, db):
        for u in (f"{URL_A}#1", f"{URL_A}#2", f"{URL_A}?utm=3"):
            uf._host_strike_sync(db, u, "captcha")
        assert HOST not in _with_conn(db, q.blocked_hosts)
        assert _with_conn(db, lambda c: q.host_state(c, HOST))["strikes"] == 1
