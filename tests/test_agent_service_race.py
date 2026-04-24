"""Concurrency test for the deep-mode SSE polling ↔ event-commit race.

Background: `_real_run` in `agents/service.py` runs the orchestrator
as a background asyncio task that calls `append_event(conn, ...)`
repeatedly, while the same coroutine drains events to SSE via
`fetch_events_after(poll_conn, ...)` polling. The two coroutines are
concurrent (driven by `asyncio.gather`), and they each hand a sync
psycopg connection to `asyncio.to_thread` so the libpq calls don't
block the event loop.

The reason `_real_run` opens TWO `get_connection` handles instead of
one is the polling ↔ commit race:

  * Historically (psycopg2 / psycopg3 async / psycopg3 with pipeline
    mode) two simultaneous `execute()`s on the SAME connection raise
    `OperationalError: another command is already in progress`. The
    in-tree comment in `service.py` is written to that history.

  * psycopg3 sync connections (the variant we use today) carry an
    internal lock that *serializes* concurrent operations from
    multiple threads instead of raising. So under our current
    runtime, sharing a connection wouldn't crash — but it WOULD
    mean every `fetch_events_after` poll blocks behind every
    `append_event` commit and vice versa, defeating the whole point
    of running them concurrently. The poller is meant to drain rows
    in PARALLEL with the writer, not after it.

This test proves both halves of the contract at the `append_event` +
`fetch_events_after` layer (the lowest layer that exhibits it),
without requiring the live ADK pipeline:

  1. SEPARATE-CONNECTION case (the production design): 50 writes +
     concurrent reader, two connections. No errors. Every event is
     delivered exactly once, in seq order. THIS IS THE INVARIANT
     `_real_run` RELIES ON.

  2. SAME-CONNECTION case (the anti-pattern): same workload, one
     shared connection. No data loss either (psycopg3 sync's
     internal lock saves us from the OperationalError seen on older
     stacks), but the lock contention is observable: the reader's
     polls and the writer's commits serialize. We assert that
     correctness still holds — duplicates and missing events would
     be a true bug, lock-induced slowness would not — so this
     branch is here as a regression guard: if a future psycopg
     update reintroduces the OperationalError on threaded shared
     connections, this test will fail loudly and tell the next
     person why the two-connection design exists.

Skips automatically when Postgres isn't reachable on 127.0.0.1:5544
(the autouse fixture in conftest.py handles that).
"""

from __future__ import annotations

import threading
import time

from gmail_search.agents.session import append_event, create_session, fetch_events_after, new_session_id
from gmail_search.store.db import get_connection, init_db

# 50 was the number named in the review note. Each event has a tiny
# payload so the test still completes in well under a second.
NUM_WRITES = 50


def _make_session(db_path):
    """Create the agent_sessions row this test will hang events off
    of. Returns the session id and closes its bootstrap connection so
    the actual test threads each hold their own."""
    conn = get_connection(db_path)
    try:
        sid = new_session_id()
        create_session(conn, session_id=sid, conversation_id=None, mode="deep", question="race test")
        return sid
    finally:
        conn.close()


def _run_writer_thread(conn, session_id: str, n: int, errors: list, start_evt: threading.Event) -> None:
    """Append `n` events back-to-back. Waits on `start_evt` so writer +
    reader fire at the same instant — overlapping is what would expose
    a concurrency bug if one existed."""
    start_evt.wait()
    for i in range(n):
        try:
            append_event(
                conn,
                session_id=session_id,
                agent_name="planner",
                kind="plan",
                payload={"i": i},
            )
        except Exception as e:
            errors.append(e)
            return  # one failure already proves the design is unsafe


def _run_reader_thread(
    conn,
    session_id: str,
    target_count: int,
    seen: list,
    errors: list,
    start_evt: threading.Event,
    deadline_s: float,
) -> None:
    """Tight-loop poller: drain rows newer than `last_seq` until we've
    seen `target_count` events, the writer has finished + drained, or
    the deadline expires. Sub-millisecond sleep keeps the poll loop
    actively colliding with the writer's commits."""
    start_evt.wait()
    last_seq = 0
    end = time.time() + deadline_s
    while time.time() < end:
        try:
            rows = list(fetch_events_after(conn, session_id, after_seq=last_seq))
        except Exception as e:
            errors.append(e)
            return
        for ev in rows:
            seen.append(ev.seq)
            last_seq = max(last_seq, ev.seq)
        if len(seen) >= target_count:
            return
        time.sleep(0.0001)


def _drive_concurrent_workload(write_conn, read_conn, session_id: str, deadline_s: float = 5.0):
    """Common harness: spin up writer + reader threads, release them
    together, return (write_errs, read_errs, seen_seqs, wall_seconds).
    Used by both the same-conn and separate-conn tests so the
    workloads are byte-identical apart from the connection topology."""
    write_errs: list[Exception] = []
    read_errs: list[Exception] = []
    seen: list[int] = []
    start = threading.Event()

    t_writer = threading.Thread(
        target=_run_writer_thread,
        args=(write_conn, session_id, NUM_WRITES, write_errs, start),
        daemon=True,
    )
    t_reader = threading.Thread(
        target=_run_reader_thread,
        args=(read_conn, session_id, NUM_WRITES, seen, read_errs, start, deadline_s),
        daemon=True,
    )
    t_writer.start()
    t_reader.start()
    t0 = time.time()
    start.set()
    t_writer.join(timeout=deadline_s + 5)
    t_reader.join(timeout=deadline_s + 5)
    wall = time.time() - t0
    return write_errs, read_errs, seen, wall


# ── Separate connections: the production design — must succeed ─────────


def test_writer_and_reader_on_separate_connections_succeed(db_backend):
    """`_real_run` opens `conn` and `poll_conn` via two separate
    `get_connection` calls (see service.py, just inside `_real_run`).
    Under that topology, 50 concurrent writes + a tight-loop reader
    must produce: no errors, every event delivered exactly once, seq
    order preserved.

    This is the SLA the rest of the deep-mode pipeline is built on:
    SSE clients see every row; replay-after-disconnect doesn't skip
    rows; UI never shows a gap.
    """
    db_path = db_backend["db_path"]
    init_db(db_path)
    session_id = _make_session(db_path)

    write_conn = get_connection(db_path)
    read_conn = get_connection(db_path)
    try:
        write_errs, read_errs, seen, _wall = _drive_concurrent_workload(write_conn, read_conn, session_id)
    finally:
        write_conn.close()
        read_conn.close()

    assert not write_errs, f"writer hit psycopg errors with separate connections: {write_errs!r}"
    assert not read_errs, f"reader hit psycopg errors with separate connections: {read_errs!r}"

    # Every event delivered exactly once, in monotonic seq order.
    assert len(seen) == NUM_WRITES, f"reader missed events: saw {len(seen)} of {NUM_WRITES}"
    assert len(set(seen)) == NUM_WRITES, f"reader saw duplicate seqs: {seen}"
    assert seen == sorted(seen), f"reader saw events out of order: {seen}"
    assert seen == list(range(1, NUM_WRITES + 1)), f"seqs aren't 1..N: {seen}"


# ── Same connection: regression guard ─────────────────────────────────


def test_writer_and_reader_on_same_connection_do_not_corrupt_data(db_backend):
    """Regression guard for the historical reason `_real_run` opens
    two connections.

    On older stacks (psycopg2, psycopg3 async, psycopg3 with
    pipeline mode), two threads doing concurrent `execute()` on the
    same connection raise
    `OperationalError: another command is already in progress`. On
    psycopg3 sync (our current runtime) the connection has an
    internal lock that serializes operations instead — so a shared
    connection produces correct data but no parallelism between
    reader and writer.

    What we assert here:

      * If a future psycopg upgrade re-introduces the
        OperationalError under threaded shared-conn usage, this test
        catches it and the message points the next person at
        `_real_run`'s two-connection design.
      * If shared-conn ever silently DROPS or DUPLICATES rows
        (visibility surprises across an open transaction) we catch
        that too.

    We DELIBERATELY do not assert "shared conn must raise" — that's
    psycopg-version-specific behavior and we'd rather have a stable
    test than a flaky one. The separate-conn test is the load-
    bearing assertion for the production code.
    """
    db_path = db_backend["db_path"]
    init_db(db_path)
    session_id = _make_session(db_path)

    shared = get_connection(db_path)
    try:
        write_errs, read_errs, seen, _wall = _drive_concurrent_workload(shared, shared, session_id)
    finally:
        # Same-conn run can leave the connection in an in-progress
        # / failed-tx state on older psycopg; rollback before close
        # so we don't poison subsequent tests.
        try:
            shared.rollback()
        except Exception:
            pass
        shared.close()

    # If psycopg ever DOES raise on shared-conn concurrency again,
    # surface that explicitly with a pointer to the design doc.
    all_errs = write_errs + read_errs
    if all_errs:
        # Match the historical "another command is already in
        # progress" family. If we see it, the test fails loudly
        # with a directional message.
        msgs = " | ".join(str(e) for e in all_errs)
        raise AssertionError(
            f"Shared psycopg connection raised under threaded concurrency: {msgs!r}\n"
            "This is the historical OperationalError that drove the two-connection "
            "design in agents/service.py:_real_run. The separate-connection variant "
            "of this test should still pass; if it doesn't, the orchestration race "
            "fix has regressed."
        )

    # No data corruption under serialization either: every row is
    # written and read exactly once, in order.
    assert len(seen) == NUM_WRITES, f"shared-conn workload lost events: saw {len(seen)} of {NUM_WRITES}"
    assert len(set(seen)) == NUM_WRITES, f"shared-conn workload duplicated seqs: {seen}"
    assert seen == list(range(1, NUM_WRITES + 1)), f"shared-conn workload reordered seqs: {seen}"
