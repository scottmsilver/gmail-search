"""Tests for the heartbeat-based daemon liveness + supervisor.

Exercises the bits the supervisor and HTTP `_daemon_status` rely on:
`JobProgress.__init__` recording the current pid, `heartbeat()` bumping
only `updated_at`, and stale/missing rows being treated as dead.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from gmail_search.store.db import JobProgress, get_connection, init_db


def _row(conn, job_id: str) -> dict | None:
    row = conn.execute(
        "SELECT * FROM job_progress WHERE job_id=%s",
        (job_id,),
    ).fetchone()
    return dict(row) if row else None


def test_job_progress_init_records_current_pid(db_backend):
    """On construction JobProgress should record os.getpid() into the
    `pid` column so the supervisor / stop endpoints know which process
    to signal.
    """
    db_path = db_backend["db_path"]
    init_db(db_path)

    JobProgress(db_path, "watch", start_completed=0)

    conn = get_connection(db_path)
    try:
        row = _row(conn, "watch")
        assert row is not None
        assert row["pid"] == os.getpid()
        assert row["status"] == "running"
    finally:
        conn.close()


def test_heartbeat_bumps_updated_at_only(db_backend):
    """`heartbeat()` must only refresh `updated_at` — stage/status/
    completed/detail are all preserved. The watch daemon's idle sleep
    relies on this so an idle frontfill doesn't clobber its last
    in-progress stage reading.
    """
    db_path = db_backend["db_path"]
    init_db(db_path)

    job = JobProgress(db_path, "watch", start_completed=0)
    job.update("extract", 5, 10, "processing attachments")

    conn = get_connection(db_path)
    try:
        before = _row(conn, "watch")
    finally:
        conn.close()

    # Force `updated_at` into the past so we can see the heartbeat move
    # it without relying on sub-millisecond clock resolution.
    past = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    conn = get_connection(db_path)
    try:
        conn.execute(
            "UPDATE job_progress SET updated_at=%s WHERE job_id='watch'",
            (past,),
        )
        conn.commit()
    finally:
        conn.close()

    job.heartbeat()

    conn = get_connection(db_path)
    try:
        after = _row(conn, "watch")
    finally:
        conn.close()

    # Mutating field moved forward…
    assert after["updated_at"] > before["updated_at"] or after["updated_at"] != past

    # …and nothing else changed.
    for field in ("stage", "status", "total", "completed", "detail", "pid", "started_at"):
        assert after[field] == before[field], f"{field} should not have changed"


def test_stale_row_detected_as_dead(db_backend):
    """A `running` row whose `updated_at` is 120s old should be
    considered dead by the heartbeat-age check the supervisor and
    HTTP endpoint both use.
    """
    db_path = db_backend["db_path"]
    init_db(db_backend["db_path"])

    JobProgress(db_path, "watch", start_completed=0)

    # Force the row to look 2 minutes old.
    stale_iso = (datetime.now(timezone.utc) - timedelta(seconds=120)).isoformat()
    conn = get_connection(db_path)
    try:
        conn.execute(
            "UPDATE job_progress SET updated_at=%s WHERE job_id='watch'",
            (stale_iso,),
        )
        conn.commit()
    finally:
        conn.close()

    row = JobProgress.get(db_path, "watch")
    assert row is not None
    updated = datetime.fromisoformat(row["updated_at"])
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - updated).total_seconds()

    _STALE_SECONDS = 90
    is_running = row["status"] == "running" and age < _STALE_SECONDS
    assert not is_running, "120s-old row should not count as running"


def test_missing_row_treated_as_dead(db_backend):
    """If a job has never written a row, the heartbeat check must say
    'not running' rather than throwing — that's how a cold daemon
    triggers the supervisor's first spawn.
    """
    db_path = db_backend["db_path"]
    init_db(db_path)

    row = JobProgress.get(db_path, "watch")
    assert row is None

    # Mirror the supervisor's logic: missing row => dead.
    def is_alive(r: dict | None) -> bool:
        if not r:
            return False
        if r.get("status") != "running":
            return False
        try:
            updated = datetime.fromisoformat(r["updated_at"])
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - updated).total_seconds()
        except Exception:
            return False
        return age < 90

    assert not is_alive(row)


def test_fresh_row_detected_as_alive(db_backend):
    """Sanity check the inverse of the stale-row test: a row that was
    just written should read as alive."""
    db_path = db_backend["db_path"]
    init_db(db_path)

    JobProgress(db_path, "watch", start_completed=0)
    row = JobProgress.get(db_path, "watch")
    assert row is not None
    updated = datetime.fromisoformat(row["updated_at"])
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - updated).total_seconds()
    assert age < 90
    assert row["status"] == "running"
