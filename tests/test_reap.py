"""Zombie-job reaper tests.

The reaper walks `job_progress` for stale `running` rows, tries to
find a matching process by cmdline via psutil, SIGTERMs it if alive,
then marks the row stopped. We test the orchestration with a fake
`process_iter` + fake killer so no real processes are touched.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from gmail_search.store.db import get_connection, init_db


def _insert_job(conn, job_id: str, status: str, updated_at_iso: str) -> None:
    conn.execute(
        "INSERT INTO job_progress (job_id, stage, status, total, completed, detail, started_at, updated_at) "
        "VALUES (%s, 'reindex', %s, 100, 50, 'test', %s, %s)",
        (job_id, status, updated_at_iso, updated_at_iso),
    )
    conn.commit()


def _fake_proc(pid: int, cmdline: list[str]) -> MagicMock:
    p = MagicMock()
    p.pid = pid
    p.cmdline.return_value = cmdline
    p.is_running.return_value = True
    return p


# ─── find_process_for_job_id ──────────────────────────────────────────────


def test_find_process_matches_by_job_id_in_cmdline():
    from gmail_search.reap import find_process_for_job_id

    processes = [
        _fake_proc(100, ["python", "-m", "unrelated"]),
        _fake_proc(200, ["/usr/bin/gmail-search", "update", "--max-messages", "500"]),
        _fake_proc(300, ["/usr/bin/gmail-search", "watch", "--interval", "120"]),
    ]
    found = find_process_for_job_id("update", process_iter=lambda attrs=None: processes)
    assert found is not None
    assert found.pid == 200


def test_find_process_returns_none_when_no_match():
    from gmail_search.reap import find_process_for_job_id

    processes = [_fake_proc(100, ["python", "-m", "unrelated"])]
    found = find_process_for_job_id("update", process_iter=lambda attrs=None: processes)
    assert found is None


def test_find_process_skips_processes_without_gmail_search():
    """A random `update` in someone else's tool must not be a false match."""
    from gmail_search.reap import find_process_for_job_id

    processes = [
        _fake_proc(100, ["yum", "update"]),
        _fake_proc(200, ["apt-get", "update"]),
    ]
    found = find_process_for_job_id("update", process_iter=lambda attrs=None: processes)
    assert found is None


# ─── reap_zombie_jobs ──────────────────────────────────────────────────────


def test_reap_marks_stale_row_stopped_when_no_matching_process(tmp_path):
    from gmail_search.reap import reap_zombie_jobs

    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = get_connection(db_path)
    stale = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    _insert_job(conn, "update", "running", stale)
    conn.close()

    report = reap_zombie_jobs(
        db_path,
        staleness_seconds=600,
        process_iter=lambda attrs=None: [],  # nothing running
    )
    assert report.rows_reaped == 1
    assert report.processes_killed == 0

    conn = get_connection(db_path)
    row = conn.execute("SELECT status FROM job_progress WHERE job_id='update'").fetchone()
    conn.close()
    assert row["status"] == "stopped"


def test_reap_kills_stale_running_process_and_marks_row(tmp_path):
    from gmail_search.reap import reap_zombie_jobs

    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = get_connection(db_path)
    stale = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    _insert_job(conn, "update", "running", stale)
    conn.close()

    victim = _fake_proc(200, ["/usr/bin/gmail-search", "update"])

    report = reap_zombie_jobs(
        db_path,
        staleness_seconds=600,
        process_iter=lambda attrs=None: [victim],
    )
    assert report.rows_reaped == 1
    assert report.processes_killed == 1
    victim.terminate.assert_called_once()

    conn = get_connection(db_path)
    row = conn.execute("SELECT status FROM job_progress WHERE job_id='update'").fetchone()
    conn.close()
    assert row["status"] == "stopped"


def test_reap_leaves_fresh_running_row_alone(tmp_path):
    from gmail_search.reap import reap_zombie_jobs

    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = get_connection(db_path)
    fresh = datetime.now(timezone.utc).isoformat()
    _insert_job(conn, "watch", "running", fresh)
    conn.close()

    alive = _fake_proc(300, ["/usr/bin/gmail-search", "watch"])
    report = reap_zombie_jobs(
        db_path,
        staleness_seconds=600,
        process_iter=lambda attrs=None: [alive],
    )
    assert report.rows_reaped == 0
    assert report.processes_killed == 0
    alive.terminate.assert_not_called()

    conn = get_connection(db_path)
    row = conn.execute("SELECT status FROM job_progress WHERE job_id='watch'").fetchone()
    conn.close()
    assert row["status"] == "running"


def test_reap_no_rows_no_error(tmp_path):
    """Empty job_progress must not crash the reaper."""
    from gmail_search.reap import reap_zombie_jobs

    db_path = tmp_path / "test.db"
    init_db(db_path)
    report = reap_zombie_jobs(db_path, staleness_seconds=600, process_iter=lambda attrs=None: [])
    assert report.rows_reaped == 0
    assert report.processes_killed == 0
