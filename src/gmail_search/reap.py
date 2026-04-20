"""Zombie-job reaper.

Walks `job_progress` for `running` rows whose `updated_at` is older
than `staleness_seconds`. For each stale row, tries to find a matching
process by cmdline via psutil and SIGTERMs it, then marks the row as
`stopped`. Rows without a living process still get marked — that's the
common post-OOM case.

Design intent:
- `store.db.reap_stale_jobs` is pure DB; it stays reusable as the
  fallback when psutil isn't available or no process match is found.
- This module owns the process-inspection layer. Keeping psutil
  contained here means the DB module doesn't learn about processes
  and there's one place to swap out psutil if we ever need to.
- `process_iter` is injected so tests can exercise the control flow
  without spawning real subprocesses.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

from gmail_search.store.db import get_connection, reap_stale_jobs

logger = logging.getLogger(__name__)


@dataclass
class ReapReport:
    rows_reaped: int = 0
    processes_killed: int = 0
    details: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.details is None:
            self.details = []


def _default_process_iter(attrs=None):
    import psutil

    return psutil.process_iter(attrs=attrs)


def find_process_for_job_id(
    job_id: str,
    process_iter: Callable[..., Iterable] = _default_process_iter,
) -> Optional[object]:
    """Find a running process whose cmdline looks like `gmail-search <job_id>`.

    We require BOTH tokens to match so a bare `apt-get update` isn't
    mistaken for our `update` job. Returns the first match or None.
    """
    needle = job_id.lower()
    for proc in process_iter(attrs=["cmdline"]):
        try:
            cmdline = proc.cmdline() if callable(proc.cmdline) else (proc.info or {}).get("cmdline") or []
        except Exception:
            continue
        if not cmdline:
            continue
        joined = " ".join(cmdline).lower()
        if "gmail-search" in joined and needle in joined:
            return proc
    return None


def _terminate_with_fallback(proc, kill_timeout_sec: float) -> bool:
    """SIGTERM the process; if it's still alive after kill_timeout_sec,
    SIGKILL it. Returns True if the process was signalled (regardless
    of whether it died cleanly).
    """
    try:
        proc.terminate()
    except Exception as e:
        logger.warning(f"terminate() failed on pid={getattr(proc, 'pid', '?')}: {e}")
        return False

    try:
        # psutil.Process has .wait(timeout); the fake in tests doesn't.
        if hasattr(proc, "wait"):
            proc.wait(timeout=kill_timeout_sec)
    except Exception:
        try:
            proc.kill()
        except Exception as e:
            logger.warning(f"kill() failed on pid={getattr(proc, 'pid', '?')}: {e}")
    return True


def reap_zombie_jobs(
    db_path: Path,
    staleness_seconds: int = 600,
    kill_timeout_sec: float = 5.0,
    process_iter: Callable[..., Iterable] = _default_process_iter,
) -> ReapReport:
    """Find stale `running` job_progress rows, kill any matching live
    process, and mark the rows stopped. Returns a report of what was
    reaped for CLI display.
    """
    report = ReapReport()
    cutoff = (datetime.now(timezone.utc) - timedelta(seconds=staleness_seconds)).isoformat()

    conn = get_connection(db_path)
    try:
        stale_rows = conn.execute(
            "SELECT job_id, stage, updated_at FROM job_progress "
            "WHERE status='running' AND updated_at < %s ORDER BY updated_at",
            (cutoff,),
        ).fetchall()

        for row in stale_rows:
            job_id = row["job_id"]
            proc = find_process_for_job_id(job_id, process_iter=process_iter)
            if proc is not None:
                if _terminate_with_fallback(proc, kill_timeout_sec):
                    report.processes_killed += 1
                    report.details.append(f"killed pid={proc.pid} for job {job_id}")
            else:
                report.details.append(f"no live process for stale job {job_id}")

        # One UPDATE to mark all the stale rows stopped.
        reaped = reap_stale_jobs(conn, staleness_seconds=staleness_seconds)
        report.rows_reaped = reaped
    finally:
        conn.close()

    return report
