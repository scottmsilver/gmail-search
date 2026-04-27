"""Agent-artifact garbage collection.

Each deep-mode turn can produce several MB of artifacts (plot PNGs,
rendered HTML, CSV blobs). Over time those pile up in
`agent_artifacts` and fill the DB. We keep artifacts for a window
(default 30 days) after the session they belong to finished, then
drop them. The session row + event log stick around longer —
they're tiny and valuable for analytics / repro.

The window is tunable via `agent.artifact_retention_days` in
config.yaml; the default matches the design doc.

This module is the pure-logic half. The CLI subcommand
`gmail-search prune-artifacts` is the human-facing half; a
systemd-timer / cron can call it nightly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


DEFAULT_RETENTION_DAYS = 30
_SCRATCH_ROOT = "data/agent_scratch"


@dataclass
class PruneResult:
    """What the GC did. `artifacts_deleted` is the row count (not
    byte count — PG doesn't expose pre-delete size cheaply) so CI
    can assert the behaviour without flaky numbers."""

    artifacts_deleted: int
    bytes_freed_estimate: int  # sum of size_bytes-like for reporting
    sessions_considered: int


@dataclass
class ScratchPruneResult:
    """What `prune_scratch_dirs` did. `dirs_deleted` is the count of
    per-conversation scratch directories removed; `bytes_freed` is a
    pre-delete size estimate (best-effort — failures fold to 0)."""

    dirs_deleted: int
    bytes_freed: int


def _scratch_dir_size_bytes(path) -> int:
    """Sum file sizes under `path`. Best-effort — any unreadable file
    is skipped so a permissions blip doesn't fail the GC sweep."""
    import os as _os

    total = 0
    for root, _dirs, files in _os.walk(path):
        for f in files:
            try:
                total += _os.path.getsize(_os.path.join(root, f))
            except OSError:
                continue
    return total


def _scratch_dir_is_stale(path, retention_seconds: float) -> bool:
    """A scratch dir is stale when its newest mtime (across the dir
    itself and any file under it) is older than the retention window.
    We walk because the dir mtime alone misses bind-mount writes that
    only touch leaf files."""
    import os as _os
    import time as _time

    newest = _os.path.getmtime(path)
    for root, _dirs, files in _os.walk(path):
        for name in files:
            try:
                m = _os.path.getmtime(_os.path.join(root, name))
            except OSError:
                continue
            if m > newest:
                newest = m
    return (_time.time() - newest) > retention_seconds


def prune_scratch_dirs(
    retention_days: int = DEFAULT_RETENTION_DAYS,
    *,
    scratch_root: str | None = None,
) -> ScratchPruneResult:
    """Remove `data/agent_scratch/<id>/` dirs whose newest file is
    older than `retention_days`. Leaves the parent root in place.

    `scratch_root` lets tests point at a tmp_path without monkeypatching.
    Returns the count + estimated bytes freed; missing root is a no-op.
    """
    import os as _os
    import shutil as _shutil
    from pathlib import Path as _Path

    root = _Path(scratch_root or _SCRATCH_ROOT)
    if not root.is_dir():
        return ScratchPruneResult(0, 0)
    retention_seconds = float(retention_days) * 86400.0
    dirs_deleted = 0
    bytes_freed = 0
    for entry in _os.scandir(root):
        if not entry.is_dir(follow_symlinks=False):
            continue
        dir_path = _Path(entry.path)
        if not _scratch_dir_is_stale(dir_path, retention_seconds):
            continue
        size = _scratch_dir_size_bytes(dir_path)
        try:
            _shutil.rmtree(dir_path)
        except OSError as exc:
            logger.warning("prune_scratch_dirs: failed to remove %s: %s", dir_path, exc)
            continue
        dirs_deleted += 1
        bytes_freed += size
    if dirs_deleted:
        logger.info(
            "prune_scratch_dirs: removed %d scratch dir(s) (~%d bytes) older than %dd",
            dirs_deleted,
            bytes_freed,
            retention_days,
        )
    return ScratchPruneResult(dirs_deleted, bytes_freed)


def prune_artifacts(conn, *, retention_days: int = DEFAULT_RETENTION_DAYS) -> PruneResult:
    """Delete artifact rows whose session finished more than
    `retention_days` ago. Keeps the session + event log intact.

    Only considers sessions with status='done' or 'error' — a
    session still 'running' past the threshold is suspect (a dead
    daemon), and we'd rather leave those rows alone than destroy
    evidence of the failure mode.
    """
    # Pre-count for reporting. We use LENGTH(data) as the size
    # estimate; BYTEA doesn't carry a cheap length metadata column
    # so we pay one table scan. At expected volumes (tens to low
    # thousands of rows) that's cheap.
    row = conn.execute(
        """SELECT
               COUNT(*) AS n,
               COALESCE(SUM(LENGTH(aa.data)), 0) AS bytes,
               COUNT(DISTINCT aa.session_id) AS sessions
           FROM agent_artifacts aa
           JOIN agent_sessions s ON s.id = aa.session_id
           WHERE s.finished_at IS NOT NULL
             AND s.finished_at < NOW() - (%s || ' days')::interval
             AND s.status IN ('done', 'error')""",
        (str(retention_days),),
    ).fetchone()
    to_delete = int(row["n"])
    bytes_freed = int(row["bytes"])
    sessions = int(row["sessions"])

    if to_delete == 0:
        logger.info(f"prune_artifacts: nothing to delete (retention={retention_days}d)")
        return PruneResult(0, 0, sessions)

    conn.execute(
        """DELETE FROM agent_artifacts aa
           USING agent_sessions s
           WHERE s.id = aa.session_id
             AND s.finished_at IS NOT NULL
             AND s.finished_at < NOW() - (%s || ' days')::interval
             AND s.status IN ('done', 'error')""",
        (str(retention_days),),
    )
    conn.commit()
    logger.info(
        f"prune_artifacts: deleted {to_delete} artifacts "
        f"({bytes_freed:,} bytes estimate) across {sessions} sessions older than {retention_days}d"
    )
    return PruneResult(to_delete, bytes_freed, sessions)
