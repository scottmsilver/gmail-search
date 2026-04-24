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


@dataclass
class PruneResult:
    """What the GC did. `artifacts_deleted` is the row count (not
    byte count — PG doesn't expose pre-delete size cheaply) so CI
    can assert the behaviour without flaky numbers."""

    artifacts_deleted: int
    bytes_freed_estimate: int  # sum of size_bytes-like for reporting
    sessions_considered: int


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
