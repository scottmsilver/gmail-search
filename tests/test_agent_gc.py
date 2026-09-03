"""Tests for the deep-analysis artifact GC.

Covers the retention boundary (recent artifacts are kept, old ones
deleted), the status filter (running sessions are never touched),
and the empty-case return value.
"""

from __future__ import annotations

from gmail_search.agents.gc import DEFAULT_RETENTION_DAYS, prune_artifacts
from gmail_search.agents.session import create_session, new_session_id, save_artifact
from gmail_search.store.db import get_connection, init_db


def _make_session(conn, *, status: str = "done", finished_days_ago: int = 0) -> str:
    """Helper: insert one session row with a backdated finished_at
    and the requested status. Tests drive retention cases by varying
    those two fields."""
    sid = new_session_id()
    create_session(conn, session_id=sid, conversation_id=None, mode="deep", question="q")
    conn.execute(
        """UPDATE agent_sessions
               SET status = %s,
                   finished_at = NOW() - (%s || ' days')::interval
             WHERE id = %s""",
        (status, str(finished_days_ago), sid),
    )
    conn.commit()
    return sid


def test_prune_deletes_artifacts_older_than_retention(db_backend):
    """Artifacts whose session finished > retention_days ago go; a
    recent session's artifacts stay. This is the core contract."""
    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)

    old = _make_session(conn, status="done", finished_days_ago=60)
    recent = _make_session(conn, status="done", finished_days_ago=5)
    save_artifact(conn, session_id=old, name="old.png", mime_type="image/png", data=b"OLD")
    save_artifact(conn, session_id=recent, name="new.png", mime_type="image/png", data=b"NEW")

    result = prune_artifacts(conn, retention_days=30)
    assert result.artifacts_deleted == 1
    assert result.sessions_considered == 1

    rows = conn.execute("SELECT name FROM agent_artifacts ORDER BY id").fetchall()
    names = [r["name"] for r in rows]
    assert names == ["new.png"]  # only the recent one remains
    conn.close()


def test_prune_leaves_running_sessions_alone(db_backend):
    """A session stuck in 'running' past the retention window is
    probably a dead daemon; we don't want to destroy its artifacts
    while an operator is investigating. Status filter protects
    'running' (and anything else that's not 'done'/'error')."""
    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)

    stuck = new_session_id()
    create_session(conn, session_id=stuck, conversation_id=None, mode="deep", question="q")
    # Artificially backdate, leave status=running.
    conn.execute(
        """UPDATE agent_sessions
               SET finished_at = NOW() - INTERVAL '90 days'
             WHERE id = %s""",
        (stuck,),
    )
    conn.commit()
    save_artifact(conn, session_id=stuck, name="stuck.png", mime_type="image/png", data=b"S")

    result = prune_artifacts(conn, retention_days=30)
    assert result.artifacts_deleted == 0

    row = conn.execute("SELECT COUNT(*) AS n FROM agent_artifacts").fetchone()
    assert row["n"] == 1
    conn.close()


def test_prune_is_a_noop_on_empty_table(db_backend):
    """Zero rows in `agent_artifacts` → PruneResult(0, 0, 0).
    Important to verify because the DELETE-with-JOIN pattern will
    happily run against an empty table and we don't want the
    reporting to lie."""
    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)

    result = prune_artifacts(conn, retention_days=30)
    assert result.artifacts_deleted == 0
    assert result.bytes_freed_estimate == 0
    conn.close()


def test_prune_default_retention_is_30_days():
    """Double-check the published contract: the module-level
    DEFAULT_RETENTION_DAYS matches the design doc's 30-day window."""
    assert DEFAULT_RETENTION_DAYS == 30


class _FakeStaleConn:
    """Answers the stale-conversation SELECT with the ids given and
    accepts the mapping DELETE."""

    def __init__(self, stale_ids: list[str]) -> None:
        self._stale_ids = stale_ids
        self.committed = False

    def execute(self, sql: str, params=None):
        if sql.lstrip().upper().startswith("SELECT"):
            return _Result(rows=[{"id": i} for i in self._stale_ids], rowcount=len(self._stale_ids))
        return _Result(rows=[], rowcount=len(self._stale_ids))

    def commit(self) -> None:
        self.committed = True


class _Result:
    def __init__(self, rows, rowcount):
        self._rows = rows
        self.rowcount = rowcount

    def fetchall(self):
        return self._rows


def test_prune_conversation_workspaces_removes_pi_session_file(tmp_path):
    from gmail_search.agents import gc

    ws_root = tmp_path / "workspaces"
    (ws_root / "deep-conv-stale1").mkdir(parents=True)
    (ws_root / "deep-conv-fresh2").mkdir(parents=True)
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    (sessions / "stale1.jsonl").write_text("{}\n")
    (sessions / "fresh2.jsonl").write_text("{}\n")

    result = gc.prune_conversation_workspaces(
        _FakeStaleConn(["stale1"]), retention_days=30, workspaces_root=str(ws_root), pi_sessions_root=str(sessions)
    )

    assert result.dirs_deleted == 1
    assert not (ws_root / "deep-conv-stale1").exists() and (ws_root / "deep-conv-fresh2").exists()
    assert not (sessions / "stale1.jsonl").exists()
    assert (sessions / "fresh2.jsonl").exists()
