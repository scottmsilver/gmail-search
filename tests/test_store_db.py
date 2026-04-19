import sqlite3

from gmail_search.store.db import get_connection, init_db


def test_init_db_creates_tables(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    assert "messages" in tables
    assert "attachments" in tables
    assert "embeddings" in tables
    assert "costs" in tables
    assert "sync_state" in tables


def test_init_db_idempotent(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    init_db(db_path)  # Should not raise
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    assert "messages" in tables


def test_get_connection(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = get_connection(db_path)
    assert conn is not None
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode == "wal"
    conn.close()


# ─── zombie job reaping ────────────────────────────────────────────────────


def _insert_job(conn, job_id: str, status: str, updated_at_iso: str) -> None:
    conn.execute(
        "INSERT INTO job_progress (job_id, stage, status, total, completed, detail, started_at, updated_at) "
        "VALUES (?, 'reindex', ?, 100, 50, 'test', ?, ?)",
        (job_id, status, updated_at_iso, updated_at_iso),
    )
    conn.commit()


def test_reap_stale_jobs_marks_old_running_as_stopped(tmp_path):
    from datetime import datetime, timedelta, timezone

    from gmail_search.store.db import reap_stale_jobs

    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = get_connection(db_path)

    stale = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    _insert_job(conn, "update", "running", stale)

    reaped = reap_stale_jobs(conn, staleness_seconds=600)
    assert reaped == 1

    row = conn.execute("SELECT status, detail FROM job_progress WHERE job_id='update'").fetchone()
    assert row["status"] == "stopped"
    assert "stale" in row["detail"].lower()
    conn.close()


def test_reap_stale_jobs_leaves_recent_running_alone(tmp_path):
    from datetime import datetime, timezone

    from gmail_search.store.db import reap_stale_jobs

    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = get_connection(db_path)

    fresh = datetime.now(timezone.utc).isoformat()
    _insert_job(conn, "watch", "running", fresh)

    reaped = reap_stale_jobs(conn, staleness_seconds=600)
    assert reaped == 0

    row = conn.execute("SELECT status FROM job_progress WHERE job_id='watch'").fetchone()
    assert row["status"] == "running"
    conn.close()


def test_reap_stale_jobs_leaves_non_running_alone(tmp_path):
    """Rows already in done/stopped/error must never be rewritten, even
    if very old — that's history, not a zombie.
    """
    from datetime import datetime, timedelta, timezone

    from gmail_search.store.db import reap_stale_jobs

    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = get_connection(db_path)

    ancient = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    _insert_job(conn, "old_done", "done", ancient)
    _insert_job(conn, "old_stopped", "stopped", ancient)
    _insert_job(conn, "old_error", "error", ancient)

    reaped = reap_stale_jobs(conn, staleness_seconds=600)
    assert reaped == 0

    rows = dict((r["job_id"], r["status"]) for r in conn.execute("SELECT job_id, status FROM job_progress").fetchall())
    assert rows == {"old_done": "done", "old_stopped": "stopped", "old_error": "error"}
    conn.close()
