from gmail_search.store.db import get_connection, init_db


def test_init_db_creates_tables(db_backend):
    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT table_name FROM information_schema.tables " "WHERE table_schema = current_schema() ORDER BY table_name"
    ).fetchall()
    tables = [r["table_name"] for r in rows]
    conn.close()
    assert "messages" in tables
    assert "attachments" in tables
    assert "embeddings" in tables
    assert "costs" in tables
    assert "sync_state" in tables


def test_init_db_idempotent(db_backend):
    db_path = db_backend["db_path"]
    init_db(db_path)
    init_db(db_path)  # Should not raise
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = current_schema()"
    ).fetchall()
    tables = [r["table_name"] for r in rows]
    conn.close()
    assert "messages" in tables


def test_get_connection(db_backend):
    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)
    assert conn is not None
    # Prove we got a live, queryable PG connection.
    row = conn.execute("SELECT 1 AS ok").fetchone()
    assert row[0] == 1
    conn.close()


# ─── zombie job reaping ────────────────────────────────────────────────────


def _insert_job(conn, job_id: str, status: str, updated_at_iso: str) -> None:
    conn.execute(
        "INSERT INTO job_progress (job_id, stage, status, total, completed, detail, started_at, updated_at) "
        "VALUES (%s, 'reindex', %s, 100, 50, 'test', %s, %s)",
        (job_id, status, updated_at_iso, updated_at_iso),
    )
    conn.commit()


def test_reap_stale_jobs_marks_old_running_as_stopped(db_backend):
    from datetime import datetime, timedelta, timezone

    from gmail_search.store.db import reap_stale_jobs

    db_path = db_backend["db_path"]
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


def test_reap_stale_jobs_leaves_recent_running_alone(db_backend):
    from datetime import datetime, timezone

    from gmail_search.store.db import reap_stale_jobs

    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)

    fresh = datetime.now(timezone.utc).isoformat()
    _insert_job(conn, "watch", "running", fresh)

    reaped = reap_stale_jobs(conn, staleness_seconds=600)
    assert reaped == 0

    row = conn.execute("SELECT status FROM job_progress WHERE job_id='watch'").fetchone()
    assert row["status"] == "running"
    conn.close()


def test_reap_stale_jobs_leaves_non_running_alone(db_backend):
    """Rows already in done/stopped/error must never be rewritten, even
    if very old — that's history, not a zombie.
    """
    from datetime import datetime, timedelta, timezone

    from gmail_search.store.db import reap_stale_jobs

    db_path = db_backend["db_path"]
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
