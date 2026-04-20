"""Adversarial tests for the /api/sql endpoint validator.

These exercise the safety gate (`_validate_sql`) directly, plus a couple of
end-to-end checks against Postgres to confirm the READ ONLY transaction
actually blocks writes even if the gate were bypassed.
"""

from __future__ import annotations

import pytest

from gmail_search.server import _run_sql_with_timeout, _validate_sql
from gmail_search.store.db import get_connection, init_db

# ─── _validate_sql gate ───────────────────────────────────────────────


def test_rejects_empty():
    assert _validate_sql("") == "query required"
    assert _validate_sql("   ") == "query required"


def test_rejects_too_long():
    err = _validate_sql("SELECT 1" + " " * 6000)
    assert err is not None and "too long" in err


def test_rejects_non_select_or_with():
    for q in ("INSERT INTO x VALUES (1)", "DROP TABLE messages", "UPDATE messages SET x=1"):
        assert _validate_sql(q) == "only SELECT / WITH queries are allowed"


def test_rejects_forbidden_keywords():
    # Real attack surfaces (both SQLite- and Postgres-specific).
    cases = [
        "SELECT 1; DROP TABLE messages",
        "WITH x AS (SELECT 1) INSERT INTO y SELECT * FROM x",
        "SELECT 1 PRAGMA writable_schema=1",
        "SELECT 1 ATTACH DATABASE 'foo' AS bar",
        "SELECT load_extension('/tmp/evil')",
        "SELECT readfile('/etc/passwd')",
        "SELECT writefile('/tmp/x', 'pwn')",
        "SELECT 1; COPY messages FROM '/tmp/evil'",
        "WITH x AS (SELECT 1) GRANT ALL ON messages TO public",
    ]
    for q in cases:
        err = _validate_sql(q)
        assert err is not None, f"missed forbidden in: {q}"


def test_rejects_keyword_via_comment_split():
    # Both `--` and block comments must be stripped before keyword scanning.
    bad = [
        "SELECT 1 /* sneaky */ INSERT INTO x VALUES (1)",
        "SELECT 1\n-- still inline\nDROP TABLE m",
        "WITH x AS (SELECT 1) /* hide */ DELETE FROM y",
    ]
    for q in bad:
        assert _validate_sql(q) is not None, f"missed via comment in: {q!r}"


def test_keyword_split_inside_comment_is_caught_after_strip():
    # /* */ inside the leading keyword breaks the SELECT/WITH match.
    assert _validate_sql("SE/* */LECT 1") == "only SELECT / WITH queries are allowed"


def test_string_literal_does_not_trip_filter():
    # Keywords inside a quoted string should NOT trigger the filter.
    q = "SELECT * FROM messages WHERE subject = 'INSERT happens here' LIMIT 10"
    assert _validate_sql(q) is None


def test_rejects_internal_tables_sqlite_names():
    """Legacy sqlite_* block list. Still in place so a future backend
    flip never accidentally unblocks the old surface."""
    for tbl in ("sqlite_master", "sqlite_schema", "sqlite_sequence", "sqlite_stat1", "sqlite_dbpage"):
        err = _validate_sql(f"SELECT * FROM {tbl}")
        assert err is not None and tbl in err, f"missed introspection of {tbl}"


def test_rejects_postgres_catalog_leaks():
    """pg_catalog / information_schema / pg_shadow leak role hashes and
    every other session's query text. Read-only mode alone doesn't
    protect against reading them."""
    for tbl in ("pg_catalog", "pg_shadow", "pg_roles", "pg_stat_activity", "information_schema"):
        err = _validate_sql(f"SELECT * FROM {tbl}.something")
        assert err is not None and tbl in err, f"missed introspection of {tbl}"


def test_rejects_negative_limit():
    cases = [
        "SELECT * FROM messages LIMIT -1",
        "SELECT * FROM messages LIMIT  -10",
        "SELECT * FROM messages WHERE 1 LIMIT-5",
    ]
    for q in cases:
        err = _validate_sql(q)
        assert err is not None and "negative" in err, f"missed negative LIMIT in: {q}"


def test_allows_normal_queries():
    cases = [
        "SELECT * FROM messages LIMIT 10",
        "WITH t AS (SELECT 1) SELECT * FROM t",
        "SELECT COUNT(*) FROM messages",
        "SELECT thread_id, subject FROM thread_summary ORDER BY date_last DESC LIMIT 50",
    ]
    for q in cases:
        assert _validate_sql(q) is None, f"false positive on: {q}"


def test_rejects_multiple_statements_after_strip():
    assert _validate_sql("SELECT 1; SELECT 2") is not None


# ─── End-to-end: read-only transaction actually blocks writes ─────────


@pytest.fixture
def fake_db(db_backend):
    """Seed a single message row into the per-test PG schema so the
    SELECT-it-back tests have something to read.
    """
    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)
    conn.execute(
        "INSERT INTO messages (id, thread_id, from_addr, to_addr, date) "
        "VALUES ('m1', 't1', 'a@x.com', 'b@x.com', '2026-01-01T00:00:00+00:00')"
    )
    conn.commit()
    conn.close()
    return db_path


def test_readonly_connection_rejects_write_even_if_gate_bypassed(fake_db):
    """Bypass `_validate_sql` by feeding an INSERT straight to the
    runner — the READ ONLY transaction set inside
    `_open_readonly_connection` must still reject it.
    """
    import psycopg

    with pytest.raises(psycopg.Error):
        _run_sql_with_timeout(
            fake_db,
            "INSERT INTO messages (id, thread_id, from_addr, to_addr, date) VALUES ('x','t','a','b','2026-01-01T00:00:00+00:00')",
        )


def test_readonly_select_works(fake_db):
    out = _run_sql_with_timeout(fake_db, "SELECT id FROM messages")
    assert out["row_count"] == 1
    assert out["rows"][0][0] == "m1"


def test_row_cap_enforced(fake_db):
    """Stuff in a few hundred rows and confirm we cap at SQL_MAX_ROWS=500."""
    conn = get_connection(fake_db)
    for i in range(700):
        conn.execute(
            "INSERT INTO messages (id, thread_id, from_addr, to_addr, date) "
            "VALUES (?, ?, 'a', 'b', '2026-01-01T00:00:00+00:00')",
            (f"m{i:04d}", f"t{i:04d}"),
        )
    conn.commit()
    conn.close()
    out = _run_sql_with_timeout(fake_db, "SELECT id FROM messages")
    assert out["row_count"] == 500
    assert out["truncated"] is True
