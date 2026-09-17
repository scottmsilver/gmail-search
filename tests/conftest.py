import os
from types import SimpleNamespace
import uuid

import pytest

from gmail_search.config import load_config


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    return d


@pytest.fixture
def test_config(tmp_path, data_dir):
    return load_config(config_path=tmp_path / "nonexistent.yaml", data_dir=data_dir)


# ─── Backend-parametrized DB fixture (Postgres only) ─────────────────────
#
# The SQLite backend was retired on 2026-04-20 (see Stage 2 cleanup).
# Tests that accept `db_backend` now run once against Postgres only. The
# fixture yields `{"kind": "postgres", "db_path": Path, "schema": str}`
# where `db_path` is an unused placeholder — call sites thread it into
# `init_db(...)` / `get_connection(...)` unchanged, and the PG layer
# ignores it in favour of `DB_DSN`.
#
# Each test gets a fresh `test_<uuid8>` schema; `DB_DSN` is wired with an
# `options=-csearch_path=...` query param so every connection opened by
# `get_connection()` automatically lands in that schema. Teardown drops
# the schema with CASCADE. If the server isn't reachable on
# the explicit GMS_TEST_PG_DSN is unset the fixture skips (i.e. any test using it is
# skipped) so `pytest` still exits cleanly on dev machines that haven't
# brought up the paradedb container.

_PG_BASE_DSN = os.environ.get("GMS_TEST_PG_DSN", "")

# The production mailbox lives in a database with this name. The isolation
# fixture pins `search_path` to `<schema>,public`, so any relation the
# disposable schema does not have resolves to `public` instead of failing --
# harmless against a scratch cluster, and how the live database twice ended up
# with test writes in it: 28 stray schemas one night, and the production owner's
# ScaNN index pointer redirected to a pytest temp directory the next, which
# disabled semantic search for the main mailbox until it was noticed.
#
# The fallback is only dangerous because the base DSN can be production. Refuse
# that, and a missing relation becomes an ordinary error instead of a write to
# real mail.
_PRODUCTION_DBNAME = "gmail_search"


def reject_production_dsn(dsn: str) -> None:
    """Raise if `dsn` names the production mailbox database.

    Matches the database name rather than host or port: the same cluster is
    reachable as localhost, 127.0.0.1, a container name and a unix socket, and
    any of those would otherwise slip through.
    """
    if not dsn:
        return
    try:
        from psycopg.conninfo import conninfo_to_dict

        dbname = conninfo_to_dict(dsn).get("dbname")
    except Exception:
        # An unparseable DSN is not a production DSN; let psycopg report it.
        return
    if dbname == _PRODUCTION_DBNAME:
        raise RuntimeError(
            f"GMS_TEST_PG_DSN names the production database ({_PRODUCTION_DBNAME!r}). "
            "Point it at a disposable cluster instead -- the isolation fixture falls "
            "back to the `public` schema, so tests would write into real mail."
        )


reject_production_dsn(_PG_BASE_DSN)

# Deliberately unconnectable. Port 1 is never a PostgreSQL server, and the
# database name is the instruction — psycopg puts it in the error text, so a
# test that needs a database says what to set instead of failing obscurely.
_NO_TEST_DATABASE_DSN = (
    "host=127.0.0.1 port=1 connect_timeout=1 user=none "
    "dbname=set_GMS_TEST_PG_DSN_to_run_database_tests"
)


def _pg_server_reachable() -> bool:
    """Cheap TCP probe so we skip PG tests cleanly on dev machines
    that haven't started the docker compose stack. Full psycopg dial
    would also work but prints a scary traceback on skip."""
    if not _PG_BASE_DSN:
        return False
    import psycopg
    try:
        with psycopg.connect(_PG_BASE_DSN, connect_timeout=2):
            return True
    except psycopg.OperationalError:
        return False


def _make_pg_schema(schema_name: str) -> None:
    """Create a fresh isolated schema on the shared PG test database."""
    import psycopg

    with psycopg.connect(_PG_BASE_DSN, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'CREATE SCHEMA "{schema_name}"')


def _drop_pg_schema(schema_name: str) -> None:
    """Drop the isolated schema and everything in it. Called in teardown."""
    import psycopg

    try:
        with psycopg.connect(_PG_BASE_DSN, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE')
    except Exception:
        # Teardown must never mask a real test failure.
        pass


def _pg_dsn_for_schema(schema_name: str) -> str:
    """Build a DSN that pins search_path to the isolated schema on every
    connection psycopg opens. `options=-csearch_path=...` is the standard
    trick — the server applies it at connection start so every statement
    resolves unqualified names against our fresh schema, not `public`.
    """
    # Pin a tight idle-in-transaction timeout (60s) on every test connection.
    # A test that leaks a connection (opens a txn, never commits/closes) would
    # otherwise pin the cluster-wide xmin horizon and starve autovacuum on the
    # SHARED production tables — exactly the leak that bloated prod
    # thread_summary to 25 GB / 68M dead tuples. 60s self-terminates any such
    # zombie. (The database-level default is a looser 10min for app code.)
    # %20 = space between -c options, %3D = '='.
    from psycopg.conninfo import make_conninfo
    return make_conninfo(
        _PG_BASE_DSN,
        options=f"-csearch_path={schema_name},public -cidle_in_transaction_session_timeout=60000",
    )


@pytest.fixture(scope="session")
def pg_schema_tools():
    """The disposable-schema helpers, handed to tests that build their own.

    A fixture rather than `from conftest import _make_pg_schema`: in a
    full-suite run the bare name `conftest` resolves to whichever conftest.py
    pytest registered first, which is `tests/perf/conftest.py`, and collection
    fails with a confusing ImportError. Per-file runs happen to work, so the
    breakage only appears on the full sweep.
    """
    return SimpleNamespace(
        reachable=_pg_server_reachable,
        make=_make_pg_schema,
        drop=_drop_pg_schema,
        dsn_for=_pg_dsn_for_schema,
    )


@pytest.fixture(autouse=True)
def _isolated_pg_schema(request, tmp_path, monkeypatch):
    """Autouse PG-schema isolation for every test.

    Stage 2 dropped the SQLite backend, which means every `init_db` /
    `get_connection` call now routes to Postgres. Without isolation,
    tests that use `tmp_path / "test.db"` end up writing straight into
    the production `public` schema on the dev machine.

    This fixture creates a fresh `test_<uuid8>` schema per test and wires
    `DB_DSN` with an `options=-csearch_path=...` query param so every
    connection opened by `get_connection()` lands in that schema.
    Teardown drops the schema with CASCADE. Skips when PG isn't
    reachable (dev machines that haven't started the paradedb container)
    — tests that don't touch the DB still run.

    Tests that need the schema name (e.g. integration smoke tests) can
    depend on the `db_backend` fixture, which is a thin wrapper around
    this one.
    """
    if not _pg_server_reachable():
        # No disposable cluster configured. Crucially, do NOT leave `DB_DSN`
        # alone here: its default in `store/db.py` is the *live* mailbox, so a
        # test that opens a connection without this fixture's rewrite lands in
        # production. That is not hypothetical — it left 28 `test_<uuid8>`
        # schemas, 326 tables and 87 MB inside the production database before
        # anyone noticed, because each one is isolated enough not to corrupt
        # `public` and therefore never announced itself.
        #
        # Point at something that cannot connect and whose error names the fix.
        monkeypatch.setenv("DB_BACKEND", "postgres")
        monkeypatch.setenv("DB_DSN", _NO_TEST_DATABASE_DSN)
        yield None
        return

    schema_name = f"test_{uuid.uuid4().hex[:8]}"
    _make_pg_schema(schema_name)

    monkeypatch.setenv("DB_BACKEND", "postgres")
    monkeypatch.setenv("DB_DSN", _pg_dsn_for_schema(schema_name))
    # The suite as a whole runs on the NUMERIC shape: the gateway search-reader
    # fixtures and `migrate_owner_partitions.py` are written against it. Since
    # the profile fences landed this is a *choice* rather than the only thing
    # `pg_schema.sql` can install — a test that wants the live TEXT shape sets
    # the variable itself and gets it (see `test_fresh_install_matches_profile`).
    monkeypatch.setenv("GMS_SCHEMA_PROFILE", "numeric-key-v1")

    # Multi-tenant Phase 2/3: every per-user table requires a non-NULL
    # user_id and `resolve_write_user_id` looks up `users` by email.
    # Tests don't sign anyone in, so seed a bootstrap row so write
    # paths don't blow up with "bootstrap user not found." Clear the
    # process-level cache too — a previous test in this process may
    # have memo'd a user_id that doesn't exist in this fresh schema.
    from gmail_search.auth import write_user as _write_user_mod

    _write_user_mod._BOOTSTRAP_CACHE.clear()
    _seed_bootstrap_user(schema_name)

    try:
        yield {"kind": "postgres", "schema": schema_name}
    finally:
        _write_user_mod._BOOTSTRAP_CACHE.clear()
        _drop_pg_schema(schema_name)


def _seed_bootstrap_user(schema_name: str) -> None:
    """Insert a `users` row matching `GMS_BOOTSTRAP_EMAIL` (default
    scott) so `resolve_write_user_id` succeeds in tests that touch any
    write path. Uses init_db (which runs pg_schema.sql) via the
    project's get_connection helper — same path tests use, same
    transaction handling. Idempotent: ON CONFLICT keeps one row."""
    import os
    from pathlib import Path

    from gmail_search.store.db import get_connection, init_db

    # init_db is the canonical schema applier. Tests typically call it
    # themselves; doing it here too is a harmless no-op (idempotent
    # CREATE TABLE IF NOT EXISTS) but ensures the bootstrap insert
    # below has `users` to write to even for tests that never call
    # init_db directly.
    init_db(Path("unused.db"))
    email = os.environ.get("GMS_BOOTSTRAP_EMAIL", "scottmsilver@gmail.com").lower()
    conn = get_connection(Path("unused.db"))
    try:
        conn.execute(
            "INSERT INTO users (id, email) VALUES (%s, %s) ON CONFLICT (email) DO NOTHING",
            ("u_test_bootstrap", email),
        )
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def db_backend(_isolated_pg_schema, tmp_path):
    """Postgres-only DB fixture. Tests see `{"kind": "postgres",
    "db_path": Path, "schema": str}`.

    A fresh `test_<uuid8>` schema is created per test (via the autouse
    `_isolated_pg_schema`); teardown drops it.
    """
    if _isolated_pg_schema is None:
        pytest.skip("Set GMS_TEST_PG_DSN to a disposable PostgreSQL instance to enable database tests")

    # db_path is ignored, but we still pass one for API compatibility
    # with the existing (db_path) call sites.
    db_path = tmp_path / "unused.db"
    yield {
        "kind": "postgres",
        "db_path": db_path,
        "schema": _isolated_pg_schema["schema"],
    }
