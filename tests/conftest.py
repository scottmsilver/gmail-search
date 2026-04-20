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
# 127.0.0.1:5544 the whole fixture skips (i.e. any test using it is
# skipped) so `pytest` still exits cleanly on dev machines that haven't
# brought up the paradedb container.

_PG_BASE_DSN = "postgresql://gmail_search:gmail_search@127.0.0.1:5544/gmail_search"


def _pg_server_reachable() -> bool:
    """Cheap TCP probe so we skip PG tests cleanly on dev machines
    that haven't started the docker compose stack. Full psycopg dial
    would also work but prints a scary traceback on skip."""
    import socket

    try:
        with socket.create_connection(("127.0.0.1", 5544), timeout=0.5):
            return True
    except OSError:
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
    return f"{_PG_BASE_DSN}?options=-csearch_path%3D{schema_name}"


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
        yield None
        return

    schema_name = f"test_{uuid.uuid4().hex[:8]}"
    _make_pg_schema(schema_name)

    monkeypatch.setenv("DB_BACKEND", "postgres")
    monkeypatch.setenv("DB_DSN", _pg_dsn_for_schema(schema_name))

    try:
        yield {"kind": "postgres", "schema": schema_name}
    finally:
        _drop_pg_schema(schema_name)


@pytest.fixture
def db_backend(_isolated_pg_schema, tmp_path):
    """Postgres-only DB fixture. Tests see `{"kind": "postgres",
    "db_path": Path, "schema": str}`.

    A fresh `test_<uuid8>` schema is created per test (via the autouse
    `_isolated_pg_schema`); teardown drops it.
    """
    if _isolated_pg_schema is None:
        pytest.skip("Postgres not reachable at 127.0.0.1:5544 — run `docker compose up pg` to enable")

    # db_path is ignored, but we still pass one for API compatibility
    # with the existing (db_path) call sites.
    db_path = tmp_path / "unused.db"
    yield {
        "kind": "postgres",
        "db_path": db_path,
        "schema": _isolated_pg_schema["schema"],
    }
