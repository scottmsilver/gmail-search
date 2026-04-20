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


# ─── Backend-parametrized DB fixture ──────────────────────────────────────
#
# Tests that accept `db_backend` auto-run twice: once against SQLite and
# once against Postgres. The fixture yields a dict with a `db_path` that
# tests pass through to `init_db(...)` and `get_connection(...)` — same
# function signature on both backends thanks to the shim in store/db.py.
#
# SQLite parametrization: `DB_BACKEND` is unset and we hand back a
# per-test tmp file.
#
# Postgres parametrization: we create a fresh `test_<uuid8>` schema per
# test and wire `DB_DSN` with an `options=-csearch_path=...` query param
# so every connection opened by `get_connection()` automatically lands in
# that schema. Teardown drops the schema with CASCADE. If the server
# isn't reachable on 127.0.0.1:5544 we skip just the PG parametrization —
# developers who haven't run `docker compose up pg` still get SQLite
# coverage.

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
    # psycopg accepts libpq-style `options` via the query string; the
    # `-c` prefix tells the server to apply a config parameter for the
    # session. We don't URL-encode the equals sign — libpq handles it.
    return f"{_PG_BASE_DSN}?options=-csearch_path%3D{schema_name}"


@pytest.fixture(params=["sqlite", "postgres"])
def db_backend(request, tmp_path, monkeypatch):
    """Dual-backend fixture. Tests see `{"kind": str, "db_path": Path|str}`.

    On SQLite: a fresh tmp file per test, `DB_BACKEND` cleared so the
    dispatch gate routes to the legacy path.

    On Postgres: a fresh `test_<uuid8>` schema per test. `DB_BACKEND` and
    `DB_DSN` are set for the duration; teardown drops the schema.
    """
    if request.param == "sqlite":
        monkeypatch.delenv("DB_BACKEND", raising=False)
        db_path = tmp_path / "test.db"
        yield {"kind": "sqlite", "db_path": db_path}
        return

    # Postgres path.
    if not _pg_server_reachable():
        pytest.skip("Postgres not reachable at 127.0.0.1:5544 — run `docker compose up pg` to enable")

    schema_name = f"test_{uuid.uuid4().hex[:8]}"
    _make_pg_schema(schema_name)

    monkeypatch.setenv("DB_BACKEND", "postgres")
    monkeypatch.setenv("DB_DSN", _pg_dsn_for_schema(schema_name))

    # db_path is ignored on the PG path, but we still pass one for API
    # compatibility with the existing (db_path) call sites.
    db_path = tmp_path / "unused.db"
    try:
        yield {"kind": "postgres", "db_path": db_path, "schema": schema_name}
    finally:
        _drop_pg_schema(schema_name)
