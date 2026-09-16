"""A fresh install must produce the shape this process says it expects.

Before the profile fences, `pg_schema.sql` always added `search_id` and always
keyed BM25 on it, so every fresh install was the NUMERIC shape while the live
database was TEXT. Nothing failed at install time — the divergence only showed
up much later as a query against a column that isn't there.

The assertion that matters is the round trip: install under a profile, then ask
the connection binding to verify that same profile against the catalog. If the
installer and the binding ever disagree again, one of these fails at the point
of installation rather than in production.

These tests need their own schemas rather than the autouse `db_backend` one,
because that fixture has already installed NUMERIC by the time a test body
runs. Each shape is installed once for the module, not once per test — the
install builds BM25 indexes, which is the expensive part.
"""
from __future__ import annotations

import contextlib
import os
import uuid

import pytest

from gmail_search.store import schema_profile
from gmail_search.store.db import _read_pg_schema, get_connection, init_db
from gmail_search.store.schema_profile import SELECTION_ENV, SchemaProfileMismatch

from conftest import (  # noqa: F401  (module-level helpers, not fixtures)
    _drop_pg_schema,
    _make_pg_schema,
    _pg_dsn_for_schema,
    _pg_server_reachable,
)

_INSTALLABLE = ["text-key-v1", "numeric-key-v1"]


@contextlib.contextmanager
def _selected(profile_name: str, dsn: str):
    """Run a block as a process configured for one profile against one schema.

    `monkeypatch` is function-scoped and the install fixture is not, so the
    environment is saved and restored by hand.
    """
    previous = {key: os.environ.get(key) for key in (SELECTION_ENV, "DB_DSN")}
    os.environ[SELECTION_ENV] = profile_name
    os.environ["DB_DSN"] = dsn
    schema_profile.reset_verification_cache()
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        schema_profile.reset_verification_cache()


@pytest.fixture(scope="module")
def installed():
    """One freshly installed schema per profile. Returns {profile: dsn}."""
    if not _pg_server_reachable():
        pytest.skip("Set GMS_TEST_PG_DSN to a disposable PostgreSQL instance")

    schemas = {name: f"install_{uuid.uuid4().hex[:8]}" for name in _INSTALLABLE}
    dsns = {name: _pg_dsn_for_schema(schema) for name, schema in schemas.items()}
    try:
        for name, dsn in dsns.items():
            _make_pg_schema(schemas[name])
            with _selected(name, dsn):
                init_db(None)
        yield dsns
    finally:
        for schema in schemas.values():
            _drop_pg_schema(schema)


def _column_exists(conn, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM pg_attribute WHERE attrelid='messages'::regclass "
        "AND attname=%s AND NOT attisdropped",
        (name,),
    ).fetchone() is not None


@pytest.mark.parametrize("profile_name", _INSTALLABLE)
def test_fresh_install_satisfies_its_own_profile(installed, profile_name):
    """`get_connection` verifies the catalog before handing back a connection,
    so reaching one at all is the assertion. The checks below name what held."""
    expected = schema_profile.PROFILES[profile_name]
    with _selected(profile_name, installed[profile_name]):
        conn = get_connection(None)
        try:
            assert conn.profile is expected
            assert schema_profile.observed_shape(conn) == (
                expected.message_bm25_key,
                expected.partitioned,
            )
        finally:
            conn.close()


def test_text_install_has_no_search_id_column(installed):
    """The column, not just the index key: an unused identity column is still a
    sequence, a unique index and a write cost on every insert."""
    with _selected("text-key-v1", installed["text-key-v1"]):
        conn = get_connection(None)
        try:
            assert not _column_exists(conn, "search_id")
            assert _column_exists(conn, "id")
        finally:
            conn.close()


def test_numeric_install_still_has_it(installed):
    """The gateway search-reader fixtures and `migrate_owner_partitions.py`
    select NUMERIC deliberately; the fence must not remove it for everyone."""
    with _selected("numeric-key-v1", installed["numeric-key-v1"]):
        conn = get_connection(None)
        try:
            assert _column_exists(conn, "search_id")
        finally:
            conn.close()


def test_claiming_the_other_shape_is_refused_at_connect(installed):
    """Point a NUMERIC-configured process at the TEXT database. It must refuse
    rather than run `search_id` queries against a database without the column."""
    with _selected("numeric-key-v1", installed["text-key-v1"]):
        with pytest.raises(SchemaProfileMismatch) as excinfo:
            get_connection(None).close()
    message = str(excinfo.value)
    assert "expects key_field='search_id'" in message
    assert "database has key_field='id'" in message


def test_an_empty_database_is_not_read_as_the_text_shape(installed):
    """`observed_shape` used to default the key to `id` whenever it could not
    find a BM25 index, so a database with no `messages` table at all satisfied
    the TEXT profile. That is the exact failure this module exists to prevent,
    so it gets its own test rather than riding on the install cases."""
    schema = f"empty_{uuid.uuid4().hex[:8]}"
    _make_pg_schema(schema)
    try:
        with _selected("text-key-v1", _pg_dsn_for_schema(schema)):
            with pytest.raises(SchemaProfileMismatch) as excinfo:
                get_connection(None).close()
        assert "no BM25 index" in str(excinfo.value)
    finally:
        _drop_pg_schema(schema)


def test_the_installed_bm25_index_really_exists(installed):
    """Guards the test above's opposite: the install cases must be passing
    because an index is there, not because the probe found nothing."""
    for profile_name, dsn in installed.items():
        with _selected(profile_name, dsn):
            conn = get_connection(None)
            try:
                reloptions = conn.execute(
                    "SELECT c.reloptions::text FROM pg_class c "
                    "JOIN pg_index i ON i.indexrelid=c.oid JOIN pg_am am ON am.oid=c.relam "
                    "WHERE i.indrelid=to_regclass('messages') AND am.amname='bm25'"
                ).fetchone()
                assert reloptions is not None, f"{profile_name}: no BM25 index on messages"
                expected = schema_profile.PROFILES[profile_name].message_bm25_key
                assert f"key_field={expected}" in reloptions[0]
            finally:
                conn.close()


def test_a_partitioned_install_is_seen_by_observed_shape(installed):
    """`text-partitioned-v1` is the migration *target*, and nothing covered it.

    `observed_shape` looks for a BM25 index whose `indrelid` is `messages`. The
    open question was whether a partitioned install attaches that index to the
    child partitions instead, in which case the parent has none, the shape reads
    as absent, and **every connection would be refused right after the
    migration** — a self-inflicted outage at the worst possible moment.

    It does not: `CREATE INDEX` on a LIST-partitioned parent creates a
    partitioned index (`relkind='I'`) whose `indrelid` is the parent, so the
    probe finds it and carries the real `key_field`. Pinned here because the
    answer is a property of PostgreSQL's catalog, not of our code, and nothing
    else in the suite would notice if the migration changed where the index
    lands.
    """
    schema = f"partprobe_{uuid.uuid4().hex[:8]}"
    _make_pg_schema(schema)
    try:
        with _selected("text-key-v1", _pg_dsn_for_schema(schema)):
            import psycopg

            from gmail_search.store.db import _pg_dsn

            with psycopg.connect(_pg_dsn(), autocommit=True) as conn:
                conn.execute(
                    "CREATE TABLE messages(id TEXT NOT NULL, user_id TEXT NOT NULL,"
                    " subject TEXT NOT NULL DEFAULT '', body_text TEXT NOT NULL DEFAULT '',"
                    " from_addr TEXT NOT NULL DEFAULT '', to_addr TEXT NOT NULL DEFAULT '',"
                    " PRIMARY KEY (user_id, id)) PARTITION BY LIST (user_id)")
                conn.execute("CREATE TABLE messages_alice PARTITION OF messages"
                             " FOR VALUES IN ('alice')")
                conn.execute("CREATE INDEX messages_bm25_idx ON messages USING bm25"
                             " (id, subject, body_text, from_addr, to_addr)"
                             " WITH (key_field='id')")
                assert schema_profile.observed_shape(conn) == ("id", True)
                # The binding must accept the partitioned target, and must still
                # refuse the unpartitioned profile against the same database.
                schema_profile.verify(conn, schema_profile.TEXT_PARTITIONED_V1, cache_key=None)
                with pytest.raises(SchemaProfileMismatch, match="partitioned=False"):
                    schema_profile.verify(conn, schema_profile.TEXT_KEY_V1, cache_key=None)
    finally:
        _drop_pg_schema(schema)


@pytest.mark.parametrize("profile_name", _INSTALLABLE)
def test_no_fence_marker_survives_selection(monkeypatch, profile_name):
    """A marker left in the text reaches PostgreSQL as a comment, which would
    silently keep the block it was meant to remove."""
    monkeypatch.setenv(SELECTION_ENV, profile_name)
    assert "profile-only:" not in _read_pg_schema()


def test_docs_are_checked_against_the_schema_that_gets_installed(monkeypatch):
    """`assert_table_docs_cover_schema` reads the same selected text the
    installer applies, so a fenced-out table can never be documented-but-absent."""
    from gmail_search.store.db import assert_table_docs_cover_schema

    for profile_name in _INSTALLABLE:
        monkeypatch.setenv(SELECTION_ENV, profile_name)
        assert_table_docs_cover_schema()
