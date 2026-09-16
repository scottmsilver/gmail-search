"""The schema shape is selected, carried and verified — never inferred.

These tests pin the three properties the caller inventory asks for: the
selection comes from configuration, the connection carries it, and a database
that disagrees fails loudly instead of degrading.
"""
import pytest

from gmail_search.store import queries, schema_profile
from gmail_search.store.db import get_connection, init_db
from gmail_search.store.schema_profile import (
    DEFAULT_PROFILE,
    NUMERIC_KEY_V1,
    SELECTION_ENV,
    TEXT_KEY_V1,
    TEXT_PARTITIONED_V1,
    SchemaProfileMismatch,
)


@pytest.fixture(autouse=True)
def _fresh_verification_cache():
    schema_profile.reset_verification_cache()
    yield
    schema_profile.reset_verification_cache()


def test_default_selection_matches_the_live_shape():
    """Live keys BM25 on `id` and is not partitioned."""
    assert DEFAULT_PROFILE.message_bm25_key == "id"
    assert DEFAULT_PROFILE.partitioned is False


def test_selection_comes_from_configuration(monkeypatch):
    monkeypatch.setenv(SELECTION_ENV, "text-partitioned-v1")
    assert schema_profile.selected_profile() is TEXT_PARTITIONED_V1


def test_unknown_selection_is_refused(monkeypatch):
    monkeypatch.setenv(SELECTION_ENV, "whatever-we-feel-like")
    with pytest.raises(SchemaProfileMismatch) as excinfo:
        schema_profile.selected_profile()
    assert "not a known schema profile" in str(excinfo.value)


def test_connection_carries_the_declared_profile(db_backend):
    """conftest declares `numeric-key-v1`, the shape `pg_schema.sql` installs."""
    init_db(db_backend["db_path"])
    conn = get_connection(db_backend["db_path"])
    try:
        assert conn.profile is NUMERIC_KEY_V1
        assert queries._bm25_score_key(conn) == conn.profile.message_bm25_key
    finally:
        conn.close()


def test_mismatched_database_is_refused_at_connect(db_backend, monkeypatch):
    """Declaring a shape the database does not have must fail at connect.

    This is the case that matters: a half-applied migration, or a reader
    deployed ahead of its schema, is caught before any mailbox work.
    """
    init_db(db_backend["db_path"])
    monkeypatch.setenv(SELECTION_ENV, TEXT_KEY_V1.name)
    with pytest.raises(SchemaProfileMismatch) as excinfo:
        get_connection(db_backend["db_path"])
    assert "key_field='id'" in str(excinfo.value)
    assert "key_field='search_id'" in str(excinfo.value)


def test_observed_shape_reads_the_catalog(db_backend):
    init_db(db_backend["db_path"])
    conn = get_connection(db_backend["db_path"])
    try:
        key, partitioned = schema_profile.observed_shape(conn)
        assert (key, partitioned) == ("search_id", False)
    finally:
        conn.close()


def test_bound_profile_drives_the_bm25_key(db_backend):
    """A caller must take the key from the connection, not a local literal."""
    init_db(db_backend["db_path"])
    conn = get_connection(db_backend["db_path"])
    try:
        object.__setattr__(conn, "profile", TEXT_KEY_V1)
        assert queries._bm25_score_key(conn) == "id"
    finally:
        conn.close()


def test_verification_is_cached_per_process(db_backend, monkeypatch):
    """Sixty call sites opening connections should not each query the catalog."""
    init_db(db_backend["db_path"])
    conn = get_connection(db_backend["db_path"])
    calls = []
    real = schema_profile.observed_shape
    monkeypatch.setattr(schema_profile, "observed_shape",
                        lambda c: (calls.append(1), real(c))[1])
    schema_profile.verify(conn, NUMERIC_KEY_V1, cache_key="same")
    schema_profile.verify(conn, NUMERIC_KEY_V1, cache_key="same")
    conn.close()
    assert len(calls) == 1
