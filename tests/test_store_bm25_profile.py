"""The BM25 callers must use the selected key and must not hide a mismatch.

The live schema and the TEXT owner-partition profile both key the BM25 index on
`id`; there is no `search_id` column. The caller inventory requires that a
profile mismatch *propagates* rather than degrading into a valid-looking empty
result, because an empty score dict is indistinguishable from "nothing matched".

The fixture is module-scoped on purpose. `init_db` builds BM25 indexes, which
costs ~40s; per-test setup made these six tests take four minutes. Nothing here
mutates the seeded data, so one schema serves the whole module.
"""
import logging
import os
import uuid
from pathlib import Path

import pytest

from gmail_search.store import queries, schema_profile
from gmail_search.store.db import get_connection, init_db

LOGGER = logging.getLogger(__name__)

MESSAGES = [
    ("m1", "alice", "invoice for March", "the invoice body mentions payment"),
    ("m2", "alice", "holiday photos", "unrelated text about the beach"),
    ("m3", "bob", "invoice for April", "another invoice body for payment"),
]
ATTACHMENTS = [("alice", "m1", "invoice.pdf", "scanned invoice total due")]


def _seed(conn):
    for owner in sorted({row[1] for row in MESSAGES}):
        conn.execute("INSERT INTO users (id,email) VALUES (%s,%s)", (owner, owner + "@example.test"))
    for message_id, owner, subject, body in MESSAGES:
        conn.execute(
            "INSERT INTO messages (id,user_id,thread_id,from_addr,to_addr,subject,body_text,date) "
            "VALUES (%s,%s,'t','s@x','r@x',%s,%s,'2026-01-01')",
            (message_id, owner, subject, body),
        )
    for owner, message_id, filename, text in ATTACHMENTS:
        conn.execute(
            "INSERT INTO attachments (user_id,message_id,filename,mime_type,extracted_text) "
            "VALUES (%s,%s,%s,'application/pdf',%s)",
            (owner, message_id, filename, text),
        )


@pytest.fixture(scope="module")
def seeded(pg_schema_tools):
    """One schema, one `init_db`, one seed for the whole module.

    Built without the autouse per-test fixture: the connection is opened here
    and held, so the per-test `DB_DSN` rebinding that happens later cannot
    redirect it.

    The schema helpers arrive as a fixture rather than `import conftest`: that
    bare name resolves to whichever conftest.py pytest registered first, which
    in a full-suite run is `tests/perf/conftest.py`, and every test in this
    module then errors at setup. A per-file run resolves it correctly and
    passes, so the breakage appears only on the full sweep.
    """
    if not pg_schema_tools.reachable():
        pytest.skip("Set GMS_TEST_PG_DSN to a disposable PostgreSQL instance")
    schema = f"test_bm25_{uuid.uuid4().hex[:8]}"
    pg_schema_tools.make(schema)
    previous = {key: os.environ.get(key) for key in ("DB_BACKEND", "DB_DSN", "GMS_SCHEMA_PROFILE")}
    os.environ["DB_BACKEND"] = "postgres"
    os.environ["DB_DSN"] = pg_schema_tools.dsn_for(schema)
    # Module-scoped, so it runs before conftest's per-test declaration; state
    # the shape `pg_schema.sql` installs or the connect-time check will refuse.
    os.environ["GMS_SCHEMA_PROFILE"] = "numeric-key-v1"
    conn = None
    try:
        unused = Path("unused.db")  # Postgres ignores it; the DSN decides.
        init_db(unused)
        conn = get_connection(unused)
        _seed(conn)
        conn.commit()  # so the per-test rollback below cannot undo the seed
        yield conn
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if conn is not None:
            conn.close()
        pg_schema_tools.drop(schema)


@pytest.fixture(autouse=True)
def _clear_aborted_transaction(seeded):
    """A propagating error aborts the shared connection's transaction.

    Without this, the first mismatch test would poison every test after it with
    `InFailedSqlTransaction` — the cost of sharing one connection across a
    module. The seed is committed, so rolling back only discards test state.
    """
    yield
    seeded.rollback()


def test_bm25_messages_scores_on_the_selected_key(seeded):
    """The default profile keys on `id`, which is what the live index uses."""
    # Whatever shape the installer built, the caller must follow the binding.
    assert schema_profile.bm25_key(seeded) == seeded.profile.message_bm25_key
    scores = queries._pg_bm25_messages(seeded, "subject:invoice", 10, LOGGER)
    assert set(scores) == {"m1", "m3"}
    assert all(value > 0 for value in scores.values())


def test_bm25_messages_restricts_to_the_owner(seeded):
    scores = queries._pg_bm25_messages(seeded, "subject:invoice", 10, LOGGER, user_id="alice")
    assert set(scores) == {"m1"}


def test_bm25_messages_restricts_to_candidate_ids(seeded):
    scores = queries._pg_bm25_messages(seeded, "subject:invoice", 10, LOGGER, candidate_ids=["m3"])
    assert set(scores) == {"m3"}


def test_bm25_attachments_shares_the_same_key(seeded):
    """Both callers must resolve the key through one place, not two literals."""
    scores = queries._pg_bm25_attachments(seeded, "extracted_text:invoice", 10, LOGGER)
    assert set(scores) == {"m1"}


@pytest.mark.parametrize("caller", ["_pg_bm25_messages", "_pg_bm25_attachments"])
def test_profile_mismatch_propagates_instead_of_scoring_nothing(seeded, monkeypatch, caller):
    """A key the schema does not have is a deployment error, not "no results".

    Silently returning {} here is how a search outage hides: the request still
    succeeds and simply ranks nothing.
    """
    bogus = schema_profile.SchemaProfile(
        name="bogus", message_bm25_key="no_such_key", partitioned=False,
        attachment_bm25_key="no_such_key",
    )
    monkeypatch.setattr(seeded, "profile", bogus, raising=False)
    with pytest.raises(Exception) as excinfo:
        getattr(queries, caller)(seeded, "subject:invoice", 10, LOGGER)
    assert "no_such_key" in str(excinfo.value)


def test_transient_failure_still_degrades_quietly(seeded, monkeypatch):
    """Operational faults keep the old behaviour: log, return what we have.

    Only schema/profile mismatches are worth failing the request over.
    """
    def boom(*args, **kwargs):
        raise RuntimeError("connection reset by peer")

    monkeypatch.setattr(seeded, "execute", boom)
    assert queries._pg_bm25_messages(seeded, "subject:invoice", 10, LOGGER) == {}
