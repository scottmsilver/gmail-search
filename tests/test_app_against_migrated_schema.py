"""The application must work on the shape the migration leaves behind.

Everything qualified so far proves the migration *mechanism*: that it converts a
production-shaped database into owner partitions without losing rows. Nothing
had ever run the application against the result, and that is the gap that turns
a four-minute window into an outage — the post-migration table is keyed
`(user_id, id)` and partitioned by `user_id`, while the live writer upserts
`ON CONFLICT(id)`.

So this builds the target shape and drives the real code over it: the writer, the
BM25 readers, the profile binding. It is deliberately about the *application*,
not the migration; `rehearse_mixed_migration.py` owns the other half.

Disposable cluster only. Every check here is one that would otherwise be made
for the first time against 57 GB of live mail.
"""
from __future__ import annotations

import contextlib
import os
import uuid

import pytest

from datetime import datetime, timezone

from gmail_search.store import queries, schema_profile
from gmail_search.store.db import get_connection
from gmail_search.store.models import Message
from gmail_search.store.schema_profile import SELECTION_ENV, TEXT_PARTITIONED_V1

OWNERS = ("owner_dominant", "owner_minority")


@pytest.fixture(scope="module")
def migrated(pg_schema_tools):
    """A partitioned, `(user_id, id)`-keyed mailbox — the post-migration shape.

    Built directly rather than by running the migration: the migration is
    qualified elsewhere and takes minutes, and what is under test here is
    whether the application can speak to the result. The shape is asserted
    against `gateway.partition_profiles` so it cannot drift from what the
    migration actually produces.
    """
    if not pg_schema_tools.reachable():
        pytest.skip("Set GMS_TEST_PG_DSN to a disposable PostgreSQL instance")

    schema = f"migrated_{uuid.uuid4().hex[:8]}"
    pg_schema_tools.make(schema)
    dsn = pg_schema_tools.dsn_for(schema)

    import psycopg
    from psycopg import sql

    from gmail_search.store.db import init_db

    previous = {key: os.environ.get(key) for key in ("DB_BACKEND", "DB_DSN", SELECTION_ENV)}
    try:
        # The whole schema first, so the writer finds every table it maintains —
        # `thread_summary` and the rest — and only `messages` is replaced. A
        # hand-written subset would pass for the wrong reason the moment the
        # writer touched something the subset omitted.
        os.environ["DB_BACKEND"] = "postgres"
        os.environ["DB_DSN"] = dsn
        os.environ[SELECTION_ENV] = "text-key-v1"
        schema_profile.reset_verification_cache()
        init_db(None)

        with psycopg.connect(dsn, autocommit=True) as setup:
            setup.execute("INSERT INTO users (id,email) VALUES (%s,%s) ON CONFLICT (id) DO NOTHING",
                          (OWNERS[0], OWNERS[0] + "@example.test"))
            setup.execute("INSERT INTO users (id,email) VALUES (%s,%s) ON CONFLICT (id) DO NOTHING",
                          (OWNERS[1], OWNERS[1] + "@example.test"))
            columns = setup.execute(
                "SELECT string_agg(column_name || ' ' || data_type, ', ' ORDER BY ordinal_position) "
                "FROM information_schema.columns WHERE table_schema=%s AND table_name='messages'",
                (schema,)).fetchone()[0]
            setup.execute("DROP TABLE messages CASCADE")
            setup.execute(sql.SQL(
                "CREATE TABLE messages ({}, PRIMARY KEY (user_id, id)) PARTITION BY LIST (user_id)"
            ).format(sql.SQL(columns)))
            for owner in OWNERS:
                # A partition bound is DDL, not a value: psycopg cannot infer a
                # type for a placeholder here, so the literal is composed.
                setup.execute(sql.SQL('CREATE TABLE {} PARTITION OF messages FOR VALUES IN ({})')
                              .format(sql.Identifier('messages_' + owner), sql.Literal(owner)))
            setup.execute("""CREATE INDEX messages_bm25_idx ON messages
                USING bm25 (id, subject, body_text, from_addr, to_addr)
                WITH (key_field='id')""")
        yield dsn
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        schema_profile.reset_verification_cache()
        pg_schema_tools.drop(schema)


@contextlib.contextmanager
def app_connection(dsn, profile="text-partitioned-v1"):
    """A connection to the migrated schema, bound to a declared profile.

    The per-test autouse isolation fixture rewrites `DB_DSN` to its own empty
    schema *after* this module's fixture has run, so the environment has to be
    set inside the test rather than around it — otherwise every check here
    silently measures a freshly installed unpartitioned database instead of the
    migrated one, and passes for the wrong reason.
    """
    previous = {key: os.environ.get(key) for key in ("DB_BACKEND", "DB_DSN", SELECTION_ENV)}
    os.environ["DB_BACKEND"] = "postgres"
    os.environ["DB_DSN"] = dsn
    os.environ[SELECTION_ENV] = profile
    schema_profile.reset_verification_cache()
    conn = None
    try:
        conn = get_connection(None)
        yield conn
    finally:
        if conn is not None:
            conn.close()
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        schema_profile.reset_verification_cache()


def _message(message_id, subject, body):
    return Message(id=message_id, thread_id="t_" + message_id,
                   from_addr="sender@example.test", to_addr="recipient@example.test",
                   subject=subject, body_text=body, body_html="",
                   date=datetime(2026, 1, 1, tzinfo=timezone.utc), labels=[],
                   history_id=0, raw_json="{}")


def _write(conn, owner, message_id, subject, body):
    """The real writer, with the owner stated rather than resolved from the
    bootstrap user — these owners are the point of the test."""
    queries.upsert_message(conn, _message(message_id, subject, body), user_id=owner)


def _bm25(conn, term, owner, candidates):
    """Scored exactly as the search path does it.

    The query goes through `_build_bm25_query`, which field-qualifies every
    token (`subject:invoice body_text:invoice ...`). A bare term would match
    nothing — Tantivy has no default field here — so building it by hand would
    make the test pass or fail for reasons the application never encounters.
    """
    import logging

    disjunction, _ = queries._build_bm25_query([term], queries._BM25_MESSAGE_FIELDS)
    return queries._pg_bm25_messages(conn, disjunction, 50, logging.getLogger(__name__),
                                     candidates, user_id=owner)


# ── the shape itself ─────────────────────────────────────────────────────────

def test_the_built_shape_is_the_one_the_migration_produces(migrated):
    """Guards the fixture. If this drifts from `partition_profiles`, every other
    test in the file is measuring the wrong thing."""
    from gmail_search.gateway.partition_profiles import TEXT_OWNER_PARTITIONS_V1 as TEXT

    with app_connection(migrated) as conn:
        assert schema_profile.observed_shape(conn) == (TEXT.message_key, True)
        keys = conn.execute(
            "SELECT string_agg(a.attname,',' ORDER BY k.ord) FROM pg_constraint c "
            "JOIN LATERAL unnest(c.conkey) WITH ORDINALITY k(attnum,ord) ON true "
            "JOIN pg_attribute a ON a.attrelid=c.conrelid AND a.attnum=k.attnum "
            "WHERE c.conrelid='messages'::regclass AND c.contype='p'"
        ).fetchone()[0]
        assert keys == "user_id,id"


def test_the_binding_accepts_the_partitioned_profile(migrated):
    """The daemons will run with `GMS_SCHEMA_PROFILE=text-partitioned-v1`. If the
    binding refused it, every connection after the migration would fail — the
    outage this file exists to rule out."""
    with app_connection(migrated) as conn:
        assert conn.profile is TEXT_PARTITIONED_V1


# ── the writer ───────────────────────────────────────────────────────────────

def test_the_writer_upserts_into_the_partitioned_table(migrated):
    """`ON CONFLICT(user_id, id)`. The live writer says `ON CONFLICT(id)` and
    would fail here with InvalidColumnReference, which is exactly why the branch
    and the migration have to ship together."""
    with app_connection(migrated) as conn:
        _write(conn, OWNERS[0], "m1", "invoice for March", "the invoice body mentions payment")
        stored = conn.execute("SELECT subject FROM messages WHERE user_id=%s AND id=%s",
                              (OWNERS[0], "m1")).fetchone()
        assert stored["subject"] == "invoice for March"


def test_the_same_id_can_exist_for_two_owners(migrated):
    """The whole point of the composite key. Under the old `(id)` key the second
    write would have overwritten the first owner's message."""
    with app_connection(migrated) as conn:
        for owner in OWNERS:
            _write(conn, owner, "shared-id", owner + " subject", owner + " private body")
        rows = conn.execute(
            "SELECT user_id, subject FROM messages WHERE id=%s ORDER BY user_id",
            ("shared-id",)).fetchall()
        assert [(row["user_id"], row["subject"]) for row in rows] == [
            (owner, owner + " subject") for owner in sorted(OWNERS)]


def test_re_upserting_updates_rather_than_duplicating(migrated):
    with app_connection(migrated) as conn:
        _write(conn, OWNERS[0], "m2", "first", "first body")
        _write(conn, OWNERS[0], "m2", "second", "second body")
        rows = conn.execute("SELECT subject FROM messages WHERE user_id=%s AND id=%s",
                            (OWNERS[0], "m2")).fetchall()
        assert [row["subject"] for row in rows] == ["second"]


def test_rows_land_in_their_owner_partition(migrated):
    """Routing, not just storage: a row in the wrong leaf would still be
    readable through the parent and would silently break owner isolation at the
    partition level, which is what the invited runtime relies on."""
    with app_connection(migrated) as conn:
        _write(conn, OWNERS[1], "m3", "minority", "minority body")
        leaf = conn.execute(
            "SELECT tableoid::regclass::text FROM messages WHERE user_id=%s AND id=%s",
            (OWNERS[1], "m3")).fetchone()[0]
        assert leaf.endswith(OWNERS[1])


# ── the readers ──────────────────────────────────────────────────────────────

def test_bm25_scoring_works_on_the_partitioned_parent(migrated):
    """The BM25 index lives on the partitioned parent. `paradedb.score(id)`
    against a partitioned relation is the part that had never been exercised by
    the application's own query builder."""
    with app_connection(migrated) as conn:
        _write(conn, OWNERS[0], "b1", "quarterly invoice", "the invoice covers payment terms")
        _write(conn, OWNERS[0], "b2", "holiday photos", "unrelated text about the beach")
        scores = _bm25(conn, "invoice", OWNERS[0], ["b1", "b2"])
        assert scores.get("b1", 0) > scores.get("b2", 0)


def test_bm25_does_not_score_another_owners_mail(migrated):
    """Owner isolation through the read path, on the shape that will be live."""
    with app_connection(migrated) as conn:
        _write(conn, OWNERS[0], "iso-a", "shared word invoice", "dominant private body")
        _write(conn, OWNERS[1], "iso-b", "shared word invoice", "minority private body")
        scores = _bm25(conn, "invoice", OWNERS[0], ["iso-a", "iso-b"])
        assert "iso-a" in scores
        assert "iso-b" not in scores


def test_a_mismatched_profile_is_still_refused_here(migrated):
    """The binding must not become permissive just because the database is
    partitioned: claiming the unpartitioned shape has to fail."""
    with pytest.raises(schema_profile.SchemaProfileMismatch, match="partitioned=True"):
        with app_connection(migrated, profile="text-key-v1"):
            pass
