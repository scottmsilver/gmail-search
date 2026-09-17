"""The production driver, exercised against a disposable database.

`rehearse_mixed_migration.py` proves the mechanism by building its own database.
This proves the *driver* — the thing that will be pointed at the real one: the
declared-target guard, the fence that checks who is connected, the durable
registry, and the two phases run in order against a database it did not create.

The fence is the piece that most needs a test. It is the only thing standing
between "the operator stopped the daemons" and "the operator believes they
stopped the daemons", and it cannot be checked by reading it.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import secrets

import psycopg
from psycopg import sql
import pytest

ROOT = Path(__file__).parents[1]
LEGACY_SCHEMA = ROOT / 'tests/fixtures/legacy_text_owner_schema.sql'
# Deliberately ordered so that largest-first is NOT alphabetical: `zz_dominant`
# holds the most rows but sorts last. The plan binding digests the owner tuple
# and requires it sorted, so passing the report's largest-first order straight
# through is refused — which is exactly what happened against production the
# first time, because both test owners here used to be alphabetical *and* in
# count order, and the bug had nowhere to show itself.
OWNERS = ('aa_minority', 'zz_dominant')
DOMINANT, MINORITY = 'zz_dominant', 'aa_minority'


@pytest.fixture(scope='module')
def driver():
    spec = importlib.util.spec_from_file_location(
        'migrate_production_mixed', ROOT / 'deploy/public/migrate_production_mixed.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def database():
    """A disposable database in production's *starting* shape.

    Built from the frozen legacy fixture rather than `pg_schema.sql`, because
    that fixture is what production actually looks like: `id TEXT PRIMARY KEY`,
    no `search_id`, BM25 keyed on `id`. Seeding from the newer schema would
    start the migration from a shape it never has to handle.
    """
    base = os.getenv('GMS_TEST_PG_DSN')
    if not base:
        pytest.skip('Explicit disposable ParadeDB required')
    suffix = secrets.token_hex(8)
    name = 'gms_owner_partitions_test_text_' + suffix
    with psycopg.connect(base, autocommit=True) as admin:
        admin.execute(sql.SQL('CREATE DATABASE {} TEMPLATE template0').format(sql.Identifier(name)))
    from psycopg.conninfo import conninfo_to_dict, make_conninfo
    config = conninfo_to_dict(base)
    config['dbname'] = name
    dsn = make_conninfo(**config)
    try:
        with psycopg.connect(dsn, autocommit=True) as conn:
            # The fixture hardcodes the production database and role names; each
            # disposable copy needs its own so concurrent runs cannot collide on
            # a cluster-wide role. Same substitutions the rehearsal makes.
            fixture = (LEGACY_SCHEMA.read_text()
                       .replace('gmail_search_reader', 'gms_drv_reader_' + suffix)
                       .replace('gmail_analyst', 'gms_drv_analyst_' + suffix)
                       .replace('ON DATABASE gmail_search', 'ON DATABASE ' + name))
            conn.execute(fixture)
            for owner in OWNERS:
                conn.execute('INSERT INTO public.users (id,email) VALUES (%s,%s)',
                             (owner, owner + '@example.test'))
            rows = [(f'{owner}-{index}', owner, 'thread', 'a@x', 'b@x',
                     f'subject {index}', f'body text {index} invoice', '2026-01-01')
                    for owner, count in ((DOMINANT, 40), (MINORITY, 5))
                    for index in range(count)]
            conn.cursor().executemany(
                'INSERT INTO public.messages (id,user_id,thread_id,from_addr,to_addr,'
                'subject,body_text,date) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)', rows)
        yield dsn
    finally:
        with psycopg.connect(base, autocommit=True) as admin:
            admin.execute(sql.SQL('DROP DATABASE IF EXISTS {} WITH (FORCE)')
                          .format(sql.Identifier(name)))
            # Roles are cluster-wide, so they outlive the database.
            for role in ('gms_drv_reader_' + suffix, 'gms_drv_analyst_' + suffix):
                admin.execute(sql.SQL('DROP ROLE IF EXISTS {}').format(sql.Identifier(role)))


# ── the fence ────────────────────────────────────────────────────────────────

def test_the_fence_refuses_when_another_connection_is_attached(driver, database):
    """The case the fence exists for: the operator thinks the writers are down
    and one is not. Holding a single extra connection is enough."""
    fence = driver.ProductionFence(driver._connector(database))
    with psycopg.connect(database, application_name='a-daemon-someone-forgot'):
        with pytest.raises(driver.ForeignWritersPresent, match='a-daemon-someone-forgot'):
            with fence.hold(None, deadline=None):
                pass


def test_the_fence_admits_the_migrations_own_connections(driver, database):
    """The migration opens many connections of its own; the fence must not
    mistake them for writers, or it could never pass."""
    connect = driver._connector(database)
    fence = driver.ProductionFence(connect)
    with connect(), connect():
        with fence.hold(None, deadline=None):
            pass


def test_the_fence_refuses_a_writer_that_appears_mid_migration(driver, database):
    """A daemon that restarts partway through has written against a half-migrated
    table. The exit check is what stops the receipt claiming a clean run."""
    connect = driver._connector(database)
    fence = driver.ProductionFence(connect)
    with pytest.raises(driver.ForeignWritersPresent, match='while the fence was held'):
        with fence.hold(None, deadline=None):
            intruder = psycopg.connect(database, application_name='restarted-daemon')
    intruder.close()


# ── the declared target ──────────────────────────────────────────────────────

def test_apply_refuses_without_a_declared_target(driver, database, monkeypatch, tmp_path):
    """The disposable fixture name would satisfy the default target, so this
    test names an unknown one: the guard has to be consulted, not bypassed,
    from inside the driver."""
    monkeypatch.setenv('GMS_MIGRATION_TARGET', 'somewhere-else')
    with pytest.raises(ValueError, match='not a known apply target'):
        driver.apply(database, registry_path=tmp_path / 'registry.sqlite',
                     store_id='test-store', migration_id='m1', release_epoch=1,
                     dominant_owner=DOMINANT, expected_owners=OWNERS,
                     checkpoint=lambda _message: None)


# ── the whole run ────────────────────────────────────────────────────────────

def test_the_driver_migrates_a_database_it_did_not_create(driver, database, tmp_path):
    """The end this whole file exists for: both phases, against an existing
    database, leaving a partitioned owner-keyed table with every row intact."""
    before = {}
    with psycopg.connect(database, autocommit=True) as conn:
        for owner in OWNERS:
            before[owner] = conn.execute(
                'SELECT count(*) FROM public.messages WHERE user_id=%s', (owner,)).fetchone()[0]

    receipt = driver.apply(database, registry_path=tmp_path / 'registry.sqlite',
                           store_id='test-store', migration_id='m1', release_epoch=1,
                           dominant_owner=DOMINANT, expected_owners=OWNERS,
                           checkpoint=lambda _message: None)

    assert receipt['phases']['phase_one']['state'] == 'INDEX_PENDING'
    assert receipt['phases']['phase_two']['state'] == 'READY'

    with psycopg.connect(database, autocommit=True) as conn:
        relkind = conn.execute(
            "SELECT relkind::text FROM pg_class WHERE oid='public.messages'::regclass").fetchone()[0]
        assert relkind == 'p', 'messages should be partitioned after the migration'
        keys = conn.execute(
            "SELECT string_agg(a.attname,',' ORDER BY k.ord) FROM pg_constraint c "
            "JOIN LATERAL unnest(c.conkey) WITH ORDINALITY k(attnum,ord) ON true "
            "JOIN pg_attribute a ON a.attrelid=c.conrelid AND a.attnum=k.attnum "
            "WHERE c.conrelid='public.messages'::regclass AND c.contype='p'").fetchone()[0]
        assert keys == 'user_id,id'
        for owner, count in before.items():
            after = conn.execute('SELECT count(*) FROM public.messages WHERE user_id=%s',
                                 (owner,)).fetchone()[0]
            assert after == count, f'{owner} lost rows: {count} -> {after}'


def test_the_registry_is_left_on_a_durable_path(driver, database, tmp_path):
    """The rehearsal writes its registry into a temp directory. A real run has
    to leave one behind, or there is no record that the migration happened."""
    registry = tmp_path / 'nested' / 'registry.sqlite'
    driver.apply(database, registry_path=registry, store_id='test-store',
                 migration_id='m1', release_epoch=1, dominant_owner=DOMINANT,
                 expected_owners=OWNERS, checkpoint=lambda _message: None)
    assert registry.is_file() and registry.stat().st_size > 0


def test_apply_accepts_owners_in_report_order(driver, database, tmp_path):
    """The report returns owners largest-first so the dominant one can be
    picked; the plan binding digests the tuple and requires it sorted. The
    driver has to reconcile those, and it did not — production refused with
    "Unsupported synthetic TEXT migration plan" at plan capture.

    Passing the largest-first order here is the regression check.
    """
    largest_first = (DOMINANT, MINORITY)
    assert largest_first != tuple(sorted(largest_first)), 'this test is pointless if they coincide'
    receipt = driver.apply(database, registry_path=tmp_path / 'registry.sqlite',
                           store_id='test-store', migration_id='m1', release_epoch=1,
                           dominant_owner=DOMINANT, expected_owners=largest_first,
                           checkpoint=lambda _message: None)
    assert receipt['phases']['phase_two']['state'] == 'READY'


def test_a_dominant_owner_outside_the_set_is_refused(driver, database, tmp_path):
    with pytest.raises(ValueError, match='not in the owner set'):
        driver.apply(database, registry_path=tmp_path / 'registry.sqlite',
                     store_id='test-store', migration_id='m1', release_epoch=1,
                     dominant_owner='someone-else', expected_owners=OWNERS,
                     checkpoint=lambda _message: None)


def test_the_fence_ignores_postgres_background_workers(driver, database):
    """Autovacuum is not a writer and cannot be stopped by stopping daemons.

    `pg_stat_activity` lists PostgreSQL's own background processes alongside
    client connections, with a null `usename` and no application name. Counting
    them makes the fence impossible to satisfy — and phase one's table rewrite
    provokes autovacuum, so a real migration blocks its own gate check moments
    after doing the work. That is exactly what happened on the first production
    attempt.

    A fresh test database has no autovacuum activity to observe, so this asserts
    the filter directly rather than trying to race one into existence.
    """
    import re

    source = (ROOT / 'deploy/public/migrate_production_mixed.py').read_text()
    query = source[source.index('def _foreign'):source.index('def _require_closed')]
    assert "backend_type = 'client backend'" in query, (
        'the fence must count only client backends, or an autovacuum worker '
        'will be mistaken for a writer that nobody can stop')
    # And the filter has to be in the SQL, not applied afterwards in Python,
    # because the count is what the refusal message reports.
    assert re.search(r"WHERE[^\"]*backend_type = 'client backend'", query, re.S)
