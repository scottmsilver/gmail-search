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
OWNERS = ('owner_dominant', 'owner_minority')


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
                    for owner, count in ((OWNERS[0], 40), (OWNERS[1], 5))
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
                     dominant_owner=OWNERS[0], expected_owners=OWNERS,
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
                           dominant_owner=OWNERS[0], expected_owners=OWNERS,
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
                 migration_id='m1', release_epoch=1, dominant_owner=OWNERS[0],
                 expected_owners=OWNERS, checkpoint=lambda _message: None)
    assert registry.is_file() and registry.stat().st_size > 0
