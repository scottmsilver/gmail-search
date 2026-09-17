"""Run the mixed-owner TEXT partition migration against a real database.

`rehearse_mixed_migration.py` proves the mechanism, but it begins with
`CREATE DATABASE` and seeds synthetic rows: it can only ever migrate a database
it made itself. This is the driver for one that already exists.

Two modes, and the gap between them is deliberate:

* `--report` opens a read-only transaction, runs the same preflight the apply
  path runs, and prints what a migration would do. It changes nothing, needs no
  declared target, and is the thing to run first and to keep re-running until it
  is boring.
* `--apply` performs the migration. It requires every condition in
  `migrate_text_owner_partitions._require_apply_target` — `GMS_MIGRATION_TARGET`,
  a marker nonce written into the target, a verified same-cluster backup, a
  local connection — plus an external fence proving the writers are already
  stopped. It does not stop them for you: a tool that can both silence the
  writers and rewrite the tables is a tool that can do half of that.

Phase one commits the layout; phase two VACUUMs the retained leaves, rebuilds
their BM25 indexes and qualifies every owner's index. Phase two needs the
patched pg_search build — an unpatched engine asserts on a stale ctid once a
retained leaf is reindexed and searched
(docs/qualification/retained-reader-root-cause.md).
"""
import argparse
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import sys
import time

import psycopg
from psycopg.rows import tuple_row


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MECHANISM = _load('mixed_text_production', 'migrate_mixed_text_owner_partitions.py')
LEGACY = _load('text_partition_production', 'migrate_text_owner_partitions.py')


# ── read-only report ─────────────────────────────────────────────────────────

ENGINE = """SELECT current_database(), current_user,
    (SELECT setting FROM pg_settings WHERE name='server_version'),
    (SELECT extversion FROM pg_extension WHERE extname='pg_search'),
    (SELECT rolsuper FROM pg_roles WHERE rolname=current_user),
    (SELECT d.datdba=r.oid FROM pg_database d, pg_roles r
      WHERE d.datname=current_database() AND r.rolname=current_user)"""

SHAPE = """SELECT (SELECT relkind::text FROM pg_class WHERE oid='public.messages'::regclass),
    coalesce((SELECT string_agg(a.attname,',' ORDER BY k.ord)
      FROM pg_constraint c
      JOIN LATERAL unnest(c.conkey) WITH ORDINALITY k(attnum,ord) ON true
      JOIN pg_attribute a ON a.attrelid=c.conrelid AND a.attnum=k.attnum
      WHERE c.conrelid='public.messages'::regclass AND c.contype='p'),'NONE'),
    (SELECT c.reloptions::text FROM pg_class c
      JOIN pg_index i ON i.indexrelid=c.oid JOIN pg_am am ON am.oid=c.relam
      WHERE i.indrelid='public.messages'::regclass AND am.amname='bm25' LIMIT 1),
    pg_size_pretty(pg_total_relation_size('public.messages')),
    pg_size_pretty(pg_database_size(current_database()))"""


def _owners(conn):
    """Every owner with mail, largest first. The largest keeps its heap."""
    return conn.execute(
        'SELECT user_id, count(*) FROM public.messages GROUP BY 1 ORDER BY 2 DESC, 1'
    ).fetchall()


def _orphans(conn):
    """Rows whose owner has no `users` record. These block the migration: the
    partition they belong in cannot be named."""
    return conn.execute("""SELECT count(*) FROM public.messages m
        WHERE m.user_id IS NULL OR NOT EXISTS (SELECT 1 FROM public.users u WHERE u.id=m.user_id)
        """).fetchone()[0]


def report(dsn):
    with psycopg.connect(dsn, autocommit=True, row_factory=tuple_row) as conn:
        conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        database, user, version, extension, superuser, owns = conn.execute(ENGINE).fetchone()
        relkind, primary_key, reloptions, table_size, database_size = conn.execute(SHAPE).fetchone()
        owners, orphans = _owners(conn), _orphans(conn)
        # `apply=False` performs the structural checks without needing a target.
        try:
            LEGACY._require(conn, apply=False)
            requirement = 'ok'
        except ValueError as error:
            requirement = str(error)

    dominant = owners[0][0] if owners else None
    blocking = []
    if not version.startswith('16.'):
        blocking.append(f'PostgreSQL 16 required, found {version}')
    if extension != '0.23.0':
        blocking.append(f'pg_search 0.23.0 required, found {extension}')
    if not superuser or not owns:
        blocking.append('a database-owning superuser is required')
    if relkind != 'r':
        blocking.append(f'messages is already relkind={relkind!r}, not a plain table')
    if orphans:
        blocking.append(f'{orphans} message rows have no matching users row')
    if not 2 <= len(owners) <= 16:
        blocking.append(f'the migration takes 2-16 owners, found {len(owners)}')

    return {
        'database': database, 'user': user, 'engine': f'PostgreSQL {version} / pg_search {extension}',
        'superuser': superuser, 'owns_database': owns,
        'messages': {'relkind': relkind, 'primary_key': primary_key,
                     'bm25_reloptions': reloptions, 'table_size': table_size},
        'database_size': database_size,
        'owners': [{'owner_id': owner, 'messages': count} for owner, count in owners],
        'dominant_owner': dominant,
        'redistributed_rows': sum(count for _, count in owners[1:]),
        'orphan_rows': orphans,
        'connection_requirement': requirement,
        'blocking': blocking,
        'ready': not blocking,
    }


def _print_report(data):
    print(f"database        : {data['database']} ({data['database_size']})")
    print(f"engine          : {data['engine']}")
    print(f"connected as    : {data['user']}  superuser={data['superuser']} owns_db={data['owns_database']}")
    message = data['messages']
    print(f"messages        : relkind={message['relkind']} pk=({message['primary_key']}) "
          f"size={message['table_size']}")
    print(f"bm25            : {message['bm25_reloptions']}")
    print(f"owners          : {len(data['owners'])}")
    for owner in data['owners']:
        marker = '  <- dominant, keeps its heap' if owner['owner_id'] == data['dominant_owner'] else ''
        print(f"  {owner['owner_id']:<24} {owner['messages']:>9,}{marker}")
    print(f"rows to move    : {data['redistributed_rows']:,}")
    print(f"orphan rows     : {data['orphan_rows']}")
    print(f"connection      : {data['connection_requirement']}")
    if data['ready']:
        print('\nREADY — the database satisfies the structural preconditions.')
        print('This says nothing about whether the application can run against the')
        print('migrated shape, which is a separate gate.')
    else:
        print('\nNOT READY:')
        for reason in data['blocking']:
            print(f'  - {reason}')


# ── the fence ────────────────────────────────────────────────────────────────

APPLICATION_NAME = 'gms-migration'


class ForeignWritersPresent(RuntimeError):
    """Something other than the migration is connected to the database."""


class ProductionFence:
    """Proves the writers are already stopped. Never stops them itself.

    The mechanism asks a fence to confirm access is closed on entry and still
    closed on exit. `RehearsalFence` asserts a boolean it set itself, which is
    right for a synthetic database and worthless for a real one.

    What is actually observable is who is connected. Every connection this
    driver opens carries `application_name=gms-migration`; if any *other*
    backend is attached to the database, a writer survived and the migration
    must not proceed. That is evidence rather than trust: it does not matter
    whether the daemons were stopped by systemd, a supervisor, or by hand, and
    it catches the case the operator forgot — a stray psql, a cron job, an
    agent run holding a connection.

    Deliberately not a stopper. A tool that can both silence the writers and
    rewrite the tables can do the second without the first.
    """

    def __init__(self, connect):
        self._connect = connect

    def _foreign(self):
        with self._connect() as conn:
            return conn.execute(
                """SELECT pid, application_name, state, usename FROM pg_stat_activity
                   WHERE datname = current_database() AND pid <> pg_backend_pid()
                     AND coalesce(application_name,'') <> %s""", (APPLICATION_NAME,)).fetchall()

    def _require_closed(self, when):
        foreign = self._foreign()
        if foreign:
            detail = ', '.join(f'pid {row[0]} ({row[1] or "unnamed"}, {row[3]})' for row in foreign[:5])
            raise ForeignWritersPresent(
                f'{len(foreign)} connection(s) other than the migration are attached '
                f'{when}: {detail}. Stop every writer before applying.')

    @contextmanager
    def hold(self, plan, *, deadline):
        self._require_closed('before the fence was taken')
        yield
        # On the way out too: a daemon that restarted mid-migration would have
        # written against a half-migrated table, and the receipt must not claim
        # a clean run.
        self._require_closed('while the fence was held')


# ── the apply path ───────────────────────────────────────────────────────────

def _connector(dsn):
    """Fresh autocommit connections, each identifying itself to the fence."""
    def connect():
        conn = psycopg.connect(dsn, autocommit=True,
                               application_name=APPLICATION_NAME, row_factory=tuple_row)
        return conn
    return connect


def apply(dsn, *, registry_path, store_id, migration_id, release_epoch, dominant_owner,
          expected_owners, checkpoint=print):
    """Phase one then phase two, against a database that already exists.

    Every guard the rehearsal has, plus the two it cannot have: a fence that
    checks who is connected rather than a boolean it set itself, and a registry
    on a durable path rather than a temp directory. The declared-target checks
    in `migrate_text_owner_partitions` run inside `capture_plan` — this function
    cannot reach production without them.
    """
    from gmail_search.gateway.maintenance import MaintenanceAdmin, ReleaseIdentity
    from gmail_search.gateway.partition_profiles import TEXT_OWNER_PARTITIONS_V1 as TEXT

    connect = _connector(dsn)
    registry = Path(registry_path)
    registry.parent.mkdir(parents=True, mode=0o700, exist_ok=True)

    identity = ReleaseIdentity(store_id, TEXT, release_epoch)
    with connect() as conn:
        plan = MECHANISM.capture_plan(conn, identity=identity, migration_id=migration_id,
                                      dominant_owner=dominant_owner,
                                      expected_owners=tuple(expected_owners))
    checkpoint(f'plan captured: database={plan.database_name} oid={plan.database_oid} '
               f'system_identifier={plan.system_identifier}')

    admin = MaintenanceAdmin(registry)
    snapshot = admin.initialize_closed(identity, migration_id=plan.migration_id,
                                       owner_set_digest=plan.owner_set_digest,
                                       procedure_digest=plan.procedure_digest)
    checkpoint(f'registry initialised at {registry} (state={snapshot.state})')

    controller = MECHANISM.MixedTextPhaseOne(connect, registry, plan,
                                             fence=ProductionFence(connect))

    receipt = {'migration_id': migration_id, 'database': plan.database_name,
               'system_identifier': plan.system_identifier, 'phases': {}}
    for label, call, expected in (('phase_one', controller.advance, 'INDEX_PENDING'),
                                  ('phase_two', controller.publish, 'READY')):
        # The gate caps verification at 30s, so the slow work is done outside it.
        started = time.monotonic()
        prepared = controller.prepare(snapshot, seconds=86400)
        snapshot = call(snapshot)
        elapsed = round(time.monotonic() - started, 2)
        if snapshot.state != expected:
            raise RuntimeError(f'{label} ended in {snapshot.state}, expected {expected}')
        receipt['phases'][label] = {'seconds': elapsed, 'prepare_seconds': prepared,
                                    'state': snapshot.state}
        checkpoint(f'{label}: {elapsed}s -> {snapshot.state}')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dsn', required=True, help='the database to inspect or migrate')
    parser.add_argument('--report', action='store_true', help='read-only; change nothing')
    parser.add_argument('--json', action='store_true', help='machine-readable report')
    parser.add_argument('--apply', action='store_true',
                        help='perform the migration; requires every condition below')
    parser.add_argument('--registry', help='durable path for the maintenance registry')
    parser.add_argument('--store-id', help='release store id recorded in the registry')
    parser.add_argument('--migration-id', help='identifies this migration in the registry')
    parser.add_argument('--release-epoch', type=int, default=1)
    parser.add_argument('--receipt', help='write a JSON receipt here')
    arguments = parser.parse_args()

    if not (arguments.report or arguments.apply):
        parser.error('choose --report or --apply')

    data = report(arguments.dsn)
    if arguments.report:
        if arguments.json:
            print(json.dumps(data, indent=2, sort_keys=True))
        else:
            _print_report(data)
        return 0 if data['ready'] else 1

    if not data['ready']:
        _print_report(data)
        print('\nRefusing to apply: the report is not clean.', file=sys.stderr)
        return 1
    missing = [name for name in ('registry', 'store_id', 'migration_id')
               if not getattr(arguments, name)]
    if missing:
        parser.error('--apply needs ' + ', '.join('--' + name.replace('_', '-') for name in missing))

    owners = [owner['owner_id'] for owner in data['owners']]
    print(f"about to migrate {data['database']} ({data['database_size']}): "
          f"{len(owners)} owners, {data['redistributed_rows']:,} rows move, "
          f"dominant {data['dominant_owner']} keeps its heap")
    receipt = apply(arguments.dsn, registry_path=arguments.registry,
                    store_id=arguments.store_id, migration_id=arguments.migration_id,
                    release_epoch=arguments.release_epoch,
                    dominant_owner=data['dominant_owner'], expected_owners=owners)
    if arguments.receipt:
        path = Path(arguments.receipt)
        path.write_text(json.dumps(receipt, indent=2, sort_keys=True))
        path.chmod(0o600)
        print(f'receipt written to {path}')
    else:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
