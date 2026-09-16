"""Scaled mixed-owner migration rehearsal: measure disk, WAL and lock duration.

Synthetic data in a disposable database only. Never point this at production:
it refuses any database whose name is not the disposable rehearsal prefix, and
it creates and drops that database itself.

    GMS_TEST_PG_DSN=postgresql://... uv run --no-sync python \
        deploy/public/rehearse_mixed_migration.py --scale 0.1

`--scale` is a fraction of the live row counts recorded in
docs/live-partition-preflight-2026-09-15.md. Run a small scale first, then a
larger one, and extrapolate rather than assuming linearity.
"""
import argparse
import importlib.util
import json
import os
import secrets
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo

from gmail_search.gateway.maintenance import MaintenanceAdmin, ReleaseIdentity
from gmail_search.gateway.partition_profiles import TEXT_OWNER_PARTITIONS_V1 as TEXT

ROOT = Path(__file__).resolve().parents[2]
LEGACY_SCHEMA = ROOT / 'tests/fixtures/legacy_text_owner_schema.sql'
DB_PREFIX = 'gms_owner_partitions_test_text_'

# Live row counts, docs/live-partition-preflight-2026-09-15.md.
LIVE_ROWS = {
    'messages': {'alice': 421228, 'bob': 21451, 'charlie': 118},
    'attachments': {'alice': 612268, 'bob': 63235, 'charlie': 5},
    'propositions': {'alice': 1083516, 'bob': 0, 'charlie': 0},
}
# Average TOAST bytes per row, derived from the same preflight.
LIVE_TEXT_BYTES = {'messages': 40000, 'attachments': 600, 'propositions': 14000}
OWNERS = ('alice', 'bob', 'charlie')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scaled(counts, scale):
    return {owner: max(0, round(n * scale)) for owner, n in counts.items()}


class RehearsalFence:
    """Stands in for the durable external fence; access is closed regardless."""

    def __init__(self):
        self.closed = True
        self.entries = 0

    @contextmanager
    def hold(self, plan, *, deadline):
        assert self.closed
        self.entries += 1
        yield
        assert self.closed


def connect(dsn, dbname):
    return psycopg.connect(make_conninfo(dsn, dbname=dbname), autocommit=True)


def create_database(dsn, name, role_suffix):
    with connect(dsn, 'postgres') as admin:
        admin.execute(sql.SQL('CREATE DATABASE {} TEMPLATE template0').format(sql.Identifier(name)))
    with connect(dsn, name) as conn:
        text = LEGACY_SCHEMA.read_text()
        text = text.replace('gmail_search_reader', 'gms_rehearsal_reader_' + role_suffix)
        text = text.replace('gmail_analyst', 'gms_rehearsal_analyst_' + role_suffix)
        text = text.replace('ON DATABASE gmail_search', 'ON DATABASE ' + name)
        conn.execute(text)
        conn.execute("INSERT INTO users(id,email) VALUES('alice','a@example.test'),"
                     "('bob','b@example.test'),('charlie','c@example.test')")


# One shared block of random hex per table width. Real mail barely compresses,
# so `repeat('x', n)` would understate TOAST by more than an order of magnitude;
# concatenated md5s do not compress, which is what makes the disk figures mean
# anything. The block is generated once per INSERT, not per row.
_FILLER = "(SELECT string_agg(md5(random()::text),'') FROM generate_series(1,{blocks}))"


def filler_cte(width):
    return sql.SQL(_FILLER.format(blocks=max(1, round(width / 32))))


def populate(conn, scale, report):
    """Generate owner rows server-side, with representative TOAST volume."""
    for table in ('messages', 'attachments', 'propositions'):
        per_owner = scaled(LIVE_ROWS[table], scale)
        filler = filler_cte(LIVE_TEXT_BYTES[table])
        started = time.monotonic()
        for owner, count in per_owner.items():
            if not count:
                continue
            messages_for_owner = sql.Literal(max(1, scaled(LIVE_ROWS['messages'], scale)[owner]))
            if table == 'messages':
                conn.execute(sql.SQL("""WITH blk AS ({f})
                    INSERT INTO messages
                    (id,user_id,thread_id,from_addr,to_addr,subject,body_text,body_html,date,raw_json)
                    SELECT {o}||'-'||g, {o}, 't'||(g%1000), 's@x', 'r@x',
                      'needle subject '||g, 'needle body text '||g,
                      (SELECT string_agg FROM blk), '2026-01-01', '{{}}'
                    FROM generate_series(1,{n}) g""").format(
                    f=filler, o=sql.Literal(owner), n=sql.Literal(count)))
            elif table == 'attachments':
                conn.execute(sql.SQL("""WITH blk AS ({f})
                    INSERT INTO attachments
                    (user_id,message_id,filename,mime_type,extracted_text)
                    SELECT {o}, {o}||'-'||((g%{m})+1), 'f'||g||'.pdf', 'application/pdf',
                      'needle attachment '||g||' '||(SELECT string_agg FROM blk)
                    FROM generate_series(1,{n}) g""").format(
                    f=filler, o=sql.Literal(owner), n=sql.Literal(count), m=messages_for_owner))
            else:
                conn.execute(sql.SQL("""WITH blk AS ({f})
                    INSERT INTO propositions (user_id,message_id,text,model)
                    SELECT {o}, {o}||'-'||((g%{m})+1),
                      'needle proposition '||g||' '||(SELECT string_agg FROM blk), 'rehearsal'
                    FROM generate_series(1,{n}) g""").format(
                    f=filler, o=sql.Literal(owner), n=sql.Literal(count), m=messages_for_owner))
        report[f'populate/{table}'] = dict(rows=per_owner, seconds=round(time.monotonic() - started, 2))
    conn.execute('ANALYZE')


def wal_lsn(conn):
    return conn.execute('SELECT pg_current_wal_lsn()').fetchone()[0]


def wal_bytes(conn, start):
    return conn.execute('SELECT pg_wal_lsn_diff(pg_current_wal_lsn(),%s)', (start,)).fetchone()[0]


def database_bytes(conn):
    return conn.execute('SELECT pg_database_size(current_database())').fetchone()[0]


def free_bytes(path):
    return shutil.disk_usage(path).free


def gib(value):
    return round(int(value) / 1024 ** 3, 3)


def run(dsn, scale, out):
    suffix = secrets.token_hex(8)
    name = DB_PREFIX + suffix
    report = {'scale': scale, 'database': name}
    create_database(dsn, name, suffix)
    try:
        with connect(dsn, name) as conn:
            populate(conn, scale, report)
            report['size_after_populate_gib'] = gib(database_bytes(conn))

            implementation = load_module('mixed_text_rehearsal',
                                         Path(__file__).with_name('migrate_mixed_text_owner_partitions.py'))
            identity = ReleaseIdentity('rehearsal-' + suffix, TEXT, 1)
            plan = implementation.capture_plan(conn, identity=identity, migration_id='rehearsal-1',
                                               dominant_owner='alice', expected_owners=OWNERS)
            registry_dir = Path(out).parent / ('rehearsal-registry-' + suffix)
            registry_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
            registry = registry_dir / 'registry.sqlite'
            admin = MaintenanceAdmin(registry)
            snapshot = admin.initialize_closed(identity, migration_id=plan.migration_id,
                                               owner_set_digest=plan.owner_set_digest,
                                               procedure_digest=plan.procedure_digest)

            marks = {}
            controller = implementation.MixedTextPhaseOne(
                lambda: connect(dsn, name), registry, plan, fence=RehearsalFence(),
                _checkpoint=lambda stage: marks.__setitem__(stage, round(time.monotonic() - phase_start, 2)))

            for label, call, expected in (('phase_one', lambda s: controller.advance(s), 'INDEX_PENDING'),
                                          ('phase_two', lambda s: controller.publish(s), 'READY')):
                marks.clear()
                before_size, before_free = database_bytes(conn), free_bytes(ROOT)
                start_lsn = wal_lsn(conn)
                phase_start = time.monotonic()
                # The gate caps verification at 30s, so the slow work runs first.
                prepare_seconds = controller.prepare(snapshot, seconds=86400)
                snapshot = call(snapshot)
                elapsed = time.monotonic() - phase_start
                assert snapshot.state == expected, f'{label} ended in {snapshot.state}'
                report[label] = dict(
                    seconds=round(elapsed, 2), prepare_seconds=prepare_seconds,
                    checkpoints=dict(marks),
                    wal_gib=gib(wal_bytes(conn, start_lsn)),
                    size_before_gib=gib(before_size), size_after_gib=gib(database_bytes(conn)),
                    free_before_gib=gib(before_free), free_after_gib=gib(free_bytes(ROOT)))

            report['final_counts'] = {
                table: conn.execute(sql.SQL('SELECT user_id,count(*) FROM {} GROUP BY user_id ORDER BY user_id')
                                    .format(sql.Identifier('public', table))).fetchall()
                for table in ('messages', 'attachments', 'propositions')}
    finally:
        with connect(dsn, 'postgres') as admin_conn:
            admin_conn.execute(sql.SQL('DROP DATABASE IF EXISTS {} WITH (FORCE)').format(sql.Identifier(name)))
    Path(out).write_text(json.dumps(report, indent=2, default=str))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scale', type=float, default=0.02, help='fraction of live row counts')
    parser.add_argument('--out', default='/tmp/mixed-migration-rehearsal.json')
    args = parser.parse_args()
    dsn = os.environ.get('GMS_TEST_PG_DSN')
    if not dsn:
        raise SystemExit('Set GMS_TEST_PG_DSN to a disposable PostgreSQL instance')
    report = run(dsn, args.scale, args.out)
    print(json.dumps(report, indent=2, default=str))


if __name__ == '__main__':
    main()
