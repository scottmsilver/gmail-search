"""Backup and verified-restore rehearsal for the mixed-owner migration.

An unverified backup is not a backup. This builds a synthetic database at a
chosen scale, dumps it, restores it into a second database, and then *verifies*
the restored copy: per-owner row counts, a content digest over every migrated
table, index validity, and a live BM25 query. It reports the timings that set
the real rollback cost.

    GMS_TEST_PG_DSN=postgresql://... uv run --no-sync python \
        deploy/public/rehearse_backup_restore.py --scale 0.1 --container gms-parade-patched

`pg_dump`/`pg_restore` run inside the named container because the server is
PostgreSQL 16 and the host client may be older; a client older than the server
cannot dump it. The live database has the same constraint.

Synthetic disposable databases only — it creates and drops everything it uses.
"""
import argparse
import json
import os
import secrets
import subprocess
import time
from pathlib import Path

from psycopg import sql

from rehearse_mixed_migration import (  # noqa: E402  (same directory)
    DB_PREFIX, connect, create_database, database_bytes, gib, populate,
)

TABLES = ('messages', 'attachments', 'propositions')
# Per-run path. A fixed one lets a zombie run's cleanup delete a live run's
# dump — which happened, and cost a full repopulation to discover.
DUMP_TEMPLATE = '/tmp/gms-rehearsal-backup-{suffix}.dump'


def run_in_container(container, *args, capture=False):
    result = subprocess.run(('docker', 'exec', container, *args),
                            capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f'{args[0]} failed: {result.stderr.strip()[:400]}')
    return result.stdout if capture else None


def dump(container, dbname, dump_path):
    started = time.monotonic()
    run_in_container(container, 'pg_dump', '-U', 'postgres', '-d', dbname,
                     '-Fc', '--no-owner', '-f', dump_path)
    size = run_in_container(container, 'stat', '-c', '%s', dump_path, capture=True).strip()
    return round(time.monotonic() - started, 2), int(size)


def restore(container, target, dump_path):
    started = time.monotonic()
    # template0, not the default template1: the ParadeDB image preinstalls the
    # `paradedb` schema into template1, and the dump recreates it, so restoring
    # onto template1 collides. A real restore lands on an empty cluster anyway.
    run_in_container(container, 'createdb', '-U', 'postgres', '-T', 'template0', target)
    run_in_container(container, 'pg_restore', '-U', 'postgres', '-d', target,
                     '--no-owner', '-j', '4', dump_path)
    return round(time.monotonic() - started, 2)


def owner_counts(conn):
    return {table: [list(row) for row in conn.execute(sql.SQL(
        'SELECT user_id,count(*) FROM {} GROUP BY user_id ORDER BY user_id'
    ).format(sql.Identifier('public', table))).fetchall()] for table in TABLES}


def content_digest(conn):
    """One digest per table over the whole owner-visible content, order-stable."""
    digests = {}
    for table, columns in (('messages', 'id,user_id,subject,body_text,body_html'),
                           ('attachments', 'id,user_id,message_id,filename,extracted_text'),
                           ('propositions', 'id,user_id,message_id,text')):
        # Order by the primary key, never by the row text: sorting 421k rows on a
        # 40KB body spills tens of GiB to temp files and dominates the whole run.
        digests[table] = conn.execute(sql.SQL(
            'SELECT md5(string_agg(md5(t::text), {sep} ORDER BY t.id)) FROM '
            '(SELECT {cols} FROM {tbl}) t'
        ).format(sep=sql.Literal(''), cols=sql.SQL(columns),
                 tbl=sql.Identifier('public', table))).fetchone()[0]
    return digests


def invalid_indexes(conn):
    return conn.execute("""SELECT c.relname FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid
        WHERE NOT (i.indisvalid AND i.indisready) ORDER BY c.relname""").fetchall()


def bm25_probe(conn):
    """A restored BM25 index must actually answer, not merely exist."""
    return conn.execute("""SELECT count(*) FROM (
        SELECT id FROM messages WHERE id @@@ paradedb.parse_with_field('subject','needle',false,false)
        LIMIT 500) x""").fetchone()[0]


def verify(source_conn, restored_conn):
    source = dict(counts=owner_counts(source_conn), digest=content_digest(source_conn))
    restored = dict(counts=owner_counts(restored_conn), digest=content_digest(restored_conn))
    problems = []
    if source['counts'] != restored['counts']:
        problems.append('per-owner row counts differ')
    if source['digest'] != restored['digest']:
        problems.append('table content digests differ')
    broken = invalid_indexes(restored_conn)
    if broken:
        problems.append(f'invalid indexes after restore: {broken}')
    source_hits, restored_hits = bm25_probe(source_conn), bm25_probe(restored_conn)
    if source_hits != restored_hits:
        problems.append(f'BM25 probe differs: {source_hits} vs {restored_hits}')
    return dict(source=source, restored=restored, bm25_hits=restored_hits, problems=problems)


def run(dsn, scale, container, out):
    suffix = secrets.token_hex(8)
    name, target = DB_PREFIX + suffix, DB_PREFIX + 'restored_' + suffix
    dump_path = DUMP_TEMPLATE.format(suffix=suffix)
    report = {'scale': scale, 'source': name, 'restored': target, 'dump_path': dump_path}
    create_database(dsn, name, suffix)
    try:
        with connect(dsn, name) as conn:
            populate(conn, scale, report)
            report['source_size_gib'] = gib(database_bytes(conn))
            report['dump_seconds'], dump_bytes = dump(container, name, dump_path)
            report['dump_gib'] = gib(dump_bytes)
            report['restore_seconds'] = restore(container, target, dump_path)
            with connect(dsn, target) as restored_conn:
                report['restored_size_gib'] = gib(database_bytes(restored_conn))
                report['verification'] = verify(conn, restored_conn)
    finally:
        with connect(dsn, 'postgres') as admin:
            for db in (name, target):
                admin.execute(sql.SQL('DROP DATABASE IF EXISTS {} WITH (FORCE)').format(sql.Identifier(db)))
        subprocess.run(('docker', 'exec', container, 'rm', '-f', dump_path),
                       capture_output=True, check=False)
    Path(out).write_text(json.dumps(report, indent=2, default=str))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scale', type=float, default=0.1)
    parser.add_argument('--container', default='gms-parade-patched')
    parser.add_argument('--out', default='/tmp/backup-restore-rehearsal.json')
    args = parser.parse_args()
    dsn = os.environ.get('GMS_TEST_PG_DSN')
    if not dsn:
        raise SystemExit('Set GMS_TEST_PG_DSN to a disposable PostgreSQL instance')
    report = run(dsn, args.scale, args.container, args.out)
    problems = report['verification']['problems']
    print(json.dumps({k: v for k, v in report.items() if k != 'verification'}, indent=2, default=str))
    print('VERIFICATION:', 'PASSED' if not problems else 'FAILED ' + '; '.join(problems))
    raise SystemExit(1 if problems else 0)


if __name__ == '__main__':
    main()
