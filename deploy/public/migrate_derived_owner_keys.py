#!/usr/bin/env python3
"""Explicit derived-fact owner prerequisite; apply after migrate_owner_keys.

No backfill/deletion, implicit schema repair, production CLI, or RLS/grant changes.
Facts and processed markers are derived data: deleting their owner-qualified
message must cascade to them. Existing non-derived message FKs are untouched.
"""
import argparse
import os

import psycopg
from psycopg import sql
from psycopg.pq import TransactionStatus
from psycopg.rows import tuple_row

TABLES = ('propositions', 'prop_processed')


def _preflight(cur):
    relations = cur.execute("""SELECT c.relname,c.relkind FROM pg_class c
        JOIN pg_namespace n ON n.oid=c.relnamespace
        WHERE n.nspname='public' AND c.relname=ANY(%s)""", (list(TABLES) + ['messages'],)).fetchall()
    if dict(relations) != {name: 'r' for name in (*TABLES, 'messages')}:
        raise ValueError('Expected ordinary public message/derived tables')
    if cur.execute("""SELECT 1 FROM pg_inherits i JOIN pg_class c
        ON c.oid=i.inhrelid OR c.oid=i.inhparent JOIN pg_namespace n ON n.oid=c.relnamespace
        WHERE n.nspname='public' AND c.relname=ANY(%s)""", (list(TABLES) + ['messages'],)).fetchone():
        raise ValueError('Inherited derived schema is unsupported')
    nullable_owner_keys = False
    for table in (*TABLES, 'messages'):
        columns = cur.execute("""SELECT a.attname,t.typname,n.nspname,a.attnotnull FROM pg_attribute a
            JOIN pg_type t ON t.oid=a.atttypid JOIN pg_namespace n ON n.oid=t.typnamespace
            WHERE a.attrelid=%s::regclass AND a.attnum>0 AND NOT a.attisdropped""", ('public.' + table,)).fetchall()
        actual = {name: (kind, namespace) for name, kind, namespace, _ in columns}
        if table in TABLES and any(name in ('user_id','message_id') and not nonnull for name, _, _, nonnull in columns):
            nullable_owner_keys = True
        required = {'user_id': 'text', 'id' if table == 'messages' else 'message_id': 'text'}
        if table == 'propositions':
            required['id'] = 'int8'
        if any(actual.get(name) != (kind, 'pg_catalog') for name, kind in required.items()):
            raise ValueError('Unsupported derived owner/key column type')
    rows = cur.execute("""SELECT c.relname,k.contype,
        ARRAY(SELECT a.attname FROM unnest(k.conkey) WITH ORDINALITY x(num,ord)
            JOIN pg_attribute a ON a.attrelid=k.conrelid AND a.attnum=x.num ORDER BY x.ord),
        rn.nspname,rc.relname,
        ARRAY(SELECT a.attname FROM unnest(k.confkey) WITH ORDINALITY x(num,ord)
            JOIN pg_attribute a ON a.attrelid=k.confrelid AND a.attnum=x.num ORDER BY x.ord),
        k.confdeltype,k.confupdtype,k.convalidated,k.condeferrable,k.confmatchtype
        FROM pg_constraint k JOIN pg_class c ON c.oid=k.conrelid
        JOIN pg_namespace n ON n.oid=c.relnamespace
        LEFT JOIN pg_class rc ON rc.oid=k.confrelid LEFT JOIN pg_namespace rn ON rn.oid=rc.relnamespace
        WHERE n.nspname='public' AND c.relname=ANY(%s)""", (list(TABLES) + ['messages'],)).fetchall()
    keys = {table: set() for table in TABLES}
    target = set()
    message_key = False
    for table, kind, cols, remote_schema, remote_table, remote_cols, delete, update, validated, deferred, match in rows:
        if table == 'messages':
            if kind == 'p' and cols == ['user_id', 'id'] and validated and not deferred:
                message_key = True
            continue
        if kind in ('p', 'u'):
            if not validated or deferred:
                raise ValueError('Unsupported derived key validation/deferral')
            keys[table].add((kind, tuple(cols)))
        elif kind == 'f':
            if not validated or deferred or update != 'a' or match != 's' or remote_schema != 'public':
                raise ValueError('Unsupported derived foreign key')
            if cols == ['user_id'] and remote_table == 'users' and remote_cols == ['id']:
                continue  # Preserve any preexisting owner reference verbatim.
            if (cols == ['user_id', 'message_id'] and remote_table == 'messages'
                    and remote_cols == ['user_id', 'id'] and delete == 'c' and table not in target):
                target.add(table)
            else:
                raise ValueError('Unexpected derived foreign key')
        elif kind not in ('c', 'n'):
            raise ValueError('Unsupported derived constraint')
    if not message_key:
        raise ValueError('Apply the message owner-key migration first')
    if keys != {'propositions': {('p', ('id',))}, 'prop_processed': {('p', ('user_id', 'message_id'))}}:
        raise ValueError('Unexpected derived key shape')
    if target and nullable_owner_keys:
        raise ValueError('Partially migrated derived owner nullability')
    if target and target != set(TABLES):
        raise ValueError('Partially migrated derived foreign keys')
    if cur.execute("""SELECT 1 FROM pg_constraint k JOIN pg_class c ON c.oid=k.confrelid
        JOIN pg_namespace n ON n.oid=c.relnamespace
        WHERE k.contype='f' AND n.nspname='public' AND c.relname=ANY(%s)""", (list(TABLES),)).fetchone():
        raise ValueError('Unexpected inbound dependency on derived tables')
    if cur.execute('''SELECT 1 FROM pg_index i JOIN pg_class c ON c.oid=i.indrelid
        JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname='public'
        AND c.relname=ANY(%s) AND i.indisunique
        AND NOT EXISTS(SELECT 1 FROM pg_constraint k WHERE k.conindid=i.indexrelid)''', (list(TABLES),)).fetchone():
        raise ValueError('Unexpected derived unique index')
    for table in TABLES:
        qualified = sql.Identifier('public', table)
        if cur.execute(sql.SQL('SELECT 1 FROM {} WHERE user_id IS NULL OR message_id IS NULL LIMIT 1').format(qualified)).fetchone():
            raise ValueError('Derived data has null owner/message')
        if cur.execute(sql.SQL('''SELECT 1 FROM {} child LEFT JOIN public.messages parent
            ON parent.user_id=child.user_id AND parent.id=child.message_id
            WHERE parent.id IS NULL LIMIT 1''').format(qualified)).fetchone():
            raise ValueError('Derived data has orphan or mismatched owner/message')
    return bool(target)


def migrate_derived_owner_keys(conn):
    if conn.info.transaction_status != TransactionStatus.IDLE:
        raise ValueError('Migration requires an idle connection')
    with conn.transaction(), conn.cursor(row_factory=tuple_row) as cur:
        cur.execute('SET LOCAL search_path=pg_catalog')
        cur.execute("SET LOCAL lock_timeout='2s'")
        cur.execute("SET LOCAL statement_timeout='60s'")
        if cur.execute('SELECT session_user=current_user AND rolsuper FROM pg_roles WHERE rolname=current_user').fetchone() != (True,):
            raise ValueError('Migration requires a direct administrator with full RLS visibility')
        if not cur.execute('SELECT pg_try_advisory_xact_lock(72341629,2)').fetchone()[0]:
            raise ValueError('Derived migration already running')
        for table in sorted((*TABLES, 'messages')):
            cur.execute(sql.SQL('LOCK TABLE {} IN ACCESS EXCLUSIVE MODE').format(sql.Identifier('public', table)))
        if _preflight(cur):
            return False
        for table in TABLES:
            cur.execute(sql.SQL('''ALTER TABLE {} ALTER COLUMN user_id SET NOT NULL,
                ALTER COLUMN message_id SET NOT NULL,
                ADD FOREIGN KEY(user_id,message_id) REFERENCES public.messages(user_id,id) ON DELETE CASCADE''').format(sql.Identifier('public', table)))
        if not _preflight(cur):
            raise ValueError('Derived target schema verification failed')
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if not args.apply:
        parser.error('--apply is required')
    dsn = os.environ.get('DERIVED_OWNER_KEY_MIGRATION_DSN')
    if not dsn:
        parser.error('DERIVED_OWNER_KEY_MIGRATION_DSN is required')
    try:
        with psycopg.connect(dsn, autocommit=True) as conn:
            if not conn.info.dbname.startswith('gms_owner_keys_test_'):
                raise ValueError('CLI accepts disposable qualification databases only')
            changed = migrate_derived_owner_keys(conn)
    except Exception:
        parser.exit(1, 'Derived owner migration refused or failed; transaction rolled back.\n')
    print('Derived owner keys applied.' if changed else 'Derived owner keys already present.')


if __name__ == '__main__':
    main()
