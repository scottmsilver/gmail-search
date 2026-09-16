#!/usr/bin/env python3
"""Transactional owner-key prerequisite; NOT a complete ingestion deployment.

Requires stopped writers and a separately qualified caller release. Rebuilds the
existing messages BM25 index only with an explicit maintenance opt-in.
Only the explicitly enumerated legacy or target schemas are accepted. Mailbox
content stays in its existing rows; an internal numeric search identity is added.
"""
from __future__ import annotations

import argparse
import os
import json

import psycopg
from psycopg import sql
from psycopg.pq import TransactionStatus


TABLES = ('messages', 'attachments', 'embeddings', 'message_summaries', 'summary_failures', 'message_topics')
LEGACY_KEYS = {
    'messages': {('p', ('id',))},
    'attachments': {('p', ('id',)), ('u', ('message_id', 'filename'))},
    'embeddings': {('p', ('id',))},
    'message_summaries': {('p', ('message_id',))},
    'summary_failures': {('p', ('message_id',))},
    'message_topics': {('p', ('message_id', 'topic_id'))},
}
TARGET_KEYS = {
    **LEGACY_KEYS,
    'messages': {('p', ('user_id', 'id')), ('u', ('search_id',))},
    'attachments': {('p', ('id',)), ('u', ('user_id', 'message_id', 'filename')), ('u', ('user_id', 'message_id', 'id'))},
    'message_summaries': {('p', ('user_id', 'message_id'))},
    'summary_failures': {('p', ('user_id', 'message_id'))},
    'message_topics': {('p', ('user_id', 'message_id', 'topic_id'))},
}


def _constraints(conn):
    rows = conn.execute("""
        SELECT n.nspname,c.relname,k.conname,k.contype,
            ARRAY(SELECT a.attname FROM unnest(k.conkey) WITH ORDINALITY x(num,ord)
                  JOIN pg_attribute a ON a.attrelid=k.conrelid AND a.attnum=x.num ORDER BY x.ord),
            rn.nspname,rc.relname,
            ARRAY(SELECT a.attname FROM unnest(k.confkey) WITH ORDINALITY x(num,ord)
                  JOIN pg_attribute a ON a.attrelid=k.confrelid AND a.attnum=x.num ORDER BY x.ord),
            k.confdeltype,k.confupdtype,k.convalidated,k.condeferrable,k.confmatchtype
        FROM pg_constraint k JOIN pg_class c ON c.oid=k.conrelid
        JOIN pg_namespace n ON n.oid=c.relnamespace
        LEFT JOIN pg_class rc ON rc.oid=k.confrelid
        LEFT JOIN pg_namespace rn ON rn.oid=rc.relnamespace
        WHERE (n.nspname='public' AND c.relname=ANY(%s))
           OR (rn.nspname='public' AND rc.relname IN ('messages','attachments'))
    """, (list(TABLES),)).fetchall()
    return [dict(zip(('schema','table','name','kind','columns','remote_schema','remote_table',
                      'remote_columns','delete','update','validated','deferred','match'), row)) for row in rows]


def _signature(c):
    if (c['table']=='message_topics' and c['remote_table']=='topics'
            and set(zip(c['columns'],c['remote_columns'])) == {('user_id','user_id'),('topic_id','topic_id')}):
        # The earlier promotion used both column orders in different releases.
        return ('message_topics', ('user_id','topic_id'), 'topics', ('user_id','topic_id'), c['delete'])
    return (c['table'], tuple(c['columns']), c['remote_table'], tuple(c['remote_columns']), c['delete'])


def _foreign_keys(target):
    owner = ('user_id',) if target else ()
    result = {(table, owner + ('message_id',), 'messages', owner + ('id',), 'c' if table == 'summary_failures' else 'a')
              for table in TABLES if table != 'messages'}
    result.add(('embeddings', ('user_id','message_id','attachment_id') if target else ('attachment_id',),
                'attachments', ('user_id','message_id','id') if target else ('id',), 'a'))
    # The topic key was already promoted in the earlier multi-owner migration.
    result.add(('message_topics', ('user_id','topic_id'), 'topics', ('user_id','topic_id'), 'a'))
    return result


def _bm25_plan(conn, *, allow_legacy=False):
    """Accept one ordinary BM25 index; retain its complete catalog configuration."""
    rows = conn.execute("""SELECT idx.relname,am.amname,i.indisvalid,i.indisready,
        i.indexprs IS NULL AND i.indpred IS NULL AND i.indnatts=i.indnkeyatts
        AND NOT EXISTS (SELECT 1 FROM unnest(i.indoption) o WHERE o<>0)
        AND NOT EXISTS (SELECT 1 FROM pg_attribute ia WHERE ia.attrelid=i.indexrelid AND ia.attoptions IS NOT NULL),
        ARRAY(SELECT a.attname FROM unnest(i.indkey) WITH ORDINALITY x(num,ord)
              JOIN pg_attribute a ON a.attrelid=i.indrelid AND a.attnum=x.num ORDER BY x.ord),
        idx.reloptions, ts.spcname, obj_description(idx.oid,'pg_class'),
        ARRAY(SELECT quote_ident(a.attname)
            || CASE WHEN co.oid IS NOT NULL THEN ' COLLATE ' || quote_ident(cn.nspname) || '.' || quote_ident(co.collname) ELSE '' END
            || ' ' || quote_ident(onsp.nspname) || '.' || quote_ident(opc.opcname)
            FROM unnest(i.indkey) WITH ORDINALITY x(num,ord)
            JOIN pg_attribute a ON a.attrelid=i.indrelid AND a.attnum=x.num
            JOIN pg_opclass opc ON opc.oid=i.indclass[x.ord-1]
            JOIN pg_namespace onsp ON onsp.oid=opc.opcnamespace
            LEFT JOIN pg_collation co ON co.oid=i.indcollation[x.ord-1]
            LEFT JOIN pg_namespace cn ON cn.oid=co.collnamespace ORDER BY x.ord)
        FROM pg_index i JOIN pg_class t ON t.oid=i.indrelid
        JOIN pg_namespace n ON n.oid=t.relnamespace JOIN pg_class idx ON idx.oid=i.indexrelid
        JOIN pg_am am ON am.oid=idx.relam LEFT JOIN pg_tablespace ts ON ts.oid=idx.reltablespace
        WHERE n.nspname='public' AND t.relname='messages'
        AND (idx.relname='messages_bm25_idx' OR am.amname='bm25')""").fetchall()
    if not rows:
        return None
    if len(rows) != 1:
        raise ValueError('Unsupported multiple BM25 indexes')
    name, method, valid, ready, ordinary, columns, options, tablespace, comment, definitions = rows[0]
    if method != 'bm25' or not valid or not ready or not ordinary or not columns:
        raise ValueError('Unsupported BM25 index definition')
    opts = dict(option.split('=',1) for option in (options or []))
    key = opts.get('key_field')
    if key not in ('id','search_id') or key not in columns:
        raise ValueError('Unsupported BM25 key field')
    if key == 'id' and not allow_legacy:
        raise ValueError('BM25 rebuild requires explicit maintenance opt-in')
    # These definitions come from PostgreSQL's own catalog quoting, preserving
    # collations and operator classes as well as column names. Expressions are refused.
    fields = [sql.SQL(definition) for definition in definitions]
    if key == 'id':
        fields.insert(0, sql.Identifier('search_id'))
    opts['key_field'] = 'search_id'
    statement = sql.SQL('CREATE INDEX {} ON public.messages USING bm25 ({}) WITH ({})').format(
        sql.Identifier(name), sql.SQL(',').join(fields),
        sql.SQL(',').join(sql.SQL('{}={}').format(sql.Identifier(k),sql.Literal(v)) for k,v in opts.items()))
    if tablespace:
        statement += sql.SQL(' TABLESPACE {}').format(sql.Identifier(tablespace))
    return dict(name=name, key=key, statement=statement, comment=comment)


def preview_owner_keys(conn):
    """Read-only preflight. Apply repeats all checks under exclusive locks."""
    if conn.info.transaction_status != TransactionStatus.IDLE:
        raise ValueError('Preview requires an idle connection')
    with conn.transaction():
        conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        conn.execute('SET LOCAL search_path=pg_catalog')
        conn.execute("SET LOCAL statement_timeout='60s'")
        _require_admin(conn)
        target, _ = _catalog_preflight(conn, allow_bm25=True)
        _data_preflight(conn)
        plan = _bm25_plan(conn, allow_legacy=True)
        return {'target_schema': target, 'requires_bm25_rebuild': bool(plan and plan['key']=='id'),
                'bm25_sql': plan['statement'].as_string(conn) if plan and plan['key']=='id' else None}


def _require_admin(conn):
    qualified = conn.execute('SELECT session_user=current_user AND rolsuper FROM pg_roles WHERE rolname=current_user').fetchone()
    if qualified != (True,):
        raise ValueError('Migration requires a direct administrator login with complete RLS visibility')


def _catalog_preflight(conn, *, allow_bm25=False):
    relations = conn.execute("""SELECT c.relname,c.relkind FROM pg_class c
        JOIN pg_namespace n ON n.oid=c.relnamespace
        WHERE n.nspname='public' AND c.relname=ANY(%s)""", (list(TABLES) + ['topics','users'],)).fetchall()
    if dict(relations) != {table:'r' for table in (*TABLES,'topics','users')}:
        raise ValueError('Expected ordinary public mailbox tables are missing or unsupported')
    if conn.execute('''SELECT 1 FROM pg_inherits i JOIN pg_class c
        ON c.oid=i.inhrelid OR c.oid=i.inhparent JOIN pg_namespace n ON n.oid=c.relnamespace
        WHERE n.nspname='public' AND c.relname=ANY(%s)''', (list(TABLES) + ['topics','users'],)).fetchone():
        raise ValueError('Unsupported inheritance dependency on participating tables')
    bm25 = _bm25_plan(conn, allow_legacy=allow_bm25)
    if conn.execute("""SELECT 1 FROM pg_index i JOIN pg_class t ON t.oid=i.indrelid
        JOIN pg_namespace n ON n.oid=t.relnamespace WHERE n.nspname='public'
        AND t.relname=ANY(%s) AND i.indisunique
        AND NOT EXISTS(SELECT 1 FROM pg_constraint k WHERE k.conindid=i.indexrelid)""", (list(TABLES),)).fetchone():
        raise ValueError('Unexpected unique index dependency')
    columns = conn.execute("""SELECT c.relname,a.attname,tn.nspname,t.typname,a.attnotnull,a.attidentity
        FROM pg_attribute a JOIN pg_class c ON c.oid=a.attrelid
        JOIN pg_namespace n ON n.oid=c.relnamespace JOIN pg_type t ON t.oid=a.atttypid
        JOIN pg_namespace tn ON tn.oid=t.typnamespace
        WHERE n.nspname='public' AND c.relname=ANY(%s) AND a.attnum>0 AND NOT a.attisdropped
    """, (list(TABLES),)).fetchall()
    actual = {(row[0],row[1]): row[2:] for row in columns}
    required = {table:{'user_id':'text'} for table in TABLES}
    required['messages']['id'] = 'text'
    required['attachments'].update(id='int8',message_id='text',filename='text')
    required['embeddings'].update(id='int8',message_id='text',attachment_id='int8')
    for table in ('message_summaries','summary_failures','message_topics'):
        required[table]['message_id'] = 'text'
    required['message_topics']['topic_id'] = 'text'
    for table, attrs in required.items():
        for name, kind in attrs.items():
            if actual.get((table,name), ())[:2] != ('pg_catalog',kind):
                raise ValueError('Unsupported mailbox key type or missing column')
    constraints = _constraints(conn)
    keys = {table:set() for table in TABLES}
    for c in constraints:
        if c['schema']=='public' and c['table'] in keys and c['kind'] in ('p','u'):
            if not c['validated'] or c['deferred']:
                raise ValueError('Unsupported deferred or unvalidated key dependency')
            keys[c['table']].add((c['kind'],tuple(c['columns'])))
    if keys == LEGACY_KEYS and ('messages','search_id') not in actual:
        target = False
    elif keys == TARGET_KEYS and actual.get(('messages','search_id')) == ('pg_catalog','int8',True,'d'):
        target = True
        if any(not actual[(table,'user_id')][2] for table in TABLES):
            raise ValueError('Partially migrated owner nullability')
    else:
        raise ValueError('Unexpected or partially migrated key dependency')
    if bm25 and bm25['key'] != ('search_id' if target else 'id'):
        raise ValueError('BM25 key and owner schema disagree')
    expected, found, owner_fks = _foreign_keys(target), set(), set()
    derived_fks = set()
    for c in constraints:
        if c['kind'] == 'f':
            # A separately qualified derived migration may add these only after
            # owner keys exist. Idempotent reapplication must preserve them.
            if target and c['schema']=='public' and c['table'] in ('propositions','prop_processed'):
                if (c['remote_schema']!='public' or c['remote_table']!='messages'
                        or tuple(c['columns'])!=('user_id','message_id')
                        or tuple(c['remote_columns'])!=('user_id','id')
                        or c['delete']!='c' or c['update']!='a' or c['match']!='s'
                        or not c['validated'] or c['deferred'] or c['table'] in derived_fks):
                    raise ValueError('Unexpected derived foreign-key dependency')
                derived_fks.add(c['table'])
                continue
            if c['schema'] != 'public' or c['table'] not in TABLES or c['remote_schema'] != 'public':
                raise ValueError('Unexpected foreign-key dependency')
            # Existing users ownership FKs are preserved, never dropped.
            if tuple(c['columns']) == ('user_id',) and c['remote_table']=='users' and tuple(c['remote_columns'])==('id',):
                if not c['validated'] or c['table'] in owner_fks:
                    raise ValueError('Unvalidated owner foreign key')
                owner_fks.add(c['table'])
                continue
            signature = _signature(c)
            if signature not in expected or signature in found or not c['validated'] or c['deferred'] or c['update']!='a' or c['match']!='s':
                raise ValueError('Unexpected foreign-key dependency')
            found.add(signature)
        elif c['kind'] not in ('p','u','c','n'):
            raise ValueError('Unsupported constraint dependency')
    if found != expected or owner_fks != set(TABLES):
        raise ValueError('Required foreign-key dependency is missing')
    return target, constraints


def _data_preflight(conn):
    for table in TABLES:
        if conn.execute(sql.SQL('SELECT 1 FROM {} WHERE user_id IS NULL LIMIT 1').format(sql.Identifier('public',table))).fetchone():
            raise ValueError('Mailbox table has null owner: ' + table)
        if conn.execute(sql.SQL('SELECT 1 FROM {} child LEFT JOIN public.users u ON u.id=child.user_id WHERE u.id IS NULL LIMIT 1').format(sql.Identifier('public',table))).fetchone():
            raise ValueError('Mailbox table has orphan owner: ' + table)
    for table in TABLES[1:]:
        if conn.execute(sql.SQL('''SELECT 1 FROM {} child LEFT JOIN public.messages parent
            ON child.message_id=parent.id AND child.user_id=parent.user_id
            WHERE parent.id IS NULL LIMIT 1''').format(sql.Identifier('public',table))).fetchone():
            raise ValueError('Child has missing message or mismatched owner: ' + table)
    if conn.execute('''SELECT 1 FROM public.embeddings e LEFT JOIN public.attachments a
        ON e.attachment_id=a.id AND e.message_id=a.message_id AND e.user_id=a.user_id
        WHERE e.attachment_id IS NOT NULL AND a.id IS NULL LIMIT 1''').fetchone():
        raise ValueError('Embedding has mismatched attachment/message owner')
    if conn.execute('''SELECT 1 FROM public.message_topics mt LEFT JOIN public.topics t
        ON mt.topic_id=t.topic_id AND mt.user_id=t.user_id WHERE t.topic_id IS NULL LIMIT 1''').fetchone():
        raise ValueError('Message topic has mismatched topic owner')


def _verify_target(conn):
    target, _ = _catalog_preflight(conn)
    if not target:
        raise ValueError('Owner-key migration did not reach target schema')
    _data_preflight(conn)


def migrate_owner_keys(conn, *, rebuild_bm25=False) -> bool:
    """Return True when changed; False for an already qualified target schema.

    Requires an idle administrator connection and no encompassing transaction.
    It never commits another caller's work or changes role grants/RLS policies.
    """
    if conn.info.transaction_status != TransactionStatus.IDLE:
        raise ValueError('Migration requires an idle connection with no existing transaction')
    with conn.transaction():
        conn.execute("SET LOCAL search_path = pg_catalog")
        conn.execute("SET LOCAL lock_timeout = '2s'")
        conn.execute("SET LOCAL statement_timeout = '60s'")
        _require_admin(conn)
        if not conn.execute("SELECT pg_try_advisory_xact_lock(72341629,1)").fetchone()[0]:
            raise ValueError('Owner-key migration is already running')
        # Lock relations before inspecting their catalogs/data, preventing racing writers/DDL.
        for table in sorted((*TABLES,'topics','users')):
            conn.execute(sql.SQL('LOCK TABLE {} IN ACCESS EXCLUSIVE MODE').format(sql.Identifier('public',table)))
        target, constraints = _catalog_preflight(conn, allow_bm25=rebuild_bm25)
        bm25 = _bm25_plan(conn, allow_legacy=rebuild_bm25)
        _data_preflight(conn)
        if target:
            return False
        if bm25:
            conn.execute(sql.SQL('DROP INDEX {}').format(sql.Identifier('public',bm25['name'])))
        # Drop only catalog-enumerated foreign keys affected by key replacement.
        for c in constraints:
            if c['kind']=='f' and c['remote_table'] in ('messages','attachments'):
                conn.execute(sql.SQL('ALTER TABLE {} DROP CONSTRAINT {}').format(sql.Identifier('public',c['table']),sql.Identifier(c['name'])))
        for c in constraints:
            if c['kind'] in ('p','u') and c['table'] not in ('embeddings',) and not (c['table']=='attachments' and c['kind']=='p'):
                conn.execute(sql.SQL('ALTER TABLE {} DROP CONSTRAINT {}').format(sql.Identifier('public',c['table']),sql.Identifier(c['name'])))
        for table in TABLES:
            conn.execute(sql.SQL('ALTER TABLE {} ALTER COLUMN user_id SET NOT NULL').format(sql.Identifier('public',table)))
        conn.execute('ALTER TABLE public.messages ADD COLUMN search_id bigint GENERATED BY DEFAULT AS IDENTITY')
        for table in TABLES:
            for kind, columns in sorted(TARGET_KEYS[table] - LEGACY_KEYS[table]):
                conn.execute(sql.SQL('ALTER TABLE {} ADD {} ({})').format(
                    sql.Identifier('public',table),sql.SQL('PRIMARY KEY' if kind=='p' else 'UNIQUE'),
                    sql.SQL(',').join(map(sql.Identifier,columns))))
        for table, columns, parent, remote, delete in sorted(_foreign_keys(True)):
            if parent=='topics':
                continue
            conn.execute(sql.SQL('ALTER TABLE {} ADD FOREIGN KEY ({}) REFERENCES {} ({}) ON DELETE {}').format(
                sql.Identifier('public',table),sql.SQL(',').join(map(sql.Identifier,columns)),sql.Identifier('public',parent),
                sql.SQL(',').join(map(sql.Identifier,remote)),sql.SQL('CASCADE' if delete=='c' else 'NO ACTION')))
        if bm25:
            conn.execute(bm25['statement'])
            if bm25['comment'] is not None:
                conn.execute(sql.SQL('COMMENT ON INDEX {} IS {}').format(
                    sql.Identifier('public',bm25['name']),sql.Literal(bm25['comment'])))
        _verify_target(conn)
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--apply', action='store_true', help='Apply to a disposable synthetic database only')
    mode.add_argument('--preview', action='store_true', help='Read-only catalog and owner-data preflight')
    parser.add_argument('--rebuild-bm25', action='store_true', help='Explicitly rebuild the legacy BM25 index during maintenance')
    args = parser.parse_args()
    dsn = os.environ.get('OWNER_KEY_MIGRATION_DSN')
    if not dsn:
        parser.error('OWNER_KEY_MIGRATION_DSN is required')
    try:
        with psycopg.connect(dsn, autocommit=True) as conn:
            if args.preview:
                print(json.dumps(preview_owner_keys(conn), sort_keys=True))
                return
            if not conn.info.dbname.startswith('gms_owner_keys_test_'):
                raise ValueError('CLI application is restricted to disposable qualification databases')
            changed = migrate_owner_keys(conn, rebuild_bm25=args.rebuild_bm25)
    except Exception:
        # Raw PostgreSQL diagnostics may disclose row contents or connection details.
        parser.exit(1, 'Owner-key migration refused or failed; transaction rolled back.\n')
    print('Owner-key prerequisite applied.' if changed else 'Owner-key prerequisite already present.')
    print('Synthetic qualification only; production capacity and the complete caller release require separate review.')


if __name__ == '__main__':
    main()
