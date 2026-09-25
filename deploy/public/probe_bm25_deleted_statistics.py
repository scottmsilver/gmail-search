#!/usr/bin/env python3
"""Synthetic-only historical BM25 deletion/maintenance/attach qualification.

No production DSN, migration imports, source mail, or global settings mutations.
The unique database/reader created here are removed on exit.
"""
import argparse
import json
import os
from pathlib import Path
import secrets
import time

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

FUNCTIONS = (
    'paradedb.search_with_parse(anyelement,text)', 'paradedb.score(anyelement)',
    'paradedb.with_index(regclass,paradedb.searchqueryinput)',
    'paradedb.parse_with_field(paradedb.fieldname,text,boolean,boolean)',
)
PROFILES = {
    'message_text': ('id text NOT NULL,subject text,body_text text,from_addr text,to_addr text',
                     'id', ('id','subject','body_text','from_addr','to_addr'), 'body_text'),
    'message_numeric': ('id text NOT NULL,search_id bigint NOT NULL,subject text,body_text text,from_addr text,to_addr text',
                        'search_id', ('search_id','id','subject','body_text','from_addr','to_addr'), 'body_text'),
    'attachment': ('id bigint NOT NULL,message_id text,filename text,extracted_text text',
                   'id', ('id','filename','extracted_text'), 'extracted_text'),
}
OWNER_TEXTS = ('alpha alpha', 'beta', 'alpha beta filler filler filler', 'filler '*20)


def ident(name):
    return sql.Identifier(*name.split('.'))


def objects(conn,table,index):
    return conn.execute('''SELECT t.oid,t.relfilenode,t.reltoastrelid,z.relfilenode,
        pg_relation_size(t.oid),i.oid,i.relfilenode,pg_relation_size(i.oid)
        FROM pg_class t LEFT JOIN pg_class z ON z.oid=t.reltoastrelid
        CROSS JOIN pg_class i WHERE t.oid=(SELECT indrelid FROM pg_index WHERE indexrelid=%s::regclass) AND i.oid=%s::regclass''',
        (index,index)).fetchone()


def snapshot(admin,reader,table,index,key,field,reader_dsn,*,allow_errors=False):
    statement=sql.SQL("SELECT {key},paradedb.score({key}) FROM {table} WHERE user_id='alice' "
        "AND {key} OPERATOR(pg_catalog.@@@) %s ORDER BY paradedb.score({key}) DESC,{key} LIMIT 100").format(
        key=sql.Identifier(key),table=ident(table))
    def query(conn,term,*,prepare=None):
        try:return conn.execute(statement,(term,),prepare=prepare).fetchall()
        except psycopg.Error as error:
            if not allow_errors:raise
            return {'sqlstate':error.sqlstate,'error':error.diag.message_primary}
    result={}
    for mode in ('force_custom_plan','force_generic_plan'):
        reader.execute('SELECT set_config(\'plan_cache_mode\',%s,false)',(mode,))
        result[mode]=query(reader,f'{field}:alpha OR {field}:beta',prepare=True)
    with psycopg.connect(reader_dsn,autocommit=True) as fresh:
        fresh.execute("SET statement_timeout='10s'")
        result['fresh_backend']=query(fresh,f'{field}:alpha OR {field}:beta')
    if allow_errors:
        with psycopg.connect(reader_dsn,autocommit=True,prepare_threshold=0) as fresh_generic:
            fresh_generic.execute("SET statement_timeout='10s'")
            fresh_generic.execute("SET plan_cache_mode='force_generic_plan'")
            result['fresh_generic_backend']=query(fresh_generic,f'{field}:alpha OR {field}:beta',prepare=True)
            result['fresh_generic_plan']=fresh_generic.execute(sql.SQL('EXPLAIN(VERBOSE,FORMAT JSON) ')+statement,(f'{field}:alpha OR {field}:beta',),prepare=True).fetchone()[0]
    if allow_errors:
        with psycopg.connect(reader_dsn,autocommit=True,prepare_threshold=0) as serial_generic:
            serial_generic.execute("SET statement_timeout='10s'")
            serial_generic.execute("SET plan_cache_mode='force_generic_plan'")
            serial_generic.execute("SET max_parallel_workers_per_gather=0")
            result['fresh_serial_generic_backend']=query(serial_generic,f'{field}:alpha OR {field}:beta',prepare=True)
            result['fresh_serial_generic_plan']=serial_generic.execute(sql.SQL('EXPLAIN(VERBOSE,FORMAT JSON) ')+statement,(f'{field}:alpha OR {field}:beta',),prepare=True).fetchone()[0]
    result['foreign_only']=query(reader,f'{field}:foreignonly')
    result['live_rows']=reader.execute(sql.SQL('SELECT count(*) FROM {}').format(ident(table))).fetchone()[0]
    result['objects']=objects(admin,table,index)
    segments=admin.execute('SELECT * FROM paradedb.index_info(%s::regclass,true)',(index,))
    result['segments']=[dict(zip((column.name for column in segments.description),row)) for row in segments.fetchall()]
    return result


SETTLE_SECONDS=10


def delete_foreign(conn,table):
    """Delete the foreign owner's rows; returns the deleting transaction's xid.

    Runs in the caller's transaction when there is one."""
    with conn.transaction():
        conn.execute(sql.SQL("DELETE FROM {} WHERE user_id='bob'").format(ident(table)))
        return int(conn.execute('SELECT pg_current_xact_id()').fetchone()[0])


def settle(conn,delete_xid,*,seconds=SETTLE_SECONDS):
    """Wait, bounded, until no running transaction on the cluster predates the delete.

    A REINDEX indexes HEAPTUPLE_RECENTLY_DEAD rows as live documents, and the
    rebuilding session's own snapshot takes its xmin from running xids in
    *every* database on the cluster. So any other test run's open write
    transaction turns the deleted rows back into corpus statistics (#55).
    Returns the horizon, which callers record so a failure names it."""
    started=time.monotonic()
    while True:
        xmin=int(conn.execute('SELECT pg_snapshot_xmin(pg_current_snapshot())').fetchone()[0])
        if xmin>delete_xid or time.monotonic()-started>=seconds:break
        time.sleep(0.05)
    return {'delete_xid':delete_xid,'snapshot_xmin':xmin,'settled':xmin>delete_xid,
            'waited_seconds':round(time.monotonic()-started,3)}


def settle_and_vacuum(conn,leaf,delete_xid):
    """The clean-rebuild boundary: settle the horizon, then VACUUM the leaf.

    VACUUM ignores its own snapshot, so it removes the deleted tuples even
    past a cross-database writer that outlasts the bounded wait."""
    horizon=settle(conn,delete_xid)
    conn.execute(sql.SQL('VACUUM(INDEX_CLEANUP ON) {}').format(ident(leaf)))
    return horizon


def _grant(admin,role,table):
    admin.execute(sql.SQL('ALTER TABLE {} ENABLE ROW LEVEL SECURITY').format(ident(table)))
    admin.execute(sql.SQL('ALTER TABLE {} FORCE ROW LEVEL SECURITY').format(ident(table)))
    admin.execute(sql.SQL('CREATE POLICY legacy ON {} USING(true)').format(ident(table)))
    admin.execute(sql.SQL("CREATE POLICY fixed_owner ON {} AS RESTRICTIVE TO {} USING(user_id='alice')").format(ident(table),sql.Identifier(role)))
    admin.execute(sql.SQL('GRANT SELECT ON {} TO {}').format(ident(table),sql.Identifier(role)))


def attach(admin,role,mixed,index,key,fields):
    """Caller owns the transaction; retain the existing heap/index relation."""
    leaf='retained_leaves.'+mixed.split('.')[-1]
    leaf_index='retained_leaves.'+index.split('.')[-1]
    admin.execute(sql.SQL('ALTER TABLE {} SET SCHEMA retained_leaves').format(ident(mixed)))
    admin.execute(sql.SQL('REVOKE ALL ON {} FROM {}').format(ident(leaf),sql.Identifier(role)))
    admin.execute(sql.SQL("ALTER TABLE {} ADD CONSTRAINT exact_owner CHECK(user_id='alice')").format(ident(leaf)))
    admin.execute(sql.SQL('CREATE TABLE {}(LIKE {} INCLUDING DEFAULTS INCLUDING STORAGE) PARTITION BY LIST(user_id)').format(ident(mixed),ident(leaf)))
    admin.execute(sql.SQL('ALTER TABLE {} ADD PRIMARY KEY(user_id,{})').format(ident(mixed),sql.Identifier(key)))
    admin.execute(sql.SQL('CREATE INDEX {} ON {} USING bm25({}) WITH(key_field={})').format(sql.Identifier(index.split('.')[-1]),ident(mixed),sql.SQL(',').join(map(sql.Identifier,fields)),sql.Literal(key)))
    admin.execute(sql.SQL("ALTER TABLE {} ATTACH PARTITION {} FOR VALUES IN ('alice')").format(ident(mixed),ident(leaf)))
    _grant(admin,role,mixed)
    return leaf,leaf_index


def custom_cycles(admin,reader_dsn,table,leaf,key,field,profile):
    """Countermeasure experiment only; no product profile changes."""
    mid='message_id' if profile=='attachment' else 'id'
    statement=sql.SQL("SELECT user_id AS owner_id,{key} AS id,{mid} AS message_id,paradedb.score({key}) AS score "
        "FROM {table} WHERE user_id='alice' AND {key} OPERATOR(pg_catalog.@@@) %s ORDER BY score DESC,{key}").format(
        key=sql.Identifier(key),mid=sql.Identifier(mid),table=ident(table))
    wrapped=sql.SQL('SELECT CASE WHEN octet_length(row_to_json(gms_row)::text)<=%s THEN row_to_json(gms_row)::text ELSE NULL END FROM (')+statement+sql.SQL(' LIMIT %s) gms_row')
    def read():
        started=time.monotonic()
        with psycopg.connect(reader_dsn,prepare_threshold=None) as connection:
            connection.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
            connection.execute("SET LOCAL statement_timeout='10s'")
            connection.execute("SET LOCAL plan_cache_mode='force_custom_plan'")
            params=(4*1024*1024,f'{field}:alpha OR {field}:beta',201)
            with connection.cursor(name='search_1') as cursor:
                cursor.execute(wrapped,params)
                rows=[json.loads(row[0]) for row in cursor.fetchall()]
            assert all(row['owner_id']=='alice' for row in rows)
            # The unwrapped ordinary query separately checks the custom plan
            # outside named-cursor execution, still with preparation disabled.
            direct=connection.execute(statement+sql.SQL(' LIMIT 201'),(params[1],)).fetchall()
            return {'rows':rows,'direct':direct,'seconds':time.monotonic()-started}
    baseline=read();states={'baseline':baseline,'cycles':[]}
    foreign_leaf=leaf+'_bob'
    admin.execute(sql.SQL("CREATE TABLE {} PARTITION OF {} FOR VALUES IN ('bob')").format(ident(foreign_leaf),ident(table)))
    for cycle in range(3):
        names=['user_id',key,field]
        aid='cycle'+str(cycle) if profile=='message_text' else 900+cycle
        values=['alice',aid,'alpha foreignonly']
        if profile=='message_numeric':names.append('id');values.append('cycle'+str(cycle))
        admin.execute(sql.SQL('INSERT INTO {}({}) VALUES({})').format(ident(table),sql.SQL(',').join(map(sql.Identifier,names)),sql.SQL(',').join(sql.Placeholder() for _ in names)),values)
        inserted=read()
        admin.execute(sql.SQL("DELETE FROM {} WHERE user_id='alice' AND {}=%s").format(ident(table),sql.Identifier(key)),(aid,))
        deleted=read()
        admin.execute(sql.SQL('VACUUM(INDEX_CLEANUP ON) {}').format(ident(leaf)))
        vacuumed=read()
        foreign_names=['user_id',key,field]
        expressions=[sql.Literal('bob'),sql.SQL("'foreign'||n") if profile=='message_text' else sql.SQL('1000+n'),sql.Literal('alpha foreignonly')]
        if profile=='message_numeric':foreign_names.append('id');expressions.append(sql.SQL("'foreign'||n"))
        admin.execute(sql.SQL('INSERT INTO {}({}) SELECT {} FROM generate_series(1,200)n').format(ident(table),sql.SQL(',').join(map(sql.Identifier,foreign_names)),sql.SQL(',').join(expressions)))
        foreign_insert=read()
        admin.execute(sql.SQL("UPDATE {} SET {}='beta foreignonly' WHERE user_id='bob'").format(ident(table),sql.Identifier(field)))
        foreign_update=read()
        admin.execute(sql.SQL("DELETE FROM {} WHERE user_id='bob'").format(ident(table)))
        admin.execute(sql.SQL('VACUUM(INDEX_CLEANUP ON) {}').format(ident(foreign_leaf)))
        foreign_delete_vacuum=read()
        for value in (vacuumed,foreign_insert,foreign_update,foreign_delete_vacuum):
            assert value['rows']==baseline['rows'] and value['direct']==baseline['direct']
        assert len(inserted['rows'])==len(baseline['rows'])+1
        assert len(deleted['rows'])==len(baseline['rows'])
        states['cycles'].append({'insert':inserted,'delete':deleted,'vacuum':vacuumed,
            'foreign_insert':foreign_insert,'foreign_update':foreign_update,'foreign_delete_vacuum':foreign_delete_vacuum})
    return states


APPROVED_FIXTURE=('127.0.0.1','55440','postgres','postgres')
FORBIDDEN_CONNINFO_KEYS={'hostaddr','service','options'}


def dsn_is_approved_fixture(dsn):
    """True only for the one disposable loopback fixture this probe may touch.

    This probe CREATEs databases and roles, so it must never be aimed at a
    real server. Exposed so callers can SKIP when off-fixture rather than
    driving `run()` into a ValueError: checking merely that GMS_TEST_PG_DSN is
    *set* turned CI into 171 setup errors (2026-09-22)."""
    config=conninfo_to_dict(dsn)
    return ((config.get('host'),config.get('port'),config.get('dbname'),config.get('user'))==APPROVED_FIXTURE
            and not set(config)&FORBIDDEN_CONNINFO_KEYS)


def run(*,atomic=False,staged=False,churn=None):
    if type(atomic) is not bool or type(staged) is not bool or (atomic and staged):raise ValueError("Invalid probe mode")
    if churn not in (None,'observed','unobserved','custom_cycles') or (churn and atomic) or (churn in ('observed','unobserved') and staged):raise ValueError('Invalid probe mode')
    if churn=='custom_cycles':staged=True
    base=os.environ['GMS_TEST_PG_DSN']
    if not dsn_is_approved_fixture(base):
        raise ValueError('Only the approved disposable loopback fixture is supported')
    suffix=secrets.token_hex(8)
    database='gms_deleted_bm25_'+suffix;role='gms_deleted_reader_'+suffix
    password=secrets.token_urlsafe(40)
    report={'synthetic_only':True,'profiles':{},'object_fields':['heap_oid','heap_relfilenode','toast_oid','toast_relfilenode','heap_bytes','bm25_oid','bm25_relfilenode','bm25_main_bytes']}
    created_role=False
    with psycopg.connect(base,autocommit=True) as root:
        if root.info.hostaddr!='127.0.0.1':raise ValueError('Unexpected fixture address')
        root.execute(sql.SQL('CREATE DATABASE {} TEMPLATE template0').format(sql.Identifier(database)))
    target=make_conninfo(base,dbname=database)
    try:
        with psycopg.connect(target,autocommit=True) as admin:
            admin.execute("SET statement_timeout='30s'");admin.execute("SET lock_timeout='3s'")
            admin.execute('CREATE EXTENSION pg_search')
            version=admin.execute("SELECT version(),extversion FROM pg_extension WHERE extname='pg_search'").fetchone()
            assert version[1]=='0.23.0' and admin.info.server_version==160015
            report['version']=version
            admin.execute('CREATE SCHEMA retained_leaves')
            admin.execute(sql.SQL('CREATE ROLE {} LOGIN NOSUPERUSER NOBYPASSRLS NOINHERIT PASSWORD {}').format(sql.Identifier(role),sql.Literal(password)))
            created_role=True
            admin.execute(sql.SQL('REVOKE TEMP ON DATABASE {} FROM PUBLIC').format(sql.Identifier(database)))
            admin.execute('REVOKE CREATE ON SCHEMA public,paradedb,pdb FROM PUBLIC')
            admin.execute('REVOKE ALL ON SCHEMA retained_leaves FROM PUBLIC')
            routines=admin.execute("SELECT p.oid::regprocedure::text FROM pg_proc p JOIN pg_depend d ON d.classid='pg_proc'::regclass AND d.objid=p.oid JOIN pg_extension e ON e.oid=d.refobjid WHERE d.refclassid='pg_extension'::regclass AND e.extname='pg_search'").fetchall()
            for routine, in routines:
                admin.execute(sql.SQL('REVOKE ALL ON ROUTINE {} FROM PUBLIC').format(sql.SQL(routine)))
            admin.execute(sql.SQL('GRANT USAGE ON SCHEMA public,paradedb,pdb TO {}').format(sql.Identifier(role)))
            for fn in FUNCTIONS:
                admin.execute(sql.SQL('GRANT EXECUTE ON FUNCTION {} TO {}').format(sql.SQL(fn),sql.Identifier(role)))
            reader_dsn=make_conninfo(target,user=role,password=password)
            with psycopg.connect(reader_dsn,autocommit=True,prepare_threshold=0) as reader:
                assert reader.execute('SELECT current_user,session_user').fetchone()==(role,role)
                reader.execute("SET statement_timeout='10s'")
                reader.execute("SELECT set_config('app.user_id','bob',false)")
                for profile,(columns,key,fields,field) in PROFILES.items():
                    clean=f'public.clean_{profile}';mixed=f'public.mixed_{profile}'
                    clean_index=f'public.clean_{profile}_bm25';index=f'public.mixed_{profile}_bm25'
                    for table,foreign in ((clean,False),(mixed,True)):
                        admin.execute(sql.SQL('CREATE TABLE {}(user_id text NOT NULL,{},PRIMARY KEY(user_id,{})) WITH(autovacuum_enabled=false)').format(ident(table),sql.SQL(columns),sql.Identifier(key)))
                        for n,text in enumerate(OWNER_TEXTS,1):
                            names=['user_id',key,field];values=['alice',f'owner{n}' if key=='id' and profile=='message_text' else n,text]
                            if profile=='message_numeric':names.append('id');values.append(f'owner{n}')
                            admin.execute(sql.SQL('INSERT INTO {}({}) VALUES({})').format(ident(table),sql.SQL(',').join(map(sql.Identifier,names)),sql.SQL(',').join(sql.Placeholder() for _ in names)),values)
                        if foreign:
                            names=['user_id',key,field];expr=[sql.Literal('bob'),sql.SQL("'foreign'||n") if profile=='message_text' else sql.SQL('100+n'),sql.Literal('alpha foreignonly')]
                            if profile=='message_numeric':names.append('id');expr.append(sql.SQL("'foreign'||n"))
                            admin.execute(sql.SQL('INSERT INTO {}({}) SELECT {} FROM generate_series(1,200)n').format(ident(table),sql.SQL(',').join(map(sql.Identifier,names)),sql.SQL(',').join(expr)))
                        ix=table+'_bm25'
                        admin.execute(sql.SQL('CREATE INDEX {} ON {} USING bm25({}) WITH(key_field={})').format(sql.Identifier(ix.split('.')[-1]),ident(table),sql.SQL(',').join(map(sql.Identifier,fields)),sql.Literal(key)))
                        _grant(admin,role,table)
                    evidence={}
                    evidence['clean']=snapshot(admin,reader,clean,clean_index,key,field,reader_dsn)
                    evidence['mixed_before_delete']=snapshot(admin,reader,mixed,index,key,field,reader_dsn)
                    if atomic or staged:
                        for outcome in ('rollback','commit'):
                            start_lsn=admin.execute('SELECT pg_current_wal_insert_lsn()').fetchone()[0]
                            started=time.monotonic()
                            try:
                                with admin.transaction():
                                    delete_xid=delete_foreign(admin,mixed)
                                    leaf,leaf_index=attach(admin,role,mixed,index,key,fields)
                                    if atomic:admin.execute(sql.SQL('REINDEX INDEX {}').format(ident(leaf_index)))
                                    during=objects(admin,leaf,leaf_index)
                                    during_rows=admin.execute(sql.SQL("SELECT {key},paradedb.score({key}) FROM {table} WHERE user_id='alice' AND {key} OPERATOR(pg_catalog.@@@) %s ORDER BY paradedb.score({key}) DESC,{key} LIMIT 100").format(key=sql.Identifier(key),table=ident(mixed)),(f'{field}:alpha OR {field}:beta',)).fetchall()
                                    if outcome=='rollback':raise RuntimeError('intentional atomic rollback')
                            except RuntimeError:
                                if outcome!='rollback':raise
                            selected_index=index if outcome=='rollback' else leaf_index
                            result=snapshot(admin,reader,mixed,selected_index,key,field,reader_dsn)
                            result['during_objects']=during
                            result['during_owner_rows']=during_rows
                            result['all_owner_counts']=admin.execute(sql.SQL('SELECT user_id,count(*) FROM {} GROUP BY user_id ORDER BY user_id').format(ident(mixed))).fetchall()
                            result['seconds']=time.monotonic()-started
                            result['cluster_wal_bytes']=int(admin.execute('SELECT pg_wal_lsn_diff(pg_current_wal_insert_lsn(),%s::pg_lsn)',(start_lsn,)).fetchone()[0])
                            evidence[('phase1_' if staged else 'atomic_')+outcome]=result
                        # Only the committed delete's xid matters: the loop ends on 'commit'.
                        evidence['rebuild_horizon']=settle_and_vacuum(admin,leaf,delete_xid)
                        if staged:
                            # New database connections simulate loss of phase-one
                            # process/session state. No cached migration object is
                            # needed by the index rebuild itself.
                            with psycopg.connect(target,autocommit=True) as restarted:
                                restarted.execute("SET lock_timeout='3s'")
                                restarted.execute("SET statement_timeout='30s'")
                                try:
                                    with restarted.transaction():
                                        restarted.execute(sql.SQL('REINDEX INDEX {}').format(ident(leaf_index)))
                                        during=objects(restarted,leaf,leaf_index)
                                        raise RuntimeError('intentional phase-two rollback')
                                except RuntimeError:
                                    pass
                                result=snapshot(restarted,reader,mixed,leaf_index,key,field,reader_dsn)
                                result['during_objects']=during
                                result['all_owner_counts']=restarted.execute(sql.SQL('SELECT user_id,count(*) FROM {} GROUP BY user_id ORDER BY user_id').format(ident(mixed))).fetchall()
                                evidence['phase2_rollback']=result
                            with psycopg.connect(target,autocommit=True) as retry:
                                retry.execute("SET lock_timeout='3s'")
                                retry.execute("SET statement_timeout='30s'")
                                retry.execute(sql.SQL('REINDEX INDEX {}').format(ident(leaf_index)))
                                evidence['phase2_retry_commit']=snapshot(retry,reader,mixed,leaf_index,key,field,reader_dsn)
                        else:
                            admin.execute(sql.SQL('REINDEX INDEX {}').format(ident(leaf_index)))
                            evidence['separate_post_commit_reindex']=snapshot(admin,reader,mixed,leaf_index,key,field,reader_dsn)
                        if churn=='custom_cycles':
                            evidence['custom_unprepared_cycles']=custom_cycles(admin,reader_dsn,mixed,leaf,key,field,profile)
                        report['profiles'][profile]=evidence
                        continue
                    delete_xid=delete_foreign(admin,mixed)
                    # Every VACUUM and the REINDEX below follow this boundary.
                    evidence['rebuild_horizon']=settle(admin,delete_xid)
                    evidence['after_delete']=snapshot(admin,reader,mixed,index,key,field,reader_dsn)
                    admin.execute(sql.SQL('ANALYZE {}').format(ident(mixed)))
                    evidence['after_analyze']=snapshot(admin,reader,mixed,index,key,field,reader_dsn)
                    admin.execute(sql.SQL('VACUUM (ANALYZE,INDEX_CLEANUP ON) {}').format(ident(mixed)))
                    evidence['after_vacuum']=snapshot(admin,reader,mixed,index,key,field,reader_dsn)
                    with admin.transaction():
                        leaf,leaf_index=attach(admin,role,mixed,index,key,fields)
                    reader.execute('DEALLOCATE ALL');reader.prepare_threshold=None
                    evidence['after_attach']=snapshot(admin,reader,mixed,leaf_index,key,field,reader_dsn)
                    evidence['leaf_after_attach_objects']=objects(admin,leaf,leaf_index)
                    try:
                        reader.execute(sql.SQL('SELECT * FROM {}').format(ident(leaf)))
                        raise AssertionError('Direct child access unexpectedly allowed')
                    except psycopg.errors.InsufficientPrivilege:
                        evidence['direct_child_denied']=True
                    evidence['attach_plan']=reader.execute(sql.SQL("EXPLAIN(ANALYZE,VERBOSE,FORMAT JSON) SELECT {},paradedb.score({}) FROM {} WHERE user_id='alice' AND {} OPERATOR(pg_catalog.@@@) %s ORDER BY paradedb.score({}) DESC,{} LIMIT 100").format(*(sql.Identifier(x) for x in (key,key)),ident(mixed),*(sql.Identifier(x) for x in (key,key,key))),(f'{field}:alpha OR {field}:beta',)).fetchone()[0]
                    for step,statement in (
                        ('attached_vacuum',sql.SQL('VACUUM(ANALYZE,INDEX_CLEANUP ON) {}').format(ident(leaf))),
                        ('force_merge',sql.SQL('SELECT paradedb.force_merge({}::regclass,1::bigint)').format(sql.Literal(leaf_index))),
                        ('after_merge_vacuum',sql.SQL('VACUUM(ANALYZE,INDEX_CLEANUP ON) {}').format(ident(leaf))),
                        ('reindex',sql.SQL('REINDEX INDEX {}').format(ident(leaf_index))),
                    ):
                        if step=='reindex':
                            before=snapshot(admin,reader,mixed,leaf_index,key,field,reader_dsn)
                            try:
                                with admin.transaction():
                                    admin.execute(sql.SQL('REINDEX INDEX {}').format(ident(leaf_index)))
                                    during=objects(admin,leaf,leaf_index)
                                    raise RuntimeError('intentional synthetic rollback')
                            except RuntimeError:
                                pass
                            after=snapshot(admin,reader,mixed,leaf_index,key,field,reader_dsn)
                            evidence['reindex_rollback']={'before':before,'during_objects':during,'after':after}
                        start_lsn=admin.execute('SELECT pg_current_wal_insert_lsn()').fetchone()[0]
                        started=time.monotonic()
                        try:
                            result=admin.execute(statement)
                            maintenance=result.fetchall() if result.description else None
                        except psycopg.Error as error:
                            maintenance={'sqlstate':error.sqlstate,'error':error.diag.message_primary}
                        evidence[step]=snapshot(admin,reader,mixed,leaf_index,key,field,reader_dsn)
                        evidence[step]['maintenance_return']=maintenance
                        evidence[step]['seconds']=time.monotonic()-started
                        evidence[step]['cluster_wal_bytes']=int(admin.execute('SELECT pg_wal_lsn_diff(pg_current_wal_insert_lsn(),%s::pg_lsn)',(start_lsn,)).fetchone()[0])
                    if churn:
                        extra={}
                        names=['user_id',key,field]
                        values=['alice','churn' if profile=='message_text' else 999,'alpha foreignonly']
                        if profile=='message_numeric':names.append('id');values.append('churn')
                        admin.execute(sql.SQL('INSERT INTO {}({}) VALUES({})').format(ident(mixed),sql.SQL(',').join(map(sql.Identifier,names)),sql.SQL(',').join(sql.Placeholder() for _ in names)),values)
                        if churn=='observed':extra['after_insert']=snapshot(admin,reader,mixed,leaf_index,key,field,reader_dsn,allow_errors=True)
                        admin.execute(sql.SQL('DELETE FROM {} WHERE {}=%s').format(ident(mixed),sql.Identifier(key)),(values[1],))
                        if churn=='observed':extra['after_delete']=snapshot(admin,reader,mixed,leaf_index,key,field,reader_dsn,allow_errors=True)
                        admin.execute(sql.SQL('VACUUM(INDEX_CLEANUP ON) {}').format(ident(leaf)))
                        extra['after_vacuum']=snapshot(admin,reader,mixed,leaf_index,key,field,reader_dsn,allow_errors=True)
                        # Retry on the same and a new connection, without further
                        # writes, to distinguish transient statement state.
                        extra['read_retry']=snapshot(admin,reader,mixed,leaf_index,key,field,reader_dsn,allow_errors=True)
                        evidence['post_reindex_churn']={'mode':churn,'stages':extra}
                    report['profiles'][profile]=evidence
    finally:
        with psycopg.connect(base,autocommit=True) as root:
            root.execute(sql.SQL('DROP DATABASE {} WITH(FORCE)').format(sql.Identifier(database)))
            if created_role:root.execute(sql.SQL('DROP ROLE {}').format(sql.Identifier(role)))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--atomic',action='store_true',help='Probe same-transaction delete/attach/reindex, including rollback')
    modes.add_argument('--staged',action='store_true',help='Probe committed delete/attach then separate rebuild with rollback/restart')
    modes.add_argument('--churn',choices=('observed','unobserved','custom_cycles'),help='Separately probe owner insert/delete/vacuum after committed reindex')
    args=parser.parse_args()
    report=run(atomic=args.atomic,staged=args.staged,churn=args.churn)
    target=Path(os.environ.get('GMS_PROBE_OUTPUT','/tmp/gms-churn-bm25-report.json' if args.churn else '/tmp/gms-staged-deleted-bm25-report.json' if args.staged else '/tmp/gms-atomic-deleted-bm25-report.json' if args.atomic else '/tmp/gms-deleted-bm25-report.json'))
    target.write_text(json.dumps(report,indent=2,default=str)+'\n')
    print(str(target))
    for profile,evidence in report['profiles'].items():
        baseline=evidence['clean']['force_custom_plan']
        print(profile,{stage:value['force_custom_plan']==baseline for stage,value in evidence.items() if isinstance(value,dict) and 'force_custom_plan' in value})
