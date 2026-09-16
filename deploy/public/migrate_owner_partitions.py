"""Synthetic-only, transactional single-owner heap attachment rehearsal.

This module intentionally refuses production application and populated mixed
owner heaps. Existing scalar numeric IDs become owner-qualified identities.
"""
import argparse
import os
import re

import psycopg
from psycopg import sql
from psycopg.pq import TransactionStatus

from gmail_search.gateway.partitions import (
    PARTITION_SCHEMA, partition_name, partition_binding, inspect_owner_partitions,
    verify_owner_partitions,
)

TABLES = ('messages', 'attachments', 'propositions')
DEPENDENTS = ('embeddings', 'message_topics', 'message_summaries', 'summary_failures', 'prop_processed')
KEYS = {
    'messages': [('PRIMARY KEY', ('user_id', 'id')), ('UNIQUE', ('user_id', 'search_id'))],
    'attachments': [('PRIMARY KEY', ('user_id', 'id')), ('UNIQUE', ('user_id', 'message_id', 'filename')),
                    ('UNIQUE', ('user_id', 'message_id', 'id'))],
    'propositions': [('PRIMARY KEY', ('user_id', 'id'))],
}
SOURCE_KEYS = {
    'messages': {('p', ('user_id','id')), ('u', ('search_id',))},
    'attachments': {('p', ('id',)), ('u', ('user_id','message_id','filename')), ('u', ('user_id','message_id','id'))},
    'propositions': {('p', ('id',))},
}
BM25 = {
    'messages': ('messages_bm25_idx', ('search_id','id','subject','body_text','from_addr','to_addr'), 'search_id'),
    'attachments': ('attachments_bm25_idx', ('id','filename','extracted_text'), 'id'),
    'propositions': ('props_bm25_idx', ('id','text'), 'id'),
}


# Known canonical columns only. Older heaps may lack additive nullable/state
# columns; required keys and BM25 columns are checked separately below.
_COLUMN_TYPES = {
    'messages': {
        **dict.fromkeys(('id','user_id','thread_id','from_addr','to_addr','subject','body_text','body_html','date','labels','raw_json','crawl_blocked_reason'), 'text'),
        'search_id':'int8','history_id':'int8',
    },
    'attachments': {
        **dict.fromkeys(('message_id','user_id','filename','mime_type','extracted_text','image_path','raw_path','fetch_status','embed_status','embed_error'), 'text'),
        'id':'int8','size_bytes':'int8','crawl_attempts':'int4','embed_attempts':'int4','crawl_last_attempt':'timestamptz',
    },
    'propositions': {
        **dict.fromkeys(('user_id','message_id','thread_id','text','model','date'), 'text'),
        'id':'int8','embedding':'bytea','created_at':'timestamptz',
    },
}
_DEFAULTS = {
    'messages': {'subject':"''::text",'body_text':"''::text",'body_html':"''::text",'labels':"'[]'::text",'raw_json':"'{}'::text",'history_id':'0'},
    'attachments': {'size_bytes':'0','crawl_attempts':'0','embed_attempts':'0','fetch_status':"'ok'::text"},
    'propositions': {'created_at':'now()'},
}
_NULLABLE = {
    'messages': {'crawl_blocked_reason'},
    'attachments': {'extracted_text','image_path','raw_path','crawl_last_attempt','embed_status','embed_error'},
    'propositions': {'thread_id','embedding','date','created_at'},
}
_REQUIRED = {
    'messages': {'user_id','search_id','id','thread_id','subject','body_text','from_addr','to_addr'},
    'attachments': {'user_id','message_id','id','filename','mime_type','extracted_text'},
    'propositions': {'user_id','id','message_id','text','model'},
}
_EXPECTED_FKS = {
    ('messages','users',('user_id',),('id',),'c'),
    ('attachments','users',('user_id',),('id',),'c'),
    ('attachments','messages',('user_id','message_id'),('user_id','id'),'a'),
    ('propositions','messages',('user_id','message_id'),('user_id','id'),'c'),
    ('embeddings','attachments',('user_id','message_id','attachment_id'),('user_id','message_id','id'),'a'),
    *((table,'messages',('user_id','message_id'),('user_id','id'),'c' if table in ('summary_failures','prop_processed') else 'a') for table in DEPENDENTS),
}


def _owner_column(conn, relation, column):
    row = conn.execute("""SELECT a.atttypid='pg_catalog.text'::regtype,
        a.attcollation='pg_catalog."default"'::regcollation,c.collisdeterministic,a.attnotnull
        FROM pg_attribute a JOIN pg_collation c ON c.oid=a.attcollation
        WHERE a.attrelid=%s AND a.attname=%s AND NOT a.attisdropped""", (relation,column)).fetchone()
    if row != (True,True,True,True):
        raise ValueError('Unsupported owner comparison collation/type')


def _policy_expression(expression):
    return expression is None or expression == "(user_id = current_setting('app.user_id'::text, true))" or re.fullmatch(
        r"\(user_id = '(?:[^']|'')*'::text\)", expression) is not None


def _ident(schema, name):
    return sql.Identifier(schema, name)


def _require(conn, *, apply=False):
    if conn.info.transaction_status != TransactionStatus.IDLE:
        raise ValueError('Requires idle connection')
    if apply and (conn.info.host != '127.0.0.1' or conn.info.hostaddr != '127.0.0.1' or conn.info.port != 55440
                  or not conn.info.dbname.startswith('gms_owner_partitions_test_')):
        raise ValueError('Apply accepts disposable owner-partition fixture databases only')


def _acl(conn, oid):
    return conn.execute("""SELECT 0::smallint,x.grantee,x.privilege_type,x.is_grantable
        FROM pg_class c CROSS JOIN LATERAL aclexplode(c.relacl) x WHERE c.oid=%s
        UNION ALL SELECT a.attnum,x.grantee,x.privilege_type,x.is_grantable
        FROM pg_attribute a CROSS JOIN LATERAL aclexplode(a.attacl) x WHERE a.attrelid=%s""", (oid,oid)).fetchall()


def _preflight(conn, owner_id, *, readonly=False):
    partition_name('messages', owner_id)
    who = conn.execute("""SELECT session_user=current_user,r.rolsuper,r.oid,d.datdba
        FROM pg_roles r JOIN pg_database d ON d.datname=current_database() WHERE r.rolname=current_user""").fetchone()
    if who[:2] != (True,True) or who[2] != who[3] or conn.execute('SELECT 1 FROM pg_auth_members WHERE roleid=%s',(who[2],)).fetchone():
        raise ValueError('Direct database-owning administrator without members required')
    if conn.info.server_version // 10000 != 16 or conn.execute("SELECT extversion FROM pg_extension WHERE extname='pg_search'").fetchone() != ('0.23.0',):
        raise ValueError('Requires qualified PostgreSQL16/pg_search0.23.0')
    if conn.execute("SELECT current_setting('standard_conforming_strings')").fetchone() != ('on',):
        raise ValueError('Standard conforming strings required')
    if conn.execute("""SELECT 1 FROM pg_default_acl d,LATERAL aclexplode(d.defaclacl) a
        WHERE d.defaclrole=%s AND d.defaclobjtype IN ('r','S','n') AND a.grantee<>%s""", (who[2],who[2])).fetchone():
        raise ValueError('Unsupported default object ACL')
    users = conn.execute("SELECT oid FROM pg_class WHERE oid='public.users'::regclass AND relowner=%s AND relkind='r'",(who[2],)).fetchone()
    if not users:
        raise ValueError('Unsupported owner identity relation')
    _owner_column(conn,users[0],'id')
    if not conn.execute('SELECT 1 FROM public.users WHERE id=%s', (owner_id,)).fetchone():
        raise ValueError('Explicit existing owner required')
    relations = conn.execute("SELECT relname,oid,relkind,relowner,relrowsecurity,relforcerowsecurity,obj_description(oid,'pg_class') FROM pg_class WHERE relnamespace='public'::regnamespace AND relname=ANY(%s)", (list(TABLES),)).fetchall()
    if len(relations) != 3:
        raise ValueError('Missing source tables')
    if all(row[2] == 'p' for row in relations):
        (inspect_owner_partitions if readonly else verify_owner_partitions)(conn, owner_id)
        return {'mode':'already_partitioned'}
    if any(row[2] != 'r' or row[3] != who[2] or not all(row[4:6]) for row in relations):
        raise ValueError('Unsupported relation ownership/type/RLS')
    state = {'mode':'attach_single_owner','tables':{},'owner':who[2]}
    oids = [row[1] for row in relations]
    if conn.execute('SELECT 1 FROM pg_inherits WHERE inhrelid=ANY(%s) OR inhparent=ANY(%s)', (oids,oids)).fetchone():
        raise ValueError('Unexpected inheritance')
    if conn.execute('SELECT 1 FROM pg_trigger WHERE tgrelid=ANY(%s) AND NOT tgisinternal', (oids,)).fetchone():
        raise ValueError('Unexpected trigger')
    if conn.execute('SELECT 1 FROM pg_rewrite WHERE ev_class=ANY(%s)', (oids,)).fetchone():
        raise ValueError('Unexpected rule')
    if conn.execute("SELECT 1 FROM pg_depend WHERE refclassid='pg_class'::regclass AND refobjid=ANY(%s) AND classid='pg_rewrite'::regclass", (oids,)).fetchone():
        raise ValueError('Unexpected view dependency')
    if conn.execute('SELECT 1 FROM pg_publication_rel WHERE prrelid=ANY(%s)', (oids,)).fetchone():
        raise ValueError('Unexpected publication')
    if conn.execute('SELECT 1 FROM pg_publication WHERE puballtables').fetchone():
        raise ValueError('Unsupported all-table publication')
    if conn.execute('SELECT 1 FROM pg_namespace WHERE nspname=%s', (PARTITION_SCHEMA,)).fetchone():
        raise ValueError('Private schema already exists without qualified parents')
    if conn.execute("SELECT 1 FROM pg_class WHERE oid=ANY(%s) AND (relpersistence<>'p' OR reloptions IS NOT NULL OR reloftype<>0)",(oids,)).fetchone():
        raise ValueError('Unsupported heap persistence/options')
    if conn.execute('SELECT 1 FROM pg_statistic_ext WHERE stxrelid=ANY(%s)',(oids,)).fetchone():
        raise ValueError('Unsupported extended statistics')
    if conn.execute("SELECT 1 FROM pg_seclabel WHERE classoid='pg_class'::regclass AND objoid=ANY(%s)",(oids,)).fetchone():
        raise ValueError('Unsupported security label')
    if conn.execute("""SELECT 1 FROM pg_depend d JOIN pg_class c ON c.reltype=d.refobjid
        WHERE c.oid=ANY(%s) AND d.refclassid='pg_type'::regclass AND d.classid='pg_proc'::regclass""",(oids,)).fetchone():
        raise ValueError('Unsupported composite row type dependency')
    for table, oid, _, _, _, _, comment in relations:
        _owner_column(conn,oid,'user_id')
        ident = _ident('public', table)
        if conn.execute(sql.SQL('SELECT 1 FROM {} WHERE user_id IS NULL OR user_id<>%s LIMIT 1').format(ident), (owner_id,)).fetchone():
            raise ValueError('Source contains another owner; heap redistribution refused')
        attrs = conn.execute("""SELECT a.attnum,a.attname,n.nspname,t.typname,a.attnotnull,a.attidentity,a.attgenerated,
            pg_get_expr(d.adbin,d.adrelid) FROM pg_attribute a JOIN pg_type t ON t.oid=a.atttypid
            JOIN pg_namespace n ON n.oid=t.typnamespace LEFT JOIN pg_attrdef d ON d.adrelid=a.attrelid AND d.adnum=a.attnum
            WHERE a.attrelid=%s AND a.attnum>0 AND NOT a.attisdropped ORDER BY a.attnum""", (oid,)).fetchall()
        if any(row[2] != 'pg_catalog' or row[3] not in ('text','int8','int4','bytea','timestamptz') or row[6] for row in attrs):
            raise ValueError('Unsupported column type/generation')
        attrmap = {row[1]:row for row in attrs}
        if not _REQUIRED[table] <= attrmap.keys():
            raise ValueError('Missing canonical columns')
        for row in attrs:
            _, name, _, typ, notnull, identity, _, default = row
            if name not in _COLUMN_TYPES[table] or typ != _COLUMN_TYPES[table][name] or notnull != (name not in _NULLABLE[table]):
                raise ValueError('Unsupported source column profile')
            if identity and (table,name,identity) != ('messages','search_id','d'):
                raise ValueError('Unsupported identity column')
            if name != ('search_id' if table == 'messages' else 'id') and default != _DEFAULTS[table].get(name):
                raise ValueError('Unsupported source default')
        if conn.execute("""SELECT 1 FROM pg_attribute a WHERE a.attrelid=%s AND a.attnum>0
            AND NOT a.attisdropped AND a.attcollation<>0 AND a.attcollation<>'pg_catalog."default"'::regcollation""",(oid,)).fetchone():
            raise ValueError('Unsupported source column collation')
        constraints = conn.execute("""SELECT k.conname,k.contype,k.convalidated,k.condeferrable,
            ARRAY(SELECT a.attname FROM unnest(k.conkey) WITH ORDINALITY x(n,o) JOIN pg_attribute a ON a.attrelid=k.conrelid AND a.attnum=x.n ORDER BY x.o)
            FROM pg_constraint k WHERE k.conrelid=%s AND k.contype<>'f'""", (oid,)).fetchall()
        if any(not valid or deferred or kind not in ('p','u') for _,kind,valid,deferred,_ in constraints):
            raise ValueError('Unsupported source constraint')
        if {(kind,tuple(cols)) for _,kind,_,_,cols in constraints} != SOURCE_KEYS[table]:
            raise ValueError('Unexpected source identity keys')
        index_rows = conn.execute("""SELECT c.relname,c.oid,am.amname,i.indisvalid,i.indisready,i.indisunique,
            i.indexprs IS NOT NULL,i.indnatts=i.indnkeyatts,
            ARRAY(SELECT pg_get_indexdef(i.indexrelid,j,true) FROM generate_series(1,i.indnkeyatts) j),
            pg_get_expr(i.indpred,i.indrelid),c.reloptions,t.spcname,obj_description(c.oid,'pg_class'),
            EXISTS(SELECT 1 FROM pg_constraint k WHERE k.conindid=i.indexrelid)
            FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid JOIN pg_am am ON am.oid=c.relam
            LEFT JOIN pg_tablespace t ON t.oid=c.reltablespace WHERE i.indrelid=%s""", (oid,)).fetchall()
        indexes = []
        for index in index_rows:
            name, _, method, valid, ready, unique, expression, noinclude, fields, predicate, opts, space, icomment, constraint = index
            if not valid or not ready or expression or not noinclude or method not in ('btree','bm25') or unique and not constraint:
                raise ValueError('Unsupported index dependency')
            if any(not re.fullmatch(r'[a-z_]+(?: DESC)?',field) or field.split()[0] not in attrmap for field in fields):
                raise ValueError('Unsupported index ordering/operator class')
            if predicate not in (None, "(fetch_status <> 'ok'::text)", "((mime_type = 'text/html'::text) AND (extracted_text IS NULL))", "(labels ~~ '%\"INBOX\"%'::text)"):
                raise ValueError('Unsupported index predicate')
            if constraint:
                continue
            if method == 'bm25' and (name != BM25[table][0] or tuple(fields) != BM25[table][1] or predicate
                                    or dict(item.split('=',1) for item in (opts or [])) != {'key_field':BM25[table][2]}):
                raise ValueError('Unsupported BM25 profile')
            indexes.append(index)
        if sum(index[2] == 'bm25' for index in indexes) != 1:
            raise ValueError('Missing canonical BM25 index')
        policies = conn.execute("SELECT polname,polpermissive,polcmd,polroles,pg_get_expr(polqual,polrelid),pg_get_expr(polwithcheck,polrelid) FROM pg_policy WHERE polrelid=%s", (oid,)).fetchall()
        if any(not _policy_expression(using) or not _policy_expression(check) for _,_,_,_,using,check in policies):
            raise ValueError('Unsupported policy expression/dependency')
        acl = _acl(conn, oid)
        if any(grantable and grantee != who[2] for _,grantee,_,grantable in acl):
            raise ValueError('Unsupported delegated ACL')
        key = 'search_id' if table == 'messages' else 'id'
        sequence = conn.execute('SELECT pg_get_serial_sequence(%s,%s)', ('public.'+table,key)).fetchone()[0]
        if not sequence or attrmap[key][3] != 'int8' or attrmap[key][5] != ('d' if table == 'messages' else ''):
            raise ValueError('Unexpected numeric allocation default')
        seq_oid, seq_schema, seq_name = conn.execute('SELECT c.oid,n.nspname,c.relname FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace WHERE c.oid=%s::regclass', (sequence,)).fetchone()
        if seq_schema != 'public' or seq_name != table+'_'+key+'_seq':
            raise ValueError('Unsupported sequence schema/name')
        seq = conn.execute("""SELECT c.relowner,c.relpersistence,s.seqtypid='int8'::regtype,s.seqstart,s.seqincrement,s.seqmax,s.seqmin,s.seqcache,s.seqcycle
            FROM pg_class c JOIN pg_sequence s ON s.seqrelid=c.oid WHERE c.oid=%s""",(seq_oid,)).fetchone()
        if seq != (who[2],'p',True,1,1,9223372036854775807,1,1,False):
            raise ValueError('Unsupported sequence parameters/owner')
        if any(grantee != who[2] for _,grantee,_,_ in _acl(conn,seq_oid)):
            raise ValueError('Unsupported sequence ACL')
        if conn.execute("""SELECT 1 FROM pg_depend d WHERE d.refclassid='pg_class'::regclass AND d.refobjid=%s
            AND NOT (d.classid='pg_attrdef'::regclass AND d.objid IN
                (SELECT oid FROM pg_attrdef WHERE adrelid=%s AND adnum=%s))""",(seq_oid,oid,attrmap[key][0])).fetchone():
            raise ValueError('Unsupported external sequence dependency')
        if conn.execute("""SELECT classid='pg_class'::regclass,refclassid='pg_class'::regclass,refobjid,refobjsubid,deptype
            FROM pg_depend WHERE classid='pg_class'::regclass AND objid=%s AND deptype IN ('a','i')""",(seq_oid,)).fetchall() != [(True,True,oid,attrmap[key][0],'i' if table=='messages' else 'a')]:
            raise ValueError('Unsupported sequence ownership dependency')
        expected_default = None if table=='messages' else sql.SQL('nextval({}::regclass)').format(sql.Literal('public.'+seq_name)).as_string(conn)
        if attrmap[key][7] != expected_default:
            raise ValueError('Unsupported numeric allocation default')
        last, called = conn.execute(sql.SQL('SELECT last_value,is_called FROM {}').format(_ident(seq_schema,seq_name))).fetchone()
        maximum,row_count = conn.execute(sql.SQL('SELECT max({}),count(*) FROM {}').format(sql.Identifier(key), ident)).fetchone()
        next_value = max(last + int(called), (maximum + 1) if maximum is not None else 1)
        if next_value > 9223372036854775807:
            raise ValueError('Sequence exhausted')
        state['tables'][table] = dict(oid=oid,attrs=attrs,constraints=constraints,indexes=indexes,policies=policies,acl=acl,comment=comment,
            sequence=(seq_oid,seq_name,last,called,maximum),key=key,row_count=row_count,
            sequence_comment=conn.execute("SELECT obj_description(%s,'pg_class')",(seq_oid,)).fetchone()[0])
    fks = conn.execute("""SELECT n.nspname,c.relname,k.conname,rn.nspname,r.relname,pg_get_constraintdef(k.oid,true),
        k.convalidated,k.condeferrable,k.confupdtype,k.confdeltype,k.confmatchtype,
        ARRAY(SELECT a.attname FROM unnest(k.conkey) WITH ORDINALITY x(n,o) JOIN pg_attribute a ON a.attrelid=k.conrelid AND a.attnum=x.n ORDER BY x.o),
        ARRAY(SELECT a.attname FROM unnest(k.confkey) WITH ORDINALITY x(n,o) JOIN pg_attribute a ON a.attrelid=k.confrelid AND a.attnum=x.n ORDER BY x.o)
        FROM pg_constraint k JOIN pg_class c ON c.oid=k.conrelid JOIN pg_namespace n ON n.oid=c.relnamespace
        JOIN pg_class r ON r.oid=k.confrelid JOIN pg_namespace rn ON rn.oid=r.relnamespace
        WHERE k.contype='f' AND (k.conrelid=ANY(%s) OR k.confrelid=ANY(%s))""", (oids,oids)).fetchall()
    actual_fks = set()
    for schema,table,_,remote_schema,remote,_,valid,deferred,update,delete,match,cols,rcols in fks:
        identity = (table,remote,tuple(cols),tuple(rcols),delete)
        optional = ('propositions','users',('user_id',),('id',),'c')
        if schema != 'public' or remote_schema != 'public' or identity not in _EXPECTED_FKS | {optional} or not valid or deferred or update != 'a' or match != 's' or identity in actual_fks:
            raise ValueError('Unsupported foreign-key dependency')
        actual_fks.add(identity)
    if not _EXPECTED_FKS <= actual_fks:
        raise ValueError('Missing required foreign-key dependency')
    # Moving a heap preserves its OID. Any unhandled stored dependency would
    # therefore keep reading the bootstrap leaf instead of the logical parent.
    # Permit only objects inventoried above (plus PostgreSQL's own row/TOAST
    # machinery), rather than just searching for familiar view dependencies.
    related = oids + [data['sequence'][0] for data in state['tables'].values()]
    related += [row[0] for row in conn.execute('SELECT indexrelid FROM pg_index WHERE indrelid=ANY(%s)',(oids,)).fetchall()]
    allowed = {
        'pg_class': set(related) | {row[0] for row in conn.execute('SELECT reltoastrelid FROM pg_class WHERE oid=ANY(%s)',(oids,)).fetchall()},
        'pg_attrdef': {row[0] for row in conn.execute('SELECT oid FROM pg_attrdef WHERE adrelid=ANY(%s)',(oids,)).fetchall()},
        'pg_constraint': {row[0] for row in conn.execute('SELECT oid FROM pg_constraint WHERE conrelid=ANY(%s) OR (contype=\'f\' AND confrelid=ANY(%s))',(oids,oids)).fetchall()},
        'pg_policy': {row[0] for row in conn.execute('SELECT oid FROM pg_policy WHERE polrelid=ANY(%s)',(oids,)).fetchall()},
        'pg_type': {row[0] for row in conn.execute('SELECT reltype FROM pg_class WHERE oid=ANY(%s)',(oids,)).fetchall()},
    }
    dependencies = conn.execute("""SELECT classid::regclass::text,objid FROM pg_depend
        WHERE refclassid='pg_class'::regclass AND refobjid=ANY(%s)""",(related,)).fetchall()
    if any(objid not in allowed.get(kind,set()) for kind,objid in dependencies):
        raise ValueError('Unsupported OID-bound relation dependency')
    row_types = list(allowed['pg_type'])
    arrays = [row[0] for row in conn.execute('SELECT typarray FROM pg_type WHERE oid=ANY(%s)',(row_types,)).fetchall()]
    if conn.execute("""SELECT 1 FROM pg_depend WHERE refclassid='pg_type'::regclass AND refobjid=ANY(%s)
        AND NOT (classid='pg_type'::regclass AND objid=ANY(%s) AND deptype='i')""",(row_types+arrays,arrays)).fetchone():
        raise ValueError('Unsupported OID-bound row type dependency')
    if conn.execute("""SELECT 1 FROM pg_depend WHERE classid='pg_class'::regclass AND objid=ANY(%s)
        AND deptype IN ('e','x')""",(related,)).fetchone():
        raise ValueError('Unsupported extension membership')
    state['fks'] = fks
    return state


def preview_owner_partitions(conn, *, owner_id):
    _require(conn)
    with conn.transaction():
        conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        conn.execute('SET LOCAL search_path=pg_catalog')
        conn.execute("SET LOCAL statement_timeout='60s'")
        conn.execute("SET LOCAL lock_timeout='2s'")
        state = _preflight(conn,owner_id,readonly=True)
        inventory = []
        for table, data in state.get('tables',{}).items():
            oid, node, heap_bytes, toast_bytes = conn.execute("""SELECT oid,relfilenode,pg_relation_size(oid),
                CASE WHEN reltoastrelid=0 THEN 0 ELSE pg_total_relation_size(reltoastrelid) END
                FROM pg_class WHERE oid=%s""",(data['oid'],)).fetchone()
            indexes = conn.execute("""SELECT c.relname,c.oid,c.relfilenode,a.amname,pg_relation_size(c.oid)
                FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid JOIN pg_am a ON a.oid=c.relam
                WHERE i.indrelid=%s ORDER BY c.relname""",(oid,)).fetchall()
            inventory.append({'table':table,'row_count':data['row_count'],'heap_oid':oid,'heap_file':node,
                'heap_bytes':heap_bytes,'toast_bytes':toast_bytes,'indexes':indexes,
                'sequence':{'oid':data['sequence'][0],'name':data['sequence'][1],
                    'last_value':data['sequence'][2],'is_called':data['sequence'][3],'max_stored_id':data['sequence'][4]}})
        return {'mode':state['mode'],'owner_id':owner_id,'tables':list(TABLES),'source_inventory':inventory}


def _role(conn, oid):
    return sql.SQL('PUBLIC') if oid == 0 else sql.Identifier(conn.execute('SELECT rolname FROM pg_roles WHERE oid=%s',(oid,)).fetchone()[0])


def migrate_owner_partitions(conn, *, owner_id, _checkpoint=lambda stage: None):
    _require(conn, apply=True)
    with conn.transaction():
        conn.execute('SET LOCAL search_path=pg_catalog')
        conn.execute("SET LOCAL lock_timeout='2s'")
        conn.execute("SET LOCAL statement_timeout='60s'")
        if not conn.execute('SELECT pg_try_advisory_xact_lock(72341629,2)').fetchone()[0]:
            raise ValueError('Partition migration already running')
        # Match provisioning's owner -> catalog -> relation lock order.
        partition_name('messages',owner_id)
        for lock in ('gmail-search partition owner '+owner_id,'gmail-search partition catalog v1'):
            conn.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s,0))',(lock,))
        for table in sorted((*TABLES,*DEPENDENTS,'users')):
            conn.execute(sql.SQL('LOCK TABLE {} IN ACCESS EXCLUSIVE MODE').format(_ident('public',table)))
        state = _preflight(conn,owner_id)
        if state['mode'] == 'already_partitioned':
            return False
        conn.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(PARTITION_SCHEMA)))
        conn.execute(sql.SQL('REVOKE ALL ON SCHEMA {} FROM PUBLIC').format(sql.Identifier(PARTITION_SCHEMA)))
        for schema,table,name,*_ in state['fks']:
            conn.execute(sql.SQL('ALTER TABLE {} DROP CONSTRAINT {}').format(_ident(schema,table),sql.Identifier(name)))
        for table, data in state['tables'].items():
            child_name = partition_name(table,owner_id)
            conn.execute(sql.SQL('ALTER TABLE {} SET SCHEMA {}').format(_ident('public',table),sql.Identifier(PARTITION_SCHEMA)))
            conn.execute(sql.SQL('ALTER TABLE {} RENAME TO {}').format(_ident(PARTITION_SCHEMA,table),sql.Identifier(child_name)))
        _checkpoint('renamed')
        for table, data in state['tables'].items():
            child = _ident(PARTITION_SCHEMA,partition_name(table,owner_id))
            parent = _ident('public',table)
            conn.execute(sql.SQL('CREATE TABLE {} (LIKE {} INCLUDING DEFAULTS INCLUDING IDENTITY INCLUDING STORAGE INCLUDING COMMENTS) PARTITION BY LIST(user_id)').format(parent,child))
            for name,kind,_,_,columns in data['constraints']:
                if (kind,tuple(columns)) not in {(('p' if kind=='PRIMARY KEY' else 'u'),cols) for kind,cols in KEYS[table]}:
                    conn.execute(sql.SQL('ALTER TABLE {} DROP CONSTRAINT {}').format(child,sql.Identifier(name)))
            for kind,columns in KEYS[table]:
                conn.execute(sql.SQL('ALTER TABLE {} ADD {} ({})').format(parent,sql.SQL(kind),sql.SQL(',').join(map(sql.Identifier,columns))))
            for index in data['indexes']:
                name,_,method,_,_,_,_,_,fields,predicate,opts,space,comment,_ = index
                statement = sql.SQL('CREATE INDEX {} ON ONLY {} USING {} ({})').format(sql.Identifier(name),parent,sql.Identifier(method),sql.SQL(',').join(sql.SQL(field) for field in fields))
                if opts:
                    statement += sql.SQL(' WITH ({})').format(sql.SQL(',').join(sql.SQL('{}={}').format(sql.Identifier(k),sql.Literal(v)) for k,v in (item.split('=',1) for item in opts)))
                if space:
                    statement += sql.SQL(' TABLESPACE {}').format(sql.Identifier(space))
                if predicate:
                    statement += sql.SQL(' WHERE ') + sql.SQL(predicate)
                conn.execute(statement)
                if comment is not None:
                    conn.execute(sql.SQL('COMMENT ON INDEX {} IS {}').format(_ident('public',name),sql.Literal(comment)))
            seq_oid,seq_name,last,called,maximum = data['sequence']
            if table == 'messages':
                sequence = conn.execute('SELECT pg_get_serial_sequence(%s,%s)',('public.'+table,data['key'])).fetchone()[0]
                next_value = max(last + int(called), (maximum + 1) if maximum is not None else 1)
                conn.execute('SELECT setval(%s::regclass,%s,false)',(sequence,next_value))
                conn.execute(sql.SQL('ALTER TABLE {} ALTER COLUMN {} DROP IDENTITY').format(child,sql.Identifier(data['key'])))
                if data['sequence_comment'] is not None:
                    conn.execute(sql.SQL('COMMENT ON SEQUENCE {} IS {}').format(_ident('public',seq_name),sql.Literal(data['sequence_comment'])))
            else:
                conn.execute(sql.SQL('ALTER SEQUENCE {} OWNED BY NONE').format(_ident(PARTITION_SCHEMA,seq_name)))
                conn.execute(sql.SQL('ALTER SEQUENCE {} SET SCHEMA public').format(_ident(PARTITION_SCHEMA,seq_name)))
                conn.execute(sql.SQL('ALTER SEQUENCE {} OWNED BY {}.{}').format(_ident('public',seq_name),parent,sql.Identifier(data['key'])))
                # ALTER RESTART is transactional; setval on the old sequence is not.
                next_value = max(last + int(called), (maximum + 1) if maximum is not None else 1)
                conn.execute(sql.SQL('ALTER SEQUENCE {} RESTART WITH {}').format(_ident('public',seq_name),sql.Literal(next_value)))
            conn.execute(sql.SQL('ALTER TABLE {} ADD CONSTRAINT gms_owner_bound CHECK(user_id={})').format(child,sql.Literal(owner_id)))
            conn.execute(sql.SQL('ALTER TABLE {} ATTACH PARTITION {} FOR VALUES IN ({})').format(parent,child,sql.Literal(owner_id)))
            conn.execute(sql.SQL('ALTER TABLE {} ENABLE ROW LEVEL SECURITY').format(parent))
            conn.execute(sql.SQL('ALTER TABLE {} FORCE ROW LEVEL SECURITY').format(parent))
            for name,permissive,command,roles,using,check in data['policies']:
                statement = sql.SQL('CREATE POLICY {} ON {} AS {} FOR {} TO {}').format(sql.Identifier(name),parent,
                    sql.SQL('PERMISSIVE' if permissive else 'RESTRICTIVE'),sql.SQL({'*':'ALL','r':'SELECT','a':'INSERT','w':'UPDATE','d':'DELETE'}[command]),
                    sql.SQL(',').join(_role(conn,role) for role in roles))
                if using:
                    statement += sql.SQL(' USING ({})').format(sql.SQL(using))
                if check:
                    statement += sql.SQL(' WITH CHECK ({})').format(sql.SQL(check))
                conn.execute(statement)
            for attnum,grantee,privilege,_ in data['acl']:
                if grantee == state['owner']:
                    continue
                role = _role(conn,grantee)
                priv = sql.SQL(privilege)
                if attnum:
                    column = next(row[1] for row in data['attrs'] if row[0] == attnum)
                    priv += sql.SQL('({})').format(sql.Identifier(column))
                conn.execute(sql.SQL('REVOKE {} ON {} FROM {}').format(priv,child,role))
                conn.execute(sql.SQL('GRANT {} ON {} TO {}').format(priv,parent,role))
            if data['comment'] is not None:
                conn.execute(sql.SQL('COMMENT ON TABLE {} IS {}').format(parent,sql.Literal(data['comment'])))
            conn.execute(sql.SQL('COMMENT ON TABLE {} IS {}').format(child,sql.Literal(partition_binding(table,owner_id))))
        for schema,table,name,_,_,definition,*_ in state['fks']:
            conn.execute(sql.SQL('ALTER TABLE {} ADD CONSTRAINT {} {}').format(_ident(schema,table),sql.Identifier(name),sql.SQL(definition)))
        _checkpoint('attached')
        verify_owner_partitions(conn,owner_id)
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--owner-id',required=True)
    parser.add_argument('--apply',action='store_true')
    args = parser.parse_args()
    try:
        with psycopg.connect(os.environ['OWNER_PARTITION_MIGRATION_DSN'],autocommit=True) as conn:
            result = migrate_owner_partitions(conn,owner_id=args.owner_id) if args.apply else preview_owner_partitions(conn,owner_id=args.owner_id)
    except Exception:
        parser.exit(1,'Owner partition migration refused or failed. No successful conversion reported.\n')
    print(result)


if __name__ == '__main__':
    main()
