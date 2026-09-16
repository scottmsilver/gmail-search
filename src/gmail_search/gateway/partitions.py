"""Trusted owner-partition layout shared by migration and admission tooling.

Names identify candidate objects only. Administrator provisioning must also
verify catalog ancestry, exact owner bounds, indexes and access controls.
"""
from contextlib import contextmanager
import hashlib
import json

from psycopg import sql

from .partition_profiles import NUMERIC_OWNER_PARTITIONS_V1 as NUMERIC, TEXT_OWNER_PARTITIONS_V1 as TEXT, require_profile

PARTITION_SCHEMA = 'gms_mail_partitions'
_PREFIXES = {'messages': 'm', 'attachments': 'a', 'propositions': 'p'}


def partition_name(table: str, owner_id: str) -> str:
    if not isinstance(table, str) or table not in _PREFIXES:
        raise ValueError('Unsupported partitioned mail table')
    if not isinstance(owner_id, str) or not owner_id or len(owner_id) > 2048 or '\x00' in owner_id:
        raise ValueError('A valid stable owner ID is required')
    try:
        digest = hashlib.sha256(owner_id.encode('utf-8')).hexdigest()[:48]
    except UnicodeEncodeError:
        raise ValueError('A valid stable owner ID is required') from None
    return 'gms_' + _PREFIXES[table] + '_' + digest


# The first reviewed profile is deliberately narrow: PostgreSQL 16 / pg_search
# 0.23, direct superuser administration, canonical default BM25 definitions.
# Noncanonical index options require a separately reviewed profile.
_INDEXES = NUMERIC.indexes  # Numeric migration compatibility.
_KEYS = NUMERIC.keys
_FKS = {
    'messages': {(('user_id',),'public.users',('id',),'c')},
    'attachments': {(('user_id',),'public.users',('id',),'c'), (('user_id','message_id'),'public.messages',('user_id','id'),'a')},
    'propositions': {(('user_id','message_id'),'public.messages',('user_id','id'),'c')},
}


def partition_binding(table: str, owner_id: str) -> str:
    """Canonical administrator metadata; a comment alone never proves a bound."""
    partition_name(table, owner_id)
    return 'gmail-search owner partition v1 ' + json.dumps(
        {'owner_id': owner_id, 'table': table}, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def _administrator(conn):
    row = conn.execute('''SELECT r.oid,r.rolsuper,session_user=current_user,d.datdba
        FROM pg_catalog.pg_roles r JOIN pg_catalog.pg_database d ON d.datname=pg_catalog.current_database()
        WHERE r.rolname=current_user''').fetchone()
    if not row or not row[1] or not row[2] or row[0] != row[3]:
        raise ValueError('Direct database-owning superuser administration required')
    if conn.execute('SELECT 1 FROM pg_catalog.pg_auth_members WHERE roleid=%s', (row[0],)).fetchone():
        raise ValueError('Administrator owner role must have no members')
    if conn.info.server_version // 10000 != 16 or conn.execute(
        "SELECT extversion FROM pg_catalog.pg_extension WHERE extname='pg_search'"
    ).fetchone() != ('0.23.0',):
        raise ValueError('Unsupported partition search version')
    return row[0]


def _private_acl(conn, owner, namespace):
    if namespace is None:
        return
    if conn.execute('SELECT 1 FROM pg_catalog.pg_namespace WHERE oid=%s AND nspowner<>%s', (namespace,owner)).fetchone():
        raise ValueError('Unexpected private schema owner')
    if conn.execute('SELECT 1 FROM pg_catalog.pg_class WHERE relnamespace=%s AND relowner<>%s', (namespace,owner)).fetchone():
        raise ValueError('Unexpected private relation owner')
    if conn.execute('''SELECT 1 FROM pg_catalog.pg_namespace n,
        LATERAL pg_catalog.aclexplode(coalesce(n.nspacl,pg_catalog.acldefault('n',n.nspowner))) a
        WHERE n.oid=%s AND (n.nspowner<>%s OR a.grantee<>%s)''', (namespace, owner, owner)).fetchone():
        raise ValueError('Unexpected private schema owner or grant')
    if conn.execute('''SELECT 1 FROM pg_catalog.pg_class c,
        LATERAL pg_catalog.aclexplode(coalesce(c.relacl,pg_catalog.acldefault(
            CASE WHEN c.relkind='S' THEN 'S'::"char" ELSE 'r'::"char" END,c.relowner))) a
        WHERE c.relnamespace=%s AND (c.relowner<>%s OR a.grantee<>%s)''', (namespace, owner, owner)).fetchone():
        raise ValueError('Unexpected private relation owner or grant')
    if conn.execute('''SELECT 1 FROM pg_catalog.pg_attribute c
        JOIN pg_catalog.pg_class t ON t.oid=c.attrelid,
        LATERAL pg_catalog.aclexplode(c.attacl) a
        WHERE t.relnamespace=%s AND a.grantee<>%s''', (namespace, owner)).fetchone():
        raise ValueError('Unexpected private column grant')


def _default_acl(conn, owner, namespace):
    if conn.execute('''SELECT 1 FROM pg_catalog.pg_default_acl d,
        LATERAL pg_catalog.aclexplode(d.defaclacl) a
        WHERE d.defaclrole=%s AND (d.defaclnamespace=0 OR d.defaclnamespace=%s)
        AND d.defaclobjtype IN ('r','S','n') AND a.grantee<>%s''', (owner, namespace or 0, owner)).fetchone():
        raise ValueError('Unexpected default private object grant')


def _index(conn, oid):
    return conn.execute('''SELECT c.oid,c.relkind,c.relowner,c.relpersistence,a.amname,
        i.indisvalid,i.indisready,i.indislive,i.indisunique,i.indisprimary,
        i.indnkeyatts,i.indnatts,i.indexprs IS NULL,i.indpred IS NULL,
        ARRAY(SELECT x.attname FROM pg_catalog.unnest(i.indkey) WITH ORDINALITY k(attnum,n)
            JOIN pg_catalog.pg_attribute x ON x.attrelid=i.indrelid AND x.attnum=k.attnum ORDER BY k.n),
        i.indclass::oid[],i.indcollation::oid[],i.indoption::smallint[],c.reloptions,i.indrelid
        FROM pg_catalog.pg_class c JOIN pg_catalog.pg_index i ON i.indexrelid=c.oid
        JOIN pg_catalog.pg_am a ON a.oid=c.relam WHERE c.oid=%s''', (oid,)).fetchone()


def _parent_index(conn, table, parent, owner, profile=NUMERIC):
    name, fields, key = profile.indexes[table]
    rows = conn.execute('''SELECT i.indexrelid FROM pg_catalog.pg_index i
        JOIN pg_catalog.pg_class c ON c.oid=i.indexrelid JOIN pg_catalog.pg_am a ON a.oid=c.relam
        WHERE i.indrelid=%s AND a.amname='bm25' ''', (parent,)).fetchall()
    named = conn.execute('SELECT pg_catalog.to_regclass(%s)::oid', ('public.' + name,)).fetchone()[0]
    if len(rows) != 1 or named != rows[0][0]:
        raise ValueError('Missing or unexpected parent BM25 index: ' + table)
    index = _index(conn, named)
    if (index[1:5] != ('I',owner,'p','bm25') or index[5:10] != (True,True,True,False,False)
            or index[10:14] != (len(fields),len(fields),True,True)
            or tuple(index[14]) != fields or any(index[17])
            or sorted(index[18] or []) != ['key_field=' + key]):
        raise ValueError('Unsupported parent BM25 definition: ' + table)
    if conn.execute('''SELECT 1 FROM pg_catalog.unnest(%s::oid[]) x(oid)
        JOIN pg_catalog.pg_opclass c ON c.oid=x.oid WHERE NOT c.opcdefault''', (index[15],)).fetchone():
        raise ValueError('Unsupported BM25 operator class')
    return index


def _constraints(conn, table, oid, profile=NUMERIC):
    constraints = conn.execute('''SELECT c.contype,
        ARRAY(SELECT a.attname FROM pg_catalog.unnest(c.conkey) WITH ORDINALITY k(num,n)
            JOIN pg_catalog.pg_attribute a ON a.attrelid=c.conrelid AND a.attnum=k.num ORDER BY k.n),
        n.nspname||'.'||t.relname,
        ARRAY(SELECT a.attname FROM pg_catalog.unnest(c.confkey) WITH ORDINALITY k(num,n)
            JOIN pg_catalog.pg_attribute a ON a.attrelid=c.confrelid AND a.attnum=k.num ORDER BY k.n),
        c.confdeltype,c.confupdtype,c.confmatchtype,c.condeferrable,c.convalidated
        FROM pg_catalog.pg_constraint c LEFT JOIN pg_catalog.pg_class t ON t.oid=c.confrelid
        LEFT JOIN pg_catalog.pg_namespace n ON n.oid=t.relnamespace
        WHERE c.conrelid=%s AND c.conparentid=0 AND c.contype IN ('p','u','f')''', (oid,)).fetchall()
    keys = {(c[0],tuple(c[1])) for c in constraints if c[0] in ('p','u')}
    if keys != profile.keys[table] or any(c[7] or not c[8] for c in constraints):
        raise ValueError('Unsupported owner-qualified parent keys: ' + table)
    fks = {(tuple(c[1]),c[2],tuple(c[3]),c[4]) for c in constraints if c[0]=='f'}
    expected = _FKS[table]
    optional = {(('user_id',),'public.users',('id',),'c')} if table=='propositions' else set()
    if not expected <= fks or not fks <= expected | optional or any(c[5:7] != ('a','s') for c in constraints if c[0]=='f'):
        raise ValueError('Unsupported owner-qualified parent foreign keys: ' + table)



def _owner_comparison(conn, relation, column):
    row = conn.execute('''SELECT a.atttypid='pg_catalog.text'::regtype,
        a.attcollation='pg_catalog."default"'::regcollation,c.collisdeterministic,a.attnotnull
        FROM pg_catalog.pg_attribute a JOIN pg_catalog.pg_collation c ON c.oid=a.attcollation
        WHERE a.attrelid=%s AND a.attname=%s AND NOT a.attisdropped''', (relation,column)).fetchone()
    if row != (True,True,True,True):
        raise ValueError('Unsupported owner comparison type or collation')


@contextmanager
def _operation(conn):
    # Savepoints restore caller settings on failure. On success restore them
    # explicitly, since RELEASE SAVEPOINT preserves SET LOCAL changes.
    with conn.transaction():
        original = conn.execute("SELECT name,pg_catalog.current_setting(name),setting::int FROM pg_catalog.pg_settings WHERE name IN ('lock_timeout','statement_timeout')").fetchall()
        for name, _, milliseconds in original:
            ceiling = 2000 if name == 'lock_timeout' else 30000
            bounded = min(milliseconds,ceiling) if milliseconds else ceiling
            conn.execute('SELECT pg_catalog.set_config(%s,%s,true)', (name,str(bounded)+'ms'))
        yield
        for name, value, _ in original:
            conn.execute('SELECT pg_catalog.set_config(%s,%s,true)', (name,value))

def _parents(conn, owner, profile=NUMERIC):
    parents = {}
    for table in _PREFIXES:
        row = conn.execute('''SELECT c.oid,c.relkind,c.relowner,c.relpersistence,c.relrowsecurity,c.relforcerowsecurity,
            c.relispartition,p.partstrat,p.partnatts,p.partexprs IS NULL,
            pg_catalog.pg_get_partkeydef(c.oid),a.atttypid='pg_catalog.text'::regtype,a.attnotnull
            FROM pg_catalog.pg_class c JOIN pg_catalog.pg_partitioned_table p ON p.partrelid=c.oid
            JOIN pg_catalog.pg_attribute a ON a.attrelid=c.oid AND a.attname='user_id' AND NOT a.attisdropped
            WHERE c.oid=pg_catalog.to_regclass(%s)''', ('public.'+table,)).fetchone()
        if not row or row[1:] != ('p',owner,'p',True,True,False,'l',1,True,'LIST (user_id)',True,True):
            raise ValueError('Unsupported owner LIST parent or RLS: ' + table)
        _owner_comparison(conn,row[0],'user_id')
        if not conn.execute("""SELECT p.partcollation[0]='pg_catalog.\"default\"'::regcollation::oid
            AND p.partclass[0]=(SELECT o.oid FROM pg_catalog.pg_opclass o
                JOIN pg_catalog.pg_namespace n ON n.oid=o.opcnamespace
                JOIN pg_catalog.pg_am a ON a.oid=o.opcmethod
                WHERE n.nspname='pg_catalog' AND o.opcname='text_ops' AND o.opcdefault AND a.amname='btree')
            FROM pg_catalog.pg_partitioned_table p WHERE p.partrelid=%s""", (row[0],)).fetchone()[0]:
            raise ValueError('Unsupported owner partition comparison collation or operator class')
        # Key identity is a fixed profile contract, not merely a matching index
        # option. TEXT deliberately excludes a hidden numeric allocator column.
        columns = dict(conn.execute('''SELECT a.attname,t.typname FROM pg_catalog.pg_attribute a
            JOIN pg_catalog.pg_type t ON t.oid=a.atttypid JOIN pg_catalog.pg_namespace n ON n.oid=t.typnamespace
            WHERE a.attrelid=%s AND a.attnum>0 AND NOT a.attisdropped AND n.nspname='pg_catalog' ''', (row[0],)).fetchall())
        if columns.get('id') != ('text' if table == 'messages' else 'int8'):
            raise ValueError('Unsupported partition key column type')
        if table == 'messages':
            has_search_id = conn.execute("SELECT 1 FROM pg_catalog.pg_attribute WHERE attrelid=%s AND attname='search_id' AND NOT attisdropped", (row[0],)).fetchone()
            if (profile is TEXT and has_search_id) or (profile is NUMERIC and columns.get('search_id') != 'int8'):
                raise ValueError('Partition schema profile mismatch')
        _constraints(conn, table, row[0], profile)
        parents[table] = (row[0], _parent_index(conn, table, row[0], owner, profile))
    return parents


def _children(conn, parents, owner, namespace):
    found = set()
    for table, (parent, index) in parents.items():
        rows = conn.execute('''SELECT c.oid,c.relnamespace,c.relname,c.relkind,c.relowner,c.relpersistence,
            c.relrowsecurity,c.relforcerowsecurity,c.relispartition,
            pg_catalog.pg_get_expr(c.relpartbound,c.oid),pg_catalog.obj_description(c.oid,'pg_class')
            FROM pg_catalog.pg_inherits i JOIN pg_catalog.pg_class c ON c.oid=i.inhrelid
            WHERE i.inhparent=%s''', (parent,)).fetchall()
        for row in rows:
            try:
                comment = row[10]
                binding = json.loads(comment.removeprefix('gmail-search owner partition v1 '))
                bound_owner = binding['owner_id']
                valid_binding = comment == partition_binding(table, bound_owner)
            except (TypeError,ValueError,KeyError,AttributeError):
                raise ValueError('Invalid partition binding metadata') from None
            # pg_get_expr emits a plain text constant with doubled quotes under
            # standard_conforming_strings=on, unlike format(%L), which may use E.
            # This is compared as data only; it is never executed as SQL.
            expected_bound = "FOR VALUES IN ('" + bound_owner.replace("'", "''") + "')"
            if (not valid_binding or row[1:9] != (namespace,partition_name(table,bound_owner),'r',owner,'p',True,True,True)
                    or row[9] != expected_bound):
                raise ValueError('Invalid single-owner partition catalog binding')
            if not conn.execute('SELECT 1 FROM public.users WHERE id=%s', (bound_owner,)).fetchone():
                raise ValueError('Partition owner does not exist')
            _owner_comparison(conn,row[0],'user_id')
            ancestry = conn.execute('SELECT inhparent FROM pg_catalog.pg_inherits WHERE inhrelid=%s', (row[0],)).fetchall()
            if ancestry != [(parent,)] or conn.execute('SELECT 1 FROM pg_catalog.pg_inherits WHERE inhparent=%s', (row[0],)).fetchone():
                raise ValueError('Unexpected partition ancestry or descendants')
            child_indexes = conn.execute('''SELECT i.indexrelid FROM pg_catalog.pg_index i
                JOIN pg_catalog.pg_class c ON c.oid=i.indexrelid JOIN pg_catalog.pg_am a ON a.oid=c.relam
                WHERE i.indrelid=%s AND a.amname='bm25' ''', (row[0],)).fetchall()
            if len(child_indexes) != 1:
                raise ValueError('Missing partition BM25 index')
            child_index = _index(conn, child_indexes[0][0])
            ancestry = conn.execute('SELECT inhparent FROM pg_catalog.pg_inherits WHERE inhrelid=%s', (child_index[0],)).fetchall()
            if (child_index[1]!='i' or child_index[2:19]!=index[2:19] or ancestry!=[(index[0],)]):
                raise ValueError('Invalid partition BM25 definition or ancestry')
            found.add((table,bound_owner))
    if namespace is not None:
        actual = set(conn.execute("SELECT relname FROM pg_catalog.pg_class WHERE relnamespace=%s AND relkind IN ('r','p','v','m','f')", (namespace,)).fetchall())
        expected = {(partition_name(table,bound_owner),) for table,bound_owner in found}
        if actual != expected:
            raise ValueError('Unexpected detached or unrelated private relation')
    return found


def _prepare(conn, owner_id, *, create, lock=True, profile=NUMERIC):
    require_profile(profile)
    partition_name('messages', owner_id)
    owner = _administrator(conn)
    if conn.execute("SELECT pg_catalog.current_setting('standard_conforming_strings')").fetchone() != ('on',):
        raise ValueError('Standard conforming strings required for canonical bound verification')
    users = conn.execute("SELECT oid FROM pg_catalog.pg_class WHERE oid='public.users'::regclass AND relowner=%s AND relkind='r'", (owner,)).fetchone()
    if not users:
        raise ValueError('Unsupported owner identity relation')
    _owner_comparison(conn,users[0],'id')
    if lock:
        conn.execute('SELECT pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtextextended(%s,0))', ('gmail-search partition owner '+owner_id,))
        # Shared schema/parent DDL is serialized across different owners too.
        conn.execute('SELECT pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtextextended(%s,0))', ('gmail-search partition catalog v1',))
    owner_query = 'SELECT 1 FROM public.users WHERE id=%s' + (' FOR KEY SHARE' if lock else '')
    if not conn.execute(owner_query, (owner_id,)).fetchone():
        raise ValueError('Owner must already exist')
    if lock:
        for table in _PREFIXES:
            if conn.execute('SELECT pg_catalog.to_regclass(%s)', ('public.'+table,)).fetchone()[0] is None:
                raise ValueError('Missing partition parent')
            conn.execute(sql.SQL('LOCK TABLE ONLY {} IN SHARE ROW EXCLUSIVE MODE').format(sql.Identifier('public',table)))
    parents = _parents(conn, owner, profile)
    namespace = conn.execute('SELECT pg_catalog.to_regnamespace(%s)::oid', (PARTITION_SCHEMA,)).fetchone()[0]
    _default_acl(conn, owner, namespace)
    _private_acl(conn, owner, namespace)
    found = _children(conn, parents, owner, namespace)
    if create and namespace is None:
        conn.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(PARTITION_SCHEMA)))
    return parents, found


def verify_owner_partitions(conn, owner_id: str, *, profile=NUMERIC) -> None:
    """Fail closed unless all three native-indexed owner partitions are ready.

    Uses a transaction/savepoint and administrator catalog locks. It grants no
    access, repairs nothing, and does not replace immutable reader ACL/RLS audits.
    """
    with _operation(conn):
        _, found = _prepare(conn, owner_id, create=False, profile=profile)
        if any((table,owner_id) not in found for table in _PREFIXES):
            raise ValueError('Missing owner partition')


def provision_owner_partitions(conn, owner_id: str, *, profile=NUMERIC) -> None:
    """Atomically create missing private owner leaves, refusing existing drift.

    Call on a trusted administrator connection only. Caller transactions retain
    commit/rollback ownership; this function never publishes runtime credentials.
    """
    with _operation(conn):
        _, found = _prepare(conn, owner_id, create=True, profile=profile)
        for table in _PREFIXES:
            if (table,owner_id) in found:
                continue
            child = sql.Identifier(PARTITION_SCHEMA,partition_name(table,owner_id))
            conn.execute(sql.SQL('CREATE TABLE {} PARTITION OF {} FOR VALUES IN ({})').format(
                child,sql.Identifier('public',table),sql.Literal(owner_id)))
            conn.execute(sql.SQL('ALTER TABLE {} ENABLE ROW LEVEL SECURITY').format(child))
            conn.execute(sql.SQL('ALTER TABLE {} FORCE ROW LEVEL SECURITY').format(child))
            conn.execute(sql.SQL('COMMENT ON TABLE {} IS {}').format(child,sql.Literal(partition_binding(table,owner_id))))
        verify_owner_partitions(conn, owner_id, profile=profile)


def inspect_owner_partitions(conn, owner_id: str, *, profile=NUMERIC) -> None:
    """Preview the same layout in a caller-owned stable read-only snapshot.

    This never acquires admission/DDL locks and is not an admission gate.
    Call verify_owner_partitions for readiness immediately before publication.
    """
    settings = conn.execute("SELECT pg_catalog.current_setting('transaction_isolation'),pg_catalog.current_setting('transaction_read_only')").fetchone()
    if settings not in (('repeatable read','on'),('serializable','on')):
        raise ValueError('Preview requires a stable read-only snapshot')
    with _operation(conn):
        _, found = _prepare(conn,owner_id,create=False,lock=False,profile=profile)
        if any((table,owner_id) not in found for table in _PREFIXES):
            raise ValueError('Missing owner partition')


def remove_empty_owner_partitions(conn, owner_id: str, *, profile=NUMERIC) -> None:
    """Remove only a verified, empty, complete owner set before user deletion.

    The owner must still exist. No leaves is an idempotent no-op; a partial set
    is refused. Caller-owned transactions can combine this with deleting the
    owner row. Mail deletion/retention and runtime-role retirement are separate.
    """
    with _operation(conn):
        _, found = _prepare(conn,owner_id,create=False,profile=profile)
        present = [table for table in _PREFIXES if (table,owner_id) in found]
        if not present:
            return
        if len(present) != len(_PREFIXES):
            raise ValueError('Cannot remove a partial owner partition set')
        children = {table:sql.Identifier(PARTITION_SCHEMA,partition_name(table,owner_id)) for table in present}
        # Parent locks block routed writes; leaf locks also block direct trusted
        # administrator writes between the empty check and DETACH/DROP.
        for table in present:
            conn.execute(sql.SQL('LOCK TABLE {} IN ACCESS EXCLUSIVE MODE').format(children[table]))
        for table in present:
            if conn.execute(sql.SQL('SELECT 1 FROM {} LIMIT 1').format(children[table])).fetchone():
                raise ValueError('Owner partitions must all be empty before removal')
        # Referencing leaves first. DETACH removes PostgreSQL's generated FK
        # partition dependencies while retaining each declared parent FK.
        for table in ('propositions','attachments','messages'):
            conn.execute(sql.SQL('ALTER TABLE {} DETACH PARTITION {}').format(sql.Identifier('public',table),children[table]))
            conn.execute(sql.SQL('DROP TABLE {}').format(children[table]))
        _, remaining = _prepare(conn,owner_id,create=False,profile=profile)
        if any((table,owner_id) in remaining for table in _PREFIXES):
            raise ValueError('Owner partition removal did not complete')
