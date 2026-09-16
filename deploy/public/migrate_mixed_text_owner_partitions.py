"""Guarded synthetic mixed-owner migration: phase one, then phase two.

`advance` commits the layout (MAINTENANCE -> INDEX_PENDING) and stays idempotent
so crash re-entry never publishes readiness. `publish` runs phase two
(INDEX_PENDING -> READY): VACUUM the retained leaves so ambulkdelete sees phase
one's row removal, rebuild their BM25 indexes, then qualify every owner's index
against its live rows. Phase two requires the patched pg_search build; an
unpatched engine asserts on a stale ctid once a retained leaf is reindexed and
searched (docs/qualification/retained-reader-root-cause.md).

Synthetic rehearsal only: no production DSN override and no credential
publication. The injected fence inspects an already durable external fence;
releasing that inspection must never re-enable writers/services. Administrators
remain trusted.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

from psycopg import sql
from psycopg.pq import TransactionStatus

from gmail_search.gateway.maintenance import (
    GateSnapshot, MaintenanceAdmin, ReleaseIdentity, _publication_lock,
)
from gmail_search.gateway.partition_profiles import TEXT_OWNER_PARTITIONS_V1 as TEXT
from gmail_search.gateway.partitions import (
    PARTITION_SCHEMA, _administrator, _default_acl, _private_acl,
    inspect_owner_partitions, verify_owner_partitions, partition_binding, partition_name,
)
from gmail_search.gateway.registry import AccessDenied


_spec = importlib.util.spec_from_file_location('mixed_text_legacy_mechanics',
    Path(__file__).with_name('migrate_text_owner_partitions.py'))
legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(legacy)

WITNESS_SCHEMA = 'gms_migration_control'
WITNESS_TABLE = 'text_phase_one'
PROCEDURE_DIGEST = hashlib.sha256(Path(__file__).read_bytes() + b'\x00'
    + Path(__file__).with_name('migrate_text_owner_partitions.py').read_bytes()).hexdigest()
_WITNESS = sql.Identifier(WITNESS_SCHEMA, WITNESS_TABLE)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def _digest(value):
    return hashlib.sha256(_json(value).encode('ascii')).hexdigest()


def _duration_seconds(value, maximum=86400):
    if type(value) not in (int, float) or not 0 < value <= maximum:
        raise ValueError('Bounded trusted preparation timeout required')
    return value


def _before(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('Synthetic phase-one deadline')


class _DeadlineConnection:
    """Bound each SQL statement by the one phase deadline, including refactored SQL."""
    def __init__(self, connection, deadline):
        self._raw,self._deadline = connection,deadline

    def __getattr__(self,name):
        return getattr(self._raw,name)

    def execute(self,statement,*args,**kwargs):
        _before(self._deadline)
        milliseconds = max(1,int((self._deadline-time.monotonic())*1000))
        local = 'LOCAL' if self._raw.info.transaction_status != TransactionStatus.IDLE else ''
        # SET, unlike SELECT set_config(), does not acquire an MVCC snapshot
        # before the caller's SET TRANSACTION ISOLATION LEVEL READ ONLY.
        self._raw.execute(sql.SQL('SET {} statement_timeout = {}').format(
            sql.SQL(local),sql.Literal(str(milliseconds))))
        cursor = self._raw.execute(statement,*args,**kwargs)
        _before(self._deadline)
        return cursor

    @contextmanager
    def transaction(self):
        _before(self._deadline)
        with self._raw.transaction():
            yield
            _before(self._deadline)
        _before(self._deadline)


@dataclass(frozen=True)
class MixedTextPlan:
    identity: ReleaseIdentity
    migration_id: str
    dominant_owner: str
    expected_owners: tuple[str, ...]
    database_name: str
    database_oid: int
    system_identifier: str
    procedure_digest: str = PROCEDURE_DIGEST

    @property
    def owner_set_digest(self):
        return _digest(self.expected_owners)

    def binding(self):
        if (type(self.identity) is not ReleaseIdentity or self.identity.profile is not TEXT
                or type(self.expected_owners) is not tuple or not 2 <= len(self.expected_owners) <= 16
                or self.expected_owners != tuple(sorted(set(self.expected_owners)))
                or self.dominant_owner not in self.expected_owners
                or self.procedure_digest != PROCEDURE_DIGEST):
            raise ValueError('Unsupported synthetic TEXT migration plan')
        for owner in self.expected_owners:
            partition_name('messages', owner)
        result = dict(store_id=self.identity.store_id, profile=self.identity.profile.value,
            epoch=self.identity.release_epoch, operation=self.migration_id,
            dominant=self.dominant_owner, owners=self.expected_owners,
            procedure=self.procedure_digest, database=self.database_name,
            database_oid=self.database_oid, system_identifier=self.system_identifier)
        if len(_json(result)) > 8192:
            raise ValueError('Migration binding exceeds fixed bound')
        return result


def _database(conn):
    return conn.execute('''SELECT current_database(),(SELECT oid FROM pg_database
        WHERE datname=current_database()),system_identifier::text FROM pg_control_system()''').fetchone()


def capture_plan(conn, *, identity, migration_id, dominant_owner, expected_owners):
    legacy._require(conn, apply=True)
    if conn.autocommit is not True:
        raise ValueError('Fresh idle autocommit administrator connection required')
    with conn.transaction():
        conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        _administrator(conn)
        plan = MixedTextPlan(identity,migration_id,dominant_owner,expected_owners,*_database(conn))
        plan.binding()
        return plan


def _witness_schema(conn, *, create=False):
    """Audit exact fixed catalog and dependencies before reading witness bytes."""
    owner = _administrator(conn)
    namespace = conn.execute('SELECT oid FROM pg_namespace WHERE nspname=%s',(WITNESS_SCHEMA,)).fetchone()
    if namespace is None:
        if not create:
            return None
        _default_acl(conn,owner,None)
        conn.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(WITNESS_SCHEMA)))
        conn.execute(sql.SQL('REVOKE ALL ON SCHEMA {} FROM PUBLIC').format(sql.Identifier(WITNESS_SCHEMA)))
        conn.execute(sql.SQL('''CREATE TABLE {} (
            singleton integer NOT NULL PRIMARY KEY CHECK(singleton=1),
            phase text NOT NULL CHECK(phase IN ('LAYOUT_COMMITTED','INDEXES_REBUILT')),
            binding text NOT NULL CHECK(octet_length(binding)<=8192),
            inventory text NOT NULL CHECK(octet_length(inventory)<=131072))''').format(_WITNESS))
        namespace = conn.execute('SELECT oid FROM pg_namespace WHERE nspname=%s',(WITNESS_SCHEMA,)).fetchone()
    ns = namespace[0]
    _private_acl(conn,owner,ns)
    _default_acl(conn,owner,ns)
    relations = conn.execute('SELECT oid,relname,relkind FROM pg_class WHERE relnamespace=%s ORDER BY relname',(ns,)).fetchall()
    if [(name,kind) for _,name,kind in relations] != [(WITNESS_TABLE,'r'),(WITNESS_TABLE+'_pkey','i')]:
        raise ValueError('Unexpected witness schema objects')
    oid,index_oid = relations[0][0],relations[1][0]
    row = conn.execute('''SELECT relowner,relpersistence,relrowsecurity,relforcerowsecurity,
        reloptions,reloftype,relispartition,relreplident,relam=(SELECT oid FROM pg_am WHERE amname='heap'),reltablespace
        FROM pg_class WHERE oid=%s''',(oid,)).fetchone()
    if row != (owner,'p',False,False,None,0,False,'d',True,0):
        raise ValueError('Unexpected witness relation profile')
    attrs = conn.execute('''SELECT a.attname,a.atttypid::regtype::text,a.attnotnull,a.attidentity,
        a.attgenerated,a.attcollation=CASE WHEN a.atttypid='text'::regtype
            THEN 'pg_catalog."default"'::regcollation ELSE 0 END,
        pg_get_expr(d.adbin,d.adrelid) FROM pg_attribute a LEFT JOIN pg_attrdef d
        ON d.adrelid=a.attrelid AND d.adnum=a.attnum WHERE a.attrelid=%s AND a.attnum>0
        ORDER BY a.attnum''',(oid,)).fetchall()
    if attrs != [(name,kind,True,'','',True,None) for name,kind in
            [('singleton','integer'),('phase','text'),('binding','text'),('inventory','text')]]:
        raise ValueError('Unexpected witness columns/defaults/collation')
    constraints = conn.execute('''SELECT contype,convalidated,condeferrable,pg_get_constraintdef(oid)
        FROM pg_constraint WHERE conrelid=%s''',(oid,)).fetchall()
    expected = {('p',True,False,'PRIMARY KEY (singleton)'),
        ('c',True,False,'CHECK ((singleton = 1))'),
        ('c',True,False,"CHECK ((phase = ANY (ARRAY['LAYOUT_COMMITTED'::text, 'INDEXES_REBUILT'::text])))"),
        ('c',True,False,'CHECK ((octet_length(binding) <= 8192))'),
        ('c',True,False,'CHECK ((octet_length(inventory) <= 131072))')}
    if len(constraints) != len(expected) or set(constraints) != expected:
        raise ValueError('Unexpected witness constraints')
    index = conn.execute('''SELECT i.indisvalid,i.indisready,i.indisunique,i.indisprimary,
        i.indnatts,i.indnkeyatts,i.indkey::text,i.indexprs,i.indpred,c.reloptions,c.reltablespace,
        am.amname,opc.opcname,opc.opcnamespace='pg_catalog'::regnamespace,i.indcollation::text,
        i.indoption::text,i.indislive,i.indimmediate,c.relowner,c.relkind,c.relpersistence
        FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid JOIN pg_am am ON am.oid=c.relam
        JOIN pg_opclass opc ON opc.oid=i.indclass[0] WHERE i.indrelid=%s''',(oid,)).fetchall()
    if index != [(True,True,True,True,1,1,'1',None,None,None,0,'btree','int4_ops',True,'0','0',True,True,owner,'i','p')]:
        raise ValueError('Unexpected witness index')
    for statement in (
        'SELECT 1 FROM pg_trigger WHERE tgrelid=%s',
        'SELECT 1 FROM pg_policy WHERE polrelid=%s',
        'SELECT 1 FROM pg_rewrite WHERE ev_class=%s',
        'SELECT 1 FROM pg_publication_rel WHERE prrelid=%s',
        "SELECT 1 FROM pg_seclabel WHERE classoid='pg_class'::regclass AND objoid=%s",
        'SELECT 1 FROM pg_statistic_ext WHERE stxrelid=%s',
    ):
        if conn.execute(statement,(oid,)).fetchone():
            raise ValueError('Unexpected witness execution/catalog dependency')
    if conn.execute('SELECT 1 FROM pg_publication WHERE puballtables').fetchone():
        raise ValueError('Witness cannot belong to an all-table publication')
    if conn.execute('SELECT 1 FROM pg_inherits WHERE inhrelid=ANY(%s) OR inhparent=ANY(%s)',([oid,index_oid],[oid,index_oid])).fetchone():
        raise ValueError('Unexpected witness inheritance')
    # Only automatic table/index/constraint/TOAST/row-type dependents are allowed.
    toast,rowtype = conn.execute('SELECT reltoastrelid,reltype FROM pg_class WHERE oid=%s',(oid,)).fetchone()
    array = conn.execute('SELECT typarray FROM pg_type WHERE oid=%s',(rowtype,)).fetchone()[0]
    keys = [row[0] for row in conn.execute('SELECT oid FROM pg_constraint WHERE conrelid=%s',(oid,))]
    allowed = {'pg_class':{oid,index_oid,toast},'pg_constraint':set(keys),'pg_type':{rowtype}}
    dependencies = conn.execute('''SELECT classid::regclass::text,objid FROM pg_depend
        WHERE refclassid='pg_class'::regclass AND refobjid=ANY(%s)''',([oid,index_oid],)).fetchall()
    if any(value not in allowed.get(kind,set()) for kind,value in dependencies):
        raise ValueError('Unexpected witness relation dependent')
    if conn.execute('''SELECT 1 FROM pg_depend WHERE refclassid='pg_type'::regclass
        AND refobjid=ANY(%s) AND NOT(classid='pg_type'::regclass AND objid=%s AND deptype='i')''',([rowtype,array],array)).fetchone():
        raise ValueError('Unexpected witness row-type dependent')
    if conn.execute('''SELECT 1 FROM pg_depend WHERE refclassid='pg_constraint'::regclass
        AND refobjid=ANY(%s) AND NOT(classid='pg_class'::regclass AND objid=%s AND deptype='i')''',(keys,index_oid)).fetchone():
        raise ValueError('Unexpected witness constraint dependent')
    if conn.execute('''SELECT 1 FROM pg_depend WHERE classid='pg_class'::regclass
        AND objid=ANY(%s) AND deptype IN ('e','x')''',([oid,index_oid],)).fetchone():
        raise ValueError('Witness cannot be an extension member')
    if conn.execute('SELECT 1 FROM pg_proc WHERE pronamespace=%s',(ns,)).fetchone():
        raise ValueError('Unexpected witness schema routine')
    if conn.execute('SELECT 1 FROM pg_type WHERE typnamespace=%s AND oid<>ALL(%s)',(ns,[rowtype,array])).fetchone():
        raise ValueError('Unexpected witness schema type')
    schema_dependents = conn.execute('''SELECT classid::regclass::text,objid FROM pg_depend
        WHERE refclassid='pg_namespace'::regclass AND refobjid=%s''',(ns,)).fetchall()
    if any(value not in {'pg_class':{oid,index_oid},'pg_type':{rowtype,array}}.get(kind,set())
            for kind,value in schema_dependents):
        raise ValueError('Unexpected witness namespace dependent')
    return oid


def _inventory(conn, plan, *, readonly=True):
    """Exact metadata, not mailbox bytes; unexpected children/indexes change it."""
    expected_names = {partition_name(table,owner) for table in legacy.TABLES for owner in plan.expected_owners}
    actual_names = {row[0] for row in conn.execute("SELECT relname FROM pg_class WHERE relnamespace=%s::regnamespace AND relkind IN ('r','p')",(PARTITION_SCHEMA,))}
    if actual_names != expected_names:
        raise ValueError('Owner leaf inventory mismatch')
    result = {}
    for owner in plan.expected_owners:
        (inspect_owner_partitions if readonly else verify_owner_partitions)(conn,owner,profile=TEXT)
    legacy._dependent_contract(conn,_administrator(conn),target=True)
    legacy._foreign_keys(conn,target=True)
    names = [('public',table) for table in (*legacy.TABLES,*legacy.DEPENDENTS,'users','topics')]
    names += [(PARTITION_SCHEMA,name) for name in sorted(expected_names)]
    for schema,name in names:
        relation = conn.execute('''SELECT c.oid,c.relfilenode,c.reltoastrelid,t.relfilenode,
            c.relkind,c.relowner,c.relrowsecurity,c.relforcerowsecurity
            FROM pg_class c LEFT JOIN pg_class t ON t.oid=c.reltoastrelid
            WHERE c.oid=to_regclass(%s)''',(sql.Identifier(schema,name).as_string(conn),)).fetchone()
        if relation is None:
            raise ValueError('Missing inventoried relation')
        oid = relation[0]
        indexes = conn.execute('''SELECT i.indexrelid,c.relname,c.relfilenode,pg_get_indexdef(i.indexrelid),
            i.indisvalid,i.indisready,i.indisunique,i.indisprimary,c.reloptions,
            ARRAY(SELECT inhparent FROM pg_inherits WHERE inhrelid=i.indexrelid ORDER BY inhparent)
            FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid
            WHERE i.indrelid=%s ORDER BY i.indexrelid''',(oid,)).fetchall()
        constraints = conn.execute('SELECT conname,pg_get_constraintdef(oid),convalidated,condeferrable FROM pg_constraint WHERE conrelid=%s ORDER BY conname',(oid,)).fetchall()
        counts = conn.execute(sql.SQL('SELECT user_id,count(*) FROM {} GROUP BY user_id ORDER BY user_id').format(sql.Identifier(schema,name))).fetchall() if name not in ('users','topics') else None
        result[schema+'.'+name] = dict(relation=relation,indexes=indexes,constraints=constraints,counts=counts)
    result['sequences'] = [conn.execute('''SELECT c.oid,n.nspname,c.relname,s.seqstart,s.seqincrement,
        s.seqmax,s.seqmin,s.seqcache,s.seqcycle,pg_get_serial_sequence(%s,'id')
        FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace JOIN pg_sequence s ON s.seqrelid=c.oid
        WHERE c.oid=pg_get_serial_sequence(%s,'id')::regclass''',('public.'+table,'public.'+table)).fetchone()
        for table in ('attachments','propositions')]
    result['allocation'] = {table:conn.execute(sql.SQL('SELECT last_value,is_called FROM {}').format(
        sql.Identifier('public',table+'_id_seq'))).fetchone() for table in ('attachments','propositions')}
    if len(_json(result)) > 131072:
        raise ValueError('Bounded migration inventory exceeded')
    return result


class MixedTextPhaseOne:
    def __init__(self, connect, registry_path, plan, *, fence, _checkpoint=lambda stage: None):
        self._connect,self.path,self._plan,self._fence = connect,Path(registry_path),plan,fence
        self._checkpoint_callback = _checkpoint
        self._seen_connections = []
        self._seen_backends = set()
        self._cleanup_failed = False
        self._phase_deadline = None

    @property
    def plan(self):
        return self._plan

    @contextmanager
    def _fresh(self, *, deadline=None):
        deadline = time.monotonic()+30 if deadline is None else deadline
        _before(deadline)
        if self._cleanup_failed or len(self._seen_connections) >= 128:
            raise ValueError('Fresh connection lifecycle is unavailable')
        conn = self._connect()
        try:
            if any(previous is conn for previous in self._seen_connections) or conn.info.backend_pid in self._seen_backends:
                raise ValueError('Fresh physical connection required')
            self._seen_connections.append(conn)
            self._seen_backends.add(conn.info.backend_pid)
            legacy._require(conn,apply=True)
            if conn.autocommit is not True or conn.info.transaction_status != TransactionStatus.IDLE:
                raise ValueError('Fresh idle autocommit connection required')
            bounded = _DeadlineConnection(conn,deadline)
            if _database(bounded) != (self.plan.database_name,self.plan.database_oid,self.plan.system_identifier):
                raise ValueError('Pinned database identity mismatch')
            yield bounded
        finally:
            try:
                conn.close()
            except BaseException:
                self._cleanup_failed = True
                raise
        _before(deadline)

    @contextmanager
    def _owned_fence(self, deadline):
        manager = self._fence.hold(self.plan,deadline=deadline)
        try:
            manager.__enter__()
        except BaseException:
            self._cleanup_failed = True
            raise
        try:
            _before(deadline)
            yield
        finally:
            try:
                manager.__exit__(*sys.exc_info())
            except BaseException:
                self._cleanup_failed = True
                raise
        _before(deadline)

    def _checkpoint(self,stage):
        _before(self._phase_deadline)
        self._checkpoint_callback(stage)
        _before(self._phase_deadline)

    def _settings(self, conn, deadline, *, readonly=False):
        if time.monotonic() >= deadline:
            raise TimeoutError('Synthetic phase-one deadline')
        if readonly:
            conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        conn.execute('SET LOCAL search_path=pg_catalog')
        conn.execute("SELECT set_config('statement_timeout',%s,true)",(str(max(1,int((deadline-time.monotonic())*1000))),))
        conn.execute("SET LOCAL lock_timeout='2s'")

    def _copy(self, conn, table, data, retained, parent):
        columns = sql.SQL(',').join(sql.Identifier(row[1]) for row in data['attrs'])
        for owner in self.plan.expected_owners:
            if owner == self.plan.dominant_owner:
                continue
            child = sql.Identifier(PARTITION_SCHEMA,partition_name(table,owner))
            conn.execute(sql.SQL('CREATE TABLE {} PARTITION OF {} FOR VALUES IN ({})').format(child,parent,sql.Literal(owner)))
            conn.execute(sql.SQL('ALTER TABLE {} ENABLE ROW LEVEL SECURITY').format(child))
            conn.execute(sql.SQL('ALTER TABLE {} FORCE ROW LEVEL SECURITY').format(child))
            conn.execute(sql.SQL('COMMENT ON TABLE {} IS {}').format(child,sql.Literal(partition_binding(table,owner))))
            inserted = conn.execute(sql.SQL('INSERT INTO {} ({}) SELECT {} FROM {} WHERE user_id=%s').format(child,columns,columns,retained),(owner,)).rowcount
            self._checkpoint('copied:'+table+':'+owner)
            difference = sql.SQL('''SELECT 1 FROM (
                (SELECT {} FROM {} WHERE user_id=%s EXCEPT ALL SELECT {} FROM {})
                UNION ALL (SELECT {} FROM {} EXCEPT ALL SELECT {} FROM {} WHERE user_id=%s)) differences LIMIT 1''').format(
                    columns,retained,columns,child,columns,child,columns,retained)
            if conn.execute(difference,(owner,owner)).fetchone():
                raise ValueError('Minority copy content mismatch')
            deleted = conn.execute(sql.SQL('DELETE FROM {} WHERE user_id=%s').format(retained),(owner,)).rowcount
            if deleted != inserted:
                raise ValueError('Minority copy count mismatch')
            self._checkpoint('deleted:'+table+':'+owner)

    def _read_witness(self, conn):
        """Return (phase, digest) for the committed witness, or None when absent."""
        if _witness_schema(conn) is None:
            return None
        rows = conn.execute(sql.SQL('SELECT singleton,phase,binding,inventory FROM {}').format(_WITNESS)).fetchall()
        if len(rows) != 1 or rows[0][:1] != (1,) or rows[0][2] != _json(self.plan.binding()):
            raise ValueError('Witness operation/phase binding mismatch')
        phase = rows[0][1]
        if phase not in ('LAYOUT_COMMITTED','INDEXES_REBUILT'):
            raise ValueError('Witness operation/phase binding mismatch')
        rebuilt = phase == 'INDEXES_REBUILT'
        stored = json.loads(rows[0][3])
        expected_keys = {'source','target','rebuilt'} if rebuilt else {'source','target'}
        if type(stored) is not dict or set(stored) != expected_keys or _json(stored) != rows[0][3]:
            raise ValueError('Noncanonical witness inventory')
        current = _inventory(conn,self.plan)
        self._preserved(stored['source'],current,rebuilt=rebuilt)
        parts = dict(source=stored['source'],target=current)
        if rebuilt:
            if _json(stored['rebuilt']) != _json(self._retained_bm25(conn)):
                raise ValueError('Rebuilt retained index identity changed after phase two')
            parts['rebuilt'] = stored['rebuilt']
        actual = _json(parts)
        if rows[0][3] != actual:
            raise ValueError('Witness relation/index/content-count inventory mismatch')
        return phase, _digest((rows[0][2],actual))

    def _witness_row(self, conn, phase):
        """Witness bytes for the write path, where a read-only inventory cannot run."""
        if _witness_schema(conn) is None:
            raise ValueError('Missing committed witness')
        rows = conn.execute(sql.SQL('SELECT singleton,phase,binding,inventory FROM {}').format(_WITNESS)).fetchall()
        if len(rows) != 1 or rows[0][:3] != (1,phase,_json(self.plan.binding())):
            raise ValueError('Witness operation/phase binding mismatch')
        stored = json.loads(rows[0][3])
        if type(stored) is not dict or 'source' not in stored or _json(stored) != rows[0][3]:
            raise ValueError('Noncanonical witness inventory')
        return stored

    def _retained_bm25(self, conn):
        """Exact identity of each retained leaf's BM25 index, by table."""
        result = {}
        for table in sorted(legacy.TABLES):
            leaf = sql.Identifier(PARTITION_SCHEMA,partition_name(table,self.plan.dominant_owner)).as_string(conn)
            rows = conn.execute('''SELECT c.relname,c.relfilenode,i.indisvalid,i.indisready,pg_get_indexdef(i.indexrelid)
                FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid JOIN pg_am am ON am.oid=c.relam
                WHERE i.indrelid=to_regclass(%s) AND am.amname='bm25' ORDER BY c.relname''',(leaf,)).fetchall()
            if len(rows) != 1 or rows[0][2:4] != (True,True):
                raise ValueError('Retained leaf has no single valid BM25 index: '+table)
            result[table] = list(rows[0])
        return result

    def _lock_migration_scope(self, conn):
        """Advisory locks in the established order; callers add their table locks."""
        if not conn.execute('SELECT pg_try_advisory_xact_lock(72341629,2)').fetchone()[0]:
            raise ValueError('Migration already running')
        for lock in tuple('gmail-search partition owner '+owner for owner in self.plan.expected_owners)+('gmail-search partition catalog v1',):
            conn.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s,0))',(lock,))

    def _commit_rebuild(self, deadline):
        """Vacuum, rebuild and record the retained indexes in one committed step."""
        with self._fresh(deadline=deadline) as conn:
            self._vacuum_retained(conn)
        with self._fresh(deadline=deadline) as conn:
            with conn.transaction():
                self._settings(conn,deadline)
                self._lock_migration_scope(conn)
                for table in sorted(legacy.TABLES):
                    for owner in self.plan.expected_owners:
                        conn.execute(sql.SQL('LOCK TABLE {} IN ACCESS EXCLUSIVE MODE').format(
                            sql.Identifier(PARTITION_SCHEMA,partition_name(table,owner))))
                stored = self._witness_row(conn,'LAYOUT_COMMITTED')
                self._rebuild_retained_bm25(conn)
                inventory = dict(source=stored['source'],target=self._write_inventory(conn),
                    rebuilt=self._retained_bm25(conn))
                conn.execute(sql.SQL("UPDATE {} SET phase='INDEXES_REBUILT',inventory=%s WHERE singleton=1").format(_WITNESS),
                    (_json(inventory),))
                self._checkpoint('indexes_rebuilt')
        self._checkpoint('phase2_committed')

    def _vacuum_retained(self, conn):
        """Let ambulkdelete see phase one's row removal before the rebuild.

        Without this the retained BM25 index keeps the relocated owners'
        documents as live corpus statistics, and a rebuild preserves them.
        VACUUM cannot run inside a transaction block, so this runs on its own.
        """
        for table in sorted(legacy.TABLES):
            conn.execute(sql.SQL('VACUUM (INDEX_CLEANUP ON) {}').format(
                sql.Identifier(PARTITION_SCHEMA,partition_name(table,self.plan.dominant_owner))))
        self._checkpoint('retained_vacuumed')

    def _rebuild_retained_bm25(self, conn):
        """Purge foreign ranking history the phase-one row removal leaves behind.

        DELETE, ANALYZE and ordinary VACUUM do not clear it; only a rebuild does.
        """
        for table, identity in sorted(self._retained_bm25(conn).items()):
            conn.execute(sql.SQL('REINDEX INDEX {}').format(sql.Identifier(PARTITION_SCHEMA,identity[0])))

    def _qualify_bm25(self, conn):
        """Every owner's index must hold exactly that owner's live rows.

        Content-independent: a retained segment carrying foreign or deleted
        documents shows up as a doc-count or deletion mismatch.
        """
        report = {}
        for table in sorted(legacy.TABLES):
            for owner in self.plan.expected_owners:
                leaf = sql.Identifier(PARTITION_SCHEMA,partition_name(table,owner))
                index = conn.execute('''SELECT c.relname FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid
                    JOIN pg_am am ON am.oid=c.relam WHERE i.indrelid=to_regclass(%s) AND am.amname='bm25'
                    ''',(leaf.as_string(conn),)).fetchall()
                if len(index) != 1:
                    raise ValueError('Owner leaf has no single BM25 index: '+table)
                # Only visible segments are what a reader scores against; the
                # pre-rebuild segments linger as non-visible until recycled.
                docs,deleted = conn.execute('''SELECT coalesce(sum(num_docs),0),coalesce(sum(num_deleted),0)
                    FROM paradedb.index_info(to_regclass(%s),true) WHERE visible''',
                    (sql.Identifier(PARTITION_SCHEMA,index[0][0]).as_string(conn),)).fetchone()
                live, = conn.execute(sql.SQL('SELECT count(*) FROM {}').format(leaf)).fetchone()
                foreign, = conn.execute(sql.SQL('SELECT count(*) FROM {} WHERE user_id<>%s').format(leaf),(owner,)).fetchone()
                if foreign:
                    raise ValueError('Foreign rows present in owner leaf: '+table)
                if int(deleted) or int(docs) != int(live):
                    raise ValueError('Owner index retains foreign or deleted documents: '+table)
                report[table+'/'+owner] = [int(live),int(docs),int(deleted)]
        return report

    def _verify(self, snapshot, target, *, deadline):
        if self._fence is None or self._cleanup_failed or target not in ('INDEX_PENDING','READY'):
            raise ValueError('Qualified external fence and a supported target required')
        if target == 'READY':
            return self._verify_ready(snapshot, deadline=deadline)
        self._phase_deadline = deadline
        with self._owned_fence(deadline):
            with self._fresh(deadline=deadline) as conn:
                with conn.transaction():
                    self._settings(conn,deadline,readonly=True)
                    witness = self._read_witness(conn)
            if witness is not None and witness[0] != 'LAYOUT_COMMITTED':
                raise ValueError('Phase two already committed; phase one cannot be reverified')
            if witness is None:
                if snapshot.state != 'MAINTENANCE':
                    raise ValueError('Missing witness after phase-one state')
                with self._fresh(deadline=deadline) as conn:
                    with conn.transaction():
                        self._settings(conn,deadline)
                        self._lock_migration_scope(conn)
                        for table in sorted((*legacy.TABLES,*legacy.DEPENDENTS,'users','topics')):
                            conn.execute(sql.SQL('LOCK TABLE {} IN ACCESS EXCLUSIVE MODE').format(sql.Identifier('public',table)))
                        if _witness_schema(conn) is not None:
                            raise ValueError('Witness appeared after initial snapshot')
                        state = legacy._preflight(conn,self.plan.dominant_owner,_expected_owners=self.plan.expected_owners)
                        if state['mode'] != 'attach_single_owner':
                            raise ValueError('Partitioned target has no phase-one witness')
                        original = self._source_inventory(conn,state)
                        legacy._apply_layout(conn,self.plan.dominant_owner,state,
                            _checkpoint=self._checkpoint,_redistribute=self._copy)
                        # Reuse stable-snapshot catalog inspection in this write
                        # transaction via the same underlying strict validators.
                        _witness_schema(conn,create=True)
                        target_inventory = self._write_inventory(conn)
                        for table,data in state['tables'].items():
                            original[table]['sequence'] = data['sequence']
                        self._preserved(original,target_inventory)
                        inventory = dict(source=original,target=target_inventory)
                        conn.execute(sql.SQL('INSERT INTO {} VALUES(1,%s,%s,%s)').format(_WITNESS),
                            ('LAYOUT_COMMITTED',_json(self.plan.binding()),_json(inventory)))
                        self._checkpoint('witness_written')
                self._checkpoint('phase1_committed')
            with self._fresh(deadline=deadline) as conn:
                with conn.transaction():
                    self._settings(conn,deadline,readonly=True)
                    committed = self._read_witness(conn)
                    if committed is None or committed[0] != 'LAYOUT_COMMITTED':
                        raise ValueError('Missing committed phase-one witness')
                    self._checkpoint('phase1_verified')
                    return committed[1]

    def _verify_ready(self, snapshot, *, deadline):
        """Phase two: rebuild retained BM25 indexes, then qualify every owner.

        The rebuild must commit separately from phase one — a combined
        transaction leaves foreign-influenced scores behind after commit.
        """
        if snapshot.state != 'INDEX_PENDING':
            raise ValueError('Phase two requires a committed phase-one gate')
        self._phase_deadline = deadline
        with self._owned_fence(deadline):
            with self._fresh(deadline=deadline) as conn:
                with conn.transaction():
                    self._settings(conn,deadline,readonly=True)
                    witness = self._read_witness(conn)
            if witness is None:
                raise ValueError('Missing committed phase-one witness')
            if witness[0] == 'LAYOUT_COMMITTED':
                self._commit_rebuild(deadline)
            with self._fresh(deadline=deadline) as conn:
                with conn.transaction():
                    self._settings(conn,deadline,readonly=True)
                    committed = self._read_witness(conn)
                    if committed is None or committed[0] != 'INDEXES_REBUILT':
                        raise ValueError('Missing committed phase-two witness')
                    report = self._qualify_bm25(conn)
                    self._checkpoint('phase2_verified')
                    return _digest((committed[1],_json(report)))

    def _write_inventory(self, conn):
        return _inventory(conn,self.plan,readonly=False)

    def _source_inventory(self, conn, state):
        result = {}
        for table,data in state['tables'].items():
            heap = conn.execute('''SELECT c.oid,c.relfilenode,c.reltoastrelid,t.relfilenode
                FROM pg_class c LEFT JOIN pg_class t ON t.oid=c.reltoastrelid WHERE c.oid=%s''',(data['oid'],)).fetchone()
            indexes = [conn.execute('SELECT oid,relfilenode,pg_get_indexdef(oid) FROM pg_class WHERE oid=%s',(index[1],)).fetchone()
                for index in data['indexes']]
            counts = conn.execute(sql.SQL('SELECT user_id,count(*) FROM {} GROUP BY user_id ORDER BY user_id').format(sql.Identifier('public',table))).fetchall()
            result[table] = dict(heap=heap,indexes=indexes,counts=counts,sequence=None)
        return result

    def _preserved(self, original, target, *, rebuilt=False):
        if type(original) is not dict or set(original) != set(legacy.TABLES):
            raise ValueError('Missing original retained inventory')
        for table in legacy.TABLES:
            data = original[table]
            if type(data) is not dict or set(data) != {'heap','indexes','counts','sequence'}:
                raise ValueError('Unexpected original retained inventory')
            retained = target[PARTITION_SCHEMA+'.'+partition_name(table,self.plan.dominant_owner)]
            if _json(data['heap']) != _json(retained['relation'][:4]):
                raise ValueError('Retained heap/TOAST identity changed during phase one')
            indexes = {index[0]:index for index in retained['indexes']}
            # Phase two rebuilds the retained BM25 index on purpose, so its
            # storage moves. Every other original index must still be untouched,
            # and the rebuilt one is pinned separately by the witness.
            bm25 = {index[0] for index in retained['indexes']
                if rebuilt and ' USING bm25 ' in (index[3] or '')}
            if any(index[0] not in indexes or (index[0] not in bm25 and index[1] != indexes[index[0]][2])
                    for index in data['indexes']):
                raise ValueError('Original nonconstraint index changed during phase one')
            if _json(data['counts']) != _json(target['public.'+table]['counts']):
                raise ValueError('Owner row counts changed during phase one')
            if table != 'messages':
                oid,name,last,called,maximum = data['sequence']
                sequence = next(item for item in target['sequences'] if item[2] == name)
                if sequence[0] != oid or tuple(target['allocation'][table]) != (max(last+int(called),1 if maximum is None else maximum+1),False):
                    raise ValueError('Original serial identity/allocation was not preserved')

    def advance(self, expected):
        try:
            if self._cleanup_failed:
                raise ValueError('Prior cleanup was not acknowledged')
            self.plan.binding()
            if (type(expected) is not GateSnapshot or expected.identity != self.plan.identity
                    or expected.migration_id != self.plan.migration_id
                    or expected.owner_set_digest != self.plan.owner_set_digest
                    or expected.procedure_digest != self.plan.procedure_digest):
                raise ValueError('Gate/plan binding mismatch')
            admin = MaintenanceAdmin(self.path,verifier=self._verify)
            current = admin.status()
            if current.state == 'INDEX_PENDING':
                if current != expected and not (expected.state == 'MAINTENANCE'
                        and current.revision == expected.revision+1
                        and current.identity == expected.identity
                        and current.migration_id == expected.migration_id):
                    raise ValueError('Stale phase-one state')
                with _publication_lock(self.path,exclusive=True,timeout=2):
                    if admin.status() != current:
                        raise ValueError('Concurrent gate transition')
                    digest = self._verify(current,'INDEX_PENDING',deadline=time.monotonic()+30)
                    if digest != current.phase1_digest:
                        raise ValueError('Gate/witness digest mismatch')
                    return current
            if expected.state != 'MAINTENANCE':
                raise ValueError('Phase two is published by publish(), not advance()')
            return admin.record_index_pending(expected)
        except Exception:
            raise AccessDenied() from None

    def prepare(self, snapshot, *, seconds=3600):
        """Run a phase's slow work before entering the gate's 30s budget.

        `MaintenanceAdmin` bounds verification at 30 seconds and refuses any
        larger value, so a full-size migration cannot do its work inside the
        verifier. Doing it here first is not a weakening: both phases commit a
        durable witness, and the verifier then only re-reads and qualifies that
        witness, which is fast at any scale. Calling this is optional at fixture
        scale and required at production scale.
        """
        target = 'INDEX_PENDING' if snapshot.state == 'MAINTENANCE' else 'READY'
        started = time.monotonic()
        self._verify(snapshot, target, deadline=started + _duration_seconds(seconds))
        return round(time.monotonic() - started, 2)

    def publish(self, expected):
        """Phase two: rebuild retained BM25 indexes and publish READY.

        Deliberately separate from `advance`, which stays idempotent so that
        crash re-entry never publishes readiness as a side effect.
        """
        try:
            if self._cleanup_failed:
                raise ValueError('Prior cleanup was not acknowledged')
            self.plan.binding()
            if (type(expected) is not GateSnapshot or expected.state != 'INDEX_PENDING'
                    or expected.identity != self.plan.identity
                    or expected.migration_id != self.plan.migration_id
                    or expected.owner_set_digest != self.plan.owner_set_digest
                    or expected.procedure_digest != self.plan.procedure_digest):
                raise ValueError('Gate/plan binding mismatch')
            return MaintenanceAdmin(self.path,verifier=self._verify).publish_ready(expected)
        except Exception:
            raise AccessDenied() from None
