"""Direct TEXT attach from frozen checked-in legacy schema; synthetic PG only."""
import hashlib
import importlib.util
import os
from pathlib import Path
import secrets

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
import pytest

from gmail_search.gateway.partitions import partition_name, PARTITION_SCHEMA, provision_owner_partitions, verify_owner_partitions
from gmail_search.gateway.partition_profiles import TEXT_OWNER_PARTITIONS_V1 as TEXT

ROOT = Path(__file__).parents[1]
LEGACY = ROOT/'tests/fixtures/legacy_text_owner_schema.sql'


@pytest.fixture
def migration():
    spec = importlib.util.spec_from_file_location('text_partition_migration', ROOT/'deploy/public/migrate_text_owner_partitions.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def source():
    assert hashlib.sha256(LEGACY.read_bytes()).hexdigest() == '53fef22e7d33837d749a5baf9ab25532968268857590406c36f767b2396e86fb'
    dsn = os.getenv('GMS_TEST_PG_DSN')
    if not dsn:
        pytest.skip('Explicit disposable ParadeDB required')
    cfg = conninfo_to_dict(dsn)
    assert (cfg.get('host'),cfg.get('port'),cfg.get('dbname'),cfg.get('user')) == ('127.0.0.1','55440','postgres','postgres')
    assert not set(cfg)&{'hostaddr','service','options'}
    suffix = secrets.token_hex(8)
    name = 'gms_owner_partitions_test_text_'+suffix
    role = 'gms_legacy_text_reader_'+suffix
    analyst = 'gms_legacy_text_analyst_'+suffix
    with psycopg.connect(dsn,autocommit=True) as conn:
        assert conn.info.hostaddr == '127.0.0.1'
        conn.execute(sql.SQL('CREATE DATABASE {} TEMPLATE template0').format(sql.Identifier(name)))
    try:
        with psycopg.connect(make_conninfo(dsn,dbname=name),autocommit=True) as conn:
            # Substitute only cluster-global fixture roles and the fixed database grant.
            fixture_sql = LEGACY.read_text().replace('gmail_search_reader',role).replace('gmail_analyst',analyst)
            fixture_sql = fixture_sql.replace('ON DATABASE gmail_search', 'ON DATABASE '+name)
            conn.execute(fixture_sql)
            conn.execute("INSERT INTO users(id,email) VALUES('alice','alice@example.test'),('bob','bob@example.test')")
            conn.execute("INSERT INTO messages(id,user_id,thread_id,from_addr,to_addr,date,subject,body_text) VALUES('same','alice','thread','sender','recipient','2026-09-15','needle','original body')")
            conn.execute("INSERT INTO attachments(user_id,message_id,filename,mime_type,extracted_text) VALUES('alice','same','file','text/plain','needle attachment')")
            conn.execute("INSERT INTO propositions(user_id,message_id,text,model) VALUES('alice','same','needle fact','fixture')")
            conn.execute("INSERT INTO embeddings(user_id,message_id,attachment_id,chunk_type,chunk_text,embedding,model) VALUES('alice','same',1,'body','needle','\\x00000000','fixture')")
            conn.execute("INSERT INTO topics(user_id,topic_id,label) VALUES('alice','topic','fixture')")
            conn.execute("INSERT INTO message_topics(user_id,message_id,topic_id) VALUES('alice','same','topic')")
            conn.execute("INSERT INTO message_summaries(user_id,message_id,summary,model) VALUES('alice','same','summary','fixture')")
            conn.execute("INSERT INTO summary_failures(user_id,message_id,model,error) VALUES('alice','same','fixture','error')")
            conn.execute("INSERT INTO prop_processed VALUES('alice','same')")
            yield conn
    finally:
        with psycopg.connect(dsn,autocommit=True) as conn:
            conn.execute(sql.SQL('DROP DATABASE {} WITH(FORCE)').format(sql.Identifier(name)))
            conn.execute(sql.SQL('DROP ROLE IF EXISTS {}').format(sql.Identifier(role)))
            conn.execute(sql.SQL('DROP ROLE IF EXISTS {}').format(sql.Identifier(analyst)))


def inventory(conn):
    result = {}
    for table in ('messages','attachments','propositions'):
        result[table] = conn.execute("SELECT c.oid,c.relfilenode,c.reltoastrelid,t.relfilenode FROM pg_class c LEFT JOIN pg_class t ON t.oid=c.reltoastrelid WHERE c.oid=%s::regclass",('public.'+table,)).fetchone()
    for name in ('messages_bm25_idx','attachments_bm25_idx','props_bm25_idx'):
        result[name] = conn.execute('SELECT oid,relfilenode FROM pg_class WHERE oid=%s::regclass',('public.'+name,)).fetchone()
    return result


def rows(conn):
    return {table:conn.execute(sql.SQL('SELECT * FROM public.{} ORDER BY id').format(sql.Identifier(table))).fetchall() for table in ('messages','attachments','propositions')}


def test_direct_attach_preserves_existing_text_storage_and_data(source,migration):
    before = inventory(source)
    data = rows(source)
    preview = migration.preview_text_owner_partitions(source,owner_id='alice')
    assert preview['mode'] == 'attach_single_owner'
    assert inventory(source) == before
    assert migration.migrate_text_owner_partitions(source,owner_id='alice')
    assert rows(source) == data
    for table in ('messages','attachments','propositions'):
        child = PARTITION_SCHEMA+'.'+partition_name(table,'alice')
        assert source.execute('SELECT oid,relfilenode,reltoastrelid FROM pg_class WHERE oid=%s::regclass',(child,)).fetchone() == before[table][:3]
    for name in ('messages_bm25_idx','attachments_bm25_idx','props_bm25_idx'):
        oid,node = before[name]
        assert source.execute('SELECT relfilenode FROM pg_class WHERE oid=%s',(oid,)).fetchone() == (node,)
    assert source.execute("SELECT 1 FROM pg_attribute WHERE attrelid='public.messages'::regclass AND attname='search_id'").fetchone() is None
    verify_owner_partitions(source,'alice',profile=TEXT)
    assert migration.migrate_text_owner_partitions(source,owner_id='alice') is False
    assert migration.preview_text_owner_partitions(source,owner_id='alice')['mode'] == 'already_partitioned'


def test_colliding_owner_crud_and_all_foreign_keys(source,migration):
    migration.migrate_text_owner_partitions(source,owner_id='alice')
    provision_owner_partitions(source,'bob',profile=TEXT)
    source.execute("INSERT INTO messages(id,user_id,thread_id,from_addr,to_addr,date) VALUES('same','bob','thread','b','b','2026-09-15')")
    source.execute("INSERT INTO attachments(id,user_id,message_id,filename,mime_type) VALUES(1,'bob','same','file','text/plain')")
    source.execute("INSERT INTO propositions(id,user_id,message_id,text,model) VALUES(1,'bob','same','foreign','fixture')")
    source.execute("INSERT INTO message_summaries(user_id,message_id,summary,model) VALUES('bob','same','foreign','fixture')")
    assert source.execute("SELECT user_id FROM messages WHERE id='same' ORDER BY user_id").fetchall() == [('alice',),('bob',)]
    assert source.execute("SELECT id FROM messages WHERE user_id='alice' AND id @@@ 'subject:needle'").fetchall() == [('same',)]
    for table in ('propositions','prop_processed'):
        statement = "INSERT INTO propositions(user_id,message_id,text,model) VALUES('bob','absent','x','fixture')" if table=='propositions' else "INSERT INTO prop_processed VALUES('bob','absent')"
        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            source.execute(statement)


@pytest.mark.parametrize('stage',['renamed','attached'])
def test_midway_failure_rolls_back_original_storage_and_keys(source,migration,stage):
    before = inventory(source)
    data = rows(source)
    def fail(current):
        if current == stage:
            raise RuntimeError('synthetic rollback')
    with pytest.raises(RuntimeError,match='synthetic rollback'):
        migration.migrate_text_owner_partitions(source,owner_id='alice',_checkpoint=fail)
    assert inventory(source) == before and rows(source) == data
    assert source.execute("SELECT to_regnamespace('gms_mail_partitions')").fetchone() == (None,)


def test_live_observed_source_variant_is_explicit_and_preserved(source,migration):
    for table in ('messages','attachments','embeddings','message_topics','message_summaries','summary_failures'):
        source.execute(sql.SQL('ALTER TABLE {} ALTER COLUMN user_id SET NOT NULL').format(sql.Identifier(table)))
    source.execute('ALTER TABLE embeddings ENABLE ROW LEVEL SECURITY')
    source.execute('ALTER TABLE message_topics DROP CONSTRAINT message_topics_topic_id_fkey')
    source.execute('ALTER TABLE message_topics ADD CONSTRAINT message_topics_topic_id_fkey FOREIGN KEY(user_id,topic_id) REFERENCES topics(user_id,topic_id)')
    assert migration.migrate_text_owner_partitions(source,owner_id='alice')
    assert source.execute("SELECT relrowsecurity,relforcerowsecurity FROM pg_class WHERE oid='embeddings'::regclass").fetchone() == (True,False)
    assert source.execute("SELECT pg_get_constraintdef(oid) FROM pg_constraint WHERE conrelid='message_topics'::regclass AND conname='message_topics_topic_id_fkey'").fetchone()[0].startswith('FOREIGN KEY (user_id, topic_id)')


@pytest.mark.parametrize('fault', ['null_owner','other_owner','orphan_fact','missing_fk','external_dependent_fk','extra_numeric_key','sequence_default','sequence_dependency','function','view','derived_fk_present','bad_embeddings_rls'])
def test_source_drift_refuses_preview_and_apply_before_mutation(source,migration,fault):
    if fault == 'null_owner':
        source.execute("UPDATE messages SET user_id=NULL WHERE id='same'")
    elif fault == 'other_owner':
        source.execute("INSERT INTO messages(id,user_id,thread_id,from_addr,to_addr,date) VALUES('foreign','bob','t','b','b','2026')")
    elif fault == 'orphan_fact':
        source.execute("UPDATE propositions SET message_id='absent' WHERE user_id='alice'")
    elif fault == 'missing_fk':
        source.execute('ALTER TABLE embeddings DROP CONSTRAINT embeddings_message_id_fkey')
    elif fault == 'external_dependent_fk':
        source.execute('CREATE TABLE public.external_dependency(id text REFERENCES message_summaries(message_id))')
    elif fault == 'extra_numeric_key':
        source.execute('ALTER TABLE messages ADD COLUMN search_id bigint GENERATED BY DEFAULT AS IDENTITY')
    elif fault == 'sequence_default':
        source.execute('ALTER TABLE attachments ALTER COLUMN id SET DEFAULT 1')
    elif fault == 'sequence_dependency':
        source.execute("CREATE TABLE public.external_dependency(id bigint DEFAULT nextval('attachments_id_seq'))")
    elif fault == 'function':
        source.execute('CREATE FUNCTION public.external_dependency() RETURNS bigint LANGUAGE sql BEGIN ATOMIC SELECT count(*) FROM messages; END')
    elif fault == 'view':
        source.execute('CREATE VIEW public.external_dependency AS SELECT id FROM messages')
    elif fault == 'derived_fk_present':
        source.execute('ALTER TABLE propositions ADD FOREIGN KEY(message_id) REFERENCES messages(id)')
    else:
        source.execute('ALTER TABLE embeddings FORCE ROW LEVEL SECURITY')
    before = inventory(source)
    for method in (migration.preview_text_owner_partitions,migration.migrate_text_owner_partitions):
        with pytest.raises(ValueError):
            method(source,owner_id='alice')
        assert inventory(source) == before
        assert source.execute("SELECT to_regnamespace('gms_mail_partitions')").fetchone() == (None,)


def test_serial_advancement_and_rollback_preserve_existing_sequences(source,migration):
    source.execute("INSERT INTO attachments(id,user_id,message_id,filename,mime_type) VALUES(100,'alice','same','high','text/plain')")
    before = source.execute("SELECT oid FROM pg_class WHERE oid='attachments_id_seq'::regclass").fetchone()[0]
    sequence = source.execute('SELECT last_value,is_called FROM attachments_id_seq').fetchone()
    def fail(stage):
        if stage == 'attached':raise RuntimeError('rollback')
    with pytest.raises(RuntimeError):
        migration.migrate_text_owner_partitions(source,owner_id='alice',_checkpoint=fail)
    assert source.execute('SELECT last_value,is_called FROM attachments_id_seq').fetchone() == sequence
    migration.migrate_text_owner_partitions(source,owner_id='alice')
    assert source.execute("SELECT oid FROM pg_class WHERE oid='attachments_id_seq'::regclass").fetchone()[0] == before
    assert source.execute("INSERT INTO attachments(user_id,message_id,filename,mime_type) VALUES('alice','same','after','text/plain') RETURNING id").fetchone()[0] == 101


def test_toast_and_acl_parent_projection_survive_direct_attach(source,migration):
    import random
    rng = random.Random(314159)
    body = ''.join(rng.choices('abcdefghijklmnopqrstuvwxyz0123456789',k=120000))
    source.execute("UPDATE messages SET body_html=%s WHERE id='same'",(body,))
    before = inventory(source)
    assert source.execute('SELECT pg_relation_size(%s::oid)',(before['messages'][2],)).fetchone()[0] > 0
    role = source.execute("SELECT grantee::regrole::text FROM pg_class c,LATERAL aclexplode(c.relacl) a WHERE c.oid='messages'::regclass AND grantee<>c.relowner LIMIT 1").fetchone()[0]
    migration.migrate_text_owner_partitions(source,owner_id='alice')
    old_oid,_,toast,toast_file = before['messages']
    assert source.execute('SELECT reltoastrelid FROM pg_class WHERE oid=%s',(old_oid,)).fetchone()[0] == toast
    assert source.execute('SELECT relfilenode FROM pg_class WHERE oid=%s',(toast,)).fetchone()[0] == toast_file
    assert source.execute("SELECT body_html FROM public.messages WHERE user_id='alice' AND id='same'").fetchone()[0] == body
    assert source.execute("SELECT has_table_privilege(%s,'public.messages','SELECT'),has_table_privilege(%s,%s::oid,'SELECT')",(role,role,old_oid)).fetchone() == (True,False)


def test_external_constraint_dependency_on_changed_dependent_key_is_refused(source,migration):
    source.execute("""CREATE FUNCTION public.external_constraint_dependency() RETURNS void LANGUAGE sql
        BEGIN ATOMIC INSERT INTO message_summaries(message_id,user_id,summary,model)
        VALUES('same','alice','test','fixture') ON CONFLICT ON CONSTRAINT message_summaries_pkey DO NOTHING; END""")
    before = inventory(source)
    with pytest.raises(ValueError):
        migration.preview_text_owner_partitions(source,owner_id='alice')
    assert inventory(source) == before


def test_apply_guard_uses_actual_address_without_connecting(migration):
    from types import SimpleNamespace
    from psycopg.pq import TransactionStatus
    for hostaddr,dbname in [('192.0.2.1','gms_owner_partitions_test_synthetic'),('127.0.0.1','postgres')]:
        conn = SimpleNamespace(info=SimpleNamespace(transaction_status=TransactionStatus.IDLE,host='127.0.0.1',hostaddr=hostaddr,port=55440,dbname=dbname))
        with pytest.raises(ValueError,match='disposable'):
            migration.migrate_text_owner_partitions(conn,owner_id='alice')


@pytest.mark.asyncio
async def test_direct_migration_composes_with_actual_text_reader(source,migration):
    import asyncio
    from gmail_search.gateway.data_admission import DataAdmission
    from gmail_search.gateway.provision_search_reader import provision_search_reader
    from gmail_search.gateway.search_reader import SearchCredential, SearchProfile, SearchReader, SearchRegistry, search_role
    owner = 'reader_owner_'+secrets.token_hex(8)
    source.execute('INSERT INTO users(id,email) VALUES(%s,%s)',(owner,owner+'@example.test'))
    source.execute("INSERT INTO topics(user_id,topic_id,label) VALUES(%s,'topic','fixture')",(owner,))
    for table in ('messages','attachments','propositions','embeddings','message_topics','message_summaries','summary_failures','prop_processed'):
        source.execute(sql.SQL("UPDATE {} SET user_id=%s WHERE user_id='alice'").format(sql.Identifier(table)),(owner,))
    migration.migrate_text_owner_partitions(source,owner_id=owner)
    # Restrict ambient privileges in this fixture only before reader publication.
    source.execute(sql.SQL('REVOKE TEMP ON DATABASE {} FROM PUBLIC').format(sql.Identifier(source.info.dbname)))
    source.execute('REVOKE ALL ON ALL SEQUENCES IN SCHEMA public,paradedb,pdb FROM PUBLIC')
    source.execute('REVOKE ALL ON ALL TABLES IN SCHEMA public,paradedb,pdb FROM PUBLIC')
    source.execute('REVOKE CREATE ON SCHEMA public,paradedb,pdb FROM PUBLIC')
    routines = source.execute("SELECT p.oid::regprocedure::text FROM pg_proc p JOIN pg_depend d ON d.classid='pg_proc'::regclass AND d.objid=p.oid JOIN pg_extension e ON e.oid=d.refobjid WHERE d.refclassid='pg_extension'::regclass AND e.extname='pg_search'").fetchall()
    for routine, in routines:
        source.execute(sql.SQL('REVOKE ALL ON ROUTINE {} FROM PUBLIC').format(sql.SQL(routine)))
    # The unique synthetic owner also yields a unique cluster role. Cleanup
    # occurs before the outer fixture drops its two legacy roles/database.
    role = search_role(owner,profile=TEXT)
    password = secrets.token_urlsafe(40)
    try:
        provision_search_reader(source,owner,password,profile=TEXT)
        credential = SearchCredential(owner,make_conninfo(source.info.dsn,user=role,password=password),schema_profile=TEXT)
        admission = DataAdmission()
        reader = SearchReader(SearchRegistry({owner:credential},is_active=lambda _:True),profile=SearchProfile('fixture','fixture',1,schema_profile=TEXT),admission=admission)
        async with reader.session(owner,deadline=asyncio.get_running_loop().time()+10,check_active=lambda:None) as q:
            for method in (q.lexical_messages,q.lexical_attachments,q.lexical_facts):
                assert [row.message_id for row in (await method(('needle',))).rows] == ['same']
            assert (await q.lexical_messages(('needle',),candidate_ids=('same',))).rows[0].id == 'same'
        assert not admission.active
        assert migration.migrate_text_owner_partitions(source,owner_id=owner) is False
    finally:
        # DROP OWNED is bounded to this single freshly provisioned fixture role.
        source.execute(sql.SQL('DROP OWNED BY {}').format(sql.Identifier(role)))
        source.execute(sql.SQL('DROP ROLE IF EXISTS {}').format(sql.Identifier(role)))


def test_missing_owner_column_is_not_a_valid_nonnullable_column(migration):
    from types import SimpleNamespace
    conn = SimpleNamespace(execute=lambda *args:SimpleNamespace(fetchone=lambda:None))
    with pytest.raises(ValueError):
        migration._owner_column(conn,1,'user_id')


@pytest.mark.parametrize('rollback',[False,True])
def test_serial_lock_blocks_nextval_and_refreshes_prelock_allocation(source,migration,rollback):
    seen = []
    def checkpoint(stage):
        if stage == 'before_sequence_locks':
            with psycopg.connect(make_conninfo(os.environ['GMS_TEST_PG_DSN'],dbname=source.info.dbname),autocommit=True) as other:
                assert other.execute("SELECT nextval('attachments_id_seq')").fetchone()[0] == 2
            seen.append(stage)
        if stage == 'sequences_locked':
            with psycopg.connect(make_conninfo(os.environ['GMS_TEST_PG_DSN'],dbname=source.info.dbname),autocommit=True) as other:
                other.execute("SET lock_timeout='100ms'")
                with pytest.raises(psycopg.errors.LockNotAvailable):
                    other.execute("SELECT nextval('attachments_id_seq')")
            seen.append(stage)
            if rollback:raise RuntimeError('synthetic sequence rollback')
    if rollback:
        with pytest.raises(RuntimeError,match='synthetic sequence rollback'):
            migration.migrate_text_owner_partitions(source,owner_id='alice',_checkpoint=checkpoint)
    else:
        migration.migrate_text_owner_partitions(source,owner_id='alice',_checkpoint=checkpoint)
    assert seen == ['before_sequence_locks','sequences_locked']
    assert source.execute("SELECT nextval('attachments_id_seq')").fetchone()[0] == 3
