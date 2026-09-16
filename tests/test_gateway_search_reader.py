"""Immutable fixed-operation readers, qualified only with synthetic mail."""
from types import SimpleNamespace
import asyncio
import importlib.util
import os
from pathlib import Path
import secrets
import struct

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
import pytest

from gmail_search.gateway.search_reader import SearchCredential, SearchProfile, SearchReader, SearchRegistry, search_role
from gmail_search.gateway.provision_search_reader import provision_search_reader, verify_search_reader_access


@pytest.fixture
def search_database(request):
    dsn = os.environ.get('GMS_TEST_PG_DSN')
    if not dsn:
        pytest.skip('Explicit disposable ParadeDB required')
    cfg = conninfo_to_dict(dsn)
    assert (cfg.get('host'),cfg.get('port'),cfg.get('dbname'),cfg.get('user'))==('127.0.0.1','55440','postgres','postgres')
    assert not set(cfg)&{'hostaddr','service','options'}
    suffix = secrets.token_hex(8)
    name = 'gms_owner_partitions_test_'+suffix
    owners = (("alice'\\quoted_" if getattr(request,'param',None)=='quoted' else 'alice_')+suffix,'bob_'+suffix)
    roles = [search_role(owner) for owner in owners]
    with psycopg.connect(dsn,autocommit=True) as conn:
        assert conn.info.hostaddr=='127.0.0.1'
        conn.execute(sql.SQL('CREATE DATABASE {} TEMPLATE template0').format(sql.Identifier(name)))
    target = make_conninfo(dsn,dbname=name)
    try:
        with psycopg.connect(target,autocommit=True) as conn:
            root = Path(__file__).parents[1]
            conn.execute((root/'src/gmail_search/store/pg_schema.sql').read_text())
            # These revocations affect only this new synthetic database. The
            # provisioner must refuse unsafe ambient grants, not repair globals.
            conn.execute(sql.SQL('REVOKE TEMP ON DATABASE {} FROM PUBLIC').format(sql.Identifier(name)))
            conn.execute('REVOKE ALL ON ALL SEQUENCES IN SCHEMA public,paradedb,pdb FROM PUBLIC')
            conn.execute('REVOKE ALL ON ALL TABLES IN SCHEMA public,paradedb,pdb FROM PUBLIC')
            conn.execute('REVOKE CREATE ON SCHEMA public,paradedb,pdb FROM PUBLIC')
            functions = conn.execute("SELECT p.oid::regprocedure::text FROM pg_proc p JOIN pg_depend d ON d.classid='pg_proc'::regclass AND d.objid=p.oid JOIN pg_extension e ON e.oid=d.refobjid WHERE d.refclassid='pg_extension'::regclass AND e.extname='pg_search'").fetchall()
            for function, in functions:
                conn.execute(sql.SQL('REVOKE ALL ON ROUTINE {} FROM PUBLIC').format(sql.SQL(function)))
            for owner in owners:
                conn.execute('INSERT INTO users(id,email) VALUES(%s,%s)',(owner,owner+'@example.test'))
            spec = importlib.util.spec_from_file_location('search_fixture_migration',root/'deploy/public/migrate_owner_partitions.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            module.migrate_owner_partitions(conn,owner_id=owners[0])
            from gmail_search.gateway.partitions import provision_owner_partitions
            provision_owner_partitions(conn,owners[1])
            for index,owner in enumerate(owners):
                word = 'needle' if index==0 else 'foreignonly'
                conn.execute("INSERT INTO messages(user_id,id,search_id,thread_id,subject,body_text,from_addr,to_addr,date) VALUES(%s,'same',1,'thread',%s,%s,'sender','recipient','2026-09-15')",(owner,word,word+' body'))
                conn.execute("INSERT INTO attachments(user_id,id,message_id,filename,mime_type,extracted_text) VALUES(%s,1,'same','file','text/plain',%s)",(owner,word))
                conn.execute("INSERT INTO propositions(user_id,id,message_id,thread_id,text,embedding,model) VALUES(%s,1,'same','thread',%s,%s,'fixture+v1')",(owner,word,struct.pack('<ff',1,0)))
                conn.execute("INSERT INTO embeddings(user_id,message_id,attachment_id,chunk_type,chunk_text,embedding,model) VALUES(%s,'same',1,'attachment_text',%s,%s,'fixture')",(owner,word,struct.pack('<ff',1,0)))
                conn.execute('INSERT INTO term_aliases(user_id,term,expansions) VALUES(%s,%s,%s)',(owner,word,'["alias"]'))
                conn.execute('INSERT INTO contact_frequency(user_id,email,score) VALUES(%s,%s,0.5)',(owner,word+'@example.test'))
            yield SimpleNamespace(conn=conn,dsn=target,owners=owners)
    finally:
        with psycopg.connect(dsn,autocommit=True) as conn:
            conn.execute(sql.SQL('DROP DATABASE {} WITH(FORCE)').format(sql.Identifier(name)))
            for role in roles:
                conn.execute(sql.SQL('DROP ROLE IF EXISTS {}').format(sql.Identifier(role)))


def provisioned(db, owner):
    password=secrets.token_urlsafe(40)
    provision_search_reader(db.conn,owner,password)
    return SearchCredential(owner,make_conninfo(db.dsn,user=search_role(owner),password=password))


def test_credentials_are_separate_and_cannot_select_role_settings():
    owner='alice'
    assert search_role(owner).startswith('gms_search_')
    assert search_role(owner)==search_role(owner)
    with pytest.raises(ValueError):
        SearchCredential(owner,'user=postgres dbname=example')
    with pytest.raises(ValueError):
        SearchCredential(owner,f'user={search_role(owner)} options=-csearch_path=public')
    credential=SearchCredential(owner,f'user={search_role(owner)} password=secret')
    assert 'secret' not in repr(credential)


def test_provisioned_direct_reader_is_owner_bound_and_has_exact_privileges(search_database):
    db=search_database
    for owner in db.owners:
        credential=provisioned(db,owner)
        verify_search_reader_access(db.conn,owner)
        with psycopg.connect(credential.dsn,autocommit=True) as reader:
            assert reader.execute('SELECT session_user,current_user').fetchone()==(search_role(owner),)*2
            reader.execute("SELECT set_config('app.user_id',%s,false)",(db.owners[1],))
            assert reader.execute('SELECT DISTINCT user_id FROM public.messages').fetchall()==[(owner,)]
            assert reader.execute('SELECT DISTINCT user_id FROM public.embeddings').fetchall()==[(owner,)]
            reader.execute('SET default_transaction_read_only=off')
            for statement in ('SELECT raw_json FROM public.messages','SELECT raw_path FROM public.attachments','SELECT * FROM public.users',
                              'SELECT * FROM public.query_cache',"SELECT nextval('public.attachments_id_seq')",'SET ROLE postgres',
                              "UPDATE public.messages SET subject='bad'",'CREATE TEMP TABLE bad(id int)'):
                with pytest.raises(psycopg.errors.InsufficientPrivilege):
                    reader.execute(statement)


def test_unexpected_public_function_grant_is_refused_without_repair(search_database):
    db=search_database
    db.conn.execute("CREATE FUNCTION public.unsafe() RETURNS int LANGUAGE sql AS $$SELECT 1$$")
    original=db.conn.execute("SELECT proacl FROM pg_proc WHERE oid='public.unsafe()'::regprocedure").fetchone()
    with pytest.raises(ValueError):
        provisioned(db,db.owners[0])
    assert db.conn.execute("SELECT proacl FROM pg_proc WHERE oid='public.unsafe()'::regprocedure").fetchone()==original


@pytest.mark.asyncio
async def test_reader_session_queries_and_releases_admission(search_database):
    db=search_database; owner=db.owners[0]
    credential=provisioned(db,owner)
    from gmail_search.gateway.data_admission import DataAdmission
    admission=DataAdmission()
    reader=SearchReader(SearchRegistry({owner:credential},is_active=lambda _:True),profile=SearchProfile('fixture','fixture+v1',2),admission=admission)
    async with reader.session(owner,deadline=asyncio.get_running_loop().time()+5,check_active=lambda:None) as queries:
        result=await queries.lexical_messages(('needle',),limit=20)
        assert [row.message_id for row in result.rows]==['same']
        assert result.complete
        assert (await queries.lexical_messages(('foreignonly',),limit=20)).rows==()
        assert (await queries.fact_count())==1


@pytest.mark.parametrize('statement',[
    'GRANT SELECT(raw_json) ON public.messages TO {role}',
    'GRANT SELECT ON public.users TO {role}',
    'GRANT EXECUTE ON FUNCTION paradedb.index_size(regclass) TO {role}',
    "ALTER ROLE {role} SET temp_file_limit='128MB'",
])
def test_provision_refuses_existing_role_drift(search_database,statement):
    db=search_database;owner=db.owners[0];provisioned(db,owner)
    if 'index_size' in statement:
        # Select an actual unapproved extension routine from this pinned build.
        function=db.conn.execute("SELECT p.oid::regprocedure::text FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace WHERE n.nspname='paradedb' AND p.prokind='f' AND p.proname NOT IN ('search_with_parse','score','with_index','parse_with_field') ORDER BY p.oid LIMIT 1").fetchone()[0]
        statement='GRANT EXECUTE ON FUNCTION '+function+' TO {role}'
    db.conn.execute(sql.SQL(statement).format(role=sql.Identifier(search_role(owner))))
    with pytest.raises(ValueError):
        provisioned(db,owner)


@pytest.mark.asyncio
async def test_message_and_chunk_clipping_is_explicit_before_transfer(search_database):
    db=search_database;owner=db.owners[0];credential=provisioned(db,owner)
    db.conn.execute('UPDATE public.messages SET body_text=%s WHERE user_id=%s',('z'*10000,owner))
    db.conn.execute('UPDATE public.embeddings SET chunk_text=%s WHERE user_id=%s',('z'*10000,owner))
    from gmail_search.gateway.data_admission import DataAdmission
    reader=SearchReader(SearchRegistry({owner:credential},is_active=lambda _:True),profile=SearchProfile('fixture','fixture+v1',2),admission=DataAdmission())
    async with reader.session(owner,deadline=asyncio.get_running_loop().time()+10,check_active=lambda:None) as queries:
        row=(await queries.hydrate_messages(('same',))).rows[0]
        assert len(row.body_text)==200 and row.body_bytes==10000 and row.body_complete is False
        eid=db.conn.execute('SELECT id FROM public.embeddings WHERE user_id=%s',(owner,)).fetchone()[0]
        row=(await queries.hydrate_embeddings((eid,))).rows[0]
        assert len(row.chunk_text)==200 and row.chunk_bytes==10000 and row.chunk_complete is False


@pytest.mark.asyncio
async def test_repeated_cancel_waits_for_close_before_admission_release(search_database,monkeypatch):
    from gmail_search.gateway.search_reader import _Session
    from gmail_search.gateway.data_admission import DataAdmission
    db=search_database;owner=db.owners[0];credential=provisioned(db,owner)
    admission=DataAdmission();entered=asyncio.Event();closing=asyncio.Event();ack=asyncio.Event()
    original=_Session.close
    async def gated_close(self):
        closing.set()
        await ack.wait()
        await original(self)
    monkeypatch.setattr(_Session,'close',gated_close)
    reader=SearchReader(SearchRegistry({owner:credential},is_active=lambda _:True),profile=SearchProfile('fixture','fixture+v1',2),admission=admission)
    async def run():
        async with reader.session(owner,deadline=asyncio.get_running_loop().time()+10,check_active=lambda:None):
            entered.set();await asyncio.Event().wait()
    task=asyncio.create_task(run());await asyncio.wait_for(entered.wait(),3)
    task.cancel();await asyncio.wait_for(closing.wait(),3);task.cancel();await asyncio.sleep(.02)
    assert admission.active[owner]==1 and not task.done()
    ack.set()
    with pytest.raises(asyncio.CancelledError):await task
    assert not admission.active


@pytest.mark.asyncio
async def test_revocation_and_deadline_close_snapshot(search_database):
    from gmail_search.gateway.data_admission import DataAdmission
    from gmail_search.gateway.search_reader import SearchLimits
    db=search_database;owner=db.owners[0];credential=provisioned(db,owner);active=True
    admission=DataAdmission()
    reader=SearchReader(SearchRegistry({owner:credential},is_active=lambda _:active),profile=SearchProfile('fixture','fixture+v1',2),admission=admission)
    with pytest.raises(PermissionError):
        async with reader.session(owner,deadline=asyncio.get_running_loop().time()+10,check_active=lambda:None):
            active=False;await asyncio.sleep(1)
    assert not admission.active
    active=True
    reader.limits=SearchLimits(deadline_seconds=.15)
    with pytest.raises(TimeoutError):
        async with reader.session(owner,deadline=asyncio.get_running_loop().time()+10,check_active=lambda:None):
            await asyncio.sleep(1)
    assert not admission.active


@pytest.mark.asyncio
@pytest.mark.parametrize('search_database',['quoted'],indirect=True)
async def test_quoted_owner_binding_and_fixed_literal_are_safe(search_database):
    from gmail_search.gateway.data_admission import DataAdmission
    db=search_database;owner=db.owners[0];credential=provisioned(db,owner)
    reader=SearchReader(SearchRegistry({owner:credential},is_active=lambda _:True),profile=SearchProfile('fixture','fixture+v1',2),admission=DataAdmission())
    async with reader.session(owner,deadline=asyncio.get_running_loop().time()+10,check_active=lambda:None) as q:
        await q._session.conn.execute('SET LOCAL plan_cache_mode=force_generic_plan')
        assert len((await q.lexical_messages(('needle',))).rows)==1
        assert (await q.lexical_messages(('foreignonly',))).rows==()


@pytest.mark.asyncio
async def test_cancel_running_database_query_closes_backend(search_database,monkeypatch):
    from gmail_search.gateway.search_reader import _Session
    from gmail_search.gateway.data_admission import DataAdmission
    db=search_database;owner=db.owners[0];credential=provisioned(db,owner)
    admission=DataAdmission();entered=asyncio.Event();pid=[]
    reader=SearchReader(SearchRegistry({owner:credential},is_active=lambda _:True),profile=SearchProfile('fixture','fixture+v1',2),admission=admission)
    original=_Session.count
    async def slow_count(self,statement,params):
        pid.append(self.conn.info.backend_pid);entered.set()
        return await original(self,'SELECT count(*) FROM pg_sleep(10)',[])
    monkeypatch.setattr(_Session,'count',slow_count)
    async def run():
        async with reader.session(owner,deadline=asyncio.get_running_loop().time()+10,check_active=lambda:None) as q:
            await q.fact_count()
    task=asyncio.create_task(run());await asyncio.wait_for(entered.wait(),3);await asyncio.sleep(.03)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):await asyncio.wait_for(task,3)
    assert not admission.active
    assert db.conn.execute('SELECT count(*) FROM pg_stat_activity WHERE pid=%s',(pid[0],)).fetchone()==(0,)


@pytest.mark.asyncio
async def test_runtime_refuses_function_and_partition_drift(search_database):
    from gmail_search.gateway.data_admission import DataAdmission
    db=search_database;owner=db.owners[0];credential=provisioned(db,owner)
    reader=SearchReader(SearchRegistry({owner:credential},is_active=lambda _:True),profile=SearchProfile('fixture','fixture+v1',2),admission=DataAdmission())
    db.conn.execute('DROP INDEX public.messages_bm25_idx')
    with pytest.raises(PermissionError):
        async with reader.session(owner,deadline=asyncio.get_running_loop().time()+10,check_active=lambda:None):
            pytest.fail('Drifted partition should not be admitted')
    assert not reader.admission.active


@pytest.mark.parametrize('grant',[
    'GRANT SELECT(body_text) ON public.messages TO {role} WITH GRANT OPTION',
    'GRANT USAGE ON SCHEMA public TO {role} WITH GRANT OPTION',
    'GRANT EXECUTE ON FUNCTION paradedb.score(anyelement) TO {role} WITH GRANT OPTION',
    'GRANT CONNECT ON DATABASE {database} TO {role} WITH GRANT OPTION',
    'CREATE TABLE public.unexpected(); GRANT SELECT ON public.unexpected TO {role}',
])
def test_delegation_and_zero_column_relation_grants_are_refused(search_database,grant):
    db=search_database;owner=db.owners[0];provisioned(db,owner)
    db.conn.execute(sql.SQL(grant).format(role=sql.Identifier(search_role(owner)),database=sql.Identifier(db.conn.info.dbname)))
    with pytest.raises(ValueError):
        verify_search_reader_access(db.conn,owner)
    with pytest.raises(ValueError):
        provisioned(db,owner)


def test_provisioning_bounds_all_ddl_waits_and_restores_settings(search_database):
    db=search_database;owner=db.owners[0]
    db.conn.execute("SET statement_timeout='3s'")
    db.conn.execute("SET lock_timeout='0'")
    with psycopg.connect(db.dsn,autocommit=True) as blocker:
        with blocker.transaction():
            blocker.execute('LOCK TABLE public.embeddings IN ACCESS SHARE MODE')
            with pytest.raises(psycopg.errors.LockNotAvailable):
                provisioned(db,owner)
    assert db.conn.execute("SELECT current_setting('statement_timeout'),current_setting('lock_timeout')").fetchone()==('3s','0')
    assert db.conn.execute('SELECT 1 FROM pg_roles WHERE rolname=%s',(search_role(owner),)).fetchone() is None


def test_provisioning_preserves_stricter_caller_settings_on_success(search_database):
    db=search_database
    with db.conn.transaction():
        db.conn.execute("SET LOCAL statement_timeout='750ms'")
        db.conn.execute("SET LOCAL lock_timeout='50ms'")
        provisioned(db,db.owners[0])
        assert db.conn.execute("SELECT current_setting('statement_timeout'),current_setting('lock_timeout')").fetchone()==('750ms','50ms')


@pytest.mark.asyncio
async def test_runtime_refuses_delegated_privilege_and_releases_capacity(search_database):
    from gmail_search.gateway.data_admission import DataAdmission
    db=search_database;owner=db.owners[0];credential=provisioned(db,owner)
    db.conn.execute(sql.SQL('GRANT SELECT(body_text) ON public.messages TO {} WITH GRANT OPTION').format(sql.Identifier(search_role(owner))))
    reader=SearchReader(SearchRegistry({owner:credential},is_active=lambda _:True),profile=SearchProfile('fixture','fixture+v1',2),admission=DataAdmission())
    with pytest.raises(PermissionError):
        async with reader.session(owner,deadline=asyncio.get_running_loop().time()+10,check_active=lambda:None):
            pytest.fail('Delegated privilege must fail runtime audit')
    assert not reader.admission.active
