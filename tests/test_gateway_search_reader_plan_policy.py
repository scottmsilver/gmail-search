"""Candidate planner policy; retained-table maintenance has known failing cases.

See docs/qualification/retained-reader-maintenance-blocker.md. Passing fresh-leaf
cases do not qualify retained tables. Keep the failing maintenance checks visible.
"""
import asyncio
import secrets
import os
from types import SimpleNamespace
from psycopg.conninfo import make_conninfo
import time

from psycopg import sql
import pytest

# BM25 statistics / plan assertions: needs a quiet database (see scripts/test.sh).
pytestmark = pytest.mark.pg_exclusive

from gmail_search.gateway.data_admission import DataAdmission
from gmail_search.gateway.partitions import provision_owner_partitions, partition_name, PARTITION_SCHEMA
from gmail_search.gateway.partition_profiles import NUMERIC_OWNER_PARTITIONS_V1 as NUMERIC, TEXT_OWNER_PARTITIONS_V1 as TEXT
from gmail_search.gateway.search_reader import SearchProfile, SearchReader, SearchRegistry, search_role
from test_gateway_search_reader import search_database as search_database, provisioned
from test_gateway_partition_profiles import text_database as text_database, credential_for, seed
from test_text_owner_partitions import source as source, migration as migration


@pytest.fixture
def direct_text_database(source,migration):
    migration.migrate_text_owner_partitions(source,owner_id="alice")
    provision_owner_partitions(source,"bob",profile=TEXT)
    source.execute("INSERT INTO messages(user_id,id,thread_id,subject,body_text,from_addr,to_addr,date) VALUES('bob','same','thread','foreignonly','foreignonly','sender','recipient','2026-09-15')")
    source.execute("INSERT INTO attachments(user_id,id,message_id,filename,extracted_text,mime_type) VALUES('bob',1,'same','file','foreignonly','text/plain')")
    source.execute("INSERT INTO propositions(user_id,id,message_id,text,model) VALUES('bob',1,'same','foreignonly','fixture')")
    source.execute(sql.SQL("REVOKE TEMP ON DATABASE {} FROM PUBLIC").format(sql.Identifier(source.info.dbname)))
    source.execute("REVOKE ALL ON ALL TABLES IN SCHEMA public,paradedb,pdb FROM PUBLIC")
    source.execute("REVOKE ALL ON ALL SEQUENCES IN SCHEMA public,paradedb,pdb FROM PUBLIC")
    source.execute("REVOKE CREATE ON SCHEMA public,paradedb,pdb FROM PUBLIC")
    routines=source.execute("SELECT p.oid::regprocedure::text FROM pg_proc p JOIN pg_depend d ON d.classid='pg_proc'::regclass AND d.objid=p.oid JOIN pg_extension e ON e.oid=d.refobjid WHERE d.refclassid='pg_extension'::regclass AND e.extname='pg_search'").fetchall()
    for routine, in routines:
        source.execute(sql.SQL("REVOKE ALL ON ROUTINE {} FROM PUBLIC").format(sql.SQL(routine)))
    try:
        yield SimpleNamespace(conn=source,dsn=make_conninfo(os.environ["GMS_TEST_PG_DSN"],dbname=source.info.dbname),owners=("alice","bob"))
    finally:
        for owner in ("alice","bob"):
            role=search_role(owner,profile=TEXT)
            if source.execute("SELECT 1 FROM pg_roles WHERE rolname=%s",(role,)).fetchone():
                source.execute(sql.SQL("DROP OWNED BY {}").format(sql.Identifier(role)))
                source.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role)))


@pytest.mark.asyncio
async def test_fixed_reader_uses_audited_custom_unprepared_policy(search_database):
    db = search_database
    owner = db.owners[0]
    credential = provisioned(db,owner)
    reader = SearchReader(SearchRegistry({owner:credential},is_active=lambda _:True),
        profile=SearchProfile('fixture','fixture+v1',2),admission=DataAdmission())
    async with reader.session(owner,deadline=asyncio.get_running_loop().time()+10,check_active=lambda:None) as q:
        assert q._session.conn.prepare_threshold is None
        cursor = await q._session.conn.execute("SELECT current_setting('plan_cache_mode')")
        assert await cursor.fetchone() == ('force_custom_plan',)


@pytest.mark.asyncio
@pytest.mark.parametrize('mode',['auto','missing'])
async def test_old_or_drifted_plan_policy_is_refused_without_repair(search_database,mode):
    db = search_database
    owner = db.owners[0]
    credential = provisioned(db,owner)
    role = search_role(owner)
    statement = "ALTER ROLE {} SET plan_cache_mode='auto'" if mode=='auto' else 'ALTER ROLE {} RESET plan_cache_mode'
    db.conn.execute(sql.SQL(statement).format(sql.Identifier(role)))
    before = db.conn.execute('SELECT rolconfig FROM pg_roles WHERE rolname=%s',(role,)).fetchone()
    with pytest.raises(ValueError):
        provisioned(db,owner)
    assert db.conn.execute('SELECT rolconfig FROM pg_roles WHERE rolname=%s',(role,)).fetchone() == before
    admission = DataAdmission()
    reader = SearchReader(SearchRegistry({owner:credential},is_active=lambda _:True),
        profile=SearchProfile('fixture','fixture+v1',2),admission=admission)
    with pytest.raises(PermissionError):
        async with reader.session(owner,deadline=asyncio.get_running_loop().time()+10,check_active=lambda:None):
            pytest.fail('unqualified planner policy opened')
    assert not admission.active


@pytest.mark.asyncio
@pytest.mark.parametrize('fixture,profile', [('search_database',NUMERIC),('text_database',TEXT),('direct_text_database',TEXT)])
@pytest.mark.parametrize('branch', ['messages','attachments','facts'])
async def test_three_owner_fixed_readers_survive_three_own_and_foreign_vacuum_cycles(request,fixture,profile,branch):
    db = request.getfixturevalue(fixture)
    if fixture=='text_database':seed(db)
    third = 'charlie_'+secrets.token_hex(8)
    if fixture=='text_database':db.conn.execute('INSERT INTO public.users(id) VALUES(%s)',(third,))
    else:db.conn.execute('INSERT INTO public.users(id,email) VALUES(%s,%s)',(third,third+'@example.test'))
    provision_owner_partitions(db.conn,third,profile=profile)
    owners = (*db.owners,third)
    db.conn.execute("INSERT INTO public.messages(user_id,id,thread_id,subject,body_text,from_addr,to_addr,date) VALUES(%s,'same','thread','charlieword','charlieword','sender','recipient','2026-09-15')",(third,))
    db.conn.execute("INSERT INTO public.attachments(user_id,id,message_id,filename,mime_type,extracted_text) VALUES(%s,1,'same','file','text/plain','charlieword')",(third,))
    db.conn.execute("INSERT INTO public.propositions(user_id,id,message_id,text,model) VALUES(%s,1,'same','charlieword','fixture+v1')",(third,))
    third_role = search_role(third,profile=profile)
    assert db.conn.execute('SELECT 1 FROM pg_roles WHERE rolname=%s',(third_role,)).fetchone() is None
    readers = {}
    credential_factory = credential_for if profile is TEXT else provisioned
    started = time.monotonic()
    samples = []
    try:
        for owner in owners:
            credential = credential_factory(db,owner)
            readers[owner] = SearchReader(SearchRegistry({owner:credential},is_active=lambda _:True),
                profile=SearchProfile('fixture','fixture+v1',2,schema_profile=profile),admission=DataAdmission())

        async def snapshot(owner):
            result = []
            reader = readers[owner]
            async with reader.session(owner,deadline=asyncio.get_running_loop().time()+10,check_active=lambda:None) as q:
                assert q._session.conn.prepare_threshold is None
                method = getattr(q,'lexical_'+branch)
                for _ in range(2):
                    page = await method(('needle','foreignonly','charlieword'))
                    assert page.complete and [row.message_id for row in page.rows] == ['same']
                    result.append(tuple((row.id,row.message_id,row.score) for row in page.rows))
                    if branch != 'facts':
                        restricted = await method(('needle','foreignonly','charlieword'),candidate_ids=('same',))
                        assert restricted.complete and [row.message_id for row in restricted.rows] == ['same']
                        result.append(tuple((row.id,row.message_id,row.score) for row in restricted.rows))
                cursor = await q._session.conn.execute('SELECT count(*) FROM pg_prepared_statements')
                assert await cursor.fetchone() == (0,)
            assert not reader.admission.active
            return tuple(result)

        def churn(owner,cycle):
            mid = 'churn_'+str(cycle)
            key = 1000+cycle
            # A separate owner-local message ensures its deletion doesn't touch
            # fixture baseline rows or dependent data. Numeric IDs are preserved.
            db.conn.execute("INSERT INTO public.messages(user_id,id,thread_id,subject,body_text,from_addr,to_addr,date) VALUES(%s,%s,'thread','needle foreignonly charlieword','needle foreignonly charlieword','sender','recipient','2026-09-15')",(owner,mid))
            db.conn.execute("INSERT INTO public.attachments(user_id,id,message_id,filename,mime_type,extracted_text) VALUES(%s,%s,%s,'churn','text/plain','needle foreignonly charlieword')",(owner,key,mid))
            db.conn.execute("INSERT INTO public.propositions(user_id,id,message_id,text,model) VALUES(%s,%s,%s,'needle foreignonly charlieword','fixture+v1')",(owner,key,mid))
            for table in ('propositions','attachments'):
                db.conn.execute(sql.SQL('DELETE FROM public.{} WHERE user_id=%s AND id=%s').format(sql.Identifier(table)),(owner,key))
            db.conn.execute('DELETE FROM public.messages WHERE user_id=%s AND id=%s',(owner,mid))
            for table in ('messages','attachments','propositions'):
                db.conn.execute(sql.SQL('VACUUM(INDEX_CLEANUP ON) {}').format(sql.Identifier(PARTITION_SCHEMA,partition_name(table,owner))))

        # Match the maintenance probe's starting state: rebuilt owner indexes,
        # committed before subsequent own and foreign INSERT/DELETE/VACUUM.
        for owner in owners:
            for table in ('messages','attachments','propositions'):
                child = sql.Identifier(PARTITION_SCHEMA,partition_name(table,owner)).as_string(db.conn)
                index = db.conn.execute("SELECT n.nspname,c.relname FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid JOIN pg_namespace n ON n.oid=c.relnamespace JOIN pg_am a ON a.oid=c.relam WHERE i.indrelid=%s::regclass AND a.amname='bm25'",(child,)).fetchone()
                assert index is not None
                db.conn.execute(sql.SQL('REINDEX INDEX {}').format(sql.Identifier(*index)))
        baseline = {owner:await snapshot(owner) for owner in owners}
        for cycle in range(3):
            for changed in owners:
                before = dict(baseline)
                churn(changed,cycle)
                # Own score changes are permitted; another owner's maintenance
                # must not alter scores or results in this owner's leaf.
                for owner in owners:
                    tick = time.monotonic()
                    actual = await snapshot(owner)
                    samples.append(time.monotonic()-tick)
                    if owner != changed:assert actual == before[owner]
                    baseline[owner] = actual
        print(f'\nSynthetic {profile.value}/{branch}: 27 post-maintenance snapshots; total={time.monotonic()-started:.3f}s; mean_snapshot={sum(samples)/len(samples):.3f}s')
    finally:
        if db.conn.execute('SELECT 1 FROM pg_roles WHERE rolname=%s',(third_role,)).fetchone():
            db.conn.execute(sql.SQL('DROP OWNED BY {}').format(sql.Identifier(third_role)))
            db.conn.execute(sql.SQL('DROP ROLE {}').format(sql.Identifier(third_role)))


@pytest.mark.asyncio
async def test_effective_database_role_override_is_refused_before_local_repair(search_database):
    db = search_database
    owner = db.owners[0]
    credential = provisioned(db,owner)
    role = search_role(owner)
    # A database-specific setting overrides the otherwise correct role setting.
    db.conn.execute(sql.SQL("ALTER ROLE {} IN DATABASE {} SET plan_cache_mode='auto'").format(
        sql.Identifier(role),sql.Identifier(db.conn.info.dbname)))
    admission = DataAdmission()
    reader = SearchReader(SearchRegistry({owner:credential},is_active=lambda _:True),
        profile=SearchProfile('fixture','fixture+v1',2),admission=admission)
    with pytest.raises(PermissionError):
        async with reader.session(owner,deadline=asyncio.get_running_loop().time()+10,check_active=lambda:None):
            pytest.fail('effective generic policy silently repaired')
    assert not admission.active
