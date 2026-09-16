"""Full canonical TEXT profile; disposable pinned ParadeDB only."""
import asyncio
from dataclasses import FrozenInstanceError
import os
import secrets
from types import SimpleNamespace

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
import pytest

from gmail_search.gateway.partition_profiles import NUMERIC_OWNER_PARTITIONS_V1 as NUMERIC, TEXT_OWNER_PARTITIONS_V1 as TEXT
from gmail_search.gateway import partitions
from gmail_search.gateway.search_reader import SearchCredential, SearchProfile, SearchReader, SearchRegistry, search_role, search_columns
from gmail_search.gateway.provision_search_reader import provision_search_reader, verify_search_reader_access
from gmail_search.gateway.data_admission import DataAdmission


def test_profiles_are_closed_and_credentials_are_bound():
    assert NUMERIC.message_key == 'search_id' and TEXT.message_key == 'id'
    credential = SearchCredential('alice', f'user={search_role("alice", profile=TEXT)}', schema_profile=TEXT)
    with pytest.raises(FrozenInstanceError):
        credential.schema_profile = NUMERIC
    for value in ('text', None, object()):
        with pytest.raises(ValueError):
            SearchProfile('fixture', 'fixture+v1', 2, schema_profile=value)
    with pytest.raises(ValueError):
        SearchCredential('alice', f'user={search_role("alice")}', schema_profile=TEXT)
    assert 'search_id' not in search_columns(TEXT)['messages']
    with pytest.raises(TypeError):
        search_columns(TEXT)['messages']['id'] = 'int8'


@pytest.fixture
def text_database():
    dsn = os.getenv('GMS_TEST_PG_DSN')
    if not dsn:
        pytest.skip('Explicit synthetic ParadeDB required')
    cfg = conninfo_to_dict(dsn)
    assert (cfg.get('host'), cfg.get('port'), cfg.get('dbname'), cfg.get('user')) == ('127.0.0.1', '55440', 'postgres', 'postgres')
    assert not set(cfg) & {'hostaddr', 'service', 'options'}
    suffix = secrets.token_hex(8)
    name = 'gms_text_profile_' + suffix
    owners = ("alice'\\_" + suffix, 'bob_' + suffix)
    roles = [search_role(owner, profile=TEXT) for owner in owners]
    with psycopg.connect(dsn, autocommit=True) as conn:
        assert conn.info.hostaddr == '127.0.0.1'
        conn.execute(sql.SQL('CREATE DATABASE {} TEMPLATE template0').format(sql.Identifier(name)))
    target = make_conninfo(dsn, dbname=name)
    try:
        with psycopg.connect(target, autocommit=True) as conn:
            conn.execute('CREATE EXTENSION pg_search')
            conn.execute('CREATE TABLE public.users(id text PRIMARY KEY)')
            for owner in owners:
                conn.execute('INSERT INTO public.users VALUES(%s)', (owner,))
            conn.execute('''CREATE TABLE public.messages(user_id text NOT NULL REFERENCES public.users ON DELETE CASCADE,
                id text NOT NULL,thread_id text,subject text,body_text text,from_addr text,to_addr text,date text,labels text,
                PRIMARY KEY(user_id,id)) PARTITION BY LIST(user_id)''')
            conn.execute('''CREATE TABLE public.attachments(user_id text NOT NULL REFERENCES public.users ON DELETE CASCADE,
                id bigint NOT NULL,message_id text NOT NULL,filename text NOT NULL,extracted_text text,mime_type text,fetch_status text,size_bytes bigint,
                PRIMARY KEY(user_id,id),UNIQUE(user_id,message_id,filename),UNIQUE(user_id,message_id,id),
                FOREIGN KEY(user_id,message_id) REFERENCES public.messages(user_id,id)) PARTITION BY LIST(user_id)''')
            conn.execute('''CREATE TABLE public.propositions(user_id text NOT NULL,id bigint NOT NULL,message_id text NOT NULL,
                thread_id text,text text,model text,embedding bytea,PRIMARY KEY(user_id,id),
                FOREIGN KEY(user_id,message_id) REFERENCES public.messages(user_id,id) ON DELETE CASCADE) PARTITION BY LIST(user_id)''')
            for table, columns in search_columns(TEXT).items():
                if table not in ('messages', 'attachments', 'propositions'):
                    definitions = [sql.SQL('{} {}').format(sql.Identifier(column), sql.SQL(kind)) for column, kind in columns.items()]
                    conn.execute(sql.SQL('CREATE TABLE public.{}({})').format(sql.Identifier(table), sql.SQL(',').join(definitions)))
                conn.execute(sql.SQL('ALTER TABLE public.{} ENABLE ROW LEVEL SECURITY').format(sql.Identifier(table)))
                conn.execute(sql.SQL('ALTER TABLE public.{} FORCE ROW LEVEL SECURITY').format(sql.Identifier(table)))
            conn.execute("CREATE INDEX messages_bm25_idx ON public.messages USING bm25(id,subject,body_text,from_addr,to_addr) WITH(key_field='id')")
            conn.execute("CREATE INDEX attachments_bm25_idx ON public.attachments USING bm25(id,filename,extracted_text) WITH(key_field='id')")
            conn.execute("CREATE INDEX props_bm25_idx ON public.propositions USING bm25(id,text) WITH(key_field='id')")
            conn.execute(sql.SQL('REVOKE TEMP ON DATABASE {} FROM PUBLIC').format(sql.Identifier(name)))
            conn.execute('REVOKE ALL ON ALL TABLES IN SCHEMA public,paradedb,pdb FROM PUBLIC')
            conn.execute('REVOKE ALL ON ALL SEQUENCES IN SCHEMA public,paradedb,pdb FROM PUBLIC')
            conn.execute('REVOKE CREATE ON SCHEMA public,paradedb,pdb FROM PUBLIC')
            routines = conn.execute("SELECT p.oid::regprocedure::text FROM pg_proc p JOIN pg_depend d ON d.classid='pg_proc'::regclass AND d.objid=p.oid JOIN pg_extension e ON e.oid=d.refobjid WHERE d.refclassid='pg_extension'::regclass AND e.extname='pg_search'").fetchall()
            for routine, in routines:
                conn.execute(sql.SQL('REVOKE ALL ON ROUTINE {} FROM PUBLIC').format(sql.SQL(routine)))
            for owner in owners:
                partitions.provision_owner_partitions(conn, owner, profile=TEXT)
            yield SimpleNamespace(conn=conn, dsn=target, owners=owners)
    finally:
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(sql.SQL('DROP DATABASE {} WITH(FORCE)').format(sql.Identifier(name)))
            for role in roles:
                conn.execute(sql.SQL('DROP ROLE IF EXISTS {}').format(sql.Identifier(role)))


def credential_for(db, owner):
    password = secrets.token_urlsafe(40)
    provision_search_reader(db.conn, owner, password, profile=TEXT)
    return SearchCredential(owner, make_conninfo(db.dsn, user=search_role(owner, profile=TEXT), password=password), schema_profile=TEXT)


def seed(db):
    for owner, word in zip(db.owners, ('needle', 'foreignonly')):
        db.conn.execute("INSERT INTO public.messages(user_id,id,thread_id,subject,body_text) VALUES(%s,'same','thread',%s,%s)", (owner, word, word))
        db.conn.execute("INSERT INTO public.attachments(user_id,id,message_id,filename,extracted_text) VALUES(%s,1,'same','file',%s)", (owner, word))
        db.conn.execute("INSERT INTO public.propositions(user_id,id,message_id,text) VALUES(%s,1,'same',%s)", (owner, word))


def test_text_partition_lifecycle_and_wrong_profile(text_database):
    db = text_database
    before = db.conn.execute("SELECT oid FROM pg_class WHERE relnamespace='gms_mail_partitions'::regnamespace ORDER BY oid").fetchall()
    for operation in (partitions.verify_owner_partitions, partitions.provision_owner_partitions):
        operation(db.conn, db.owners[0], profile=TEXT)
        with pytest.raises(ValueError):
            operation(db.conn, db.owners[0])
    assert db.conn.execute("SELECT oid FROM pg_class WHERE relnamespace='gms_mail_partitions'::regnamespace ORDER BY oid").fetchall() == before
    with db.conn.transaction():
        db.conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        partitions.inspect_owner_partitions(db.conn, db.owners[0], profile=TEXT)
    partitions.remove_empty_owner_partitions(db.conn, db.owners[1], profile=TEXT)
    partitions.provision_owner_partitions(db.conn, db.owners[1], profile=TEXT)
    with pytest.raises(ValueError):
        provision_search_reader(db.conn, db.owners[0], secrets.token_urlsafe(40))
    assert db.conn.execute('SELECT 1 FROM pg_roles WHERE rolname=%s', (search_role(db.owners[0]),)).fetchone() is None


@pytest.mark.asyncio
async def test_profile_mismatch_precedes_admission_and_connection(monkeypatch):
    credential = SearchCredential('alice', f'user={search_role("alice", profile=TEXT)}', schema_profile=TEXT)
    admission = DataAdmission()
    reader = SearchReader(SearchRegistry({'alice':credential}, is_active=lambda _:True), profile=SearchProfile('fixture','fixture+v1',2), admission=admission)
    async def forbidden(*args, **kwargs):
        pytest.fail('wrong profile connected')
    monkeypatch.setattr(psycopg.AsyncConnection, 'connect', forbidden)
    with pytest.raises(PermissionError):
        async with reader.session('alice', deadline=asyncio.get_running_loop().time()+5, check_active=lambda:None):
            pytest.fail('wrong profile entered')
    assert not admission.active


@pytest.mark.asyncio
async def test_actual_text_reader_branches_and_fixed_generic_owner(text_database):
    db = text_database
    seed(db)
    credential = credential_for(db, db.owners[0])
    verify_search_reader_access(db.conn, db.owners[0], profile=TEXT)
    reader = SearchReader(SearchRegistry({db.owners[0]:credential},is_active=lambda _:True), profile=SearchProfile('fixture','fixture+v1',2,schema_profile=TEXT), admission=DataAdmission())
    for mode in ('force_custom_plan','force_generic_plan'):
        async with reader.session(db.owners[0],deadline=asyncio.get_running_loop().time()+10,check_active=lambda:None) as q:
            await q._session.conn.execute('SELECT set_config(\'plan_cache_mode\',%s,true)',(mode,))
            q._session.conn.prepare_threshold = 0
            for method in (q.lexical_messages,q.lexical_attachments,q.lexical_facts):
                rows = (await method(('needle',))).rows
                assert len(rows) == 1 and rows[0].message_id == 'same'
                assert type(rows[0].id) is (str if method == q.lexical_messages else int)
                assert (await method(('foreignonly',))).rows == ()
            assert (await q.lexical_messages(('needle',),candidate_ids=('missing',))).rows == ()
            assert [row.id for row in (await q.lexical_messages(('needle',),candidate_ids=('same',))).rows] == ['same']
    with psycopg.connect(credential.dsn,autocommit=True) as conn:
        conn.execute("SELECT set_config('app.user_id',%s,false)",(db.owners[1],))
        assert conn.execute('SELECT DISTINCT user_id FROM public.messages').fetchall() == [(db.owners[0],)]
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            conn.execute(sql.SQL('SELECT id FROM {}').format(sql.Identifier(partitions.PARTITION_SCHEMA,partitions.partition_name('messages',db.owners[0]))))


@pytest.mark.parametrize('mode', ['force_custom_plan','force_generic_plan'])
@pytest.mark.parametrize('restricted', [False, True])
def test_text_scores_and_plans_ignore_foreign_insert_update_delete(text_database, mode, restricted):
    db = text_database
    seed(db)
    owner, foreign = db.owners
    credential = credential_for(db, owner)
    with psycopg.connect(credential.dsn, autocommit=True) as conn:
        conn.execute("SELECT set_config('plan_cache_mode',%s,false)", (mode,))
        literal = sql.Literal(owner).as_string(conn)
        statement = f'SELECT id,paradedb.score(id) FROM public.messages WHERE user_id={literal} AND id OPERATOR(pg_catalog.@@@) %s ORDER BY paradedb.score(id) DESC,id LIMIT %s'
        if restricted:
            statement = statement.replace(' ORDER BY', ' AND id=ANY(%s::text[]) ORDER BY')
        params = ('subject:needle',['same'],20) if restricted else ('subject:needle',20)
        before = conn.execute(statement, params, prepare=True).fetchall()
        assert len(before) == 1
        plans = conn.execute('EXPLAIN (FORMAT JSON) ' + statement, params, prepare=True).fetchone()[0]
        import json
        plan = json.dumps(plans)
        assert partitions.partition_name('messages', owner) in plan
        assert partitions.partition_name('messages', foreign) not in plan
        db.conn.execute("INSERT INTO public.messages(user_id,id,subject,body_text) SELECT %s,'churn-'||n,'needle','needle' FROM generate_series(1,200) n", (foreign,))
        assert conn.execute(statement, params, prepare=True).fetchall() == before
        db.conn.execute("UPDATE public.messages SET subject=repeat('needle ',50) WHERE user_id=%s AND id LIKE 'churn-%%'", (foreign,))
        assert conn.execute(statement, params, prepare=True).fetchall() == before
        db.conn.execute("DELETE FROM public.messages WHERE user_id=%s AND id LIKE 'churn-%%'", (foreign,))
        assert conn.execute(statement, params, prepare=True).fetchall() == before


@pytest.mark.parametrize('fault', ['extra_numeric_column','index','grant','binding'])
def test_text_profile_drift_is_refused_before_reader_publication(text_database, fault):
    db = text_database
    owner = db.owners[0]
    child = sql.Identifier(partitions.PARTITION_SCHEMA,partitions.partition_name('messages',owner))
    if fault == 'extra_numeric_column':
        db.conn.execute('ALTER TABLE public.messages ADD COLUMN search_id bigint')
    elif fault == 'index':
        db.conn.execute('DROP INDEX public.messages_bm25_idx')
        db.conn.execute("CREATE INDEX messages_bm25_idx ON public.messages USING bm25(id,subject,body_text) WITH(key_field='id')")
    elif fault == 'grant':
        db.conn.execute(sql.SQL('GRANT SELECT(id) ON {} TO PUBLIC').format(child))
    else:
        db.conn.execute(sql.SQL("COMMENT ON TABLE {} IS 'forged'").format(child))
    with pytest.raises(ValueError):
        credential_for(db,owner)
    assert db.conn.execute('SELECT 1 FROM pg_roles WHERE rolname=%s',(search_role(owner,profile=TEXT),)).fetchone() is None


def test_text_cleanup_rollback_keeps_original_owner_leaves(text_database):
    db = text_database
    owner = db.owners[1]
    before = db.conn.execute("SELECT oid FROM pg_class WHERE relnamespace='gms_mail_partitions'::regnamespace ORDER BY oid").fetchall()
    with pytest.raises(RuntimeError):
        with db.conn.transaction():
            partitions.remove_empty_owner_partitions(db.conn,owner,profile=TEXT)
            raise RuntimeError('synthetic rollback')
    partitions.verify_owner_partitions(db.conn,owner,profile=TEXT)
    assert db.conn.execute("SELECT oid FROM pg_class WHERE relnamespace='gms_mail_partitions'::regnamespace ORDER BY oid").fetchall() == before


def test_lexical_keys_preserve_distinct_types():
    from gmail_search.gateway.search_queries import LexicalHit, TextMessageLexicalHit
    for wrong in ('1', True, None):
        with pytest.raises(ValueError):
            LexicalHit(wrong,'same',1.0)
    for wrong in (1,True,None,'foreign'):
        with pytest.raises(ValueError):
            TextMessageLexicalHit(wrong,'same',1.0)


@pytest.mark.asyncio
async def test_composed_wrong_profile_does_not_acquire_index_or_embed(tmp_path):
    from test_gateway_search_service import _service
    service, caps, token, _, indexes, embedder = _service(tmp_path)
    credential = SearchCredential('alice',f'user={search_role("alice",profile=TEXT)}',schema_profile=TEXT)
    admission = DataAdmission()
    service.reader = SearchReader(SearchRegistry({'alice':credential},is_active=lambda _:True),
        profile=SearchProfile('test-model','test-facts',2),admission=admission)
    with pytest.raises(PermissionError):
        await service.search(token.secret,query='needle')
    assert indexes.owners == [] and embedder.calls == [] and not admission.active


def test_text_helper_grants_are_exact_and_drift_refused(text_database):
    from gmail_search.gateway.search_reader import search_functions
    db = text_database
    owner = db.owners[0]
    credential_for(db,owner)
    assert len(search_functions(NUMERIC)) == 4 and len(search_functions(TEXT)) == 5
    actual = db.conn.execute("""SELECT p.oid::regprocedure::text FROM pg_proc p
        JOIN pg_depend d ON d.classid='pg_proc'::regclass AND d.objid=p.oid
        JOIN pg_extension e ON e.oid=d.refobjid WHERE d.refclassid='pg_extension'::regclass
        AND e.extname='pg_search' AND has_function_privilege(%s,p.oid,'EXECUTE')""",(search_role(owner,profile=TEXT),)).fetchall()
    expected = {db.conn.execute('SELECT %s::regprocedure::oid',(signature,)).fetchone()[0] for signature in search_functions(TEXT)}
    observed = {db.conn.execute('SELECT %s::regprocedure::oid',(signature,)).fetchone()[0] for signature, in actual}
    assert observed == expected
    signature = 'paradedb.terms_with_operator(paradedb.fieldname,text,anyelement,boolean)'
    with db.conn.transaction():
        db.conn.execute(sql.SQL('ALTER FUNCTION {} CALLED ON NULL INPUT').format(sql.SQL(signature)))
        with pytest.raises(ValueError):
            verify_search_reader_access(db.conn,owner,profile=TEXT)


@pytest.mark.asyncio
async def test_actual_schema_drift_fails_before_composed_index_or_provider(text_database,tmp_path):
    from gmail_search.gateway.capabilities import Capabilities
    from gmail_search.gateway.registry import Registry
    from gmail_search.gateway.search_service import RunSearchService, OwnerSearchContext
    from test_gateway_search_service import Indexes
    db = text_database
    owner = db.owners[0]
    credential = credential_for(db,owner)
    # Role/profile labels still match: the actual catalog must reject drift.
    db.conn.execute('ALTER TABLE public.messages ADD COLUMN search_id bigint')
    admission = DataAdmission()
    reader = SearchReader(SearchRegistry({owner:credential},is_active=lambda _:True),profile=SearchProfile('fixture','fixture+v1',2,schema_profile=TEXT),admission=admission)
    caps = Capabilities(Registry(tmp_path/'profile.sqlite',is_active=lambda _:True))
    run = caps.registry.start_run(owner,'conversation',request_key='profile',writer=False)
    token = caps.issue(run.run_id,audience='retrieval',operations={'search'})
    indexes = Indexes()
    class NeverEmbed:
        model = 'fixture'
        dimensions = 2
        async def embed(self,*args,**kwargs):
            pytest.fail('catalog drift reached provider')
    service = RunSearchService(caps,reader,indexes,NeverEmbed(),owners={owner:OwnerSearchContext(owner,(owner+'@example.test',))},reranker=None)
    with pytest.raises(PermissionError):
        await service.search(token.secret,query='needle')
    assert indexes.owners == [] and not admission.active
