"""Application writer roles tested only in generated disposable PostgreSQL DBs."""
import os
import secrets

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
import pytest

from gmail_search.gateway.provision_writer import provision_application_writer,verify_writer_access
from gmail_search.gateway.writer import application_writer_role, WriterCredential, WriterRegistry
from gmail_search.gateway.database import reader_role
from gmail_search.gateway.partition_profiles import NUMERIC_OWNER_PARTITIONS_V1 as NUMERIC


@pytest.fixture(autouse=True)
def _pg_isolation():
    yield None


@pytest.fixture
def writer_database():
    dsn=os.environ.get('GMS_GATEWAY_TEST_DSN')
    if not dsn:
        pytest.skip('Disposable GMS_GATEWAY_TEST_DSN required')
    suffix=secrets.token_hex(8)
    name='gms_writer_test_'+suffix
    owners=('alice_'+suffix,'bob_'+suffix)
    with psycopg.connect(dsn,autocommit=True) as admin:
        admin.execute(sql.SQL('CREATE DATABASE {} TEMPLATE template0').format(sql.Identifier(name)))
    config=conninfo_to_dict(dsn);config['dbname']=name
    target=make_conninfo(**config)
    try:
        with psycopg.connect(target,autocommit=True) as conn:
            conn.execute(sql.SQL('REVOKE TEMP ON DATABASE {} FROM PUBLIC').format(sql.Identifier(name)))
            conn.execute('''CREATE TABLE users(id text PRIMARY KEY,email text UNIQUE,google_sub text UNIQUE);
                CREATE TABLE conversations(id text PRIMARY KEY,title text,created_at text DEFAULT now(),updated_at text DEFAULT now(),user_id text);
                CREATE TABLE conversation_messages(id bigserial PRIMARY KEY,conversation_id text REFERENCES conversations(id) ON DELETE CASCADE,
                    seq bigint,role text,parts text,created_at text DEFAULT now(),UNIQUE(conversation_id,seq));
                CREATE TABLE agent_sessions(id text PRIMARY KEY,conversation_id text,mode text,question text,plan jsonb,final_answer text,
                    started_at timestamptz DEFAULT now(),finished_at timestamptz,status text DEFAULT 'running',user_id text);
                CREATE TABLE agent_events(id bigserial PRIMARY KEY,session_id text REFERENCES agent_sessions(id),seq integer,agent_name text,
                    kind text,payload jsonb,created_at timestamptz DEFAULT now(),UNIQUE(session_id,seq));
                CREATE TABLE costs(id bigserial PRIMARY KEY,timestamp text,operation text,model text,input_tokens bigint DEFAULT 0,
                    image_count bigint DEFAULT 0,estimated_cost_usd double precision,message_id text,output_tokens bigint DEFAULT 0,
                    cached_input_tokens bigint DEFAULT 0,cache_write_tokens bigint DEFAULT 0,user_id text);
                CREATE TABLE messages(user_id text,id text,body_text text);
                CREATE TABLE sync_state(key text PRIMARY KEY,value text)''')
            for table in ('conversations','conversation_messages','agent_sessions','agent_events','costs'):
                conn.execute(sql.SQL('ALTER TABLE {} ENABLE ROW LEVEL SECURITY').format(sql.Identifier(table)))
                conn.execute(sql.SQL('CREATE POLICY legacy_public ON {} USING(true) WITH CHECK(true)').format(sql.Identifier(table)))
            from gmail_search.gateway.browser_conversations import BrowserConversations
            BrowserConversations.install(conn)
            for owner in owners:
                conn.execute('INSERT INTO users VALUES(%s,%s,%s)',(owner,owner+'@example.test','google-'+owner))
                conn.execute('INSERT INTO conversations(id,user_id,title) VALUES(%s,%s,%s)',(owner,owner,owner+' private'))
                conn.execute('INSERT INTO agent_sessions(id,conversation_id,user_id) VALUES(%s,%s,%s)',(owner,owner,owner))
                conn.execute("INSERT INTO conversation_messages(conversation_id,seq,role,parts) VALUES(%s,1,'user',%s)",(owner,owner))
                conn.execute("INSERT INTO agent_events(session_id,seq,agent_name,kind,payload) VALUES(%s,1,'agent','text',%s::jsonb)",(owner,'{"text":"'+owner+'"}'))
        yield target,owners
    finally:
        with psycopg.connect(dsn,autocommit=True) as admin:
            admin.execute(sql.SQL('DROP DATABASE {} WITH(FORCE)').format(sql.Identifier(name)))
            for owner in owners:
                admin.execute(sql.SQL('DROP ROLE IF EXISTS {}').format(sql.Identifier(application_writer_role(owner))))
                admin.execute(sql.SQL('DROP ROLE IF EXISTS {}').format(sql.Identifier(reader_role(owner))))


def writer_dsn(dsn,owner):
    config=conninfo_to_dict(dsn);config['user']=application_writer_role(owner);config.pop('password',None)
    return make_conninfo(**config)


def provision_all(dsn,owners):
    with psycopg.connect(dsn,autocommit=True) as conn:
        for owner in owners:
            provision_application_writer(conn,owner,secrets.token_urlsafe(40))


def test_writer_verifier_accepts_valid_fixed_owner_policies(writer_database):
    dsn,(alice,_)=writer_database
    provision_all(dsn,(alice,))
    role=application_writer_role(alice)
    with psycopg.connect(dsn,autocommit=True) as conn:
        verify_writer_access(conn,role)
        verify_writer_access(conn,role,alice)
        with pytest.raises(ValueError):verify_writer_access(conn,role,'different-owner')
        conn.execute(sql.SQL('COMMENT ON ROLE {} IS {}').format(
            sql.Identifier(role),sql.Literal('gmail-search application writer v1 owner=different-owner')))
        with pytest.raises(ValueError):verify_writer_access(conn,role)


@pytest.mark.parametrize('mutation',[
    'disable_rls',
    'drop_restrictive',
    'alter_using',
    'alter_check',
    'change_policy_role',
    'make_bound_permissive',
    'own_table',
])
def test_writer_verifier_rejects_rls_and_policy_drift(writer_database,mutation):
    dsn,(alice,_)=writer_database
    provision_all(dsn,(alice,))
    role=application_writer_role(alice)
    bound=role+'_bound'
    with psycopg.connect(dsn,autocommit=True) as conn:
        if mutation=='disable_rls':
            conn.execute('ALTER TABLE public.conversations DISABLE ROW LEVEL SECURITY')
        elif mutation=='drop_restrictive':
            conn.execute(sql.SQL('DROP POLICY {} ON public.conversations').format(sql.Identifier(bound)))
        elif mutation=='alter_using':
            conn.execute(sql.SQL('ALTER POLICY {} ON public.conversations USING(user_id={}) WITH CHECK(user_id={})').format(
                sql.Identifier(bound),sql.Literal('different-owner'),sql.Literal(alice)))
        elif mutation=='alter_check':
            conn.execute(sql.SQL('ALTER POLICY {} ON public.conversations USING(user_id={}) WITH CHECK(user_id={})').format(
                sql.Identifier(bound),sql.Literal(alice),sql.Literal('different-owner')))
        elif mutation=='change_policy_role':
            conn.execute(sql.SQL('ALTER POLICY {} ON public.conversations TO PUBLIC').format(sql.Identifier(bound)))
        elif mutation=='make_bound_permissive':
            conn.execute(sql.SQL('DROP POLICY {} ON public.conversations').format(sql.Identifier(bound)))
            conn.execute(sql.SQL("CREATE POLICY {} ON public.conversations AS PERMISSIVE FOR ALL TO {} USING(user_id={}) WITH CHECK(user_id={})").format(
                sql.Identifier(bound),sql.Identifier(role),sql.Literal(alice),sql.Literal(alice)))
        elif mutation=='own_table':
            conn.execute(sql.SQL('ALTER TABLE public.conversations OWNER TO {}').format(sql.Identifier(role)))
        with pytest.raises(ValueError):
            verify_writer_access(conn,role,alice)


def test_writer_verifier_rejects_shadow_schema_policy_and_restores_search_path(writer_database):
    dsn,(alice,_)=writer_database
    provision_all(dsn,(alice,))
    role=application_writer_role(alice)
    bound=role+'_bound'
    with psycopg.connect(dsn,autocommit=True) as conn:
        conn.execute('CREATE SCHEMA shadow')
        conn.execute('CREATE TABLE shadow.conversations(id text,user_id text)')
        with conn.transaction():
            conn.execute('SET search_path=shadow,public')
            hostile_path=conn.execute("SELECT current_setting('search_path')").fetchone()[0]

            verify_writer_access(conn,role,alice)
            assert conn.execute("SELECT current_setting('search_path')").fetchone()[0]==hostile_path

            predicate=sql.SQL('EXISTS(SELECT 1 FROM shadow.conversations c WHERE c.id=conversation_messages.conversation_id AND c.user_id={})').format(sql.Literal(alice))
            conn.execute(sql.SQL('ALTER POLICY {} ON public.conversation_messages USING({}) WITH CHECK({})').format(
                sql.Identifier(bound),predicate,predicate))
            with pytest.raises(ValueError):
                verify_writer_access(conn,role,alice)
            assert conn.execute("SELECT current_setting('search_path')").fetchone()[0]==hostile_path


def test_fixed_roles_write_their_own_conversations_and_parent_bound_events(writer_database):
    dsn,(alice,bob)=writer_database
    provision_all(dsn,(alice,bob))
    with psycopg.connect(writer_dsn(dsn,alice),autocommit=True) as conn:
        assert conn.execute('SELECT id FROM public.conversations').fetchall()==[(alice,)]
        conn.execute("SELECT set_config('app.user_id',%s,false)",(bob,))
        assert conn.execute('SELECT parts FROM public.conversation_messages').fetchall()==[(alice,)]
        conn.execute('INSERT INTO public.conversations(id,user_id,title) VALUES(%s,%s,%s)',('new',alice,'own'))
        conn.execute("INSERT INTO public.agent_sessions(id,conversation_id,user_id,mode,question,status) VALUES('new','new',%s,'deep','question','running')",(alice,))
        conn.execute("INSERT INTO public.agent_events(session_id,seq,agent_name,kind,payload) VALUES('new',1,'agent','text','{}')")
        conn.execute("UPDATE public.agent_sessions SET status='done',final_answer='answer',finished_at=now() WHERE id='new'")
        for query,args in [
            ('INSERT INTO public.conversations(id,user_id) VALUES(%s,%s)',('foreign',bob)),
            ('INSERT INTO public.conversation_messages(conversation_id,seq,role,parts) VALUES(%s,2,%s,%s)',(bob,'user','foreign')),
            ('INSERT INTO public.agent_sessions(id,conversation_id,user_id) VALUES(%s,%s,%s)',('foreign-session',bob,alice)),
            ('INSERT INTO public.agent_events(session_id,seq,agent_name,kind,payload) VALUES(%s,2,%s,%s,%s)',(bob,'agent','text','{}')),
        ]:
            with pytest.raises(psycopg.errors.InsufficientPrivilege):conn.execute(query,args)
        assert conn.execute('DELETE FROM public.conversations WHERE id=%s',(bob,)).rowcount==0
        with pytest.raises(psycopg.errors.InsufficientPrivilege):conn.execute('UPDATE public.conversations SET user_id=%s WHERE id=%s',(bob,alice))
        for statement in ('SELECT * FROM public.messages','SELECT * FROM public.users',"INSERT INTO public.sync_state VALUES('anything','x')",'CREATE TABLE public.forbidden(id text)','CREATE TEMP TABLE forbidden(id text)',f'SET ROLE "{application_writer_role(bob)}"'):
            with pytest.raises(psycopg.errors.InsufficientPrivilege):conn.execute(statement)


@pytest.mark.parametrize('hazard',["GRANT SELECT ON messages TO PUBLIC","GRANT TEMP ON DATABASE {database} TO PUBLIC",
    "CREATE FUNCTION public.unsafe() RETURNS integer LANGUAGE sql SECURITY DEFINER AS 'SELECT 1'",
    'ALTER TABLE agent_events DISABLE ROW LEVEL SECURITY'])
def test_provision_refuses_inherited_grants_and_missing_rls(writer_database,hazard):
    dsn,(alice,_)=writer_database
    with psycopg.connect(dsn,autocommit=True) as conn:
        conn.execute(hazard.format(database=sql.Identifier(conn.info.dbname).as_string(conn)))
        with pytest.raises(ValueError):provision_application_writer(conn,alice,secrets.token_urlsafe(40))
        assert conn.execute('SELECT 1 FROM pg_roles WHERE rolname=%s',(application_writer_role(alice),)).fetchone() is None


def test_writer_registry_has_no_owner_or_privileged_login_fallback(writer_database):
    dsn,(alice,bob)=writer_database
    provision_all(dsn,(alice,bob))
    credential=WriterCredential(alice,writer_dsn(dsn,alice))
    active={alice}
    registry=WriterRegistry({alice:credential},is_active=lambda owner:owner in active)
    with registry.connection(alice) as conn:
        assert conn.execute('SELECT current_user').fetchone()[0]==application_writer_role(alice)
    with pytest.raises(PermissionError):registry.credential(bob)
    with pytest.raises(ValueError):WriterCredential(alice,dsn)
    active.clear()
    with pytest.raises(PermissionError):registry.credential(alice)


def test_writer_cost_ledger_is_append_only_and_owner_bound(writer_database):
    from gmail_search.store.cost import record_cost
    dsn,(alice,bob)=writer_database
    provision_all(dsn,(alice,bob))
    with psycopg.connect(writer_dsn(dsn,alice),autocommit=True) as conn:
        record_cost(conn,'inference','synthetic',10,0,0.01,'run',user_id=alice)
        assert conn.execute('SELECT user_id FROM public.costs').fetchall()==[(alice,)]
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            record_cost(conn,'inference','synthetic',10,0,0.01,'run',user_id=bob)
        with pytest.raises(psycopg.errors.InsufficientPrivilege):conn.execute('DELETE FROM public.costs')


def test_revocation_before_context_commit_rolls_back_writer_transaction(writer_database):
    dsn,(alice,bob)=writer_database
    provision_all(dsn,(alice,bob))
    active=[True]
    registry=WriterRegistry({alice:WriterCredential(alice,writer_dsn(dsn,alice))},is_active=lambda owner:active[0])
    with pytest.raises(PermissionError):
        with registry.connection(alice) as conn:
            conn.execute("INSERT INTO public.conversations(id,user_id,title) VALUES('revoked',%s,'never published')",(alice,))
            active[0]=False
    with psycopg.connect(dsn) as conn:
        assert conn.execute("SELECT 1 FROM conversations WHERE id='revoked'").fetchone() is None


def test_role_reuse_cannot_change_owner_or_inherit_another_writer(writer_database):
    dsn,(alice,bob)=writer_database
    provision_all(dsn,(alice,bob))
    with psycopg.connect(dsn,autocommit=True) as conn:
        conn.execute(sql.SQL('GRANT {} TO {}').format(sql.Identifier(application_writer_role(bob)),sql.Identifier(application_writer_role(alice))))
        with pytest.raises(ValueError):provision_application_writer(conn,alice,secrets.token_urlsafe(40))
        conn.execute(sql.SQL('REVOKE {} FROM {}').format(sql.Identifier(application_writer_role(bob)),sql.Identifier(application_writer_role(alice))))
        conn.execute(sql.SQL('COMMENT ON ROLE {} IS {}').format(sql.Identifier(application_writer_role(alice)),sql.Literal('wrong owner binding')))
        with pytest.raises(ValueError):provision_application_writer(conn,alice,secrets.token_urlsafe(40))


def add_reader_schema(dsn):
    from gmail_search.gateway.schema import ANALYTICAL_SCHEMA
    with psycopg.connect(dsn,autocommit=True) as conn:
        for table,columns in ANALYTICAL_SCHEMA.items():
            if table=='messages':
                for column,kind in columns.items():
                    conn.execute(sql.SQL('ALTER TABLE public.messages ADD COLUMN IF NOT EXISTS {} {}').format(sql.Identifier(column),sql.SQL(kind)))
            else:
                conn.execute(sql.SQL('CREATE TABLE public.{} ({})').format(sql.Identifier(table),sql.SQL(',').join(
                    sql.SQL('{} {}').format(sql.Identifier(column),sql.SQL(kind)) for column,kind in columns.items())))
            conn.execute(sql.SQL('ALTER TABLE public.{} ENABLE ROW LEVEL SECURITY').format(sql.Identifier(table)))


def test_admission_provisioner_validates_canonical_identity_before_installing_credentials(writer_database, monkeypatch):
    # Partition DDL is exercised by test_partitioned_admission on ParadeDB.
    monkeypatch.setattr('gmail_search.gateway.admission_provision.provision_owner_partitions', lambda conn, owner, *, profile: None)
    from gmail_search.auth.identity_store import Account,VerifiedGoogleIdentity
    from gmail_search.gateway.admission_provision import AdmissionProvisioner
    dsn,(alice,_)=writer_database
    add_reader_schema(dsn)
    config=conninfo_to_dict(dsn);config.pop('user',None);config.pop('password',None)
    installed=[]
    provision=AdmissionProvisioner(lambda:psycopg.connect(dsn),runtime_dsn=make_conninfo(**config),
                                   install_credentials=lambda *credentials:installed.append(credentials),partition_profile=NUMERIC)
    account=Account(alice,alice+'@example.test',1)
    verified=VerifiedGoogleIdentity(account.email,'google-'+alice,True)
    assert provision(account,verified) is True
    reader,writer=installed[0]
    assert reader.owner_id==writer.owner_id==alice
    assert conninfo_to_dict(writer.dsn)['user']==application_writer_role(alice)
    assert conninfo_to_dict(reader.dsn)['user']==reader_role(alice)
    with pytest.raises(ValueError):provision(account,VerifiedGoogleIdentity(account.email,'different-subject',True))
    assert len(installed)==1


def test_legacy_subject_requires_explicit_audited_binding_not_email_claim(writer_database, monkeypatch):
    # Partition DDL is exercised by test_partitioned_admission on ParadeDB.
    monkeypatch.setattr('gmail_search.gateway.admission_provision.provision_owner_partitions', lambda conn, owner, *, profile: None)
    from gmail_search.auth.identity_store import Account,VerifiedGoogleIdentity
    from gmail_search.gateway.admission_provision import AdmissionProvisioner,bind_existing_subject
    dsn,(alice,bob)=writer_database
    add_reader_schema(dsn)
    with psycopg.connect(dsn,autocommit=True) as conn:
        conn.execute('UPDATE public.users SET google_sub=NULL WHERE id=%s',(alice,))
    cfg=conninfo_to_dict(dsn);cfg.pop('user',None);cfg.pop('password',None)
    provision=AdmissionProvisioner(lambda:psycopg.connect(dsn),runtime_dsn=make_conninfo(**cfg),install_credentials=lambda *a:None,partition_profile=NUMERIC)
    account=Account(alice,alice+'@example.test',1);claims=VerifiedGoogleIdentity(account.email,'google-'+alice,True)
    with pytest.raises(ValueError):provision(account,claims)
    with psycopg.connect(dsn) as conn:
        with pytest.raises(ValueError):bind_existing_subject(conn,bob,claims)
        bind_existing_subject(conn,alice,claims)
    assert provision(account,claims) is True


def test_admission_vault_failure_never_reports_success(writer_database, monkeypatch):
    # Partition DDL is exercised by test_partitioned_admission on ParadeDB.
    monkeypatch.setattr('gmail_search.gateway.admission_provision.provision_owner_partitions', lambda conn, owner, *, profile: None)
    from gmail_search.auth.identity_store import Account,VerifiedGoogleIdentity
    from gmail_search.gateway.admission_provision import AdmissionProvisioner
    dsn,(alice,_)=writer_database
    add_reader_schema(dsn)
    cfg=conninfo_to_dict(dsn);cfg.pop('user',None);cfg.pop('password',None)
    def unavailable(*credentials):
        raise RuntimeError('vault unavailable')
    provision=AdmissionProvisioner(lambda:psycopg.connect(dsn),runtime_dsn=make_conninfo(**cfg),install_credentials=unavailable,partition_profile=NUMERIC)
    with pytest.raises(RuntimeError,match='vault unavailable'):
        provision(Account(alice,alice+'@example.test',1),VerifiedGoogleIdentity(alice+'@example.test','google-'+alice,True))
    installed=[]
    retry=AdmissionProvisioner(lambda:psycopg.connect(dsn),runtime_dsn=make_conninfo(**cfg),install_credentials=lambda *a:installed.append(a),partition_profile=NUMERIC)
    assert retry(Account(alice,alice+'@example.test',1),VerifiedGoogleIdentity(alice+'@example.test','google-'+alice,True))
    registry=WriterRegistry({alice:installed[0][1]},is_active=lambda _:True)
    with registry.connection(alice) as conn:
        assert conn.execute('SELECT count(*) FROM public.conversations').fetchone()[0]==1


def test_browser_transcripts_use_fixed_owner_writer_and_scoped_receipts(writer_database):
    from gmail_search.gateway.browser_conversations import BrowserConversations
    dsn,owners=writer_database
    provision_all(dsn,owners)
    writers=WriterRegistry({owner:WriterCredential(owner,writer_dsn(dsn,owner)) for owner in owners},is_active=lambda owner:owner in owners)
    transcripts=BrowserConversations(None,is_active=lambda owner:owner in owners,
        authorize_persistence=lambda *args:True,read_events=lambda *args:[],
        can_edit=lambda *args:True,transaction_factory=writers.connection)
    alice,bob=owners
    transcripts.save(alice,'owned-chat',{'messages':[{'role':'user','parts':[{'type':'text','text':'Hello'}]}]})
    assert transcripts.persist(alice,'owned-chat','receipt-a','Private answer') is True
    assert transcripts.persist(alice,'owned-chat','receipt-a','Private answer') is True
    assert len(transcripts.get(alice,'owned-chat')['messages'])==2
    assert transcripts.get(bob,'owned-chat') is None
    with writers.connection(bob) as db:
        assert db.execute('SELECT run_id FROM public.browser_answer_receipts').fetchall()==[]
    with pytest.raises((PermissionError,RuntimeError)):
        transcripts.persist(bob,'owned-chat','receipt-b','Bad answer')
    with psycopg.connect(writer_dsn(dsn,bob),autocommit=True) as db:
        with pytest.raises(psycopg.Error):
            db.execute('INSERT INTO browser_answer_receipts VALUES(%s,%s,%s,%s)',('foreign',alice,'owned-chat','hash'))
        with pytest.raises(psycopg.Error):
            db.execute("UPDATE browser_answer_receipts SET content_hash='changed'")
    assert transcripts.delete(alice,'owned-chat') is True
    with writers.connection(alice) as db:
        assert db.execute('SELECT run_id FROM public.browser_answer_receipts').fetchall()==[]
