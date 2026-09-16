"""Production assembly invariants using synthetic controller state only."""
import json
from types import SimpleNamespace

import pytest

from gmail_search.gateway.capabilities import Capabilities
from gmail_search.gateway.full_agent_remote import validate_bootstrap
from gmail_search.gateway.registry import AccessDenied, Registry


def test_envelope_has_full_tool_profile_and_only_run_scoped_authority(tmp_path):
    from gmail_search.invited_runtime import envelope_factory
    tmp_path.chmod(0o700)
    registry=Registry(tmp_path/'registry.sqlite',is_active=lambda owner:owner=='alice')
    budget=registry.create_budget('alice',10**9)
    lease=registry.start_run('alice','chat',request_key='request',budget_id=budget)
    bound=[]
    provider=SimpleNamespace(bind_profile=lambda run,profile:bound.append((run,profile)))
    rates=SimpleNamespace(input_units_per_token=2,output_units_per_token=3)
    caps=Capabilities(registry)
    packet=envelope_factory(caps,provider,rates)(lease,'Find my receipts')
    validate_bootstrap(packet,'Find my receipts')
    value=json.loads(packet[4:])
    assert value['tool_config']['tool_profile']=='mail-raw-mcp-v3'
    assert set(value['tool_config']['capabilities'])=={'sql','retrieval','artifact','attachment'}
    assert bound[0][0]==lease.run_id
    assert bound[0][1].client_profile=='pi-0.84.4'
    assert bound[0][1].model=='claude-sonnet-4-6'
    with registry._transaction() as db:
        rows=db.execute('SELECT run_id,audience FROM capabilities').fetchall()
        assert {row['audience'] for row in rows}=={'sql','retrieval','artifact','attachment','inference','events'}
        assert {row['run_id'] for row in rows}=={lease.run_id}


def test_runtime_lock_excludes_second_process_and_releases(tmp_path):
    from gmail_search.invited_runtime import runtime_lock
    tmp_path.chmod(0o700)
    with runtime_lock(tmp_path):
        with pytest.raises(RuntimeError):
            with runtime_lock(tmp_path):
                pass
    with runtime_lock(tmp_path):
        pass


def test_absent_release_gate_prevents_any_runtime_initialization(tmp_path):
    from gmail_search.invited_runtime import require_release
    cfg=SimpleNamespace(state_dir=tmp_path,release=SimpleNamespace(store_id='test-store',release_epoch=1))
    with pytest.raises(AccessDenied):
        require_release(cfg)
    assert list(tmp_path.iterdir())==[]


def test_budget_binding_cannot_select_another_owner_or_create_budget(tmp_path):
    from gmail_search.invited_runtime import verify_budgets
    tmp_path.chmod(0o700)
    registry=Registry(tmp_path/'registry.sqlite',is_active=lambda _:True)
    budget=registry.create_budget('alice',100)
    verify_budgets(registry,[SimpleNamespace(id='alice',budget_id=budget)])
    with pytest.raises(AccessDenied):
        verify_budgets(registry,[SimpleNamespace(id='bob',budget_id=budget)])
    with registry._transaction() as db:
        assert db.execute('SELECT count(*) FROM budgets').fetchone()[0]==1


@pytest.fixture
def assembled(tmp_path,monkeypatch):
    """Real stores/services, with DB, native index and network boundaries doubled."""
    from gmail_search import invited_runtime as module
    from gmail_search.auth.identity_store import IdentityStore,VerifiedGoogleIdentity
    from gmail_search.gateway.database import reader_role
    from gmail_search.gateway.search_reader import search_role
    from gmail_search.gateway.writer import application_writer_role
    from gmail_search.gateway.maintenance import MaintenanceAdmin,ReleaseIdentity
    from gmail_search.gateway.partition_profiles import TEXT_OWNER_PARTITIONS_V1 as TEXT
    from gmail_search.gateway.search_index import LoadedIndex
    from gmail_search.gateway import full_agent_rpc as rpc
    tmp_path.chmod(0o700)
    (tmp_path/'artifacts').mkdir(mode=0o700)
    (tmp_path/'attachments').mkdir(mode=0o700)
    identities=IdentityStore(tmp_path/'identities.sqlite')
    identity=VerifiedGoogleIdentity('alice@example.test','subject-alice',True)
    identities.import_existing_account(owner_id='alice',identity=identity)
    identities.invite(identity.email)
    identities.mark_provisioned('alice')
    cookie=identities.admit(identity)
    old=Registry(tmp_path/'registry.sqlite',is_active=lambda _:True)
    budget=old.create_budget('alice',10**9)
    release=ReleaseIdentity('test-store',TEXT,1)
    admin=MaintenanceAdmin(old.path,verifier=lambda *args,**kwargs:'e'*64)
    snapshot=admin.initialize_closed(release,migration_id='test-migration',owner_set_digest='a'*64,procedure_digest='b'*64)
    snapshot=admin.publish_ready(admin.record_index_pending(snapshot))
    owner=SimpleNamespace(id='alice',email=identity.email,google_subject=identity.subject,budget_id=budget,
        reader_dsn='dbname=synthetic user='+reader_role('alice'),
        search_dsn='dbname=synthetic user='+search_role('alice',profile=TEXT),
        writer_dsn='dbname=synthetic user='+application_writer_role('alice'),
        index=SimpleNamespace(path=tmp_path/'index',generation='one',source_id='source-one'))
    cfg=SimpleNamespace(state_dir=tmp_path,attachment_root=tmp_path/'attachments',owners=(owner,),
        release=SimpleNamespace(store_id='test-store',release_epoch=1),
        worker=SimpleNamespace(host='worker.test',port=22,private_key=tmp_path/'key',known_hosts=tmp_path/'hosts'),
        broker=SimpleNamespace(origin='https://broker.test',bearer='b'*48,signing_secret='h'*48),
        provider=SimpleNamespace(anthropic_key='a'*40,gemini_key='g'*40,input_units_per_token=1,
            output_units_per_token=1,embedding_units_per_token=1,rerank_input_units_per_token=1,
            rerank_output_units_per_token=1,fact_model_tag='facts-v1'))
    for key,value in {'GMAIL_MULTI_TENANT':'1','GMS_PUBLIC_ORIGIN':'https://gms.example.test',
        'GMS_PUBLIC_ALLOWED_EMAILS':'alice@example.test','GMS_IDENTITY_BROKER_URL':'https://identity.example.test',
        'GMS_IDENTITY_HANDOFF_SECRET':'i'*48,'GMS_SESSION_SECRET':'s'*48}.items():
        monkeypatch.setenv(key,value)
    observed={};closed=[]
    async def databases(owners,writers,gateway,search_reader):
        assert gateway.admission is search_reader.admission
        observed.update(writers=writers,gateway=gateway,search_reader=search_reader)
    monkeypatch.setattr(module,'verify_databases',databases)
    class Searcher:
        def search(self,*args,**kwargs):return []
        def close(self):closed.append('index')
    monkeypatch.setattr(module,'load_scann_index',lambda binding,path:LoadedIndex(binding,Searcher(),'resource-one'))
    class Transport:
        def __init__(self,*args,**kwargs):pass
        async def aclose(self):closed.append('provider')
    for name in ('AnthropicHTTPTransport','GeminiEmbeddingHTTPTransport','GeminiRerankerHTTPTransport'):
        monkeypatch.setattr(module,name,Transport)
    class SSH:
        def __init__(self,**kwargs):pass
        def exchange(self,header,payload,cancelled):
            assert header['op']=='inventory'
            return rpc.reply(header,'ok')
    monkeypatch.setattr(module,'SSHTransport',SSH)
    return SimpleNamespace(config=cfg,observed=observed,closed=closed,cookie=cookie,admin=admin,snapshot=snapshot)


@pytest.mark.asyncio
async def test_complete_assembly_separates_routes_closes_resources_and_observes_gate(assembled):
    import httpx
    from gmail_search.invited_runtime import open_runtime
    from gmail_search.gateway.maintenance import ReleaseIdentity
    from gmail_search.gateway.partition_profiles import TEXT_OWNER_PARTITIONS_V1 as TEXT
    from gmail_search.auth.public import SESSION_COOKIE
    s=assembled
    async with open_runtime(s.config) as runtime:
        async with runtime.browser_app.router.lifespan_context(runtime.browser_app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=runtime.browser_app),base_url='https://gms.example.test') as browser:
                response=await browser.get('/api/auth/me',headers={'Cookie':SESSION_COOKIE+'='+s.cookie})
                assert response.status_code==200
                assert response.json()['user']['id']=='alice'
                assert (await browser.get('/v1/schema')).status_code==404
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=runtime.gateway_app),base_url='http://127.0.0.1') as gateway:
                assert (await gateway.get('/api/auth/me')).status_code==404
                assert (await gateway.get('/v1/schema')).status_code==401
            s.observed['writers'].credential('alice')
            s.admin.begin_maintenance(ReleaseIdentity('test-store',TEXT,2),migration_id='next',owner_set_digest='a'*64,
                procedure_digest='b'*64,expected_revision=s.snapshot.revision)
            with pytest.raises((AccessDenied,PermissionError)):
                s.observed['writers'].credential('alice')
    assert s.closed.count('provider')==3
    assert s.closed.count('index')==1


@pytest.mark.asyncio
async def test_assembly_failure_after_resources_open_closes_all(assembled,monkeypatch):
    from gmail_search import invited_runtime as module
    def fail(**kwargs):raise RuntimeError('failed browser composition')
    monkeypatch.setattr(module,'create_invited_app',fail)
    with pytest.raises(RuntimeError,match='failed browser composition'):
        async with module.open_runtime(assembled.config):
            pytest.fail('must not publish a partial runtime')
    assert assembled.closed.count('provider')==3
    assert assembled.closed.count('index')==1
    with module.runtime_lock(assembled.config.state_dir):
        pass


@pytest.mark.asyncio
async def test_native_index_load_cancellation_waits_then_closes_resource(monkeypatch):
    import asyncio
    import threading
    from gmail_search import invited_runtime as module
    entered=threading.Event();release=threading.Event();closed=[]
    resource=SimpleNamespace(searcher=SimpleNamespace(close=lambda:closed.append(True)))
    def load(*args):
        entered.set()
        assert release.wait(5)
        return resource
    monkeypatch.setattr(module,'load_scann_index',load)
    task=asyncio.create_task(module._publish_index(None,None,None))
    while not entered.is_set():
        await asyncio.sleep(.005)
    task.cancel()
    await asyncio.sleep(.01)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):await task
    assert closed==[True]


@pytest.mark.asyncio
async def test_prepared_identity_subject_must_match_config(assembled):
    from gmail_search.invited_runtime import open_runtime
    assembled.config.owners[0].google_subject='another-subject'
    with pytest.raises(AccessDenied):
        async with open_runtime(assembled.config):
            pytest.fail('identity mismatch must prevent startup')
    assert assembled.observed=={}


def test_executable_validates_config_but_missing_gate_exits_without_fallback(tmp_path):
    import os
    import subprocess
    import sys
    from test_invited_config import _value,_write
    config=_write(tmp_path,_value(tmp_path))
    env={**os.environ,'GMS_INVITED_CONFIG':str(config)}
    check=subprocess.run([sys.executable,'-m','gmail_search.invited_server','--check-config'],
        env=env,capture_output=True,text=True,timeout=10)
    assert check.returncode==0,check.stderr
    assert 'deployment readiness was not checked' in check.stdout
    launch=subprocess.run([sys.executable,'-m','gmail_search.invited_server'],
        env=env,capture_output=True,text=True,timeout=10)
    assert launch.returncode==1
    assert launch.stderr=='Invited service startup or shutdown failed; public admission is closed.\n'
    assert list((tmp_path/'state').iterdir())==[]
    assert 'password' not in launch.stderr


@pytest.mark.asyncio
async def test_recovery_finishes_before_either_application_is_published(assembled,monkeypatch):
    from gmail_search.gateway.browser_runs import BrowserRuns
    from gmail_search.invited_runtime import open_runtime
    original=BrowserRuns.recover
    recovered=[]
    async def recover(self):
        await original(self)
        recovered.append(True)
    monkeypatch.setattr(BrowserRuns,'recover',recover)
    async with open_runtime(assembled.config) as runtime:
        assert recovered==[True]
        async with runtime.browser_app.router.lifespan_context(runtime.browser_app):
            assert recovered==[True]


@pytest.mark.asyncio
async def test_recovery_failure_never_publishes_gateway(assembled,monkeypatch):
    from gmail_search.gateway.browser_runs import BrowserRuns
    from gmail_search.invited_runtime import open_runtime
    async def fail(self):raise AccessDenied()
    monkeypatch.setattr(BrowserRuns,'recover',fail)
    with pytest.raises(AccessDenied):
        async with open_runtime(assembled.config):
            pytest.fail('no gateway before recovery')
    assert assembled.closed.count('provider')==3
    assert assembled.closed.count('index')==1


@pytest.mark.asyncio
async def test_gmail_consent_secrets_cannot_reuse_identity_secrets(assembled):
    from gmail_search.invited_runtime import open_runtime
    assembled.config.broker.signing_secret='s'*48
    with pytest.raises(AccessDenied):
        async with open_runtime(assembled.config):
            pytest.fail('independent credentials required')
    assert assembled.observed=={}
