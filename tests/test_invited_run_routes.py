"""Exercise browser HTTP against real identity, run, event and worker stores."""
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gmail_search.auth.identity_store import IdentityStore, VerifiedGoogleIdentity
from gmail_search.auth.public import SESSION_COOKIE
from gmail_search.gateway.browser_runs import BrowserRuns
from gmail_search.gateway.capabilities import Capabilities
from gmail_search.gateway.events import Events
from gmail_search.gateway.registry import AccessDenied, Registry
from gmail_search.gateway.worker import WorkerController
from test_browser_runs import Backend


@pytest.fixture
def app_state(tmp_path):
    from gmail_search.auth.run_routes import create_run_router
    identities = IdentityStore(tmp_path/'identities')
    accounts, tokens = {}, {}
    for name in ('alice','bob'):
        accounts[name] = identities.invite(name+'@example.test')
        identities.mark_provisioned(accounts[name].owner_id)
        tokens[name] = identities.admit(VerifiedGoogleIdentity(name+'@example.test',name,True))
    registry = Registry(tmp_path/'registry',is_active=identities.is_active)
    capabilities = Capabilities(registry)
    events = Events(capabilities)
    backend = Backend(events,capabilities)
    workers = WorkerController(registry,backend)
    budgets = {a.owner_id:registry.create_budget(a.owner_id,1000) for a in accounts.values()}
    conversations, saved = {}, []
    def claim(owner, conversation):
        if conversations.setdefault(conversation,owner) != owner:
            raise AccessDenied()
        return True
    def persist(owner, conversation, run, answer):
        claim(owner,conversation)
        saved.append((owner,conversation,run,answer))
        return True
    def prepare(lease,prompt,runtime):
        backend.prompts[lease.run_id] = prompt
        backend.runtimes[lease.run_id] = runtime
    runs = BrowserRuns(workers,events,prepare_input=prepare,
        persist_answer=persist,budget_for=budgets.__getitem__,poll_seconds=.01)
    @asynccontextmanager
    async def lifespan(app):
        yield
        await runs.close()
    app = FastAPI(lifespan=lifespan)
    app.include_router(create_run_router(identities,runs,origin='https://gms.example.test',claim_conversation=claim))
    with TestClient(app,base_url='https://gms.example.test') as client:
        yield SimpleNamespace(client=client,identities=identities,accounts=accounts,tokens=tokens,
            runs=runs,backend=backend,saved=saved)


def headers(state, name='alice'):
    return {'Cookie':SESSION_COOKIE+'='+state.tokens[name],'Origin':'https://gms.example.test'}


def frames(response):
    return [(block.splitlines()[0][7:],json.loads(block.splitlines()[1][6:]))
        for block in response.text.strip().split('\n\n') if block.startswith('event:')]


def test_browser_question_replay_and_cross_owner_denial(app_state):
    s = app_state
    response = s.client.post('/api/agent/analyze',headers=headers(s),json={
        'question':'summarize my receipts','conversation_id':'conversation','backend':'pi'})
    assert response.status_code == 200
    rows = frames(response)
    run = rows[0][1]['session_id']
    assert rows[0][0] == 'session'
    assert any(kind == 'tool_call' for kind,_ in rows)
    assert rows[-2][0] == 'final'
    assert rows[-2][1]['payload']['text'] == 'Answer for summarize my receipts'
    assert rows[-1][0] == 'persist_ok'
    assert len(s.saved) == 1 and not s.backend.handles
    path = f'/api/agent/analyze/{run}/events?conversation_id=conversation&after=1'
    replay = s.client.get(path,headers=headers(s))
    assert replay.status_code == 200
    assert all(kind != 'tool_call' for kind,_ in frames(replay))
    assert len(s.backend.prompts) == 1
    assert s.client.get(path,headers=headers(s,'bob')).status_code == 404
    assert s.client.post(f'/api/agent/analyze/{run}/cancel',headers=headers(s,'bob'),
        json={'conversation_id':'conversation'}).status_code == 404
    denied = s.client.post('/api/agent/analyze',headers=headers(s,'bob'),json={
        'question':'steal conversation','conversation_id':'conversation'})
    assert denied.status_code == 404 and len(s.backend.prompts) == 1
    assert response.headers['cache-control'] == 'private, no-store'
    assert response.headers['x-gms-transcript-owner'] == 'server'


@pytest.mark.parametrize('body',[
    {'question':'hello','conversation_id':'conversation','owner_id':'bob'},
    {'question':'hello','conversation_id':'../elsewhere'},
    {'question':'é'*8193,'conversation_id':'conversation'},
    {'question':'hello','conversation_id':'conversation','backend':'shell'},
    {'question':'hello','conversation_id':'conversation','model':7},
    {'question':'hello','conversation_id':'conversation','model':'x'*129},
])
def test_invalid_start_never_launches(app_state,body):
    response = app_state.client.post('/api/agent/analyze',headers=headers(app_state),json=body)
    assert response.status_code in (400,413)
    assert not app_state.backend.prompts


@pytest.mark.parametrize('backend,model,runtime',[
    (None,None,'pi_gemini'),('pi','google/gemini-3.8-flash','pi_gemini'),
    ('pi','openrouter/google/gemini-3.8-flash','pi_gemini'),
    ('pi','anthropic/claude-opus-5','pi_opus'),('pi','openrouter/anthropic/claude-opus-5','pi_opus'),
    ('claude_code',None,'claude'),('claude_code','sonnet','claude')])
def test_the_chosen_backend_selects_the_guest_runtime(app_state,backend,model,runtime):
    body = {'question':'hello','conversation_id':'conversation'}
    for key,value in (('backend',backend),('model',model)):
        if value is not None:
            body[key] = value
    response = app_state.client.post('/api/agent/analyze',headers=headers(app_state),json=body)
    assert response.status_code == 200
    assert list(app_state.backend.runtimes.values()) == [runtime]


@pytest.mark.parametrize('backend,model',[
    ('pi','openrouter/meta/muse-spark-1.3'),
    ('claude_code','opus'),('claude_code','haiku'),('shell',None)])
def test_a_model_without_a_gateway_route_is_refused_not_substituted(app_state,backend,model):
    """The gateway pins Claude Code to Sonnet 4.6; running Sonnet for an 'opus' pick would lie."""
    body = {'question':'hello','conversation_id':'conversation','backend':backend}
    if model is not None:
        body['model'] = model
    response = app_state.client.post('/api/agent/analyze',headers=headers(app_state),json=body)
    assert response.status_code == 400
    assert not app_state.backend.prompts


def test_every_reported_model_is_served():
    """The picker offers what /api/auth/me reports; each of those must start a run."""
    from gmail_search.auth.run_routes import _runtime, deep_models
    for backend,models in deep_models().items():
        for model in models:
            assert _runtime(backend,model) is not None


def test_mutation_requires_cookie_and_exact_origin(app_state):
    s = app_state
    body = {'question':'hello','conversation_id':'conversation'}
    for hdr,code in [({},401),({'Authorization':'Bearer guest'},401),
        ({**headers(s),'Origin':'https://evil.example'},403),
        ({**headers(s),'X-User-Id':'bob'},401)]:
        assert s.client.post('/api/agent/analyze',headers=hdr,json=body).status_code == code
    assert not s.backend.prompts


def test_revocation_during_snapshot_suppresses_private_output(app_state,monkeypatch):
    s = app_state
    run = s.client.portal.call(s.runs.start,s.accounts['alice'].owner_id,'conversation','secret')
    original = s.runs.snapshot
    def snapshot(*args,**kwargs):
        result = original(*args,**kwargs)
        s.identities.revoke_session(s.tokens['alice'])
        return result
    monkeypatch.setattr(s.runs,'snapshot',snapshot)
    result = s.client.get(f'/api/agent/analyze/{run}/events?conversation_id=conversation',headers=headers(s))
    assert result.status_code == 401
    assert 'secret' not in result.text


def test_explicit_stop_waits_for_worker_and_replay_is_terminal(app_state):
    s = app_state
    s.backend.complete = False
    run = s.client.portal.call(s.runs.start,s.accounts['alice'].owner_id,'conversation','wait')
    result = s.client.post(f'/api/agent/analyze/{run}/cancel',headers=headers(s),json={'conversation_id':'conversation'})
    assert result.status_code == 200
    assert result.json()['state'] == 'cancelled'
    assert not s.backend.handles
    replay = s.client.get(f'/api/agent/analyze/{run}/events?conversation_id=conversation',headers=headers(s))
    assert replay.status_code == 200
    assert frames(replay)[-1][0] == 'error'


def test_revocation_between_stream_frames_stops_further_output(app_state,monkeypatch):
    from gmail_search.auth import run_routes
    s = app_state
    original = run_routes.event_frame
    def frame(row):
        result = original(row)
        s.identities.revoke_session(s.tokens['alice'])
        return result
    monkeypatch.setattr(run_routes,'event_frame',frame)
    result = s.client.post('/api/agent/analyze',headers=headers(s),json={
        'question':'secret second frame','conversation_id':'conversation'})
    assert result.status_code == 200
    assert 'event: tool_call' in result.text
    assert 'secret second frame' not in result.text
    assert 'event: final' not in result.text


def test_duplicate_fields_and_body_limit_fail_before_admission(app_state):
    s = app_state
    hdr = {**headers(s),'Content-Type':'application/json'}
    duplicate = '{"question":"a","question":"b","conversation_id":"conversation"}'
    assert s.client.post('/api/agent/analyze',headers=hdr,content=duplicate).status_code == 400
    assert s.client.post('/api/agent/analyze',headers=hdr,content=' '*32769).status_code == 413
    assert not s.backend.prompts


def test_failed_stop_returns_pending_acknowledgement_for_retry(app_state):
    s = app_state
    s.backend.complete = False
    run = s.client.portal.call(s.runs.start,s.accounts['alice'].owner_id,'conversation','wait')
    s.backend.stop_fails = True
    try:
        response = s.client.post(f'/api/agent/analyze/{run}/cancel',headers=headers(s),json={'conversation_id':'conversation'})
        assert response.status_code == 202
        assert response.json() == {'state':'stopping'}
    finally:
        s.backend.stop_fails = False
        s.client.post(f'/api/agent/analyze/{run}/cancel',headers=headers(s),json={'conversation_id':'conversation'})


def test_revocation_before_first_stream_frame_reaps_unpublished_worker(app_state,monkeypatch):
    s = app_state
    s.backend.complete = False
    original = s.identities.read_session
    calls = 0
    def revoke_at_stream(token):
        nonlocal calls
        calls += 1
        if calls == 5:
            s.identities.revoke_session(token)
        return original(token)
    monkeypatch.setattr(s.identities,'read_session',revoke_at_stream)
    response = s.client.post('/api/agent/analyze',headers=headers(s),json={
        'question':'no frame delivered','conversation_id':'conversation'})
    assert response.status_code == 200 and response.text == ''
    assert not s.backend.handles
    with s.runs.registry._transaction() as db:
        assert db.execute('SELECT state FROM browser_runs').fetchone()[0] == 'cancelled'


def test_first_body_send_failure_reaps_unpublished_worker(app_state):
    from starlette.requests import ClientDisconnect
    s = app_state
    s.backend.complete = False
    async def request():
        body = json.dumps({'question':'failed send','conversation_id':'conversation'}).encode()
        scope = {'type':'http','asgi':{'version':'3.0','spec_version':'2.4'},
            'http_version':'1.1','method':'POST','scheme':'https','path':'/api/agent/analyze',
            'raw_path':b'/api/agent/analyze','query_string':b'',
            'root_path':'','server':('gms.example.test',443),'client':('127.0.0.1',12345),
            'headers':[(key.lower().encode(),value.encode()) for key,value in {
                **headers(s),'Content-Type':'application/json','Host':'gms.example.test'}.items()]}
        async def receive():
            return {'type':'http.request','body':body,'more_body':False}
        async def send(message):
            if message['type']=='http.response.body':
                raise OSError('synthetic browser disappeared')
        with pytest.raises(ClientDisconnect):
            await s.client.app(scope,receive,send)
    s.client.portal.call(request)
    assert not s.backend.handles
    with s.runs.registry._transaction() as db:
        assert db.execute('SELECT state FROM browser_runs').fetchone()[0] == 'cancelled'


def test_cancellation_while_first_send_waits_reaps_unpublished_worker(app_state):
    import asyncio
    s = app_state
    s.backend.complete = False
    async def request():
        body = json.dumps({'question':'blocked send','conversation_id':'conversation'}).encode()
        entered = asyncio.Event()
        scope = {'type':'http','asgi':{'version':'3.0','spec_version':'2.4'},
            'http_version':'1.1','method':'POST','scheme':'https','path':'/api/agent/analyze',
            'raw_path':b'/api/agent/analyze','query_string':b'',
            'root_path':'','server':('gms.example.test',443),'client':('127.0.0.1',12345),
            'headers':[(key.lower().encode(),value.encode()) for key,value in {
                **headers(s),'Content-Type':'application/json','Host':'gms.example.test'}.items()]}
        async def receive():
            return {'type':'http.request','body':body,'more_body':False}
        async def send(message):
            if message['type']=='http.response.body':
                entered.set()
                await asyncio.Event().wait()
        task = asyncio.create_task(s.client.app(scope,receive,send))
        async with asyncio.timeout(3):
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    s.client.portal.call(request)
    assert not s.backend.handles
    with s.runs.registry._transaction() as db:
        assert db.execute('SELECT state FROM browser_runs').fetchone()[0] == 'cancelled'


def test_the_table_offers_only_runtimes_the_deployment_can_launch():
    """Without an OpenRouter key there is no pi_opus binder: Opus is neither
    reported nor accepted."""
    from gmail_search.auth.run_routes import _runtime, deep_models
    runtimes = frozenset({'pi_gemini','claude'})
    assert deep_models(runtimes) == {'pi':['google/gemini-3.8-flash'],'claude_code':['sonnet']}
    assert _runtime('pi','anthropic/claude-opus-5',runtimes) is None
    assert _runtime('pi','anthropic/claude-opus-5') == 'pi_opus'
    assert deep_models(frozenset({'claude'})) == {'claude_code':['sonnet']}


def test_a_question_while_every_worker_is_busy_is_told_to_wait(app_state):
    """The worker runs one VM at a time. A second question used to be admitted
    and then fail at launch as "The run failed"; now it is refused up front."""
    s = app_state
    s.runs.workers.max_workers = 1
    owner = s.accounts['alice'].owner_id
    s.runs._open(owner,'busy-conversation')   # a live run holding the only slot
    response = s.client.post('/api/agent/analyze',headers=headers(s),json={
        'question':'hello','conversation_id':'conversation'})
    assert response.status_code == 409
    assert 'still running' in response.json()['detail']
    assert not s.backend.prompts


def test_stream_terminal_frame_carries_the_refusal_reason(app_state):
    s = app_state
    s.backend.complete = False
    run = s.client.portal.call(s.runs.start, s.accounts['alice'].owner_id, 'conversation', 'synthetic question')
    token = s.runs.events.capabilities.issue(run, audience='inference', operations=['generate']).secret
    s.client.portal.call(s.runs.refuse_capability, token, 'The run stopped: synthetic reason.')
    replay = s.client.get(f'/api/agent/analyze/{run}/events?conversation_id=conversation', headers=headers(s))
    kind, frame = frames(replay)[-1]
    assert kind == 'error'
    assert frame['payload'] == {'message': 'The run stopped: synthetic reason.', 'state': 'failed'}
