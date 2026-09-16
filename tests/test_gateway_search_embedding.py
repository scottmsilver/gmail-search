import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
import json

import pytest

from gmail_search.gateway import search_embedding as embedding
from gmail_search.gateway.capabilities import Capabilities
from gmail_search.gateway.data_admission import DataAdmission
from gmail_search.gateway.registry import AccessDenied, Registry


@pytest.fixture
def setup(tmp_path):
    tmp_path.chmod(0o700)
    registry = Registry(tmp_path / 'registry.sqlite', is_active=lambda owner: owner in ('alice','bob'))
    budget = registry.create_budget('alice', 100_000)
    run = registry.start_run('alice','conversation',request_key='start',budget_id=budget)
    caps = Capabilities(registry)
    token = caps.issue(run.run_id,audience='retrieval',operations=['search']).secret
    async def check():
        caps.authorize(token,audience='retrieval',operation='search')
        return True
    return registry, run, caps, token, check


def spend(registry):
    with registry._transaction() as db:
        return tuple(db.execute('SELECT reserved,spent FROM budgets').fetchone())


def response(**extra):
    return dict(embedding={'values':[0.25]*3072},usageMetadata={'promptTokenCount':7},**extra)


class Transport:
    def __init__(self, value=None, *, raw=None, wait=False):
        self.raw = raw if raw is not None else json.dumps(response() if value is None else value).encode()
        self.calls=[]
        self.started=asyncio.Event()
        self.closed=False
        self.wait=wait
        self.status_code=200
        self.close_gate=None
        self.closing=asyncio.Event()
    @asynccontextmanager
    async def stream(self, **kwargs):
        self.calls.append(kwargs)
        self.started.set()
        try:
            yield self
        finally:
            self.closing.set()
            if self.close_gate is not None:
                await self.close_gate.wait()
            self.closed=True
    async def __aiter__(self):
        if self.wait:
            await asyncio.Event().wait()
        for offset in range(0,len(self.raw),4096):
            yield self.raw[offset:offset+4096]


def service(setup, transport, admission=None):
    return embedding.GeminiQueryEmbedder(setup[0],transport,
        profile=embedding.GeminiEmbeddingProfile(input_units_per_token=2),
        admission=admission or DataAdmission(global_concurrency=4,owner_concurrency=2))


async def call(svc, setup, text='query', **kwargs):
    return await svc.embed(setup[1],text,deadline=asyncio.get_running_loop().time()+10,
                           check_active=setup[4],**kwargs)


@pytest.mark.asyncio
async def test_fixed_profile_request_reservation_usage_and_no_query_cache(setup):
    transport=Transport()
    original=transport.stream
    @asynccontextmanager
    async def checked(**kwargs):
        assert spend(setup[0])[0]==16384
        async with original(**kwargs) as result:
            yield result
    transport.stream=checked
    svc=service(setup,transport)
    assert svc.model=='gemini-embedding-2' and svc.dimensions==3072
    assert await call(svc,setup)==[.25]*3072
    request=transport.calls[0]
    assert request==dict(url=embedding.ENDPOINT,follow_redirects=False,body={
        'model':'models/gemini-embedding-2',
        'content':{'parts':[{'text':'task: search result | query: query'}]},
        'embedContentConfig':{'outputDimensionality':3072,'autoTruncate':False}})
    assert transport.closed and spend(setup[0])==(0,14)
    await call(svc,setup)
    assert len(transport.calls)==2 and spend(setup[0])==(0,28)


@pytest.mark.parametrize('changes',[{'model':'gemini-embedding-2-preview'},{'dimensions':768},
    {'input_units_per_token':True},{'input_units_per_token':0}])
def test_reject_unsupported_profile(changes):
    with pytest.raises((ValueError,AccessDenied)):
        embedding.GeminiEmbeddingProfile(**dict({'input_units_per_token':2},**changes))


@pytest.mark.asyncio
@pytest.mark.parametrize('text',['', ' '*3, 'x'*4096, '\ud800', None])
async def test_invalid_or_oversize_text_never_reserves(setup,text):
    transport=Transport(); svc=service(setup,transport)
    with pytest.raises((ValueError,AccessDenied)):
        await call(svc,setup,text)
    assert transport.calls==[] and spend(setup[0])==(0,0)


@pytest.mark.asyncio
@pytest.mark.parametrize('usage',[None,{}, {'promptTokenCount':True},{'promptTokenCount':8193},
    {'promptTokenCount':7,'unexpectedBilling':1}, {'promptTokenCount':7,'promptTokenDetails':[{'modality':'IMAGE','tokenCount':7}]}])
async def test_unknown_usage_full_charge(setup,usage):
    value=response(); value['usageMetadata']=usage
    assert await call(service(setup,Transport(value)),setup)==[.25]*3072
    assert spend(setup[0])==(0,16384)


@pytest.mark.asyncio
@pytest.mark.parametrize('raw',[b'{"embedding":{},"embedding":{}}',b'['*1000+b']'*1000,
    b'{"embedding":{"values":[NaN]}}',b'x'*262145,
    json.dumps({'embedding':{'values':[[1]]*3072}}).encode(),
    json.dumps({'embedding':{'values':[True]*3072}}).encode(),
    json.dumps({'embedding':{'values':[0]*3072}}).encode()])
async def test_malformed_response_settled_and_closed(setup,raw):
    transport=Transport(raw=raw)
    with pytest.raises(embedding.EmbeddingUnavailable):
        await call(service(setup,transport),setup)
    assert transport.closed and spend(setup[0])==(0,16384)


@pytest.mark.asyncio
async def test_wrong_owner_binding_and_revocation_prevent_send(setup):
    transport=Transport(); svc=service(setup,transport)
    with pytest.raises(AccessDenied):
        await svc.embed(replace(setup[1],owner_id='bob'),'query',
            deadline=asyncio.get_running_loop().time()+1,check_active=setup[4])
    setup[2].revoke(setup[3])
    with pytest.raises(AccessDenied):
        await call(svc,setup)
    assert not transport.calls and spend(setup[0])==(0,0)


@pytest.mark.asyncio
async def test_cancel_drain_holds_shared_capacity(setup):
    admission=DataAdmission(global_concurrency=1,owner_concurrency=1)
    transport=Transport(wait=True); transport.close_gate=asyncio.Event()
    svc=service(setup,transport,admission)
    task=asyncio.create_task(call(svc,setup))
    await transport.started.wait(); task.cancel()
    await transport.closing.wait(); task.cancel()
    with pytest.raises(embedding.EmbeddingUnavailable):
        await call(service(setup,Transport(),admission),setup)
    assert not task.done() and admission.active=={'alice':1}
    assert spend(setup[0])==(16384,0)
    transport.close_gate.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert transport.closed and not admission.active and spend(setup[0])==(0,16384)


@pytest.mark.asyncio
async def test_settlement_failure_quarantines_until_explicit_reconcile(setup,monkeypatch):
    admission=DataAdmission(global_concurrency=1,owner_concurrency=1)
    svc=service(setup,Transport(),admission)
    original=setup[0].settle
    def fail(*args):
        raise RuntimeError('private failure')
    monkeypatch.setattr(setup[0],'settle',fail)
    with pytest.raises(embedding.EmbeddingUnavailable,match='unavailable'):
        await call(svc,setup)
    assert admission.active=={'alice':1} and spend(setup[0])==(16384,0)
    monkeypatch.setattr(setup[0],'settle',original)
    await svc.reconcile()
    assert not admission.active and spend(setup[0])==(0,14)


@pytest.mark.asyncio
async def test_cancel_during_reservation_waits_and_settles(setup,monkeypatch):
    import threading
    entered=threading.Event(); release=threading.Event()
    original=setup[0].reserve
    def reserve(*args):
        result=original(*args)
        entered.set()
        release.wait(5)
        return result
    monkeypatch.setattr(setup[0],'reserve',reserve)
    admission=DataAdmission(global_concurrency=1,owner_concurrency=1)
    transport=Transport(); svc=service(setup,transport,admission)
    task=asyncio.create_task(call(svc,setup))
    assert await asyncio.to_thread(entered.wait,2)
    task.cancel(); await asyncio.sleep(.02); task.cancel()
    assert not task.done() and admission.active=={'alice':1}
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not transport.calls and not admission.active and spend(setup[0])==(0,16384)


@pytest.mark.asyncio
async def test_revocation_during_stream_closes_and_charges(setup):
    transport=Transport(wait=True); svc=service(setup,transport)
    task=asyncio.create_task(call(svc,setup))
    await transport.started.wait()
    setup[2].revoke(setup[3])
    with pytest.raises(AccessDenied):
        await asyncio.wait_for(task,2)
    assert transport.closed and spend(setup[0])==(0,16384)


@pytest.mark.asyncio
async def test_deadline_closes_and_charges(setup):
    transport=Transport(wait=True); svc=service(setup,transport)
    with pytest.raises(TimeoutError):
        await svc.embed(setup[1],'query',deadline=asyncio.get_running_loop().time()+.08,check_active=setup[4])
    assert transport.closed and spend(setup[0])==(0,16384)


@pytest.mark.asyncio
async def test_renewed_lease_expiry_accepted_but_mutated_budget_denied(setup):
    transport=Transport(); svc=service(setup,transport)
    setup[0].heartbeat(setup[1].run_id,ttl=100)
    assert await call(svc,setup)==[.25]*3072
    with pytest.raises(AccessDenied):
        await svc.embed(replace(setup[1],budget_id='foreign'),'query',
            deadline=asyncio.get_running_loop().time()+1,check_active=setup[4])
    assert len(transport.calls)==1


@pytest.mark.asyncio
async def test_budget_exhaustion_no_transport(setup):
    transport=Transport(); svc=service(setup,transport)
    setup[0].reserve(setup[1].run_id,'other-spend',99000)
    with pytest.raises(AccessDenied):
        await call(svc,setup)
    assert not transport.calls and not svc._admission.active


@pytest.mark.asyncio
async def test_revocation_during_settlement_never_publishes_vector(setup,monkeypatch):
    original=setup[0].settle
    def settle(*args):
        original(*args)
        setup[2].revoke(setup[3])
    monkeypatch.setattr(setup[0],'settle',settle)
    with pytest.raises(AccessDenied):
        await call(service(setup,Transport()),setup)
    assert spend(setup[0])==(0,14)


@pytest.mark.asyncio
async def test_slow_settlement_holds_capacity_through_repeated_cancel(setup,monkeypatch):
    import threading
    entered=threading.Event(); release=threading.Event()
    original=setup[0].settle
    def settle(*args):
        entered.set(); release.wait(5)
        return original(*args)
    monkeypatch.setattr(setup[0],'settle',settle)
    admission=DataAdmission(global_concurrency=1,owner_concurrency=1)
    task=asyncio.create_task(call(service(setup,Transport(),admission),setup))
    assert await asyncio.to_thread(entered.wait,2)
    task.cancel(); await asyncio.sleep(.02); task.cancel()
    assert not task.done() and admission.active=={'alice':1}
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not admission.active and spend(setup[0])==(0,14)


@pytest.mark.asyncio
async def test_http_fixed_url_headers_and_real_byte_stream(setup,monkeypatch):
    import httpx
    monkeypatch.setenv('HTTPS_PROXY','http://must-not-use.invalid')
    seen=[]
    async def handler(request):
        seen.append(request)
        return httpx.Response(200,headers={'Content-Type':'application/json'},
                              stream=httpx.ByteStream(json.dumps(response()).encode()))
    transport=embedding.GeminiEmbeddingHTTPTransport('synthetic-only-key',transport=httpx.MockTransport(handler))
    try:
        assert await call(service(setup,transport),setup)==[.25]*3072
        assert len(seen)==1 and str(seen[0].url)==embedding.ENDPOINT
        assert seen[0].headers['x-goog-api-key']=='synthetic-only-key'
        assert seen[0].headers['accept-encoding']=='identity'
        assert 'authorization' not in seen[0].headers
    finally:
        await transport.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('status,headers',[(302,{'location':'http://foreign.invalid'}),
    (200,{'Content-Type':'text/plain'}),(200,{'Content-Type':'application/json','Content-Encoding':'gzip'}),
    (429,{'Content-Type':'application/json'})])
async def test_http_denies_redirect_encoding_status_no_retry(setup,status,headers):
    import httpx
    seen=[]
    async def handler(request):
        seen.append(request)
        return httpx.Response(status,headers=headers,stream=httpx.ByteStream(b'private provider diagnostic'))
    transport=embedding.GeminiEmbeddingHTTPTransport('synthetic-only-key',transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(embedding.EmbeddingUnavailable) as caught:
            await call(service(setup,transport),setup)
        assert 'private' not in str(caught.value) and len(seen)==1
        assert spend(setup[0])==(0,16384)
    finally:
        await transport.aclose()


@pytest.mark.asyncio
async def test_cancel_during_normal_eof_close_retains_capacity_until_ack(setup):
    admission=DataAdmission(global_concurrency=1,owner_concurrency=1)
    transport=Transport(); transport.close_gate=asyncio.Event()
    task=asyncio.create_task(call(service(setup,transport,admission),setup))
    await asyncio.wait_for(transport.closing.wait(),2)
    task.cancel()
    await asyncio.sleep(.03)
    task.cancel()
    held=(not task.done(),transport.closed,dict(admission.active),spend(setup[0]))
    transport.close_gate.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert held==(True,False,{'alice':1},(16384,0))
    assert transport.closed and not admission.active and spend(setup[0])==(0,16384)


@pytest.mark.asyncio
@pytest.mark.parametrize('failure',['revoke','deadline'])
async def test_final_binding_delay_rechecks_revocation_and_deadline(setup,monkeypatch,failure):
    import threading
    svc=service(setup,Transport())
    original=svc._binding
    reached=asyncio.Event(); gate=threading.Event()
    loop=asyncio.get_running_loop()
    def binding(lease):
        if spend(setup[0])[1]:
            loop.call_soon_threadsafe(reached.set)
            assert gate.wait(3)
        return original(lease)
    monkeypatch.setattr(svc,'_binding',binding)
    deadline=loop.time()+(.2 if failure=='deadline' else 3)
    task=asyncio.create_task(svc.embed(setup[1],'query',deadline=deadline,check_active=setup[4]))
    try:
        await asyncio.wait_for(reached.wait(),2)
        if failure=='revoke':
            setup[2].revoke(setup[3])
        else:
            await asyncio.sleep(max(0,deadline-loop.time())+.02)
    finally:
        gate.set()
    with pytest.raises(AccessDenied if failure=='revoke' else TimeoutError):
        await task
    assert spend(setup[0])==(0,14)


@pytest.mark.asyncio
async def test_final_authorization_delay_rechecks_deadline(setup):
    svc=service(setup,Transport())
    reached=asyncio.Event(); gate=asyncio.Event()
    async def check():
        result=await setup[4]()
        if spend(setup[0])[1]:
            reached.set()
            await gate.wait()
        return result
    loop=asyncio.get_running_loop(); deadline=loop.time()+.2
    task=asyncio.create_task(svc.embed(setup[1],'query',deadline=deadline,check_active=check))
    await asyncio.wait_for(reached.wait(),2)
    await asyncio.sleep(max(0,deadline-loop.time())+.02)
    gate.set()
    with pytest.raises(TimeoutError):
        await task
