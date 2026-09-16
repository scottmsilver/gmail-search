import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
import json

import pytest

from gmail_search.gateway import search_reranker as reranker
from gmail_search.gateway.capabilities import Capabilities
from gmail_search.gateway.data_admission import DataAdmission
from gmail_search.gateway.registry import AccessDenied, Registry
from gmail_search.search.ranking import ThreadResult, ThreadMatch
from test_gateway_search_embedding import Transport, spend


@pytest.fixture
def setup(tmp_path):
    tmp_path.chmod(0o700)
    registry=Registry(tmp_path/'registry.sqlite',is_active=lambda owner: owner in ('alice','bob'))
    budget=registry.create_budget('alice',10_000_000)
    lease=registry.start_run('alice','conversation',request_key='run',budget_id=budget)
    caps=Capabilities(registry)
    token=caps.issue(lease.run_id,audience='retrieval',operations=['search']).secret
    async def check():
        caps.authorize(token,audience='retrieval',operation='search')
        return True
    return registry,lease,caps,token,check


def threads(n=3):
    return tuple(ThreadResult(f'thread-secret-{i}',.8,.8,f'Subject {i}',['Alice <alice@example.test>'],
        2,'2026-01-01','2026-01-02',True,
        [ThreadMatch(f'message-{i}',.8,'alice@example.test','2026-01-02','snippet','semantic')]) for i in range(n))


def response(order=None,usage=None):
    return {'modelVersion':'gemini-3.1-flash-lite','responseId':'synthetic',
        'candidates':[{'index':0,'finishReason':'STOP','content':{'role':'model','parts':[
            {'text':json.dumps({'order':[2,0,1] if order is None else order})}]}}],
        'usageMetadata':{'promptTokenCount':11,'candidatesTokenCount':9,'thoughtsTokenCount':0,'totalTokenCount':20} if usage is None else usage}


def service(setup,transport,admission=None):
    return reranker.GeminiThreadReranker(setup[0],transport,
        profile=reranker.GeminiRerankerProfile(input_units_per_token=2,output_units_per_token=3,
            reservation_policy='full-model-ceilings-v1'),
        admission=admission or DataAdmission(global_concurrency=4,owner_concurrency=2))


async def call(svc,setup,candidates=None,query='query'):
    return await svc.rerank(setup[1],query,threads() if candidates is None else candidates,
        deadline=asyncio.get_running_loop().time()+10,check_active=setup[4])


@pytest.mark.asyncio
async def test_fixed_model_ordinal_prompt_and_full_reservation(setup):
    transport=Transport(response()); original=transport.stream
    @asynccontextmanager
    async def checked(**kwargs):
        assert spend(setup[0])==(2293760,0)
        async with original(**kwargs) as result:
            yield result
    transport.stream=checked
    svc=service(setup,transport)
    assert await call(svc,setup)==('thread-secret-2','thread-secret-0','thread-secret-1')
    assert svc.model=='gemini-3.1-flash-lite' and len(transport.calls)==1
    request=transport.calls[0]
    assert request['url']==reranker.ENDPOINT and request['follow_redirects'] is False
    encoded=json.dumps(request['body'])
    assert 'thread-secret' not in encoded and 'message-' not in encoded
    assert not set(request['body']) & {'tools','toolConfig','cachedContent'}
    config=request['body']['generationConfig']
    assert config['maxOutputTokens']==512 and config['thinkingConfig']=={'thinkingLevel':'MINIMAL','includeThoughts':False}
    assert config['responseMimeType']=='application/json'
    assert transport.closed and spend(setup[0])==(0,49)


@pytest.mark.parametrize('changes',[{'model':'gemini-3.1-flash-lite-preview'},
    {'reservation_policy':'estimate'},{'input_units_per_token':0},{'output_units_per_token':True}])
def test_explicit_budget_profile_rejects_preview_and_estimates(changes):
    kwargs=dict(input_units_per_token=2,output_units_per_token=3,reservation_policy='full-model-ceilings-v1')
    kwargs.update(changes)
    with pytest.raises(ValueError):
        reranker.GeminiRerankerProfile(**kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize('order',[[0,1],[0,0,1],[0,1,3],[True,0,1],['0',1,2],[-1,0,1],None])
async def test_exact_permutation_required_no_partial_fallback(setup,order):
    value=response(order)
    if order is None:
        value['candidates'][0]['content']['parts'][0]['text']='```json\n[2,0,1]\n```'
    transport=Transport(value)
    with pytest.raises(reranker.RerankingUnavailable):
        await call(service(setup,transport),setup)
    assert transport.closed and spend(setup[0])==(0,2293760)


@pytest.mark.asyncio
@pytest.mark.parametrize('mutate',[
    lambda r:r.update(modelVersion='other-model'),
    lambda r:r['candidates'][0].update(finishReason='MAX_TOKENS'),
    lambda r:r['candidates'][0]['content']['parts'][0].update(thought=True),
    lambda r:r['candidates'][0]['content']['parts'][0].update(functionCall={'name':'exfiltrate'}),
    lambda r:r['candidates'].append(r['candidates'][0]),
    lambda r:r['candidates'][0]['content']['parts'][0].update(text='{"order":[2,0,1],"order":[0,1,2]}'),
    lambda r:r['candidates'][0]['content']['parts'][0].update(text='['*1000+']'*1000),
])
async def test_unsupported_provider_response_refused(setup,mutate):
    value=response(); mutate(value)
    with pytest.raises(reranker.RerankingUnavailable):
        await call(service(setup,Transport(value)),setup)
    assert spend(setup[0])==(0,2293760)


@pytest.mark.asyncio
@pytest.mark.parametrize('usage',[{}, {'promptTokenCount':11,'candidatesTokenCount':9,'totalTokenCount':20},
    {'promptTokenCount':11,'candidatesTokenCount':9,'thoughtsTokenCount':5,'totalTokenCount':20},
    {'promptTokenCount':11,'candidatesTokenCount':9,'thoughtsTokenCount':0,'totalTokenCount':20,'cachedContentTokenCount':1}])
async def test_unknown_billing_full_reservation(setup,usage):
    assert await call(service(setup,Transport(response(usage=usage))),setup)==tuple(t.thread_id for t in (threads()[2],threads()[0],threads()[1]))
    assert spend(setup[0])==(0,2293760)


@pytest.mark.asyncio
async def test_thinking_tokens_charged_at_output_rate(setup):
    value=response(usage={'promptTokenCount':11,'candidatesTokenCount':9,'thoughtsTokenCount':5,'totalTokenCount':25})
    await call(service(setup,Transport(value)),setup)
    assert spend(setup[0])==(0,64)


@pytest.mark.asyncio
async def test_candidate_snapshot_no_mutation_during_call(setup):
    candidates=threads(); transport=Transport(response()); original=transport.stream
    @asynccontextmanager
    async def changing(**kwargs):
        candidates[0].thread_id='foreign-after-snapshot'
        async with original(**kwargs) as result:
            yield result
    transport.stream=changing
    assert await call(service(setup,transport),setup,candidates)==('thread-secret-2','thread-secret-0','thread-secret-1')


@pytest.mark.asyncio
@pytest.mark.parametrize('candidates',[(),list(threads()),threads(31), (threads()[0],threads()[0]),
    (replace(threads()[0],subject='x'*4097),)])
async def test_invalid_candidate_inputs_no_reservation(setup,candidates):
    transport=Transport(response())
    with pytest.raises(ValueError):
        await call(service(setup,transport),setup,candidates)
    assert not transport.calls and spend(setup[0])==(0,0)


@pytest.mark.asyncio
async def test_shared_embedding_capacity_and_eof_close_cancel(setup):
    from gmail_search.gateway.search_embedding import GeminiQueryEmbedder, GeminiEmbeddingProfile
    admission=DataAdmission(global_concurrency=1,owner_concurrency=1)
    transport=Transport(response()); transport.close_gate=asyncio.Event()
    task=asyncio.create_task(call(service(setup,transport,admission),setup))
    await transport.closing.wait(); task.cancel(); await asyncio.sleep(.02); task.cancel()
    other=GeminiQueryEmbedder(setup[0],Transport(),profile=GeminiEmbeddingProfile(2),admission=admission)
    with pytest.raises(RuntimeError):
        await other.embed(setup[1],'query',deadline=asyncio.get_running_loop().time()+1,check_active=setup[4])
    assert not task.done() and admission.active=={'alice':1}
    transport.close_gate.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert transport.closed and not admission.active and spend(setup[0])==(0,2293760)


@pytest.mark.asyncio
async def test_maximum_candidate_count_exact_mapping_and_summary_profile(setup):
    candidates=threads(30)
    candidates[0].subject='s'*1000
    candidates[0].participants=['First <first@example.test>','Second','Third','ignored']
    candidates[0].matches[0].snippet='z'*200
    order=list(reversed(range(30)))
    transport=Transport(response(order))
    assert await call(service(setup,transport),setup,candidates)==tuple(candidates[i].thread_id for i in order)
    payload=json.loads(transport.calls[0]['body']['contents'][0]['parts'][0]['text'])
    assert payload['threads'][0]==dict(ordinal=0,subject='s'*512,
        participants=['First','Second','Third'],message_count=2,snippet='z'*100)


@pytest.mark.asyncio
@pytest.mark.parametrize('query',['',None,'x'*4097,'\ud800','a\x00b'])
async def test_invalid_query_never_reserves(setup,query):
    transport=Transport(response())
    with pytest.raises(ValueError):
        await call(service(setup,transport),setup,query=query)
    assert not transport.calls and spend(setup[0])==(0,0)


@pytest.mark.asyncio
async def test_http_fixed_headers_body_and_stream(setup,monkeypatch):
    import httpx
    monkeypatch.setenv('HTTPS_PROXY','http://forbidden.invalid')
    seen=[]
    async def handler(request):
        seen.append(request)
        return httpx.Response(200,headers={'Content-Type':'application/json'},
                              stream=httpx.ByteStream(json.dumps(response()).encode()))
    transport=reranker.GeminiRerankerHTTPTransport('synthetic-only',transport=httpx.MockTransport(handler))
    try:
        assert await call(service(setup,transport),setup)==('thread-secret-2','thread-secret-0','thread-secret-1')
        assert len(seen)==1 and str(seen[0].url)==reranker.ENDPOINT
        assert seen[0].headers['x-goog-api-key']=='synthetic-only'
        assert 'authorization' not in seen[0].headers
        assert seen[0].headers['accept-encoding']=='identity'
        assert json.loads(seen[0].content)['generationConfig']['maxOutputTokens']==512
    finally:
        await transport.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('status,headers',[(302,{'location':'http://foreign.invalid'}),
    (200,{'Content-Type':'text/plain'}),(200,{'Content-Type':'application/json','Content-Encoding':'gzip'}),
    (429,{'Content-Type':'application/json'})])
async def test_http_no_retry_redirect_or_provider_diagnostic(setup,status,headers):
    import httpx
    seen=[]
    async def handler(request):
        seen.append(request)
        return httpx.Response(status,headers=headers,stream=httpx.ByteStream(b'private provider diagnostic'))
    transport=reranker.GeminiRerankerHTTPTransport('synthetic-only',transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(reranker.RerankingUnavailable) as caught:
            await call(service(setup,transport),setup)
        assert 'private' not in str(caught.value) and len(seen)==1
        assert spend(setup[0])==(0,2293760)
    finally:
        await transport.aclose()


@pytest.mark.asyncio
async def test_http_rejects_changed_tools_and_endpoint_before_send(setup):
    import httpx
    seen=[]
    async def handler(request):
        seen.append(request)
        raise AssertionError('must not send')
    transport=reranker.GeminiRerankerHTTPTransport('synthetic-only',transport=httpx.MockTransport(handler))
    _,body=reranker._snapshot('query',threads())
    body['tools']=[{'googleSearch':{}}]
    try:
        for url in (reranker.ENDPOINT,'https://foreign.invalid'):
            with pytest.raises(reranker.RerankingUnavailable):
                async with transport.stream(url=url,body=body,follow_redirects=False):
                    pass
        assert not seen
    finally:
        await transport.aclose()


@pytest.mark.asyncio
async def test_shared_global_limit_across_owners(setup):
    admission=DataAdmission(global_concurrency=1,owner_concurrency=1)
    alice=Transport(response(),wait=True)
    task=asyncio.create_task(call(service(setup,alice,admission),setup))
    await alice.started.wait()
    registry=setup[0]
    budget=registry.create_budget('bob',10_000_000)
    bob=registry.start_run('bob','conversation',request_key='run',budget_id=budget)
    caps=setup[2]; token=caps.issue(bob.run_id,audience='retrieval',operations=['search']).secret
    async def check():
        caps.authorize(token,audience='retrieval',operation='search')
        return True
    other=Transport(response())
    try:
        with pytest.raises(reranker.RerankingUnavailable):
            await service(setup,other,admission).rerank(bob,'query',threads(),
                deadline=asyncio.get_running_loop().time()+1,check_active=check)
        assert not other.calls and admission.active=={'alice':1}
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_revoked_capability_after_final_binding_does_not_publish(setup,monkeypatch):
    import threading
    svc=service(setup,Transport(response()))
    original=svc._binding
    reached=asyncio.Event(); gate=threading.Event(); loop=asyncio.get_running_loop()
    def binding(lease):
        if spend(setup[0])[1]:
            loop.call_soon_threadsafe(reached.set)
            assert gate.wait(3)
        return original(lease)
    monkeypatch.setattr(svc,'_binding',binding)
    task=asyncio.create_task(call(svc,setup))
    try:
        await asyncio.wait_for(reached.wait(),2)
        setup[2].revoke(setup[3])
    finally:
        gate.set()
    with pytest.raises(AccessDenied):
        await task
    assert spend(setup[0])==(0,49)
