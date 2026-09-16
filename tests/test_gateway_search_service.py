"""Run orchestration uses capability owner, bound reader/index and injected provider.

Synthetic dependencies here qualify orchestration, not live provider behavior.
"""
import asyncio
from contextlib import asynccontextmanager
import importlib
from types import SimpleNamespace

import pytest

from gmail_search.gateway.capabilities import Capabilities
from gmail_search.gateway.registry import Registry, AccessDenied
from gmail_search.gateway.search_queries import Selection, LexicalHit, MessageCandidate, AliasRow
from gmail_search.gateway.search_reader import SearchProfile
from test_gateway_search_ranking import _embedding, _message, _summary


class Queries:
    def __init__(self, owner):
        self.owner = owner
        self.calls = []
        self.candidates = Selection((MessageCandidate('m1'),),True)

    async def owner_aliases(self,**kwargs):
        return Selection((AliasRow('ke','["kol emeth"]'),),True)

    async def owner_contacts(self,**kwargs):
        return Selection((),True)

    async def structured_candidates(self,filters,**kwargs):
        self.calls.append(('structured',filters))
        return self.candidates

    async def lexical_messages(self,tokens,**kwargs):
        self.calls.append(('lexical',tokens,kwargs))
        return Selection((LexicalHit(1,'m1',1.),) if kwargs.get('candidate_ids')!=() else (),True)

    async def lexical_attachments(self,tokens,**kwargs):
        return Selection((),True)

    async def restricted_vectors_page(self,candidates,**kwargs):
        import struct
        from gmail_search.gateway.search_queries import EmbeddingVectorRow
        return Selection((EmbeddingVectorRow(1,'m1',struct.pack('<2f',1.,0.),'test-model','ok'),),True)

    async def hydrate_embeddings(self,ids):
        return Selection(tuple(_embedding(i) for i in ids if i==1),True)

    async def hydrate_messages(self,ids,**kwargs):
        return Selection(tuple(_message(mid,'t1') for mid in ids),True)

    async def hydrate_threads(self,ids):
        return Selection(tuple(_summary(tid) for tid in ids),True)

    async def message_topics(self,ids):
        return Selection((),True)

    async def topic_facets(self,ids,**kwargs):
        return Selection((),True)


class Reader:
    profile=SearchProfile('test-model','test-facts',2)

    def __init__(self):
        self.owners=[]
        self.open=0
        self.queries=Queries('alice')

    @asynccontextmanager
    async def session(self,owner_id,*,deadline,check_active):
        self.owners.append(owner_id)
        self.open+=1
        try:
            await check_active()
            yield self.queries
        finally:
            self.open-=1


class Indexes:
    def __init__(self):
        self.owners=[]
        self.ids=([1],[.8])
        self.model='test-model'
        self.search_calls=0
        self.entered=False

    @asynccontextmanager
    async def acquire(self,owner_id):
        self.owners.append(owner_id)
        self.entered=True
        async def search(vector,*,top_k,absolute_deadline):
            self.search_calls+=1
            return self.ids
        try:
            yield SimpleNamespace(binding=SimpleNamespace(owner_id=owner_id,model=self.model,dimensions=2),search=search)
        finally:
            self.entered=False


class Embedder:
    model='test-model'
    dimensions=2

    def __init__(self,reader,indexes):
        self.reader,self.indexes=reader,indexes
        self.calls=[]
        self.block=None

    async def embed(self,lease,text,*,deadline,check_active):
        assert self.reader.open==0, 'Database snapshot held during provider call'
        assert self.indexes.entered, 'Index generation must be pinned before embedding'
        self.calls.append((lease.owner_id,text))
        if self.block is not None:
            await self.block.wait()
        await check_active()
        return [1.,0.]


def _service(tmp_path):
    module=importlib.import_module('gmail_search.gateway.search_service')
    registry=Registry(tmp_path/'search.sqlite',is_active=lambda _:True)
    caps=Capabilities(registry)
    run=registry.start_run('alice','conversation',request_key='search',writer=False)
    token=caps.issue(run.run_id,audience='retrieval',operations={'search'})
    reader,indexes=Reader(),Indexes()
    embedder=Embedder(reader,indexes)
    owner=module.OwnerSearchContext(owner_id='alice',emails=('Alice@Test',))
    service=module.RunSearchService(caps,reader,indexes,embedder,owners={'alice':owner},reranker=None)
    return service,caps,token,reader,indexes,embedder


@pytest.mark.asyncio
@pytest.mark.parametrize('failure',['revoke','deadline'])
async def test_final_watcher_drain_precedes_publication_checks(tmp_path,monkeypatch,failure):
    import threading
    module=importlib.import_module('gmail_search.gateway.search_service')
    service,caps,token,reader,indexes,_=_service(tmp_path)
    loop=asyncio.get_running_loop()
    started=asyncio.Event()
    release=threading.Event()
    watcher=[]
    authorize=service.authorize
    execute=service._execute
    original_time=loop.time

    def delayed_authorization():
        lease=caps.authorize(token.secret,audience='retrieval',operation='search')
        loop.call_soon_threadsafe(started.set)
        assert release.wait(3)
        return lease

    async def authorize_with_slow_watcher(value):
        task=asyncio.current_task()
        if task.get_coro().__qualname__.endswith('.watch'):
            watcher.append(task)
            return await module._thread(delayed_authorization)
        return await authorize(value)

    async def execute_after_watcher_starts(*args):
        await started.wait()
        return await execute(*args)

    async def wait_for_cleanup():
        await started.wait()
        while not watcher[0].cancelling():
            await asyncio.sleep(0)

    monkeypatch.setattr(service,'authorize',authorize_with_slow_watcher)
    monkeypatch.setattr(service,'_execute',execute_after_watcher_starts)
    work=asyncio.create_task(service.search(token.secret,query='draw request'))
    try:
        await asyncio.wait_for(wait_for_cleanup(),2)
        assert not work.done() and not reader.open and not indexes.entered
        if failure=='revoke':
            caps.revoke(token.secret)
        else:
            # Only the final publication check remains; advance its absolute
            # clock without waiting out the real 30-second operation deadline.
            monkeypatch.setattr(loop,'time',lambda:original_time()+31)
        release.set()
        with pytest.raises(AccessDenied if failure=='revoke' else TimeoutError):
            await work
    finally:
        monkeypatch.setattr(loop,'time',original_time)
        release.set()
        await asyncio.gather(work,return_exceptions=True)


@pytest.mark.asyncio
async def test_capability_owner_and_closed_snapshot_during_embedding(tmp_path):
    service,_,token,reader,indexes,embedder=_service(tmp_path)
    result=await service.search(token.secret,query='draw request')
    assert reader.owners==['alice','alice'] and indexes.owners==['alice']
    assert embedder.calls==[('alice','draw request')]
    assert result['results'][0]['thread_id']=='t1'
    assert result['results'][0]['cite_ref']=='t1'
    assert result['coverage']['semantic']=='approximate'
    assert result['coverage']['reranking']=='disabled'
    assert reader.open==0 and not indexes.entered


@pytest.mark.asyncio
async def test_revoked_capability_never_opens_reader_or_index(tmp_path):
    service,caps,token,reader,indexes,embedder=_service(tmp_path)
    caps.revoke(token.secret)
    with pytest.raises(AccessDenied):
        await service.search(token.secret,query='draw request')
    assert not reader.owners and not indexes.owners and not embedder.calls


@pytest.mark.asyncio
async def test_owner_or_index_model_mismatch_fails_before_provider(tmp_path):
    service,_,token,reader,indexes,embedder=_service(tmp_path)
    indexes.model='other-model'
    with pytest.raises(RuntimeError,match='Search profile is unavailable'):
        await service.search(token.secret,query='draw request')
    assert not embedder.calls and reader.open==0 and not indexes.entered


@pytest.mark.asyncio
async def test_unknown_index_identifier_does_not_reach_output(tmp_path):
    service,_,token,_,indexes,_=_service(tmp_path)
    indexes.ids=([1,999],[.8,999.])
    result=await service.search(token.secret,query='draw request')
    assert len(result['results'])==1
    assert 'unavailable_embedding' in result['coverage']['reasons']
    assert result['results'][0]['similarity']==.8
    assert [match['message_id'] for match in result['results'][0]['matches']]==['m1']


@pytest.mark.asyncio
async def test_aliases_only_use_owner_reader(tmp_path):
    service,_,token,reader,_,embedder=_service(tmp_path)
    await service.search(token.secret,query='ke board')
    assert embedder.calls==[('alice','ke kol emeth board')]
    lexical=[call for call in reader.queries.calls if call[0]=='lexical']
    assert any('kol' in call[1] for call in lexical)
    assert any(call[1]==('ke','board') for call in lexical)


@pytest.mark.asyncio
async def test_revocation_during_embedding_cancels_provider_and_drains_index(tmp_path):
    service,caps,token,reader,indexes,embedder=_service(tmp_path)
    embedder.block=asyncio.Event()
    task=asyncio.create_task(service.search(token.secret,query='draw request'))
    for _ in range(100):
        if embedder.calls:break
        await asyncio.sleep(.01)
    assert embedder.calls
    caps.revoke(token.secret)
    with pytest.raises(AccessDenied):
        await asyncio.wait_for(task,2)
    assert reader.open==0 and not indexes.entered


@pytest.mark.asyncio
@pytest.mark.parametrize('options',[{'query':''},{'query':'x','owner_id':'bob'},
    {'query':'x','top_k':True},{'query':'x','top_k':101},{'query':'x','detail':'raw'},
    {'query':'x','date_from':'2026-99-01'},{'query':'x','max_matches':101}])
async def test_invalid_options_cannot_select_identity_or_start_work(tmp_path,options):
    service,_,token,reader,indexes,embedder=_service(tmp_path)
    with pytest.raises((ValueError,TypeError)):
        await service.search(token.secret,**options)
    assert not reader.owners and not indexes.owners and not embedder.calls


@pytest.mark.asyncio
async def test_pending_index_is_distinct_from_unknown_index(tmp_path):
    from gmail_search.gateway.search_index import PendingIndex, IndexUnavailable
    service,_,token,reader,indexes,embedder=_service(tmp_path)
    @asynccontextmanager
    async def pending(owner):
        raise PendingIndex()
        yield
    indexes.acquire=pending
    result=await service.search(token.secret,query='draw request')
    assert result['pending_index'] and result['results']==[]
    assert not embedder.calls and reader.owners == ['alice']
    assert reader.open == 0  # Schema qualification precedes even a pending index.
    @asynccontextmanager
    async def unavailable(owner):
        raise IndexUnavailable()
        yield
    indexes.acquire=unavailable
    with pytest.raises(IndexUnavailable):
        await service.search(token.secret,query='draw request')


@pytest.mark.asyncio
async def test_full_detail_is_text_with_explicit_clipping(tmp_path):
    from dataclasses import replace
    service,_,token,reader,_,_=_service(tmp_path)
    async def clipped(ids,**kwargs):
        return Selection(tuple(replace(_message(mid,'t1'),body_text='first page',body_bytes=999999,
                                       body_complete=False) for mid in ids),True)
    reader.queries.hydrate_messages=clipped
    result=await service.search(token.secret,query='draw request',detail='full')
    match=result['results'][0]['matches'][0]
    assert match['body']=='first page' and match['body_format']=='text'
    assert not match['body_complete'] and match['body_bytes']==999999
    assert 'body_truncated' in result['coverage']['reasons']


@pytest.mark.asyncio
async def test_repeated_cancel_waits_for_provider_teardown(tmp_path):
    service,_,token,reader,indexes,embedder=_service(tmp_path)
    entered,closing,release=asyncio.Event(),asyncio.Event(),asyncio.Event()
    async def blocked(*args,**kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            closing.set()
            await release.wait()
    embedder.embed=blocked
    task=asyncio.create_task(service.search(token.secret,query='draw request'))
    await asyncio.wait_for(entered.wait(),1)
    task.cancel()
    await asyncio.wait_for(closing.wait(),1)
    task.cancel()
    await asyncio.sleep(.02)
    assert not task.done() and indexes.entered and reader.open==0
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not indexes.entered


@pytest.mark.asyncio
async def test_facets_count_threads_once_even_with_multiple_matching_messages(tmp_path):
    from gmail_search.gateway.search_queries import TopicRow, TopicFacet
    service,_,token,reader,indexes,_=_service(tmp_path)
    indexes.ids=([1,2],[.8,.7])
    async def hydrate(ids):
        return Selection((_embedding(1),_embedding(2,'m2')),True)
    reader.queries.hydrate_embeddings=hydrate
    async def topics(ids):
        return Selection((TopicRow('m1','topic','Budget'),TopicRow('m2','topic','Budget')),True)
    async def facets(ids,**kwargs):
        return Selection((TopicFacet('topic','Budget',2),),True)
    reader.queries.message_topics=topics
    reader.queries.topic_facets=facets
    result=await service.search(token.secret,query='draw request')
    assert result['results'][0]['topic_ids']==['topic']
    assert result['facets']==[{'topic_id':'topic','label':'Budget','count':1}]


@pytest.mark.asyncio
async def test_large_facet_labels_obey_total_response_byte_limit(tmp_path):
    import json
    from gmail_search.gateway.search_queries import TopicRow, TopicFacet
    service,_,token,reader,_,_=_service(tmp_path)
    async def topics(ids):
        return Selection(tuple(TopicRow('m1',str(i),'x') for i in range(10)),True)
    async def facets(ids,**kwargs):
        return Selection(tuple(TopicFacet(str(i),'x'*600000,1) for i in range(10)),True)
    reader.queries.message_topics=topics
    reader.queries.topic_facets=facets
    result=await service.search(token.secret,query='draw request')
    assert len(json.dumps(result,ensure_ascii=False).encode())<=4*1024*1024
    assert 'facet_response_bytes' in result['coverage']['reasons']


@pytest.mark.asyncio
async def test_small_structured_corpus_uses_exact_vectors_and_bound_filters(tmp_path):
    service,_,token,reader,indexes,_=_service(tmp_path)
    result=await service.search(token.secret,query='draw from:alice@test after:2020-01-01',date_from='2025-01-01')
    assert result['coverage']['semantic']=='exact_restricted'
    assert indexes.search_calls==0
    filters=next(call[1] for call in reader.queries.calls if call[0]=='structured')
    assert filters.from_filter=='alice@test' and filters.date_from=='2025-01-01'
    assert all(call[2]['candidate_ids']==('m1',) for call in reader.queries.calls if call[0]=='lexical')


@pytest.mark.asyncio
async def test_empty_structured_corpus_returns_no_unfiltered_fallback(tmp_path):
    service,_,token,reader,indexes,_=_service(tmp_path)
    reader.queries.candidates=Selection((),True)
    result=await service.search(token.secret,query='draw from:absent@test')
    assert result['results']==[] and indexes.search_calls==0


@pytest.mark.asyncio
async def test_structured_cap_is_visible_in_response(tmp_path):
    service,_,token,reader,_,_=_service(tmp_path)
    reader.queries.candidates=Selection((MessageCandidate('m1'),),False,'row_limit')
    result=await service.search(token.secret,query='draw from:alice@test')
    assert 'structured_row_limit' in result['coverage']['reasons']
    assert not result['coverage']['complete']


@pytest.mark.asyncio
async def test_large_lexical_union_hydrates_in_bounded_batches(tmp_path):
    service,_,token,reader,_,_=_service(tmp_path)
    ids=tuple('hit'+str(i) for i in range(16000))
    reader.queries.candidates=Selection(tuple(MessageCandidate(mid) for mid in ('m1',*ids)),True)
    serial=0
    async def lexical(tokens,**kwargs):
        nonlocal serial
        begin=serial*2000
        serial+=1
        return Selection(tuple(LexicalHit(i+1,ids[i],1.) for i in range(begin,begin+2000)),True)
    reader.queries.lexical_messages=lexical
    reader.queries.lexical_attachments=lexical
    batches=[]
    async def hydrate(ids,**kwargs):
        assert len(ids)<=1000
        batches.append(tuple(ids))
        return Selection(tuple(_message(mid,'t1') for mid in ids),True)
    reader.queries.hydrate_messages=hydrate
    async def topics(ids):
        assert len(ids)<=1000
        return Selection((),True)
    reader.queries.message_topics=topics
    result=await service.search(token.secret,query='ke board after:2020-01-01')
    assert len(result['results'])==1 and len(result['results'][0]['matches'])==3
    assert len(batches)>=16 and serial==8


@pytest.mark.asyncio
async def test_large_filtered_ann_overfetch_retains_hit_beyond_default_pool(tmp_path):
    service,_,token,reader,indexes,_=_service(tmp_path)
    reader.queries.candidates=Selection(tuple(MessageCandidate(mid) for mid in
        ('m1',*('other'+str(i) for i in range(20000)))),True)
    requested=[]
    @asynccontextmanager
    async def acquire(owner):
        indexes.entered=True
        async def search(vector,*,top_k,absolute_deadline):
            requested.append(top_k)
            return ([101],[.8]) if top_k>100 else ([],[])
        try:
            yield SimpleNamespace(binding=SimpleNamespace(owner_id=owner,model='test-model',dimensions=2),search=search)
        finally:
            indexes.entered=False
    indexes.acquire=acquire
    async def hydrate(ids):
        return Selection(tuple(_embedding(identifier) for identifier in ids),True)
    async def no_lexical(*args,**kwargs):
        return Selection((),True)
    reader.queries.hydrate_embeddings=hydrate
    reader.queries.lexical_messages=no_lexical
    reader.queries.lexical_attachments=no_lexical
    result=await service.search(token.secret,query='draw from:alice@test')
    assert requested==[10000]
    assert result['results'][0]['thread_id']=='t1'


@pytest.mark.asyncio
async def test_filtered_overfetch_does_not_hydrate_discarded_threads(tmp_path):
    service,_,token,reader,indexes,_=_service(tmp_path)
    reader.queries.candidates=Selection(tuple(MessageCandidate('m'+str(i)) for i in range(1,20002)),True)
    indexes.ids=(list(range(1,301)),[1-i/1000 for i in range(300)])
    async def hydrate(ids):
        return Selection(tuple(_embedding(i,'m'+str(i),'t'+str(i)) for i in ids),True)
    seen=[]
    async def summaries(ids):
        seen.extend(ids)
        return Selection(tuple(_summary(tid) for tid in ids),True)
    async def no_lexical(*args,**kwargs):
        return Selection((),True)
    reader.queries.hydrate_embeddings=hydrate
    reader.queries.hydrate_threads=summaries
    reader.queries.lexical_messages=no_lexical
    reader.queries.lexical_attachments=no_lexical
    await service.search(token.secret,query='draw from:sender')
    assert len(seen)==100


@pytest.mark.asyncio
@pytest.mark.parametrize('foreign',[False,True])
async def test_reranker_is_outside_snapshot_and_cannot_add_foreign_threads(tmp_path,foreign):
    service,_,token,reader,indexes,_=_service(tmp_path)
    indexes.ids=([1,2,3,4],[.8]*4)
    async def hydrate(ids):
        return Selection(tuple(_embedding(i,'m'+str(i),'t'+str(i)) for i in ids),True)
    async def lexical(*args,**kwargs):
        return Selection(tuple(LexicalHit(i,'m'+str(i),1.) for i in range(1,5)),True)
    reader.queries.hydrate_embeddings=hydrate
    reader.queries.lexical_messages=lexical
    calls=[]
    class Reranker:
        async def rerank(self,lease,query,threads,*,deadline,check_active):
            assert reader.open==0 and indexes.entered and lease.owner_id=='alice'
            await check_active()
            calls.append(tuple(thread.thread_id for thread in threads))
            return ['foreign'] if foreign else [thread.thread_id for thread in reversed(threads)]
    service.reranker=Reranker()
    if foreign:
        with pytest.raises(RuntimeError,match='Invalid search reranking response'):
            await service.search(token.secret,query='draw request')
    else:
        result=await service.search(token.secret,query='draw request')
        assert [thread['thread_id'] for thread in result['results']]==['t4','t3','t2','t1']
        assert result['coverage']['reranking']=='applied'
    assert len(calls)==1 and not indexes.entered and reader.open==0
