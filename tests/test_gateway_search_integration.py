"""Actual partitioned PG reader + strict native ScaNN + run capability.

All mail, vectors and provider responses are synthetic. Each test creates and
removes a disposable database; no production model/index or provider is used.
"""
import asyncio

import pytest
from psycopg.conninfo import make_conninfo

from gmail_search.gateway.capabilities import Capabilities
from gmail_search.gateway.data_admission import DataAdmission
from gmail_search.gateway.database import QueryGateway, QueryLimits, ReaderCredential, ReaderRegistry, reader_role
from gmail_search.gateway.registry import Registry
from gmail_search.gateway.search_index import IndexBinding, OwnerIndexRegistry, load_scann_index
from gmail_search.gateway.search_reader import SearchProfile, SearchReader, SearchRegistry
from gmail_search.gateway.search_service import OwnerSearchContext, RunSearchService
from test_gateway_search_reader import search_database as search_database, provisioned
from test_gateway_search_index import build_native_index, seal_generation


async def compose(db,tmp_path,owner_index):
    owner=db.owners[owner_index]
    foreign=db.owners[1-owner_index]
    credential=provisioned(db,owner)
    admission=DataAdmission(global_concurrency=1,owner_concurrency=1)
    reader=SearchReader(SearchRegistry({owner:credential},is_active=lambda _:True),
        profile=SearchProfile('fixture','fixture+v1',2),admission=admission)
    own_id=db.conn.execute('SELECT id FROM public.embeddings WHERE user_id=%s',(owner,)).fetchone()[0]
    foreign_id=db.conn.execute('SELECT id FROM public.embeddings WHERE user_id=%s',(foreign,)).fetchone()[0]
    path=tmp_path/'index'
    # Deliberately polluted candidate index: foreign ID has highest similarity.
    # The owner reader must reject it before normalization and publication.
    build_native_index(path,[foreign_id,own_id],dimensions=2)
    seal_generation(path)
    indexes=OwnerIndexRegistry()
    await indexes.publish(load_scann_index(IndexBinding(owner,'generation','fixture',2,'synthetic-source'),path))
    registry=Registry(tmp_path/'registry.sqlite',is_active=lambda _:True)
    caps=Capabilities(registry)
    run=registry.start_run(owner,'conversation',request_key='search',writer=False)
    token=caps.issue(run.run_id,audience='retrieval',operations={'search'})
    class Embedder:
        model='fixture'
        dimensions=2
        async def embed(self,lease,text,*,deadline,check_active):
            assert lease.owner_id==owner and not admission.active
            await check_active()
            return [1.,0.]
    service=RunSearchService(caps,reader,indexes,Embedder(),owners={owner:OwnerSearchContext(owner,(owner+'@example.test',))},reranker=None)
    return service,indexes,admission,token


@pytest.mark.asyncio
@pytest.mark.parametrize('owner_index',[0,1])
async def test_real_search_isolates_colliding_mail_and_rejects_foreign_vector(search_database,tmp_path,owner_index):
    db=search_database
    service,indexes,admission,token=await compose(db,tmp_path,owner_index)
    word,foreign=('needle','foreignonly') if owner_index==0 else ('foreignonly','needle')
    try:
        result=await service.search(token.secret,query=word,detail='full')
        assert len(result['results'])==1
        thread=result['results'][0]
        assert thread['thread_id']=='thread' and thread['subject']==word
        assert thread['similarity']==0
        assert thread['matches'][0]['body']==word+' body'
        assert foreign not in str(result['results'])
        assert 'unavailable_embedding' in result['coverage']['reasons']
        assert not admission.active
    finally:
        await indexes.aclose()


@pytest.mark.asyncio
async def test_real_search_and_sql_use_same_admission(search_database,tmp_path):
    db=search_database;owner=db.owners[0]
    service,indexes,admission,token=await compose(db,tmp_path,0)
    entered,release=asyncio.Event(),asyncio.Event()
    lexical=service._lexical
    async def gated(*args,**kwargs):
        entered.set()
        await release.wait()
        return await lexical(*args,**kwargs)
    service._lexical=gated
    # This SQL login is never opened: admission must refuse before connection.
    sql=QueryGateway(ReaderRegistry({owner:ReaderCredential(owner,make_conninfo(db.dsn,user=reader_role(owner)))},
        is_active=lambda _:True),limits=QueryLimits(global_concurrency=1,owner_concurrency=1),admission=admission)
    work=asyncio.create_task(service.search(token.secret,query='needle'))
    try:
        await asyncio.wait_for(entered.wait(),10)
        assert admission.active=={owner:1}
        with pytest.raises(RuntimeError,match='capacity'):
            await sql.query(owner,'SELECT subject FROM messages')
        release.set()
        assert (await work)['results']
        assert not sql.active_queries and not admission.active
    finally:
        release.set()
        await asyncio.gather(work,return_exceptions=True)
        await indexes.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('owner_index', [0, 1])
async def test_private_search_and_facts_routes_use_partitioned_owner_reader(search_database, tmp_path, owner_index):
    import httpx
    from gmail_search.gateway.facts_service import FactsOwnerContext, RunFactsService
    from gmail_search.gateway.http import create_gateway_app
    from gmail_search.gateway.service import RunQueryService
    search, indexes, admission, old_token = await compose(search_database, tmp_path, owner_index)
    owner = search_database.owners[owner_index]
    caps = search.capabilities
    lease = caps.authorize(old_token.secret, audience='retrieval', operation='search')
    token = caps.issue(lease.run_id, audience='retrieval', operations={'search', 'facts.find'})
    facts = RunFactsService(caps, search.reader, search.embedder,
                           owners={owner: FactsOwnerContext(owner, owner+'@example.test')})
    app = create_gateway_app(RunQueryService(caps, None), search=search, facts=facts)
    word, foreign = ('needle', 'foreignonly') if owner_index == 0 else ('foreignonly', 'needle')
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://gateway') as client:
            headers = {'authorization': 'Bearer '+token.secret}
            searched = await client.post('/v1/search', headers=headers, json={'query': word, 'detail': 'full'})
            found = await client.post('/v1/find-facts', headers=headers, json={'query': word})
        assert searched.status_code == found.status_code == 200
        assert searched.json()['results'][0]['subject'] == word
        assert word in found.json()['facts'][0]['fact']
        assert foreign not in searched.text and foreign not in found.text
        assert searched.headers['cache-control'] == found.headers['cache-control'] == 'private, no-store'
        assert not admission.active
    finally:
        await indexes.aclose()
