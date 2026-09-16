"""Optional closed search route; never selects owner or provider from a request."""
import httpx
import pytest

from gmail_search.gateway.http import create_gateway_app
from gmail_search.gateway.service import RunQueryService
from test_gateway_search_service import _service


@pytest.mark.asyncio
async def test_search_route_requires_explicit_injection(tmp_path):
    service,caps,token,*_=_service(tmp_path)
    app=create_gateway_app(RunQueryService(caps,None))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://gateway') as client:
        assert (await client.post('/v1/search',json={'query':'needle'})).status_code==404


@pytest.mark.asyncio
async def test_search_route_uses_capability_and_sends_private_bounded_json(tmp_path):
    service,caps,token,*_=_service(tmp_path)
    app=create_gateway_app(RunQueryService(caps,None),search=service)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://gateway') as client:
        response=await client.post('/v1/search',headers={'authorization':'Bearer '+token.secret},json={'query':'draw request','detail':'refs'})
        assert response.status_code==200,response.text
        assert response.json()['results'][0]['cite_ref']=='t1'
        assert response.headers['cache-control']=='private, no-store'
        assert response.headers['x-content-type-options']=='nosniff'


@pytest.mark.asyncio
@pytest.mark.parametrize('extra',[{'owner_id':'bob'},{'model':'other'},{'index_path':'/private'},
    {'query':''},{'top_k':True},{'detail':'html'},{'date_to':'invalid'},{'max_matches':101}])
async def test_search_request_rejects_identity_selectors_and_invalid_options(tmp_path,extra):
    service,caps,token,reader,indexes,embedder=_service(tmp_path)
    app=create_gateway_app(RunQueryService(caps,None),search=service)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://gateway') as client:
        response=await client.post('/v1/search',headers={'authorization':'Bearer '+token.secret},json={'query':'draw',**extra})
        assert response.status_code==400,response.text
    assert not reader.owners and not indexes.owners and not embedder.calls


@pytest.mark.asyncio
async def test_search_auth_precedes_body_and_duplicate_keys_are_refused(tmp_path):
    service,caps,token,reader,indexes,embedder=_service(tmp_path)
    app=create_gateway_app(RunQueryService(caps,None),search=service)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://gateway') as client:
        bad=await client.post('/v1/search',headers={'authorization':'Bearer '+'0'*64},content='x'*40000)
        assert bad.status_code==403
        headers={'authorization':'Bearer '+token.secret,'content-type':'application/json'}
        oversized=await client.post('/v1/search',headers=headers,content='x'*40000)
        assert oversized.status_code==413
        duplicate=await client.post('/v1/search',headers=headers,content='{"query":"a","query":"b"}')
        assert duplicate.status_code==400
    assert not reader.owners and not indexes.owners and not embedder.calls
