"""Private facts transport, with real capability checks and orchestration."""
import httpx
import pytest

from gmail_search.gateway.http import create_gateway_app
from gmail_search.gateway.service import RunQueryService
from test_gateway_facts_service import compose, fact


@pytest.mark.asyncio
async def test_facts_route_is_opt_in(tmp_path):
    service, caps, token, *_ = compose(tmp_path, [fact(1, 'A private fact')])
    app = create_gateway_app(RunQueryService(caps, None))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://gateway') as client:
        assert (await client.post('/v1/find-facts', json={'query': 'fact'})).status_code == 404


@pytest.mark.asyncio
async def test_facts_route_returns_owner_results_and_coverage(tmp_path):
    service, caps, token, reader, _ = compose(tmp_path, [fact(1, 'A private fact')])
    app = create_gateway_app(RunQueryService(caps, None), facts=service)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://gateway') as client:
        response = await client.post('/v1/find-facts', headers={'authorization': 'Bearer '+token.secret}, json={'query': 'fact'})
    assert response.status_code == 200, response.text
    assert response.json()['facts'][0]['fact'] == 'A private fact'
    assert 'coverage' in response.json()
    assert reader.owners == ['alice'] and not reader.open
    assert response.headers['cache-control'] == 'private, no-store'
    assert response.headers['x-content-type-options'] == 'nosniff'


@pytest.mark.asyncio
@pytest.mark.parametrize('extra', [{'owner_id': 'bob'}, {'model': 'other'}, {'sql': 'SELECT 1'},
    {'query': ''}, {'exhaustive': 1}, {'k': True}, {'k': 501}])
async def test_facts_route_rejects_selectors_and_invalid_options(tmp_path, extra):
    service, caps, token, reader, embedder = compose(tmp_path, [])
    app = create_gateway_app(RunQueryService(caps, None), facts=service)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://gateway') as client:
        response = await client.post('/v1/find-facts', headers={'authorization': 'Bearer '+token.secret}, json={'query': 'fact', **extra})
    assert response.status_code == 400, response.text
    assert not reader.owners and not embedder.calls


@pytest.mark.asyncio
async def test_facts_auth_before_body_and_bounded_strict_json(tmp_path):
    service, caps, token, reader, embedder = compose(tmp_path, [])
    app = create_gateway_app(RunQueryService(caps, None), facts=service)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://gateway') as client:
        bad = await client.post('/v1/find-facts', headers={'authorization': 'Bearer '+'0'*64}, content='x'*40000)
        assert bad.status_code == 403
        headers = {'authorization': 'Bearer '+token.secret, 'content-type': 'application/json'}
        assert (await client.post('/v1/find-facts', headers=headers, content='x'*40000)).status_code == 413
        assert (await client.post('/v1/find-facts', headers=headers, content='{"query":"a","query":"b"}')).status_code == 400
    assert not reader.owners and not embedder.calls
