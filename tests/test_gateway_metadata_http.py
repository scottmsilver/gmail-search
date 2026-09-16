"""Private metadata filtering route, with real run authorization."""
import httpx
import pytest

from gmail_search.gateway.http import create_gateway_app
from gmail_search.gateway.service import RunQueryService
from test_gateway_metadata_service import compose


@pytest.mark.asyncio
async def test_metadata_route_requires_explicit_injection(tmp_path):
    service, caps, token, _ = compose(tmp_path)
    app = create_gateway_app(RunQueryService(caps, None))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://gateway') as client:
        assert (await client.post('/v1/query-emails', json={})).status_code == 404


@pytest.mark.asyncio
async def test_metadata_route_returns_scoped_citations_and_coverage(tmp_path):
    service, caps, token, gateway = compose(tmp_path)
    app = create_gateway_app(RunQueryService(caps, None), metadata=service)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://gateway') as client:
        response = await client.post('/v1/query-emails', headers={'authorization': 'Bearer '+token.secret}, json={'sender': "O'Brien", 'has_attachment': False})
    assert response.status_code == 200, response.text
    assert response.json()['results'][0]['cite_ref'] == 'thread'
    assert response.json()['coverage']['selection_complete']
    assert gateway.calls[0][0] == 'alice'
    assert response.headers['cache-control'] == 'private, no-store'


@pytest.mark.asyncio
@pytest.mark.parametrize('options', [{'owner_id': 'bob'}, {'sql': 'SELECT 1'}, {'index_path': '/private'},
    {'sender': None}, {'limit': True}, {'limit': 101}, {'date_from': 'invalid'}, {'has_attachment': 1}])
async def test_metadata_route_rejects_selectors_and_invalid_options(tmp_path, options):
    service, caps, token, gateway = compose(tmp_path)
    app = create_gateway_app(RunQueryService(caps, None), metadata=service)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://gateway') as client:
        response = await client.post('/v1/query-emails', headers={'authorization': 'Bearer '+token.secret}, json=options)
    assert response.status_code == 400, response.text
    assert not gateway.calls


@pytest.mark.asyncio
async def test_metadata_auth_before_bounded_strict_body(tmp_path):
    service, caps, token, gateway = compose(tmp_path)
    app = create_gateway_app(RunQueryService(caps, None), metadata=service)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://gateway') as client:
        assert (await client.post('/v1/query-emails', headers={'authorization': 'Bearer '+'0'*64}, content='x'*40000)).status_code == 403
        headers = {'authorization': 'Bearer '+token.secret, 'content-type': 'application/json'}
        assert (await client.post('/v1/query-emails', headers=headers, content='x'*40000)).status_code == 413
        assert (await client.post('/v1/query-emails', headers=headers, content='{"limit":1,"limit":2}')).status_code == 400
    assert not gateway.calls
