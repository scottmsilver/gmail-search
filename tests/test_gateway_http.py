import asyncio

import httpx
import pytest

from gmail_search.gateway.http import create_gateway_app
from test_gateway_service import scoped, gateway
from test_gateway_database_integration import database as database_fixture

database = database_fixture


@pytest.mark.asyncio
async def test_http_gateway_derives_owner_and_refuses_override(database, tmp_path):
    _, (alice, bob) = database
    service, _, capabilities, _, token = scoped(tmp_path, gateway(database), alice)
    app = create_gateway_app(service)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://gateway') as client:
        headers = {'authorization': 'Bearer ' + token.secret}
        response = await client.post('/v1/sql', headers=headers, json={'query': 'SELECT user_id FROM messages'})
        assert response.status_code == 200
        assert response.json()['rows'] == [[alice]]
        assert response.headers['cache-control'] == 'private, no-store'
        assert (await client.post('/v1/sql', headers=headers, json={'query': 'SELECT user_id FROM messages', 'owner_id': bob})).status_code == 400
        assert (await client.get('/v1/schema', headers=headers)).status_code == 200
        capabilities.revoke(token.secret)
        assert (await client.get('/v1/schema', headers=headers)).status_code == 403
        assert (await client.post('/v1/sql', headers=headers, json={'query': 'SELECT 1'})).status_code == 403


@pytest.mark.asyncio
async def test_http_auth_precedes_body_and_no_admin_routes(tmp_path):
    service, _, _, _, token = scoped(tmp_path, None)
    app = create_gateway_app(service)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://gateway') as client:
        assert (await client.post('/v1/sql', content=b'x' * 40000)).status_code == 401
        assert (await client.post('/v1/sql', headers={'authorization': 'Bearer ' + token.secret}, content=b'x' * 40000)).status_code == 413
        for headers in (
            [('authorization', 'Bearer '+token.secret), ('authorization', 'Bearer '+token.secret)],
            {'authorization': 'Bearer '+token.secret, 'x-user-id': 'alice'},
            {'cookie': 'session='+token.secret},
        ):
            assert (await client.post('/v1/sql', headers=headers, json={'query':'SELECT 1'})).status_code in (400,401)
        for path in ('/docs','/openapi.json','/v1/admin','/v1/runs','/api/sql'):
            assert (await client.get(path)).status_code == 404


@pytest.mark.asyncio
async def test_http_rejects_nested_duplicate_or_unsupported_json(tmp_path):
    service, _, _, _, token = scoped(tmp_path, None)
    app = create_gateway_app(service)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://gateway') as client:
        headers={'authorization':'Bearer '+token.secret,'content-type':'application/json'}
        for body in ('{"query":"SELECT 1","query":"SELECT 2"}', '{"query":null}', '{"query":{}}', '[]', 'NaN', '['*2000+']'*2000):
            response = await client.post('/v1/sql', headers=headers, content=body)
            assert response.status_code == 400
            assert token.secret not in response.text


@pytest.mark.asyncio
async def test_http_upload_deadline_content_type_and_mid_upload_revocation(tmp_path):
    service, _, capabilities, _, token = scoped(tmp_path, None)
    app = create_gateway_app(service)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://gateway') as client:
        headers={'authorization':'Bearer '+token.secret,'content-type':'application/json'}
        assert (await client.post('/v1/sql', headers={**headers,'content-type':'text/plain'}, content='{}')).status_code == 400

        async def slow_body():
            yield b'{'
            await asyncio.sleep(10)
        response = await client.post('/v1/sql', headers=headers, content=slow_body())
        assert response.status_code == 408

        async def revoke_during_body():
            yield b'{"query":'
            capabilities.revoke(token.secret)
            yield b'"SELECT 1"}'
        response = await client.post('/v1/sql', headers=headers, content=revoke_during_body())
        assert response.status_code == 403
        assert token.secret not in response.text


@pytest.mark.asyncio
async def test_sql_disconnect_cancels_query_before_response(tmp_path, monkeypatch):
    from gmail_search.gateway import retrieval_http
    entered, disconnected, closed = (asyncio.Event() for _ in range(3))
    class BlockingGateway:
        async def query(self, owner, query):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                closed.set()
    async def lost_connection(request):
        await entered.wait()
        disconnected.set()
    monkeypatch.setattr(retrieval_http, '_disconnect', lost_connection)
    service, _, _, _, token = scoped(tmp_path, BlockingGateway())
    app = create_gateway_app(service)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://gateway') as client:
        work = asyncio.create_task(client.post('/v1/sql', headers={'authorization': 'Bearer '+token.secret}, json={'query': 'SELECT 1'}))
        try:
            await asyncio.wait_for(disconnected.wait(), .5)
            response = await asyncio.wait_for(work, 1)
            assert response.status_code == 499
            assert closed.is_set()
        finally:
            work.cancel()
            await asyncio.gather(work, return_exceptions=True)
