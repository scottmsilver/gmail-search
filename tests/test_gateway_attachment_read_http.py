"""Opt-in private metadata, stored-text and inventory HTTP contracts."""
import httpx
import pytest

from gmail_search.gateway.http import create_gateway_app
from gmail_search.gateway.service import RunQueryService
from test_gateway_attachment_read_service import compose
from test_gateway_attachment_reader import Gateway, row


@pytest.mark.asyncio
@pytest.mark.parametrize('path,body,text', [('/v1/attachment/meta', {'attachment_id': 1}, False),
    ('/v1/attachment/text', {'attachment_id': 1}, True),
    ('/v1/attachment/list', {'thread_id': 'thread'}, False)])
async def test_private_read_routes_are_opt_in_and_project_owner_fields(tmp_path, path, body, text):
    gateway = Gateway([row()+('body',)], text=True) if text else Gateway()
    service, caps, token, _ = compose(tmp_path, gateway)
    sql = RunQueryService(caps, None)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_gateway_app(sql)), base_url='http://gateway') as client:
        assert (await client.post(path, json=body)).status_code == 404
    app = create_gateway_app(sql, attachment_reads=service)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://gateway') as client:
        response = await client.post(path, headers={'authorization': 'Bearer '+token.secret}, json=body)
    assert response.status_code == 200, response.text
    assert 'owner_id' not in response.text and 'raw_path' not in response.text
    assert response.json()['cite_ref'] == 'thread'
    assert response.headers['cache-control'] == 'private, no-store'
    if text:
        assert response.json()['extracted_text'] == 'body'


@pytest.mark.asyncio
@pytest.mark.parametrize('path,body', [('/v1/attachment/meta', {'attachment_id': 1, 'owner_id': 'bob'}),
    ('/v1/attachment/meta', {'attachment_id': True}),
    ('/v1/attachment/text', {'attachment_id': 1, 'offset': -1}),
    ('/v1/attachment/text', {'attachment_id': 1, 'limit': 100001}),
    ('/v1/attachment/list', {'thread_id': 'thread', 'after_attachment_id': True}),
    ('/v1/attachment/list', {'thread_id': 'thread', 'raw_path': '/private'})])
async def test_invalid_attachment_read_options_never_query(tmp_path, path, body):
    service, caps, token, gateway = compose(tmp_path)
    app = create_gateway_app(RunQueryService(caps, None), attachment_reads=service)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://gateway') as client:
        response = await client.post(path, headers={'authorization': 'Bearer '+token.secret}, json=body)
    assert response.status_code == 400, response.text
    assert not gateway.calls


@pytest.mark.asyncio
async def test_attachment_read_operation_auth_precedes_body(tmp_path):
    service, caps, token, gateway = compose(tmp_path, operations={'meta'})
    app = create_gateway_app(RunQueryService(caps, None), attachment_reads=service)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://gateway') as client:
        headers = {'authorization': 'Bearer '+token.secret, 'content-type': 'application/json'}
        assert (await client.post('/v1/attachment/text', headers=headers, content='x'*40000)).status_code == 403
        assert (await client.post('/v1/attachment/meta', headers=headers, content='x'*40000)).status_code == 413
        assert (await client.post('/v1/attachment/meta', headers=headers, content='{"attachment_id":1,"attachment_id":2}')).status_code == 400
    assert not gateway.calls
