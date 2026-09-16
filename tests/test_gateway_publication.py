"""Revocation while the HTTP transport drains must prevent publication."""
import asyncio

import httpx
import pytest

from gmail_search.gateway.http import create_gateway_app
from gmail_search.gateway.service import RunQueryService
from test_gateway_facts_service import compose, fact
from test_gateway_search_service import _service


@pytest.mark.asyncio
@pytest.mark.parametrize('kind,failure', [('facts','revoke'), ('search','revoke'), ('sql','revoke'), ('attachment_reads','revoke'), ('attachment_reads','deadline'), ('metadata','revoke'), ('metadata','deadline'), ('facts','deadline'), ('search','deadline')])
async def test_http_final_authorization_follows_disconnect_watcher_cleanup(tmp_path, monkeypatch, kind, failure):
    from gmail_search.gateway import retrieval_http
    if kind == 'facts':
        service, caps, token, *_ = compose(tmp_path, [fact(1, 'private fact')])
        route = '/v1/find-facts'
    elif kind == 'attachment_reads':
        from test_gateway_attachment_read_service import compose as attachment_compose
        service, caps, token, _ = attachment_compose(tmp_path)
        route = '/v1/attachment/meta'
    elif kind == 'metadata':
        from test_gateway_metadata_service import compose as metadata_compose
        service, caps, token, _ = metadata_compose(tmp_path)
        route = '/v1/query-emails'
    elif kind == 'sql':
        from gmail_search.gateway.database import QueryResult
        from test_gateway_service import scoped
        class Gateway:
            async def query(self, owner, query):
                return QueryResult(('subject',), (('private fact',),), True)
        service, _, caps, _, token = scoped(tmp_path, Gateway())
        route = '/v1/sql'
    else:
        service, caps, token, *_ = _service(tmp_path)
        route = '/v1/search'
    app = create_gateway_app(service) if kind == 'sql' else create_gateway_app(RunQueryService(caps, None), **{kind: service})
    closing, release = asyncio.Event(), asyncio.Event()
    loop = asyncio.get_running_loop()
    original_time = loop.time

    async def slow_disconnect(request):
        try:
            await asyncio.Event().wait()
        finally:
            closing.set()
            await release.wait()

    monkeypatch.setattr(retrieval_http, '_disconnect', slow_disconnect)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://gateway') as client:
        work = asyncio.create_task(client.post(route, headers={'authorization': 'Bearer '+token.secret}, json={} if kind == 'metadata' else {'attachment_id': 1} if kind == 'attachment_reads' else {'query': 'fact'}))
        try:
            await asyncio.wait_for(closing.wait(), 2)
            if failure == 'revoke':
                caps.revoke(token.secret)
            else:
                monkeypatch.setattr(loop, 'time', lambda: original_time()+31)
            release.set()
            response = await work
            assert response.status_code == (403 if failure == 'revoke' else 504), response.text
            assert 'private fact' not in response.text and 'matches' not in response.text
        finally:
            monkeypatch.setattr(loop, 'time', original_time)
            release.set()
            await asyncio.gather(work, return_exceptions=True)
