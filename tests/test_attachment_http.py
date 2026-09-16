"""Composed attachment route: capability → owner DB reader → bytes → parser job."""
import hashlib
import json

import httpx
import psycopg
import pytest

from gmail_search.gateway.attachment_service import RunAttachmentService
from gmail_search.gateway.attachment_source import OwnerAttachmentSource, QueryAttachmentLocator
from gmail_search.gateway.http import create_gateway_app
from test_gateway_database_integration import database as database_fixture
from test_gateway_service import gateway, scoped

database = database_fixture


class SyntheticBackend:
    """Protocol fixture only: real Firecracker qualification is separate."""
    def __init__(self):
        self.active = 0
    def start(self, data, mime_type, options):
        self.active += 1
        backend = self
        class Job:
            stopped = False
            def wait(self):
                return json.dumps({'text': data.decode(), 'pages': [], 'truncated': False}).encode()
            def stop(self):
                if not self.stopped:
                    self.stopped = True
                    backend.active -= 1
        return Job()


@pytest.mark.asyncio
async def test_attachment_http_requires_explicit_service(tmp_path):
    sql, *_ = scoped(tmp_path, None)
    app = create_gateway_app(sql, attachments=None)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://gateway') as client:
        assert (await client.post('/v1/attachment/parse', json={'attachment_id': 1})).status_code == 404


@pytest.mark.asyncio
async def test_composed_attachment_route_keeps_colliding_ids_private(database, tmp_path):
    dsn, owners = database
    root = tmp_path / 'attachments'
    with psycopg.connect(dsn, autocommit=True) as conn:
        for owner in owners:
            path = root / 'owners' / hashlib.sha256(owner.encode()).hexdigest() / 'message' / 'file.pdf'
            path.parent.mkdir(parents=True)
            path.write_bytes(owner.encode())
            conn.execute("INSERT INTO attachments (id,message_id,filename,mime_type,size_bytes,fetch_status,user_id) VALUES (1,'message','file.pdf','application/pdf',%s,'ok',%s)", (len(owner), owner))
    for index, owner in enumerate(owners):
        state = tmp_path / str(index)
        state.mkdir(mode=0o700)
        api = gateway(database)
        sql, _, capabilities, run, _ = scoped(state, api, owner)
        token = capabilities.issue(run.run_id, audience='attachment', operations={'parse'})
        source = OwnerAttachmentSource(root, locate=QueryAttachmentLocator(api, root).locate)
        backend = SyntheticBackend()
        service = RunAttachmentService(capabilities, backend, load=source.load)
        app = create_gateway_app(sql, attachments=service)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://gateway') as client:
            headers = {'authorization': 'Bearer ' + token.secret}
            response = await client.post('/v1/attachment/parse', headers=headers, json={'attachment_id': 1})
            assert response.status_code == 200, response.text
            assert response.json() == {'text': owner, 'pages': [], 'truncated': False}
            assert str(root) not in response.text and 'raw_path' not in response.text
            assert backend.active == 0
            assert response.headers['cache-control'] == 'private, no-store'
            assert (await client.post('/v1/attachment/parse', headers=headers, json={'attachment_id': 1, 'owner_id': owners[1-index]})).status_code == 400
            assert (await client.post('/v1/attachment/parse', headers=headers, json={'attachment_id': 1, 'pages': '1'})).status_code == 400
            capabilities.revoke(token.secret)
            assert (await client.post('/v1/attachment/parse', headers=headers, content=b'x' * 40000)).status_code == 403
