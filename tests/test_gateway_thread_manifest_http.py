"""Actual owner readers through thread HTTP attachment pagination."""
import httpx
import psycopg
import pytest

from gmail_search.gateway.attachment_reader import OwnerAttachmentReader
from gmail_search.gateway.http import create_gateway_app
from gmail_search.gateway.retrieval import RunRetrievalService
from test_gateway_database_integration import database as database_fixture
from test_gateway_retrieval import retrieval
from test_gateway_service import gateway

database = database_fixture


@pytest.mark.asyncio
@pytest.mark.parametrize('owner_index', [0, 1])
async def test_thread_http_manifest_cursor_preserves_owner_binding(database, tmp_path, owner_index):
    dsn, owners = database
    with psycopg.connect(dsn, autocommit=True) as conn:
        for owner in owners:
            conn.execute('INSERT INTO messages(id,thread_id,subject,body_text,date,user_id) VALUES(%s,%s,%s,%s,%s,%s)',
                         ('manifest-message', 'manifest-thread', owner, owner, '2026-09-15', owner))
            for aid in (2, 3):
                conn.execute('INSERT INTO attachments(id,message_id,filename,mime_type,size_bytes,fetch_status,user_id) VALUES(%s,%s,%s,%s,0,%s,%s)',
                             (aid, 'manifest-message', owner+str(aid)+'.txt', 'text/plain', 'ok', owner))
    api = gateway(database)
    _, _, caps, _, token, sql = retrieval(tmp_path, api, owners[owner_index])
    service = RunRetrievalService(caps, api, attachment_reader=OwnerAttachmentReader(api))
    app = create_gateway_app(sql, retrieval=service)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://gateway') as client:
        response = await client.post('/v1/thread', headers={'authorization': 'Bearer '+token.secret},
                                     json={'thread_id': 'manifest-thread', 'attachment_after_id': 1, 'attachment_limit': 1})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data['messages'][0]['attachments'][0]['id'] == 2
    assert not data['messages'][0]['attachments_complete']
    assert data['attachment_inventory']['next_attachment_id'] == 2
    assert data['attachment_inventory']['same_snapshot_as_messages'] is False
    assert owners[1-owner_index] not in response.text
    assert 'owner_id' not in response.text and 'raw_path' not in response.text
