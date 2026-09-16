"""Actual owner DB lookup and opaque files through the raw gateway factory."""
import hashlib

import psycopg
import pytest

from gmail_search.gateway.attachment_raw_service import RunRawAttachmentService
from gmail_search.gateway.attachment_source import OwnerAttachmentSource, QueryAttachmentLocator
from gmail_search.gateway.capabilities import Capabilities
from gmail_search.gateway.data_admission import DataAdmission
from gmail_search.gateway.http import create_gateway_app
from gmail_search.gateway.registry import Registry
from gmail_search.gateway.service import RunQueryService
from test_gateway_attachment_raw_http import Connection, packet
from test_gateway_database_integration import database as database_fixture
from test_gateway_service import gateway

database = database_fixture


@pytest.mark.asyncio
@pytest.mark.parametrize('owner_index', [0, 1])
async def test_raw_http_same_ids_and_names_read_only_capability_owner(database, tmp_path, owner_index):
    dsn, owners = database
    root = tmp_path/'attachments'
    data = {owner: ('synthetic raw bytes '+owner).encode() for owner in owners}
    with psycopg.connect(dsn, autocommit=True) as conn:
        for owner in owners:
            path = root/'owners'/hashlib.sha256(owner.encode()).hexdigest()/'same-message'/'same file.bin'
            path.parent.mkdir(parents=True)
            path.write_bytes(data[owner])
            conn.execute('INSERT INTO attachments(id,message_id,filename,mime_type,size_bytes,fetch_status,user_id) VALUES(1,%s,%s,%s,%s,%s,%s)',
                         ('same-message', 'same file.bin', 'application/octet-stream', len(data[owner]), 'ok', owner))
    locator = QueryAttachmentLocator(gateway(database), root)
    source = OwnerAttachmentSource(root, locate=locator.locate_raw)
    registry = Registry(tmp_path/'raw.sqlite', is_active=lambda _: True)
    caps = Capabilities(registry)
    owner = owners[owner_index]
    run = registry.start_run(owner, 'conversation', request_key='raw-integration')
    token = caps.issue(run.run_id, audience='attachment', operations={'raw'})
    admission = DataAdmission(global_concurrency=2, owner_concurrency=1)
    raw = RunRawAttachmentService(caps, source, admission=admission)
    app = create_gateway_app(RunQueryService(caps, None), raw_attachments=raw)
    conn = Connection(token.secret)
    await app(conn.scope, conn.receive, conn.send)
    _, header, payload = packet(conn)
    assert payload == data[owner] and data[owners[1-owner_index]] not in payload
    assert header['sha256'] == hashlib.sha256(data[owner]).hexdigest()
    assert 'owner_id' not in header and 'filename' not in header and 'path' not in header
    assert not admission.active and not locator.gateway.active_queries
