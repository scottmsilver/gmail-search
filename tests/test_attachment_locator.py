"""New-layout attachment bytes selected using the existing restricted reader."""
import hashlib

import psycopg
import pytest

from gmail_search.gateway.attachment_sandbox import AttachmentDenied
from test_gateway_database_integration import database as database_fixture
from test_gateway_service import gateway

database = database_fixture


@pytest.mark.asyncio
async def test_same_attachment_ids_cannot_select_other_owners_bytes(database, tmp_path):
    from gmail_search.gateway.attachment_source import OwnerAttachmentSource, QueryAttachmentLocator
    dsn, owners = database
    root = tmp_path / 'attachments'
    with psycopg.connect(dsn, autocommit=True) as conn:
        for owner in owners:
            directory = root / 'owners' / hashlib.sha256(owner.encode()).hexdigest() / 'same-message'
            directory.mkdir(parents=True)
            (directory / 'same.pdf').write_bytes(owner.encode())
            conn.execute('INSERT INTO attachments (id,message_id,filename,mime_type,size_bytes,fetch_status,user_id) VALUES (1,%s,%s,%s,%s,%s,%s)',
                         ('same-message', 'same.pdf', 'application/pdf', len(owner), 'ok', owner))
    locator = QueryAttachmentLocator(gateway(database), root)
    source = OwnerAttachmentSource(root, locate=locator.locate)
    for owner in owners:
        assert (await source.load(owner, 1)).data == owner.encode()
        with pytest.raises(AttachmentDenied):
            await source.load(owner, 999)


@pytest.mark.asyncio
@pytest.mark.parametrize('field,value', [('filename', '../secret.pdf'), ('message_id', '../bob'), ('fetch_status', 'fetch_failed'), ('size_bytes', 999999999)])
async def test_locator_rejects_unsafe_or_unavailable_metadata(database, tmp_path, field, value):
    from gmail_search.gateway.attachment_source import QueryAttachmentLocator
    from psycopg import sql
    dsn, (alice, _) = database
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute("INSERT INTO attachments (id,message_id,filename,mime_type,size_bytes,fetch_status,user_id) VALUES (1,'message','file.pdf','application/pdf',10,'ok',%s)", (alice,))
        conn.execute(sql.SQL('UPDATE attachments SET {}=%s WHERE user_id=%s').format(sql.Identifier(field)), (value, alice))
    locator = QueryAttachmentLocator(gateway(database), tmp_path / 'attachments')
    with pytest.raises(AttachmentDenied):
        await locator.locate(alice, 1)


@pytest.mark.asyncio
@pytest.mark.parametrize('mime', ['text/plain', None])
async def test_generic_empty_raw_locator_is_owner_bound_and_not_parser_eligible(database, tmp_path, mime):
    from gmail_search.gateway.attachment_source import OwnerAttachmentSource, QueryAttachmentLocator
    dsn, owners = database
    root = tmp_path/'attachments'
    with psycopg.connect(dsn, autocommit=True) as conn:
        for owner in owners:
            path = root/'owners'/hashlib.sha256(owner.encode()).hexdigest()/'same-message'/'file.txt'
            path.parent.mkdir(parents=True)
            path.write_bytes(b'' if owner == owners[0] else b'foreign')
            conn.execute('INSERT INTO attachments(id,message_id,filename,mime_type,size_bytes,fetch_status,user_id) VALUES(1,%s,%s,%s,%s,%s,%s)',
                         ('same-message', 'file.txt', mime, path.stat().st_size, 'ok', owner))
    locator = QueryAttachmentLocator(gateway(database), root)
    assert callable(getattr(locator, 'locate_raw', None)), 'raw location must not depend on parser eligibility'
    source = OwnerAttachmentSource(root, locate=locator.locate_raw)
    own = await source.load_raw(owners[0], 1)
    assert own.data == b''
    assert own.mime_type == (mime or 'application/octet-stream')
    assert (await source.load_raw(owners[1], 1)).data == b'foreign'
    with pytest.raises(AttachmentDenied):
        await locator.locate(owners[0], 1)
