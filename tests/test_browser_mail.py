import pytest
import asyncio
import hashlib
from types import SimpleNamespace
from gmail_search.gateway.database import QueryGateway, ReaderCredential, ReaderRegistry
from test_gateway_database_integration import database as database_fixture, reader_dsn

database = database_fixture


@pytest.mark.asyncio
async def test_browser_thread_and_prefix_are_owner_scoped(database,tmp_path):
    import psycopg
    from gmail_search.gateway.browser_mail import BrowserMail
    from gmail_search.gateway.attachment_reader import OwnerAttachmentReader
    from gmail_search.gateway.attachment_source import OwnerAttachmentSource,QueryAttachmentLocator
    dsn, (alice,bob) = database
    root=tmp_path/'attachments'
    with psycopg.connect(dsn,autocommit=True) as db:
        for owner in (alice,bob):
            db.execute('INSERT INTO messages (id,thread_id,body_text,subject,date,user_id) VALUES (%s,%s,%s,%s,%s,%s)',
                ('same-id','abcde12345',owner+' private text',owner+' subject','2026-09-15',owner))
            data=(owner+' opaque bytes').encode()
            db.execute('INSERT INTO attachments (id,message_id,filename,mime_type,size_bytes,fetch_status,user_id) VALUES (%s,%s,%s,%s,%s,%s,%s)',
                (1,'same-id',owner+'.pdf','application/pdf',len(data),'ok',owner))
            path=root/'owners'/hashlib.sha256(owner.encode()).hexdigest()/'same-id'/(owner+'.pdf')
            path.parent.mkdir(parents=True);path.write_bytes(data)
    gateway=QueryGateway(ReaderRegistry({owner:ReaderCredential(owner,reader_dsn(dsn,owner)) for owner in (alice,bob)},is_active=lambda owner:True))
    reader=OwnerAttachmentReader(gateway);locator=QueryAttachmentLocator(gateway,root)
    service=BrowserMail(gateway,attachment_reader=reader,
        attachment_source=OwnerAttachmentSource(root,locate=locator.locate_raw))
    async def active():return True
    for owner,other in ((alice,bob),(bob,alice)):
        thread=await service.thread(owner,'abcde12345')
        assert thread['messages'][0]['body_text']==owner+' private text'
        assert other+' private' not in str(thread)
        assert thread['messages'][0]['attachments'][0]['filename']==owner+'.pdf'
        assert thread['messages'][0]['from_addr']==''
        assert (await service.lookup(owner,'abcde'))['thread_id']=='abcde12345'
        meta=await service.attachment_metadata(owner,1,deadline=asyncio.get_running_loop().time()+2,check_active=active)
        assert meta['filename']==owner+'.pdf' and other not in str(meta)
        item=await service.attachment_download(owner,1,deadline=asyncio.get_running_loop().time()+2,check_active=active)
        assert item.data==(owner+' opaque bytes').encode() and other.encode() not in item.data
    with pytest.raises(ValueError):await service.thread(alice,"'; SELECT 1")
    with pytest.raises(ValueError):await service.lookup(alice,'abc%')
    assert not gateway.active_queries


@pytest.mark.asyncio
async def test_cancelled_browser_attachment_read_drains_source_cleanup():
    from gmail_search.gateway.browser_mail import BrowserMail
    from gmail_search.gateway.attachment_reader import AttachmentMetadata
    started=asyncio.Event();closing=asyncio.Event();release=asyncio.Event();closed=asyncio.Event()
    class Registry:
        def credential(self,owner):return object()
    class Reader:
        gateway=None
        async def describe(self,owner,attachment_id,**control):
            return AttachmentMetadata(owner,attachment_id,'message','thread','file.pdf','application/pdf',3,'ok',None,'missing')
    class Source:
        async def load_raw(self,owner,attachment_id):
            started.set()
            try:await asyncio.Event().wait()
            finally:
                closing.set();await release.wait();closed.set()
    gateway=SimpleNamespace(registry=Registry());reader=Reader();reader.gateway=gateway
    service=BrowserMail(gateway,attachment_reader=reader,attachment_source=Source())
    async def active():return True
    task=asyncio.create_task(service.attachment_download('alice',1,
        deadline=asyncio.get_running_loop().time()+2,check_active=active))
    await started.wait();task.cancel();await closing.wait()
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):await task
    assert closed.is_set()
