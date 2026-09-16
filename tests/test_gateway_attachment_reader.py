import asyncio
from dataclasses import asdict

import psycopg
import pytest

from gmail_search.gateway import attachment_reader as ar
from gmail_search.gateway.database import QueryGateway, QueryLimits, QueryResult, ReaderCredential, ReaderRegistry
from gmail_search.gateway.registry import AccessDenied
from test_gateway_database_integration import database as database, reader_dsn


async def active():
    return True


def options(**extra):
    return dict(deadline=asyncio.get_running_loop().time()+3,check_active=active,**extra)


def row(*,owner='alice',aid=1,text_chars=4,filename='file.pdf'):
    return (owner,aid,'message','thread',filename,'application/pdf',123,'ok',text_chars)


class Gateway:
    def __init__(self,rows=None,*,complete=True,text=False):
        self.rows=tuple(rows if rows is not None else [row()])
        self.complete=complete
        self.columns=ar.METADATA_COLUMNS+(('text_page',) if text else ())
        self.calls=[]
    async def query(self,owner,query):
        self.calls.append((owner,query))
        return QueryResult(self.columns,self.rows,self.complete)


@pytest.mark.asyncio
async def test_metadata_is_bound_bounded_and_contains_no_path_or_extraction_claim():
    api=Gateway(); result=await ar.OwnerAttachmentReader(api).describe('alice',1,**options())
    assert result.attachment_id==1 and result.owner_id=='alice' and result.thread_id=='thread'
    assert result.text_chars==4 and result.stored_text_state=='present'
    assert result.extraction_complete is None
    owner,query=api.calls[0]
    assert owner=='alice' and 'a.user_id = m.user_id' in query and 'a.message_id = m.id' in query
    assert 'raw_path' not in query and 'pg_catalog.substr(a.filename, 1, 1025)' in query
    assert 'raw_path' not in asdict(result)


@pytest.mark.asyncio
@pytest.mark.parametrize('chars,text,state,complete',[(None,None,'missing',False),(0,'','empty',True),(4,'body','present',True)])
async def test_missing_and_empty_stored_text_are_distinct(chars,text,state,complete):
    api=Gateway([row(text_chars=chars)+(text,)],text=True)
    page=await ar.OwnerAttachmentReader(api).text_page('alice',1,**options())
    assert page.attachment.stored_text_state==state and page.text==text
    assert page.attachment.extraction_complete is None
    assert page.stored_text_complete is complete and page.page_complete is complete
    assert page.next_offset is None


@pytest.mark.asyncio
async def test_text_substring_page_and_offset_ceiling():
    api=Gateway([row(text_chars=100)+('abcdefghij',)],text=True)
    page=await ar.OwnerAttachmentReader(api).text_page('alice',1,offset=20,limit=10,**options())
    assert page.text=='abcdefghij' and page.next_offset==30 and not page.stored_text_complete
    assert page.page_complete and page.total_chars==100
    assert 'pg_catalog.substr(a.extracted_text, 21, 10)' in api.calls[0][1]
    api.rows=(row(text_chars=2147483647)+('z',),)
    page=await ar.OwnerAttachmentReader(api).text_page('alice',1,offset=2147483646,limit=1,**options())
    assert page.next_offset is None and page.page_complete
    assert '2147483647, 1' in api.calls[-1][1]


@pytest.mark.asyncio
async def test_list_paging_is_stable_and_does_not_claim_unknown_totals():
    api=Gateway([row(aid=2),row(aid=3),row(aid=4)])
    result=await ar.OwnerAttachmentReader(api).list_for_thread('alice','thread',after_attachment_id=1,limit=2,**options())
    assert [r.attachment_id for r in result.items]==[2,3]
    assert result.next_attachment_id==3 and not result.complete and not result.source_complete
    assert not result.pagination_limited
    assert 'a.id > 1' in api.calls[0][1] and 'ORDER BY a.id LIMIT 3' in api.calls[0][1]
    api.rows=(row(aid=4),)
    result=await ar.OwnerAttachmentReader(api).list_for_thread('alice','thread',after_attachment_id=3,**options())
    assert result.source_complete and not result.complete and result.next_attachment_id is None


@pytest.mark.asyncio
async def test_zero_rows_database_cap_is_explicit_unpageable():
    result=await ar.OwnerAttachmentReader(Gateway([],complete=False)).list_for_thread('alice','thread',**options())
    assert not result.complete and not result.source_complete and result.pagination_limited
    assert result.next_attachment_id is None


@pytest.mark.asyncio
@pytest.mark.parametrize('rows,complete',[((),True),((row(owner='bob'),),True),((row(aid=2),),True),
    ((row(),row()),True),((row(),),False),((row(filename='x'*1025),),True),((row(text_chars=-1),),True)])
async def test_missing_incomplete_malformed_or_foreign_metadata_refused(rows,complete):
    api=Gateway(rows,complete=complete)
    with pytest.raises(ar.AttachmentReadUnavailable):
        await ar.OwnerAttachmentReader(api).describe('alice',1,**options())


@pytest.mark.asyncio
@pytest.mark.parametrize('change',[{'offset':2147483647},{'offset':-1},{'limit':100001},{'limit':True}])
async def test_invalid_text_paging_never_queries(change):
    api=Gateway()
    with pytest.raises(ValueError):
        await ar.OwnerAttachmentReader(api).text_page('alice',1,**change,**options())
    assert not api.calls


@pytest.mark.asyncio
async def test_invalid_list_identity_and_cursor_never_queries():
    api=Gateway(); reader=ar.OwnerAttachmentReader(api)
    for thread in ("thread' OR true--",None,''):
        with pytest.raises(ValueError):
            await reader.list_for_thread('alice',thread,**options())
    with pytest.raises(ValueError):
        await reader.list_for_thread('alice','thread',after_attachment_id=True,**options())
    assert not api.calls


@pytest.mark.asyncio
@pytest.mark.parametrize('failure',['cancel','revoke','deadline'])
async def test_cancellation_revocation_and_deadline_drain_query(failure):
    started=asyncio.Event(); closing=asyncio.Event(); gate=asyncio.Event(); closed=asyncio.Event()
    allowed=True
    async def check():
        if not allowed:
            raise AccessDenied()
        return True
    class Blocking:
        async def query(self,*args):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                closing.set(); await gate.wait(); closed.set()
    reader=ar.OwnerAttachmentReader(Blocking())
    deadline=asyncio.get_running_loop().time()+(.08 if failure=='deadline' else 2)
    task=asyncio.create_task(reader.describe('alice',1,deadline=deadline,check_active=check))
    await started.wait()
    if failure=='cancel':task.cancel()
    elif failure=='revoke':allowed=False
    await asyncio.wait_for(closing.wait(),1)
    if failure=='cancel':task.cancel()
    held=not task.done(); gate.set()
    with pytest.raises({'cancel':asyncio.CancelledError,'revoke':AccessDenied,'deadline':TimeoutError}[failure]):
        await task
    assert held and closed.is_set()


@pytest.mark.asyncio
async def test_late_revocation_during_query_cleanup_does_not_publish():
    allowed=True
    async def check():
        if not allowed:raise AccessDenied()
        return True
    class Revoking(Gateway):
        async def query(self,*args):
            nonlocal allowed
            result=await super().query(*args)
            allowed=False
            return result
    with pytest.raises(AccessDenied):
        await ar.OwnerAttachmentReader(Revoking()).describe('alice',1,deadline=asyncio.get_running_loop().time()+1,check_active=check)


def real_reader(database,*,max_bytes=2000000):
    dsn,owners=database
    gateway=QueryGateway(ReaderRegistry({o:ReaderCredential(o,reader_dsn(dsn,o)) for o in owners},is_active=lambda _:True),
        limits=QueryLimits(max_bytes=max_bytes,deadline_seconds=4,lock_timeout_ms=3000))
    return ar.OwnerAttachmentReader(gateway)


def insert(database):
    dsn,owners=database
    with psycopg.connect(dsn,autocommit=True) as conn:
        for owner in owners:
            conn.execute('INSERT INTO messages(id,thread_id,user_id) VALUES(%s,%s,%s)',('shared-message','shared-thread',owner))
            for aid,text in ((1,None),(2,''),(3,owner+' stored words')):
                conn.execute('INSERT INTO attachments(id,message_id,filename,mime_type,size_bytes,fetch_status,extracted_text,user_id) VALUES(%s,%s,%s,%s,%s,%s,%s,%s)',
                    (aid,'shared-message',owner+'.unknown','application/x-unknown',0,'unfetched',text,owner))


@pytest.mark.asyncio
async def test_real_colliding_owner_metadata_and_stored_text(database):
    insert(database); reader=real_reader(database)
    alice,bob=database[1]
    for owner,foreign in ((alice,bob),(bob,alice)):
        meta=await reader.describe(owner,1,**options())
        assert meta.filename==owner+'.unknown' and meta.size_bytes==0 and meta.stored_text_state=='missing'
        page=await reader.text_page(owner,3,offset=2,limit=6,**options())
        assert page.text==(owner+' stored words')[2:8] and page.next_offset==8
        listing=await reader.list_for_thread(owner,'shared-thread',limit=2,**options())
        assert [m.attachment_id for m in listing.items]==[1,2]
        assert listing.next_attachment_id==2 and foreign not in str(listing)
        final=await reader.list_for_thread(owner,'shared-thread',after_attachment_id=2,**options())
        assert [m.attachment_id for m in final.items]==[3] and final.source_complete


@pytest.mark.asyncio
async def test_real_large_text_is_sliced_before_transfer(database):
    insert(database); dsn,(alice,bob)=database
    offset=2000000
    with psycopg.connect(dsn,autocommit=True) as conn:
        for owner in (alice,bob):
            conn.execute('UPDATE attachments SET extracted_text=%s WHERE id=3 AND user_id=%s',('x'*offset+owner+'-tail',owner))
    page=await real_reader(database,max_bytes=700).text_page(alice,3,offset=offset,limit=10,**options())
    assert page.text==alice[:10] and page.total_chars==offset+len(alice)+5 and page.next_offset==offset+10
    assert bob not in str(page)


@pytest.mark.asyncio
async def test_real_missing_owner_join_and_reader_has_no_raw_path_grant(database):
    insert(database); dsn,(alice,bob)=database
    with psycopg.connect(dsn,autocommit=True) as conn:
        conn.execute('ALTER TABLE attachments ADD COLUMN raw_path text')
        conn.execute('INSERT INTO attachments(id,message_id,filename,user_id) VALUES(9,%s,%s,%s)',(bob,'orphan',alice))
    reader=real_reader(database)
    with pytest.raises(ar.AttachmentReadUnavailable):
        await reader.describe(alice,9,**options())
    with psycopg.connect(reader_dsn(dsn,alice)) as conn:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            conn.execute('SELECT raw_path FROM public.attachments')


@pytest.mark.asyncio
@pytest.mark.parametrize('chars,text',[(None,''),(0,None),(20,'short'),(4,'oversize')])
async def test_inconsistent_database_text_page_refused(chars,text):
    api=Gateway([row(text_chars=chars)+(text,)],text=True)
    with pytest.raises(ar.AttachmentReadUnavailable):
        await ar.OwnerAttachmentReader(api).text_page('alice',1,**options())


@pytest.mark.asyncio
async def test_final_authorization_that_crosses_deadline_does_not_publish():
    api=Gateway(); reached=asyncio.Event(); gate=asyncio.Event()
    async def check():
        if api.calls:
            reached.set()
            await gate.wait()
        return True
    loop=asyncio.get_running_loop(); deadline=loop.time()+.08
    task=asyncio.create_task(ar.OwnerAttachmentReader(api).describe('alice',1,deadline=deadline,check_active=check))
    await asyncio.wait_for(reached.wait(),1)
    await asyncio.sleep(max(0,deadline-loop.time())+.01)
    gate.set()
    with pytest.raises(TimeoutError):
        await task


@pytest.mark.asyncio
async def test_real_maximum_accepted_offset_and_empty_vs_missing(database):
    insert(database); owner=database[1][0]; reader=real_reader(database)
    empty=await reader.text_page(owner,2,**options())
    missing=await reader.text_page(owner,1,**options())
    assert empty.text=='' and empty.stored_text_complete and empty.attachment.extraction_complete is None
    assert missing.text is None and not missing.stored_text_complete
    page=await reader.text_page(owner,3,offset=2147483646,limit=100000,**options())
    assert page.text=='' and page.next_offset is None and page.page_complete
