"""Typed owner attachment metadata and stored text through the analytical reader.

No filesystem, raw-path privilege, parser, provider or extraction write. Owner
identity and the fresh authorization callback come from a trusted run service.
Every SQL join includes both owner and message identity. Text availability does
not establish that extraction finished; that state is absent from this schema.
"""
import asyncio
from dataclasses import dataclass
import math
import re

from .database import QueryResult
from .registry import AccessDenied
from .retrieval import _drain

_ID=re.compile(r'[A-Za-z0-9_-]{1,256}\Z',re.ASCII)
_MAX_ID=9223372036854775807
_MAX_OFFSET=2147483646
METADATA_COLUMNS=('owner_id','attachment_id','message_id','thread_id','filename',
                  'mime_type','size_bytes','fetch_status','text_chars')
_SELECT=('pg_catalog.substr(a.user_id, 1, 2049) AS owner_id, a.id AS attachment_id, '
         'pg_catalog.substr(a.message_id, 1, 257) AS message_id, '
         'pg_catalog.substr(m.thread_id, 1, 257) AS thread_id, '
         'pg_catalog.substr(a.filename, 1, 1025) AS filename, '
         'pg_catalog.substr(a.mime_type, 1, 257) AS mime_type, a.size_bytes, '
         'pg_catalog.substr(a.fetch_status, 1, 129) AS fetch_status, '
         'length(a.extracted_text) AS text_chars')
_FROM=' FROM attachments a JOIN messages m ON a.user_id = m.user_id AND a.message_id = m.id '


class AttachmentReadUnavailable(RuntimeError):
    def __init__(self):
        super().__init__('Attachment data is unavailable.')


@dataclass(frozen=True)
class AttachmentMetadata:
    owner_id: str
    attachment_id: int
    message_id: str
    thread_id: str
    filename: str | None
    mime_type: str | None
    size_bytes: int | None
    fetch_status: str | None
    text_chars: int | None
    stored_text_state: str
    extraction_complete: None = None


@dataclass(frozen=True)
class AttachmentTextPage:
    attachment: AttachmentMetadata
    text: str | None
    offset: int
    limit: int
    total_chars: int | None
    next_offset: int | None
    page_complete: bool
    stored_text_complete: bool
    pagination_limited: bool


@dataclass(frozen=True)
class AttachmentMetadataPage:
    items: tuple[AttachmentMetadata,...]
    after_attachment_id: int
    limit: int
    source_complete: bool
    complete: bool
    next_attachment_id: int | None
    pagination_limited: bool


def _safe(value,limit,*,optional=False):
    if optional and value is None:
        return None
    if (type(value) is not str or len(value)>limit
            or any(ord(c)<32 or ord(c)==127 for c in value)):
        raise ValueError('Invalid attachment text field')
    try:
        if len(value.encode('utf-8'))>limit:
            raise ValueError('Invalid attachment text field')
    except UnicodeError:
        raise ValueError('Invalid attachment text field') from None
    return value


def _owner(owner_id):
    if not _safe(owner_id,2048):
        raise ValueError('Invalid attachment owner')


def _id(value,*,zero=False):
    if type(value) is not int or not (0 if zero else 1)<=value<=_MAX_ID:
        raise ValueError('Invalid attachment identifier')


def _metadata(row,owner_id):
    try:
        if type(row) is not tuple or len(row)!=len(METADATA_COLUMNS):
            raise ValueError()
        owner,aid,message,thread,filename,mime,size,status,chars=row
        if owner!=owner_id:
            raise ValueError()
        _id(aid)
        if type(message) is not str or not _ID.fullmatch(message) or type(thread) is not str or not _ID.fullmatch(thread):
            raise ValueError()
        _safe(filename,1024,optional=True)
        _safe(mime,256,optional=True)
        _safe(status,128,optional=True)
        if size is not None and (type(size) is not int or not 0<=size<=_MAX_ID):
            raise ValueError()
        if chars is not None and (type(chars) is not int or not 0<=chars<=2147483647):
            raise ValueError()
        state='missing' if chars is None else 'empty' if chars==0 else 'present'
        return AttachmentMetadata(owner,aid,message,thread,filename,mime,size,status,chars,state)
    except (ValueError,TypeError,UnicodeError):
        raise AttachmentReadUnavailable() from None


def _result(value,columns,max_rows):
    if (type(value) is not QueryResult or value.columns!=columns or type(value.rows) is not tuple
            or len(value.rows)>max_rows or type(value.complete) is not bool
            or any(type(row) is not tuple or len(row)!=len(columns) for row in value.rows)):
        raise AttachmentReadUnavailable()
    return value


class OwnerAttachmentReader:
    def __init__(self,gateway):
        self.gateway=gateway

    async def _read(self,owner,query,decode,*,deadline,check_active):
        _owner(owner)
        if type(deadline) not in (int,float) or not math.isfinite(deadline) or not callable(check_active):
            raise ValueError('Invalid attachment read control')
        async def check():
            if asyncio.get_running_loop().time()>=deadline:
                raise TimeoutError()
            if await check_active() is not True:
                raise AccessDenied()
            if asyncio.get_running_loop().time()>=deadline:
                raise TimeoutError()
        async def watching():
            while True:
                await asyncio.sleep(.05)
                await check()
        work=watch=None
        try:
            try:
                async with asyncio.timeout_at(deadline):
                    await check()
                    work=asyncio.create_task(self.gateway.query(owner,query))
                    watch=asyncio.create_task(watching())
                    done,_=await asyncio.wait((work,watch),return_when=asyncio.FIRST_COMPLETED)
                    if watch in done:
                        await watch
                    result=decode(await work)
            finally:
                tasks=[task for task in (work,watch) if task is not None]
                for task in tasks:
                    if not task.done() and not task.cancelling():
                        task.cancel()
                await _drain(asyncio.gather(*tasks,return_exceptions=True))
            # Decoding and all cleanup precede fresh authorization. Nothing
            # blocks between the final capability/deadline check and return.
            await check()
            return result
        except (AccessDenied,TimeoutError,asyncio.CancelledError):
            raise
        except Exception:
            raise AttachmentReadUnavailable() from None

    async def describe(self,owner_id,attachment_id,*,deadline,check_active):
        _id(attachment_id)
        query=f'SELECT {_SELECT}{_FROM}WHERE a.id = {attachment_id} LIMIT 2'
        def decode(value):
            result=_result(value,METADATA_COLUMNS,2)
            if not result.complete or len(result.rows)!=1:
                raise AttachmentReadUnavailable()
            metadata=_metadata(result.rows[0],owner_id)
            if metadata.attachment_id!=attachment_id:
                raise AttachmentReadUnavailable()
            return metadata
        return await self._read(owner_id,query,decode,deadline=deadline,check_active=check_active)

    async def text_page(self,owner_id,attachment_id,*,offset=0,limit=20000,deadline,check_active):
        _id(attachment_id)
        if (type(offset) is not int or not 0<=offset<=_MAX_OFFSET
                or type(limit) is not int or not 1<=limit<=100000):
            raise ValueError('Invalid attachment text page')
        query=(f'SELECT {_SELECT}, pg_catalog.substr(a.extracted_text, {offset+1}, {limit}) AS text_page'
               f'{_FROM}WHERE a.id = {attachment_id} LIMIT 2')
        def decode(value):
            result=_result(value,METADATA_COLUMNS+('text_page',),2)
            if not result.complete or len(result.rows)!=1:
                raise AttachmentReadUnavailable()
            metadata=_metadata(result.rows[0][:-1],owner_id)
            if metadata.attachment_id!=attachment_id:
                raise AttachmentReadUnavailable()
            text=result.rows[0][-1]
            total=metadata.text_chars
            if total is None:
                if text is not None:
                    raise AttachmentReadUnavailable()
                return AttachmentTextPage(metadata,None,offset,limit,None,None,False,False,False)
            expected=min(limit,max(0,total-offset))
            if type(text) is not str or len(text)!=expected:
                raise AttachmentReadUnavailable()
            try:
                if len(text.encode('utf-8'))>4*limit:
                    raise AttachmentReadUnavailable()
            except UnicodeError:
                raise AttachmentReadUnavailable() from None
            end=min(offset+limit,total)
            more=end<total
            limited=more and end>_MAX_OFFSET
            return AttachmentTextPage(metadata,text,offset,limit,total,
                end if more and not limited else None,True,offset==0 and not more,limited)
        return await self._read(owner_id,query,decode,deadline=deadline,check_active=check_active)

    async def list_for_thread(self,owner_id,thread_id,*,after_attachment_id=0,limit=100,deadline,check_active):
        if type(thread_id) is not str or not _ID.fullmatch(thread_id):
            raise ValueError('Invalid attachment thread identifier')
        _id(after_attachment_id,zero=True)
        if type(limit) is not int or not 1<=limit<=100:
            raise ValueError('Invalid attachment list limit')
        query=(f"SELECT {_SELECT}{_FROM}WHERE m.thread_id = '{thread_id}' "
               f'AND a.id > {after_attachment_id} ORDER BY a.id LIMIT {limit+1}')
        def decode(value):
            result=_result(value,METADATA_COLUMNS,limit+1)
            all_items=tuple(_metadata(row,owner_id) for row in result.rows)
            previous=after_attachment_id
            for item in all_items:
                if item.thread_id!=thread_id or item.attachment_id<=previous:
                    raise AttachmentReadUnavailable()
                previous=item.attachment_id
            items=all_items[:limit]
            more=len(all_items)>limit or not result.complete
            limited=more and (not items or items[-1].attachment_id==_MAX_ID)
            next_id=items[-1].attachment_id if more and not limited else None
            source_complete=not more
            return AttachmentMetadataPage(items,after_attachment_id,limit,source_complete,
                source_complete and after_attachment_id==0,next_id,limited)
        return await self._read(owner_id,query,decode,deadline=deadline,check_active=check_active)
