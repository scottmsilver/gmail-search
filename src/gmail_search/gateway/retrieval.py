"""Typed text retrieval through the existing restricted analytical executor.

No legacy application HTTP, privileged database connection, HTML parser or
filesystem access. This initial thread reader does not implement ranked search.
"""
import asyncio
import json
import re

from .analytics import QueryRejected
from .registry import AccessDenied
from .tool_deadline import tool_deadline

_ID = re.compile(r'[A-Za-z0-9_-]{1,256}\Z', re.ASCII)
_COLUMNS = 'id, thread_id, from_addr, to_addr, subject, date, labels'
_PG_INT32_MAX = 2_147_483_647
# PostgreSQL substr positions are one-indexed signed int4 values.
_MAX_BODY_OFFSET = _PG_INT32_MAX - 1


async def _drain(awaitable):
    """Acknowledge owned work before propagating repeated cancellation."""
    task = asyncio.ensure_future(awaitable)
    interrupted = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            interrupted = True
        except BaseException:
            break
    if interrupted:
        if not task.cancelled():
            task.exception()
        raise asyncio.CancelledError()
    return task.result()


def _body_response(result,thread_id,message_offset,message_limit,body_offset,body_limit):
    messages = []
    for row in result.rows[:message_limit]:
        message = dict(zip(result.columns, row, strict=True))
        body = message.pop('body_text') or ''
        body_total_chars = message.pop('body_total_chars') or 0
        if type(body) is not str or type(body_total_chars) is not int or body_total_chars < 0:
            raise RuntimeError('Database returned an invalid text page')
        end = min(body_offset + body_limit, body_total_chars)
        body_limited = end < body_total_chars and end > _MAX_BODY_OFFSET
        message.update(body_text=body, cite_ref=thread_id, body_format='text',
                       body_offset=body_offset, body_limit=body_limit,
                       body_total_chars=body_total_chars,
                       body_text_truncated=body_offset > 0 or end < body_total_chars,
                       body_pagination_limited=body_limited,
                       body_next_offset=end if end < body_total_chars and not body_limited else None)
        messages.append(message)
    more = len(result.rows) > message_limit
    source_complete = result.complete and not more and message_offset == 0
    next_offset = message_offset + len(messages)
    has_more = more or not result.complete
    pagination_limited = has_more and next_offset > 10000
    # Never invent a total count from a truncated/offset result.
    response = dict(thread_id=thread_id, cite_ref=thread_id, body_format='text',
                    messages=messages, returned_message_count=len(messages),
                    thread_message_count=len(messages) if source_complete else None,
                    source_complete=source_complete,
                    complete=source_complete and not any(m['body_text_truncated'] for m in messages),
                    pagination_limited=pagination_limited,
                    next_message_offset=next_offset if has_more and messages and not pagination_limited else None)
    return response


def _add_manifest(response,page,owner,thread,after,limit):
    # Local import avoids the reader's dependency on this module's drain helper.
    from .attachment_reader import AttachmentMetadata, AttachmentMetadataPage
    if (type(page) is not AttachmentMetadataPage or type(page.items) is not tuple
            or len(page.items)>limit or page.after_attachment_id!=after or page.limit!=limit
            or any(type(flag) is not bool for flag in (page.source_complete,page.complete,page.pagination_limited))
            or page.complete != (page.source_complete and after==0)
            or (page.next_attachment_id is not None and (type(page.next_attachment_id) is not int
                or not after < page.next_attachment_id < 2**63))
            or (page.source_complete and (page.next_attachment_id is not None or page.pagination_limited))):
        raise RuntimeError('Invalid attachment inventory')
    items=[]
    previous=after
    for item in page.items:
        if (type(item) is not AttachmentMetadata or item.owner_id!=owner or item.thread_id!=thread
                or type(item.attachment_id) is not int or not previous < item.attachment_id < 2**63
                or type(item.message_id) is not str or not _ID.fullmatch(item.message_id)):
            raise RuntimeError('Invalid attachment inventory')
        previous=item.attachment_id
        items.append(dict(id=item.attachment_id,message_id=item.message_id,thread_id=item.thread_id,
            filename=item.filename,mime_type=item.mime_type,size_bytes=item.size_bytes,
            fetch_status=item.fetch_status,text_chars=item.text_chars,
            stored_text_state=item.stored_text_state,extraction_complete=None))
    if (page.next_attachment_id is not None and (not items or page.next_attachment_id!=previous)
            or (not page.source_complete and page.next_attachment_id is None and not page.pagination_limited)):
        raise RuntimeError('Invalid attachment inventory')
    response['attachment_inventory']=dict(items=items,after_attachment_id=after,limit=limit,
        source_complete=page.source_complete,complete=page.complete,
        next_attachment_id=page.next_attachment_id,pagination_limited=page.pagination_limited,
        same_snapshot_as_messages=False)
    for message in response['messages']:
        message['attachments']=[item for item in items if item['message_id']==message['id']]
        message['attachments_complete']=page.complete


class RunRetrievalService:
    def __init__(self, capabilities, gateway, *, attachment_reader=None):
        if attachment_reader is not None and getattr(attachment_reader,'gateway',None) is not gateway:
            raise ValueError('Attachment reader must share the query gateway')
        self.capabilities, self.gateway = capabilities, gateway
        self.attachment_reader = attachment_reader

    async def authorize(self, token):
        return await _drain(asyncio.to_thread(self.capabilities.authorize, token,
                                             audience='retrieval', operation='thread.get'))

    async def _watch(self, token):
        while True:
            await asyncio.sleep(.1)
            await self.authorize(token)

    async def thread(self, token, *, thread_id, message_offset=0, message_limit=20,
                     body_offset=0, body_limit=20000, attachment_after_id=0, attachment_limit=100):
        """Read body and optional attachment pages in sequential snapshots.

        Existing top-level completion flags describe the message/body page.
        Attachment inventory flags independently describe the requested ID page;
        visible messages contain only manifests from that inventory page.
        """
        deadline = tool_deadline()
        async with asyncio.timeout_at(deadline):
            lease = await self.authorize(token)
        if (type(thread_id) is not str or not _ID.fullmatch(thread_id)
                or any(type(value) is not int or not 0 <= value <= maximum
                       for value, maximum in ((message_offset, 10000), (body_offset, _MAX_BODY_OFFSET)))
                or type(message_limit) is not int or not 1 <= message_limit <= 100
                or type(body_limit) is not int or not 1 <= body_limit <= 100000
                or type(attachment_after_id) is not int or not 0 <= attachment_after_id < 2**63
                or type(attachment_limit) is not int or not 1 <= attachment_limit <= 100
                or (self.attachment_reader is None and (attachment_after_id != 0 or attachment_limit != 100))):
            raise QueryRejected('Invalid thread retrieval options')
        # The identifier grammar excludes SQL punctuation. The analytical compiler
        # still parses this fixed query and emits fresh parameterized SQL.
        query = (f"SELECT {_COLUMNS}, pg_catalog.substr(body_text, {body_offset + 1}, {body_limit}) AS body_text, "
                 f"length(body_text) AS body_total_chars FROM messages WHERE thread_id = '{thread_id}' "
                 f'ORDER BY date, id LIMIT {message_limit + 1} OFFSET {message_offset}')
        async def check():
            current = await self.authorize(token)
            if current.owner_id != lease.owner_id or current.run_id != lease.run_id:
                raise AccessDenied()
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError('Thread retrieval deadline exceeded')
            return True
        async def execute():
            async with asyncio.timeout_at(deadline):
                await check()
                result = await self.gateway.query(lease.owner_id, query)
                response = _body_response(result,thread_id,message_offset,message_limit,body_offset,body_limit)
                if self.attachment_reader is not None:
                    page = await self.attachment_reader.list_for_thread(
                        lease.owner_id,thread_id,after_attachment_id=attachment_after_id,
                        limit=attachment_limit,deadline=deadline,check_active=check)
                    _add_manifest(response,page,lease.owner_id,thread_id,attachment_after_id,attachment_limit)
                try:
                    encoded = json.dumps(response,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode('utf-8')
                except (ValueError,TypeError,UnicodeError,RecursionError):
                    raise RuntimeError('Invalid thread response') from None
                if len(encoded) > 4*1024*1024:
                    raise RuntimeError('Thread response exceeds its byte limit')
                return response
        work = asyncio.create_task(execute())
        watch = asyncio.create_task(self._watch(token))
        try:
            done, _ = await asyncio.wait((work, watch), return_when=asyncio.FIRST_COMPLETED)
            if watch in done:
                await watch
            response = await work
        finally:
            for task in (work, watch):
                if not task.done() and not task.cancelling():
                    task.cancel()
            await _drain(asyncio.gather(work, watch, return_exceptions=True))
        # A watcher can still be draining registry work after the query closes.
        # Finish all cleanup before fresh authorization, with no later await.
        await check()
        return response
