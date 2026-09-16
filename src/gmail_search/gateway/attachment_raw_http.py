"""Fixed raw-download ASGI boundary, installed outside buffering middleware.

No resources are acquired at construction. One __call__ owns authentication,
body framing, payload admission, loader and every ASGI send through teardown.
"""
import asyncio
import hashlib
import json
import re

from .attachment_raw_service import RawAttachmentBusy, RawAttachmentUnavailable
from .provider import _finish
from .registry import AccessDenied

PATH='/v1/attachment/raw'
MAX_HEADER_BYTES=4096
CHUNK_BYTES=65536
_SECURITY=[(b'cache-control',b'private, no-store'),(b'x-content-type-options',b'nosniff')]


class RawStreamAborted(RuntimeError):
    def __init__(self):
        super().__init__('Attachment stream aborted.')


class _Disconnected(Exception):
    pass


class _RequestError(Exception):
    def __init__(self,status):
        self.status=status


def _object(pairs):
    value={}
    for key,item in pairs:
        if key in value:
            raise ValueError()
        value[key]=item
    return value


def _headers(scope):
    entries=scope.get('headers',())
    if len(entries)>64 or sum(len(k)+len(v) for k,v in entries)>16384:
        raise _RequestError(400)
    headers={}
    for key,value in entries:
        key=key.lower()
        if key in headers:
            raise _RequestError(400)
        headers[key]=value
    if any(key in headers for key in (b'cookie',b'x-user-id',b'range',b'if-range',b'raw_path',b'raw-path',
                                      b'content-encoding',b'transfer-encoding')):
        raise _RequestError(400)
    authorization=headers.get(b'authorization',b'')
    if not re.fullmatch(rb'Bearer [a-f0-9]{64}',authorization):
        raise _RequestError(401)
    if headers.get(b'content-type')!=b'application/json':
        raise _RequestError(400)
    length=headers.get(b'content-length',b'')
    if not re.fullmatch(rb'0|[1-9][0-9]{0,9}',length):
        raise _RequestError(400)
    size=int(length)
    if size>4096:
        raise _RequestError(413)
    return authorization[7:].decode('ascii'),size


async def _body(receive,size):
    data=bytearray()
    async with asyncio.timeout(3):
        while True:
            message=await receive()
            if message['type']=='http.disconnect':
                raise _Disconnected()
            if message['type']!='http.request' or type(message.get('body',b'')) is not bytes:
                raise _RequestError(400)
            chunk=message.get('body',b'')
            if len(chunk)>size-len(data):
                raise _RequestError(400)
            data.extend(chunk)
            if not message.get('more_body',False):
                break
    if len(data)!=size:
        raise _RequestError(400)
    try:
        value=json.loads(data,object_pairs_hook=_object)
        if (type(value) is not dict or set(value)!={'attachment_id'} or type(value['attachment_id']) is not int
                or not 0<value['attachment_id']<=9223372036854775807):
            raise ValueError()
        return value['attachment_id']
    except (ValueError,UnicodeError,RecursionError):
        raise _RequestError(400) from None


async def _disconnect(receive):
    while True:
        message=await receive()
        if (message['type']=='http.disconnect' or message['type']!='http.request'
                or message.get('body',b'') or message.get('more_body',False)):
            raise _Disconnected()
        # A conforming server blocks after body EOF; avoid a busy loop otherwise.
        await asyncio.sleep(.01)


async def _publish(source,check,send):
    chunk=None
    try:
        digest=hashlib.sha256()
        for offset in range(0,len(source.data),CHUNK_BYTES):
            digest.update(source.data[offset:offset+CHUNK_BYTES])
            await asyncio.sleep(0)
        header=json.dumps(dict(version=1,operation='raw',attachment_id=source.attachment_id,
            mime_type=source.mime_type,size_bytes=len(source.data),sha256=digest.hexdigest()),
            separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')
        if not 0<len(header)<=MAX_HEADER_BYTES:
            raise RawAttachmentUnavailable()
        prefix=len(header).to_bytes(4,'big')+header
        await check()
        await send(dict(type='http.response.start',status=200,headers=_SECURITY+[
            (b'content-type',b'application/octet-stream'),
            (b'content-length',str(len(prefix)+len(source.data)).encode('ascii'))]))
        await check()
        await send(dict(type='http.response.body',body=prefix,more_body=True))
        for offset in range(0,len(source.data),CHUNK_BYTES):
            await check()
            chunk=source.data[offset:offset+CHUNK_BYTES]
            await send(dict(type='http.response.body',body=chunk,more_body=True))
        await check()
        await send(dict(type='http.response.body',body=b'',more_body=False))
    finally:
        source=chunk=None


class RawAttachmentMiddleware:
    """Install last with add_middleware, outside BaseHTTPMiddleware wrappers.

    Delegates every other path unchanged. Later outer middleware must also be
    pure ASGI without detached/buffered body sends; deployment must preserve this
    ordering. The application send acknowledgment is not a remote TCP ACK.
    """
    def __init__(self,app,*,service):
        self.app,self.service=app,service

    async def _request(self,scope,receive,send,deadline):
        if (scope.get('path')!=PATH or scope.get('raw_path')!=PATH.encode('ascii')
                or scope.get('root_path','') or scope.get('query_string',b'')):
            raise _RequestError(400)
        if scope.get('method')!='POST':
            raise _RequestError(405)
        token,size=_headers(scope)
        await self.service.authorize(token)
        attachment_id=await _body(receive,size)
        async def publish(source,check):
            try:
                await _publish(source,check,send)
            finally:
                source=None
        task=asyncio.create_task(self.service.deliver(token,attachment_id,deadline=deadline,publish=publish))
        disconnect=asyncio.create_task(_disconnect(receive))
        try:
            done,_=await asyncio.wait((task,disconnect),return_when=asyncio.FIRST_COMPLETED)
            if disconnect in done:
                await disconnect
            await task
        finally:
            async def cleanup():
                for owned in (task,disconnect):
                    if not owned.done() and not owned.cancelling():
                        owned.cancel()
                await asyncio.gather(task,disconnect,return_exceptions=True)
            await _finish(cleanup())

    async def __call__(self,scope,receive,send):
        if scope['type']!='http' or scope.get('path','').rstrip('/')!=PATH:
            return await self.app(scope,receive,send)
        started=False
        async def sending(message):
            nonlocal started
            if message['type']=='http.response.start':
                # A failed start send may have reached the server: never retry it.
                started=True
            await send(message)
        deadline=asyncio.get_running_loop().time()+self.service.timeout_seconds
        try:
            async with asyncio.timeout_at(deadline):
                await self._request(scope,receive,sending,deadline)
            return
        except _Disconnected:
            return
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if started:
                raise RawStreamAborted() from None
            if isinstance(error,_RequestError): status=error.status
            elif isinstance(error,AccessDenied): status=403
            elif isinstance(error,RawAttachmentUnavailable): status=404
            elif isinstance(error,RawAttachmentBusy): status=429
            elif isinstance(error,TimeoutError): status=504
            else: status=503
        detail={400:'Invalid attachment request',401:'Run capability required',403:'Run access denied',
            404:'Attachment bytes are unavailable',405:'Method not allowed',413:'Request exceeds byte limit',
            429:'Attachment download capacity is unavailable',504:'Attachment deadline exceeded',
            503:'Attachment download is unavailable'}[status]
        body=json.dumps({'detail':detail},separators=(',',':')).encode('ascii')
        try:
            async with asyncio.timeout(3):
                await sending(dict(type='http.response.start',status=status,headers=_SECURITY+[
                    (b'content-type',b'application/json'),(b'content-length',str(len(body)).encode('ascii'))]))
                await sending(dict(type='http.response.body',body=body,more_body=False))
        except OSError:
            return


def add_raw_attachment_middleware(app,service):
    """Call after registering all buffering middleware, before serving requests."""
    app.add_middleware(RawAttachmentMiddleware,service=service)
