"""Guest-only fixed binary download transport; no tool/config integration.

The persistent downloader supplies admission and an opaque file sink. Endpoint,
method and capability audience cannot be selected by model arguments.
"""
import asyncio
import hashlib
import json
import re

MAX_FILE_BYTES=10*1024**2
MAX_PACKET_HEADER=4096
MAX_HTTP_HEADER=16384
MAX_WIRE_BYTES=4+MAX_PACKET_HEADER+MAX_FILE_BYTES
CHUNK_BYTES=65536
_TOKEN=re.compile(r'[a-f0-9]{64}\Z',re.ASCII)
_MIME=re.compile(r'[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+\Z',re.ASCII)
_HEADER=re.compile(rb'[A-Za-z0-9!#$%&\'*+.^_`|~-]+\Z')


class DownloadError(ValueError):
    def __init__(self):
        super().__init__('Attachment download failed. No retry was attempted.')


class DownloadCleanupError(DownloadError):
    def __init__(self):
        ValueError.__init__(self,'Attachment cleanup is unavailable; downloader capacity is retained.')


async def _drain(awaitable):
    task=asyncio.ensure_future(awaitable)
    interrupted=False
    while True:
        try:
            value=await asyncio.shield(task)
            break
        except asyncio.CancelledError:
            if task.cancelled():
                raise
            interrupted=True
    if interrupted:
        raise asyncio.CancelledError()
    return value


async def _open():
    return await asyncio.open_connection('127.0.0.1',18080,limit=MAX_HTTP_HEADER)


def _object(pairs):
    value={}
    for key,item in pairs:
        if key in value:
            raise DownloadError()
        value[key]=item
    return value


def _constant(value):
    raise DownloadError()


def _packet(raw,attachment_id,wire_size):
    try:
        header=json.loads(raw.decode('utf-8'),object_pairs_hook=_object,parse_constant=_constant)
        if (type(header) is not dict or set(header)!={'version','operation','attachment_id','mime_type','size_bytes','sha256'}
                or type(header['version']) is not int or header['version']!=1 or header['operation']!='raw'
                or type(header['attachment_id']) is not int or header['attachment_id']!=attachment_id
                or type(header['mime_type']) is not str or len(header['mime_type'])>255
                or not _MIME.fullmatch(header['mime_type'])
                or type(header['size_bytes']) is not int or not 0<=header['size_bytes']<=MAX_FILE_BYTES
                or type(header['sha256']) is not str or not _TOKEN.fullmatch(header['sha256'])
                or wire_size!=4+len(raw)+header['size_bytes']):
            raise DownloadError()
        return header
    except (ValueError,TypeError,UnicodeError,RecursionError):
        raise DownloadError() from None


async def _http_header(reader):
    raw=await reader.readuntil(b'\r\n\r\n')
    if len(raw)>MAX_HTTP_HEADER:
        raise DownloadError()
    lines=raw[:-4].split(b'\r\n')
    if len(lines)>65 or re.fullmatch(rb'HTTP/1\.[01] 200 [\x20-\x7e]*',lines[0]) is None:
        raise DownloadError()
    headers={}
    for line in lines[1:]:
        key,separator,value=line.partition(b':')
        if not separator or not _HEADER.fullmatch(key) or any(c<32 or c>126 for c in value):
            raise DownloadError()
        key=key.lower()
        if key in headers:
            raise DownloadError()
        headers[key]=value.strip()
    length=headers.get(b'content-length',b'')
    if (headers.get(b'content-type')!=b'application/octet-stream'
            or b'transfer-encoding' in headers or b'content-encoding' in headers
            or re.fullmatch(rb'[1-9][0-9]{0,8}',length) is None or not 4<int(length)<=MAX_WIRE_BYTES):
        raise DownloadError()
    return int(length)


async def _close_writer(writer):
    # abort must run even if close() fails; it stops buffered request writes.
    try:
        writer.close()
    finally:
        writer.transport.abort()
        try:
            await writer.wait_closed()
        except OSError:
            # A connection reset is a completed local socket teardown.
            pass


class GuestRawTransport:
    """Internal transport: caller must hold the shared core socket lease."""
    def __init__(self,capability):
        if type(capability) is not str or not _TOKEN.fullmatch(capability):
            raise DownloadError()
        self._capability=capability
        self._stranded=[]

    async def receive(self,attachment_id,*,deadline,on_header,on_chunk):
        if self._stranded:
            raise DownloadCleanupError()
        connection=reader=writer=None
        header=None
        failure=None
        try:
            async with asyncio.timeout_at(deadline):
                connection=asyncio.create_task(_open())
                reader,writer=await asyncio.shield(connection)
                body=json.dumps({'attachment_id':attachment_id},separators=(',',':')).encode('ascii')
                head=(f'POST /v1/attachment/raw HTTP/1.1\r\nHost: 127.0.0.1:18080\r\n'
                    f'Authorization: Bearer {self._capability}\r\nContent-Type: application/json\r\n'
                    f'Content-Length: {len(body)}\r\nAccept: application/octet-stream\r\n'
                    'Accept-Encoding: identity\r\nConnection: close\r\n\r\n').encode('ascii')
                writer.write(head+body)
                await writer.drain()
                wire_size=await _http_header(reader)
                length=int.from_bytes(await reader.readexactly(4),'big')
                if not 0<length<=MAX_PACKET_HEADER:
                    raise DownloadError()
                header=_packet(await reader.readexactly(length),attachment_id,wire_size)
                await on_header(header)
                left=header['size_bytes']
                digest=hashlib.sha256()
                while left:
                    chunk=await reader.readexactly(min(CHUNK_BYTES,left))
                    digest.update(chunk)
                    await on_chunk(chunk)
                    left-=len(chunk)
                if digest.hexdigest()!=header['sha256'] or await reader.read(1)!=b'':
                    raise DownloadError()
        except asyncio.CancelledError:
            failure=asyncio.CancelledError
        except TimeoutError:
            failure=TimeoutError
        except Exception:
            failure=DownloadError
        finally:
            async def close():
                nonlocal writer
                if connection is not None and writer is None:
                    if not connection.done() and not connection.cancelling():
                        connection.cancel()
                    result=(await asyncio.gather(connection,return_exceptions=True))[0]
                    if type(result) is tuple and len(result)==2:
                        _,writer=result
                if writer is not None:
                    try:
                        await _close_writer(writer)
                    except Exception:
                        self._stranded.append(writer)
                        raise DownloadCleanupError() from None
            try:
                # Separate ownership also protects cancellation that arrives
                # after EOF, while normal wait_closed() is already running.
                await _drain(close())
            except asyncio.CancelledError:
                failure=asyncio.CancelledError
            except DownloadCleanupError:
                failure=DownloadCleanupError
        if failure is not None:
            raise failure()
        return header

    async def aclose(self):
        failures=[]
        for writer in self._stranded:
            try:
                await _drain(_close_writer(writer))
            except Exception:
                failures.append(writer)
        self._stranded=failures
        if failures:
            raise DownloadCleanupError()
