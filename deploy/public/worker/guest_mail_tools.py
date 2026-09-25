"""Guest-only, standard-library capability tools; no MCP or application mounting.

Trusted startup supplies the guest workspace, fixed loopback port and separate
run capabilities. No owner, session, upstream URL or credential is accepted in
model tool arguments. Capabilities.issue currently emits 64 lowercase hex chars.

The fixed relay closes each HTTP response. This client accepts Content-Length
or close-delimited JSON, never redirects, transfer coding or content encoding.
Cancellation closes/drains local sockets; remote operation cleanup remains the
relay/gateway's responsibility, and an interrupted upload has unknown outcome.
There are no automatic retries. This module must execute INSIDE the guest.
"""
import asyncio
from datetime import date
import errno
import json
import math
import os
from pathlib import Path
import re
import stat
from urllib.parse import quote

from guest_tool_config import LEGACY_PROFILE, RAW_PROFILE, ConfigError, parse_tool_config

MAX_BATCH_ITEMS = 20
# Every mail-tool call must finish fast; matches the gateway's tool deadline.
TOOL_TIMEOUT_SECONDS = 5
MAX_EXACT_INTEGER = 9007199254740991
MAX_ARGUMENT_BYTES = 256 * 1024
MAX_RESPONSE_BYTES = 4 * 1024**2
MAX_BATCH_RESPONSE_BYTES = 8 * 1024**2
MAX_ARTIFACT_BYTES = 4 * 1024**2
MAX_HEADER_BYTES = 16 * 1024
_ID = re.compile(r'[A-Za-z0-9_-]{1,256}\Z', re.ASCII)
_HEADER = re.compile(rb'[A-Za-z0-9!#$%&\'*+.^_`|~-]+\Z')
_MIME = re.compile(r'[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+\Z', re.ASCII)


class ToolError(ValueError):
    """Only fixed, non-sensitive messages are permitted here."""


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ToolError('Invalid gateway JSON.')
        value[key] = item
    return value


def _invalid_constant(_):
    raise ToolError('Invalid gateway JSON.')


async def _drain(awaitable):
    task = asyncio.ensure_future(awaitable)
    interrupted = False
    while True:
        try:
            value = await asyncio.shield(task)
            break
        except asyncio.CancelledError:
            if task.cancelled():
                raise
            interrupted = True
    if interrupted:
        raise asyncio.CancelledError()
    return value


def _filename(value):
    if (type(value) is not str or not 1 <= len(value.encode('utf-8')) <= 200
            or value in ('.', '..') or any(ord(c) < 32 or ord(c) == 127 or c in '/\\' for c in value)):
        raise ToolError('Invalid artifact filename.')
    return value


def _search_options(item):
    allowed={'query','top_k','date_from','date_to','detail','max_matches','sender','recipient'}
    if type(item) is not dict or set(item)-allowed or 'query' not in item:
        raise ToolError('Invalid search item.')
    for name in ('sender','recipient'):
        value=item.get(name)
        if value is not None and (type(value) is not str or not value.strip() or len(value)>256 or '\x00' in value):
            raise ToolError('Search sender/recipient must be 1-256 characters.')
    query=item['query']
    if type(query) is not str or not query.strip() or len(query)>1000 or '\x00' in query:
        raise ToolError('Invalid search query.')
    query.encode('utf-8')
    if item.get('detail','snippet') not in ('refs','snippet','summary','full'):
        raise ToolError('Invalid search detail.')
    for name,low,high in (('top_k',1,100),('max_matches',0,100)):
        if name in item and (type(item[name]) is not int or not low<=item[name]<=high):
            raise ToolError('Invalid search limit.')
    for name in ('date_from','date_to'):
        value=item.get(name)
        if value is not None:
            if type(value) is not str or not re.fullmatch(r'\d{4}-\d{2}-\d{2}',value):
                raise ToolError('Invalid search date.')
            try:
                date.fromisoformat(value)
            except ValueError:
                raise ToolError('Invalid search date.') from None
    if item.get('date_from') and item.get('date_to') and item['date_from']>item['date_to']:
        raise ToolError('Invalid search date range.')
    return dict(item)


def _facts_options(item):
    if type(item) is not dict or set(item)-{'query', 'exhaustive', 'k'} or 'query' not in item:
        raise ToolError('Invalid facts arguments.')
    query = item['query']
    if type(query) is not str or not query.strip() or len(query) > 1000 or '\x00' in query:
        raise ToolError('Invalid facts query.')
    query.encode('utf-8')
    if type(item.get('exhaustive', True)) is not bool:
        raise ToolError('Invalid facts exhaustive option.')
    if type(item.get('k', 200)) is not int or not 1 <= item.get('k', 200) <= 500:
        raise ToolError('Invalid facts limit.')
    return dict(item)


# Jev judgments: bounded typed questions over state the agent supplies.
MAX_JUDGE_STATE_BYTES = 32 * 1024
MAX_JUDGE_QUESTIONS = 8
_JUDGE_TYPES = ('noul', 'choice', 'score')
_JUDGE_ID = re.compile(r'[A-Za-z][A-Za-z0-9_]{0,63}\Z')


def _judge_question(question):
    """One agent question {id, type, instructions, options?} -> (id, Jev question)."""
    if (type(question) is not dict or set(question) - {'id', 'type', 'instructions', 'options'}
            or type(question.get('id')) is not str or not _JUDGE_ID.fullmatch(question['id'])
            or question.get('type') not in _JUDGE_TYPES
            or type(question.get('instructions')) is not str or not question['instructions'].strip()):
        raise ToolError('Invalid judge question.')
    options = question.get('options', [])
    if type(options) is not list or any(type(option) is not str or not option.strip() for option in options):
        raise ToolError('Judge options must be non-empty strings.')
    judged = {'type': question['type'], 'instructions': question['instructions']}
    if question['type'] == 'choice':
        if not 2 <= len(options) <= 32:
            raise ToolError('A choice question needs 2-32 options ("label: description").')
        labels = [option.split(':', 1)[0].strip() for option in options]
        if len(set(labels)) != len(labels) or not all(labels):
            raise ToolError('Choice option labels must be distinct.')
        judged['criteria'] = {label: option.split(':', 1)[-1].strip() for label, option in zip(labels, options)}
    elif question['type'] == 'score':
        if not 2 <= len(options) <= 10:
            raise ToolError('A score question needs 2-10 ordered level descriptions.')
        judged['criteria'] = options
    elif options:
        raise ToolError('A noul (yes/no) question takes no options.')
    return question['id'], judged


def _judge_options(item):
    """Validate a judge call and convert it to the gateway's Jev request shape."""
    if type(item) is not dict or set(item) != {'state', 'questions'}:
        raise ToolError('Invalid judge arguments.')
    state, questions = item['state'], item['questions']
    if type(state) is not str or not state.strip() or len(state.encode('utf-8')) > MAX_JUDGE_STATE_BYTES:
        raise ToolError('Judge state must be non-empty text of at most 32 KiB.')
    if type(questions) is not list or not 1 <= len(questions) <= MAX_JUDGE_QUESTIONS:
        raise ToolError('A judge call needs 1-8 questions.')
    converted = dict(_judge_question(question) for question in questions)
    if len(converted) != len(questions):
        raise ToolError('Judge question ids must be distinct.')
    return {'state': state, 'questions': converted}


def _metadata_options(item):
    allowed = {'sender', 'subject_contains', 'date_from', 'date_to', 'label', 'has_attachment', 'order_by', 'limit'}
    if type(item) is not dict or set(item)-allowed:
        raise ToolError('Invalid metadata filters.')
    for name, maximum in (('sender', 1000), ('subject_contains', 1000), ('label', 256), ('date_from', 10), ('date_to', 10)):
        value = item.get(name, '')
        if type(value) is not str or len(value) > maximum or '\x00' in value:
            raise ToolError('Invalid metadata filter.')
        value.encode('utf-8')
        if name in ('date_from', 'date_to') and value:
            if not re.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}', value):
                raise ToolError('Invalid metadata date.')
            try:
                date.fromisoformat(value)
            except ValueError:
                raise ToolError('Invalid metadata date.') from None
    if item.get('date_from') and item.get('date_to') and item['date_from'] > item['date_to']:
        raise ToolError('Invalid metadata date range.')
    if item.get('has_attachment') is not None and type(item['has_attachment']) is not bool:
        raise ToolError('Invalid attachment filter.')
    if item.get('order_by', 'date_desc') not in ('date_desc', 'date_asc'):
        raise ToolError('Invalid metadata ordering.')
    if type(item.get('limit', 20)) is not int or not 1 <= item.get('limit', 20) <= 100:
        raise ToolError('Invalid metadata limit.')
    return dict(item)


def _attachment_options(item, *, allow_raw=False, exact_ids=False):
    if (type(item) is not dict or set(item)-{'attachment_id','mode','offset','limit'}
            or type(item.get('attachment_id')) is not int or not 1<=item['attachment_id']<=(MAX_EXACT_INTEGER if exact_ids else 9223372036854775807)):
        raise ToolError('Invalid attachment item.')
    mode=item.get('mode','text')
    if type(mode) is not str or mode not in (('meta','text','raw') if allow_raw else ('meta','text')):
        raise ToolError('Only attachment metadata and stored text are supported.')
    body={'attachment_id':item['attachment_id']}
    if mode in ('meta','raw'):
        if 'offset' in item or 'limit' in item:raise ToolError('This attachment mode does not accept text paging.')
    else:
        for key,default,low,high in (('offset',0,0,2147483646),('limit',20000,1,100000)):
            value=item.get(key,default)
            if type(value) is not int or not low<=value<=high:raise ToolError('Invalid attachment text paging.')
            body[key]=value
    return mode,body


# The gateway explains some 400s (e.g. a text scan that belongs in search).
# Those details are fixed gateway strings, never mail content; keep them short.
MAX_REJECTION_REASON_BYTES = 600


async def _rejection_reason(reader, status_code, headers):
    """The gateway's `detail` for a 400, or None."""
    length = headers.get(b'content-length')
    if (status_code != b'400' or length is None or not re.fullmatch(rb'[0-9]{1,4}', length)
            or int(length) > MAX_REJECTION_REASON_BYTES
            or headers.get(b'content-type', b'').split(b';', 1)[0].lower() != b'application/json'):
        return None
    try:
        detail = json.loads(await reader.readexactly(int(length))).get('detail')
    except (ValueError, AttributeError, asyncio.IncompleteReadError):
        return None
    return detail if type(detail) is str and detail.isprintable() else None


class GuestMailTools:
    def __init__(self, workspace, capabilities, *, port=18080, timeout_seconds=TOOL_TIMEOUT_SECONDS):
        if (type(port) is not int or port not in (18080, 18081)
                or type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds)
                or not 0 < timeout_seconds <= 60):
            raise ToolError('Invalid trusted guest tool configuration.')
        try:
            self._config = parse_tool_config(capabilities)
        except ConfigError:
            raise ToolError('Invalid trusted guest tool configuration.') from None
        if self._config.profile==RAW_PROFILE and (port!=18080 or timeout_seconds>30):
            raise ToolError('The raw MCP profile requires the fixed gateway and bounded deadline.')
        self._workspace = Path(workspace)
        self._capabilities = self._config.capabilities
        self._port = port
        self._timeout = timeout_seconds
        self._slots = asyncio.Semaphore(2)
        self._raw_downloader=None
        self._raw_workspace_fd=None
        self._raw_initialization_failed=False
        self._workspace_close_uncertain=False
        self._closed=False
        self._dispatches=set()
        self._close_task=None
        self._loop=None

    @property
    def tool_profile(self):
        return self._config.profile

    @property
    def tool_names(self):
        return self._config.tool_names

    async def dispatch(self, name, args):
        loop=asyncio.get_running_loop()
        if self._closed or (self._loop is not None and self._loop is not loop):
            return {'error':'Guest tools are closed or unavailable.'}
        self._loop=loop
        task=asyncio.current_task()
        self._dispatches.add(task)
        try:
            return await self._dispatch(name,args)
        finally:
            self._dispatches.discard(task)

    async def _dispatch(self, name, args):
        """Return existing batch envelopes; cancellation propagates after close.

        Each dispatch has one absolute deadline including admission waits. Batch
        errors are per item and input order is preserved. The instance permits
        two active requests across all dispatches. Supported thread content is
        text only; raw/Markdown conversion and message selection are unavailable.
        """
        try:
            if type(name) is not str or name not in self.tool_names:
                raise ToolError('Unsupported guest tool.')
            if type(args) is not dict or len(json.dumps(args, allow_nan=False).encode()) > MAX_ARGUMENT_BYTES:
                raise ToolError('Invalid or oversized tool arguments.')
            options = {}
            if name == 'describe_schema':
                if args:
                    raise ToolError('Invalid schema arguments.')
                items, key = [None], None
            elif name == 'sql_query_batch':
                if set(args) != {'queries'}:
                    raise ToolError('Invalid SQL batch arguments.')
                items, key = args['queries'], 'query'
            elif name == 'query_emails_batch':
                if set(args) != {'filters'}:
                    raise ToolError('Invalid metadata batch arguments.')
                items, key = args['filters'], 'input'
            elif name == 'find_facts':
                items, key = [_facts_options(args)], None
            elif name == 'judge':
                items, key = [_judge_options(args)], None
            elif name == 'search_emails_batch':
                if set(args) != {'searches'}:
                    raise ToolError('Invalid search batch arguments.')
                items, key = args['searches'], 'input'
            elif name == 'get_attachment_batch':
                if set(args) != {'items'}:
                    raise ToolError('Invalid attachment batch arguments.')
                items, key = args['items'], 'input'
            elif name == 'get_thread_batch':
                if self.tool_profile == LEGACY_PROFILE and {'attachment_after_id','attachment_limit'} & set(args):
                    raise ToolError('Attachment paging requires the mail-read-v2 profile.')
                allowed = {'thread_ids', 'body_format', 'message_ids', 'message_offset',
                           'message_limit', 'body_offset', 'body_limit', 'attachment_after_id', 'attachment_limit'}
                if set(args) - allowed or 'thread_ids' not in args:
                    raise ToolError('Invalid thread batch arguments.')
                if args.get('body_format', 'text') != 'text' or args.get('message_ids') is not None:
                    raise ToolError('Only text thread pages are supported; raw, Markdown and message selection are unavailable.')
                bounds = {'message_offset': (0, 10000), 'message_limit': (1, 100),
                          'body_offset': (0, 2_147_483_646), 'body_limit': (1, 100000),
                          'attachment_after_id': (0, MAX_EXACT_INTEGER if self.tool_profile==RAW_PROFILE else 9223372036854775807), 'attachment_limit': (1, 100)}
                for field, (minimum, maximum) in bounds.items():
                    if field in args:
                        value = args[field]
                        if type(value) is not int or not minimum <= value <= maximum:
                            raise ToolError('Invalid thread paging options.')
                        options[field] = value
                items, key = args['thread_ids'], 'thread_id'
            elif name == 'publish_artifact_batch':
                if set(args) != {'items'}:
                    raise ToolError('Invalid artifact batch arguments.')
                items, key = args['items'], 'input'
            else:
                raise ToolError('Unsupported guest tool.')
            if type(items) is not list or not 1 <= len(items) <= MAX_BATCH_ITEMS:
                raise ToolError('A batch must contain 1 to 20 items.')
        except ToolError as error:
            return {'error': str(error)}
        except (ValueError, TypeError, UnicodeError, RecursionError):
            return {'error': 'Invalid or oversized tool arguments.'}

        deadline = asyncio.get_running_loop().time() + self._timeout
        budget = [MAX_BATCH_RESPONSE_BYTES]

        async def one(item):
            try:
                async with asyncio.timeout_at(deadline):
                    async with self._slots:
                        return await self._item(name, item, options, budget)
            except ToolError as error:
                return {'error': str(error)}
            except TimeoutError:
                return {'error': 'Guest tool deadline exceeded; interrupted operations may have completed. No retry was attempted.'}
            except (OSError, ValueError, TypeError, UnicodeError, RecursionError, asyncio.IncompleteReadError):
                return {'error': 'Guest tool request failed; operation outcome may be unknown. No retry was attempted.'}

        if name=='get_attachment_batch' and self.tool_profile==RAW_PROFILE:
            return await self._attachment_batch(items,deadline,budget,one)

        tasks = [asyncio.create_task(one(item)) for item in items]
        try:
            results = await asyncio.gather(*tasks)
        finally:
            async def cleanup():
                for task in tasks:
                    if not task.done() and not task.cancelling():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            await _drain(cleanup())
        return results[0] if key is None else {'results': [{key: item, 'result': result} for item, result in zip(items, results)]}

    def _get_raw_downloader(self):
        if self.tool_profile!=RAW_PROFILE or self._closed or self._raw_initialization_failed:
            raise ToolError('Raw attachment downloads are unavailable.')
        if self._raw_downloader is None:
            try:
                # Trusted startup owns the path. Hold its final directory and
                # let the downloader independently verify UID/mode and create
                # its private child. Never accept a workspace from tool args.
                self._raw_workspace_fd=os.open(self._workspace,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC)
                from guest_attachment_download import GuestAttachmentDownloader
                self._raw_downloader=GuestAttachmentDownloader(self._raw_workspace_fd,self._capabilities['attachment'],
                                                               slots=self._slots)
            except (OSError,ValueError,ImportError):
                self._raw_initialization_failed=True
                raise ToolError('Raw attachment workspace is unavailable.') from None
        return self._raw_downloader

    async def _attachment_batch(self,items,deadline,budget,one):
        raw_indices=[]; raw_ids=[]
        for index,item in enumerate(items):
            try: mode,payload=_attachment_options(item,allow_raw=True,exact_ids=True)
            except (ValueError,TypeError): continue  # Existing per-item errors.
            if mode=='raw':
                raw_indices.append(index); raw_ids.append(payload['attachment_id'])
        if len(raw_ids)!=len(set(raw_ids)):
            return {'error':'Duplicate raw attachment IDs are unavailable in one batch.'}
        results=[None]*len(items)
        tasks=[]
        async def regular(index): results[index]=await one(items[index])
        async def raw():
            try:
                downloader=self._get_raw_downloader()
                values=await downloader.download_many(tuple(raw_ids),deadline=deadline)
                budget[0]-=len(json.dumps(values,allow_nan=False).encode('utf-8'))
                if budget[0]<0: raise ToolError('Guest tool result exceeds its byte limit.')
                for index,value in zip(raw_indices,values,strict=True):results[index]=value
            except TimeoutError:
                for index in raw_indices:results[index]={'error':'Guest tool deadline exceeded. No retry was attempted.'}
            except (OSError,ValueError,TypeError,UnicodeError,RecursionError):
                for index in raw_indices:results[index]={'error':'Raw attachment download is unavailable. No retry was attempted.'}
        if raw_indices:tasks.append(asyncio.create_task(raw()))
        selected=set(raw_indices)
        tasks.extend(asyncio.create_task(regular(index)) for index in range(len(items)) if index not in selected)
        joined=asyncio.gather(*tasks)
        try:
            await asyncio.shield(joined)
        finally:
            async def cleanup():
                for task in tasks:
                    if not task.done() and not task.cancelling():task.cancel()
                await asyncio.gather(*tasks,return_exceptions=True)
                await asyncio.gather(joined,return_exceptions=True)
            await _drain(cleanup())
        if asyncio.get_running_loop().time()>=deadline:
            return {'error':'Guest tool deadline exceeded. Verified unreturned files may remain charged.'}
        if asyncio.current_task().cancelling():raise asyncio.CancelledError()
        return {'results':[{'input':item,'result':result} for item,result in zip(items,results,strict=True)]}

    async def aclose(self):
        """Stop admission, drain every dispatch and own persistent raw cleanup."""
        loop=asyncio.get_running_loop()
        if (self._loop is not None and self._loop is not loop) or asyncio.current_task() in self._dispatches:
            raise ToolError('Invalid guest tool close context.')
        self._closed=True
        async def close():
            tasks=tuple(self._dispatches)
            for task in tasks:
                if not task.done() and not task.cancelling():task.cancel()
            await asyncio.gather(*tasks,return_exceptions=True)
            if self._raw_downloader is not None:
                await self._raw_downloader.aclose()
            if self._raw_workspace_fd is not None:
                if self._workspace_close_uncertain:
                    try:os.fstat(self._raw_workspace_fd)
                    except OSError as error:
                        if error.errno!=errno.EBADF:raise ToolError('Guest workspace cleanup is unavailable.') from None
                        self._raw_workspace_fd=None
                    else:raise ToolError('Guest workspace cleanup is unavailable.')
                if self._raw_workspace_fd is not None:
                    try:os.close(self._raw_workspace_fd)
                    except OSError:
                        self._workspace_close_uncertain=True
                        raise ToolError('Guest workspace cleanup is unavailable.') from None
                    self._raw_workspace_fd=None
        if self._close_task is None or (self._close_task.done() and not self._close_task.cancelled()
                                       and self._close_task.exception() is not None):
            self._close_task=asyncio.create_task(close())
        await _drain(self._close_task)

    async def _item(self, name, item, options, budget):
        if name == 'describe_schema':
            return await self._request('GET', '/v1/schema', 'sql', b'', budget)
        if name == 'sql_query_batch':
            if type(item) is not str or not item:
                raise ToolError('Each SQL item must be a nonempty query string.')
            body = json.dumps({'query': item}).encode()
            if len(body) > 32768:
                raise ToolError('SQL request exceeds the gateway byte limit.')
            return await self._request('POST', '/v1/sql', 'sql', body, budget)
        if name == 'get_attachment_batch':
            mode, payload = _attachment_options(item,exact_ids=self.tool_profile==RAW_PROFILE)
            return await self._request('POST', '/v1/attachment/' + mode, 'attachment',
                                       json.dumps(payload).encode(), budget)
        if name == 'query_emails_batch':
            body = json.dumps(_metadata_options(item), ensure_ascii=False).encode('utf-8')
            return await self._request('POST', '/v1/query-emails', 'retrieval', body, budget)
        if name == 'find_facts':
            body = json.dumps(_facts_options(item), ensure_ascii=False).encode('utf-8')
            return await self._request('POST', '/v1/find-facts', 'retrieval', body, budget)
        if name == 'judge':
            # `dispatch` already validated and converted the agent's questions.
            body = json.dumps(item, ensure_ascii=False).encode('utf-8')
            return await self._request('POST', '/v1/judge', 'retrieval', body, budget)
        if name == 'search_emails_batch':
            body = json.dumps(_search_options(item), ensure_ascii=False).encode('utf-8')
            return await self._request('POST', '/v1/search', 'retrieval', body, budget)
        if name == 'get_thread_batch':
            if type(item) is not str or not _ID.fullmatch(item):
                raise ToolError('Invalid thread identifier.')
            body = json.dumps({'thread_id': item, **options}).encode()
            return await self._request('POST', '/v1/thread', 'retrieval', body, budget)
        if type(item) is not dict or set(item) - {'path', 'name', 'mime_type'} or 'path' not in item:
            raise ToolError('Invalid artifact item.')
        requested_mime = item.get('mime_type', 'application/octet-stream')
        if requested_mime != 'application/octet-stream':
            raise ToolError('Artifact MIME overrides are unavailable; downloads use application/octet-stream as attachments.')
        filename = _filename(item.get('name', Path(item['path']).name))
        data = self._read_artifact(item['path'])
        result = await self._request('POST', '/v1/artifacts?filename=' + quote(filename, safe=''),
                                     'artifact', data, budget, status=201, content_type='application/octet-stream')
        mime = result.get('mime_type', 'application/octet-stream')
        if (type(result.get('id')) is not str or not re.fullmatch('[a-f0-9]{32}', result['id'])
                or type(result.get('size')) is not int or result['size'] != len(data)
                or type(mime) is not str or len(mime) > 127 or not _MIME.fullmatch(mime)):
            raise ToolError('Invalid artifact receipt; upload outcome is unknown. No retry was attempted.')
        return {'id': result['id'], 'name': _filename(result.get('filename', filename)),
                'size': result['size'], 'mime_type': mime, 'content_disposition': 'attachment'}

    def _read_artifact(self, path):
        if (type(path) is not str or not path or len(path) > 4096 or path.startswith('/')
                or any(part in ('', '.', '..') for part in path.split('/'))):
            raise ToolError('Artifact paths must be relative regular files inside the guest workspace.')
        directory = os.open(self._workspace, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        fd = None
        try:
            parts = path.split('/')
            for part in parts[:-1]:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
                os.close(directory)
                directory = child
            fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or not 0 <= info.st_size <= MAX_ARTIFACT_BYTES:
                raise ToolError('Artifact is not a regular file within the upload byte limit.')
            chunks, size = [], 0
            while chunk := os.read(fd, min(65536, MAX_ARTIFACT_BYTES + 1 - size)):
                size += len(chunk)
                if size > MAX_ARTIFACT_BYTES:
                    raise ToolError('Artifact exceeds the upload byte limit.')
                chunks.append(chunk)
            return b''.join(chunks)
        finally:
            if fd is not None:
                os.close(fd)
            os.close(directory)

    async def _request(self, method, path, audience, body, budget, *, status=200, content_type='application/json'):
        reader, writer = await asyncio.open_connection('127.0.0.1', self._port, limit=MAX_HEADER_BYTES)
        try:
            header = (f'{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{self._port}\r\n'
                      f'Authorization: Bearer {self._capabilities[audience]}\r\n'
                      f'Content-Type: {content_type}\r\nContent-Length: {len(body)}\r\n'
                      'Accept: application/json\r\nAccept-Encoding: identity\r\nConnection: close\r\n\r\n')
            writer.write(header.encode('ascii') + body)
            await writer.drain()
            try:
                head = await reader.readuntil(b'\r\n\r\n')
            except asyncio.LimitOverrunError:
                raise ToolError('Gateway response headers exceed their byte limit.') from None
            if len(head) > MAX_HEADER_BYTES:
                raise ToolError('Gateway response headers exceed their byte limit.')
            lines = head[:-4].split(b'\r\n')
            first = re.fullmatch(rb'HTTP/1\.[01] ([0-9]{3}) [\x20-\x7e]*', lines[0])
            if first is None:
                raise ToolError('Invalid gateway HTTP response.')
            headers = {}
            for line in lines[1:]:
                key, separator, value = line.partition(b':')
                if not separator or not _HEADER.fullmatch(key) or any(c < 32 or c > 126 for c in value):
                    raise ToolError('Invalid gateway response header.')
                key, value = key.lower(), value.strip()
                if key in headers:
                    raise ToolError('Duplicate gateway response header.')
                headers[key] = value
            if b'transfer-encoding' in headers or b'content-encoding' in headers:
                raise ToolError('Unsupported gateway response encoding.')
            length = headers.get(b'content-length')
            if length is not None and (not re.fullmatch(rb'[0-9]{1,9}', length) or int(length) > MAX_RESPONSE_BYTES):
                raise ToolError('Invalid or oversized gateway response length.')
            if int(first[1]) != status:
                reason = await _rejection_reason(reader, first[1], headers)
                raise ToolError('Gateway rejected the operation' + (f': {reason}' if reason else '.')
                                + ' No retry was attempted.')
            if headers.get(b'content-type', b'').split(b';', 1)[0].lower() != b'application/json':
                raise ToolError('Gateway response is not JSON.')
            chunks, size = [], 0
            while chunk := await reader.read(65536):
                size += len(chunk)
                budget[0] -= len(chunk)
                if size > MAX_RESPONSE_BYTES or budget[0] < 0:
                    raise ToolError('Gateway response exceeds the tool result byte limit.')
                chunks.append(chunk)
            if length is not None and size != int(length):
                raise ToolError('Gateway response length mismatch.')
            result = json.loads(b''.join(chunks), object_pairs_hook=_object, parse_constant=_invalid_constant)
            if type(result) is not dict:
                raise ToolError('Invalid gateway JSON object.')
            return result
        finally:
            # close() alone may flush buffered upload bytes indefinitely after
            # cancellation. No connection is reused; abort guarantees local
            # teardown even when the peer stopped reading the request.
            writer.close()
            writer.transport.abort()
            async def close():
                try:
                    await writer.wait_closed()
                except OSError:
                    pass
            await _drain(close())
