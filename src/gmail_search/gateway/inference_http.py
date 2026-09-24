"""Worker-only inference routes over immutable, run-bound provider services.

The caller selects neither owner nor provider configuration.  A trusted factory
binds each exact route to a preconfigured run service.  This boundary accepts
the relay-normalized Bearer capability, authenticates it before consuming a
body, then passes only bounded duplicate-free JSON to that service.
"""
import asyncio
import json
import re
import uuid

from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse

from .inference import MAX_REQUEST_BYTES
from .provider import ReplayRejected
from .registry import AccessDenied, BudgetExhausted

_REQUEST_KEY = re.compile(r'[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z', re.ASCII)
_STREAM_HEADERS = {
    'Cache-Control': 'private, no-store',
    'X-Content-Type-Options': 'nosniff',
    'X-Accel-Buffering': 'no',
}


class _Admission:
    """Small non-queuing process-local stream limit for one ASGI event loop."""
    def __init__(self):
        self._lock = asyncio.Lock()
        self._total = 0
        self._owners = {}

    async def acquire(self, owner_id):
        if type(owner_id) is not str or not owner_id:
            raise RuntimeError('Invalid authenticated run identity.')
        async with self._lock:
            count = self._owners.get(owner_id, 0)
            if self._total >= 4 or count >= 2:
                return False
            self._total += 1
            self._owners[owner_id] = count + 1
            return True

    async def release(self, owner_id):
        async with self._lock:
            count = self._owners.get(owner_id)
            if count is None or self._total <= 0:
                raise RuntimeError('Inference admission accounting failed.')
            self._total -= 1
            if count == 1:
                del self._owners[owner_id]
            else:
                self._owners[owner_id] = count - 1


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def _request_key(request):
    values = request.headers.getlist('x-gateway-request-key')
    if not values:
        # The captured native CLIs do not send a custom idempotency header.
        # Server-generated keys intentionally make separate requests separate
        # reservations, rather than guessing whether they are safe retries.
        return 'gw-' + uuid.uuid4().hex
    if len(values) != 1 or not _REQUEST_KEY.fullmatch(values[0]):
        raise HTTPException(400, 'Invalid inference request key')
    return values[0]


async def _json_body(request):
    if request.headers.get('content-type', '').split(';', 1)[0].strip().lower() != 'application/json':
        raise HTTPException(400, 'JSON inference request required')
    chunks = []
    size = 0
    try:
        async with asyncio.timeout(3):
            async for chunk in request.stream():
                if type(chunk) is not bytes:
                    raise ValueError()
                size += len(chunk)
                if size > MAX_REQUEST_BYTES:
                    raise HTTPException(413, 'Inference request exceeds its byte limit')
                chunks.append(chunk)
    except TimeoutError:
        raise HTTPException(408, 'Inference request timed out') from None
    try:
        value = json.loads(b''.join(chunks), object_pairs_hook=_object,
                           parse_constant=lambda value: (_ for _ in ()).throw(ValueError()))
        if type(value) is not dict:
            raise ValueError()
        return value
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise HTTPException(400, 'Invalid inference request') from None


async def _close(iterator):
    """Close the provider iterator even if ASGI cancels this task repeatedly."""
    closer = getattr(iterator, 'aclose', None)
    if closer is None:
        return
    draining = asyncio.create_task(closer())
    interrupted = False
    while True:
        try:
            await asyncio.shield(draining)
            break
        except asyncio.CancelledError:
            if draining.cancelled():
                raise
            interrupted = True
    if interrupted:
        raise asyncio.CancelledError()


def _http_error(error):
    if isinstance(error, ReplayRejected):
        return HTTPException(409, 'Inference request was already claimed')
    if isinstance(error, BudgetExhausted):
        return HTTPException(402, str(error))
    if isinstance(error, AccessDenied):
        return HTTPException(403, 'Run access denied')
    if isinstance(error, (ValueError, TypeError, RecursionError, UnicodeError)):
        return HTTPException(400, 'Invalid inference request')
    if isinstance(error, TimeoutError):
        return HTTPException(504, 'Inference deadline exceeded')
    return HTTPException(503, 'Inference service unavailable')


async def _open(service, token, request_key, body):
    iterator = service.stream(token, request_key, body)
    try:
        return iterator, await anext(iterator)
    except BaseException:
        await _close(iterator)
        raise


async def _disconnect(request):
    # The bounded body has already been consumed before this task is started.
    while True:
        if (await request.receive())['type'] == 'http.disconnect':
            return


async def _cancel(task):
    task.cancel()
    draining = asyncio.ensure_future(asyncio.gather(task, return_exceptions=True))
    interrupted = False
    while True:
        try:
            await asyncio.shield(draining)
            break
        except asyncio.CancelledError:
            if draining.cancelled():
                raise
            interrupted = True
    if interrupted:
        raise asyncio.CancelledError()


async def _release(admission, owner_id):
    """Settle an in-memory slot before propagating repeated ASGI cancellation."""
    draining = asyncio.create_task(admission.release(owner_id))
    interrupted = False
    while True:
        try:
            await asyncio.shield(draining)
            break
        except asyncio.CancelledError:
            if draining.cancelled():
                raise
            interrupted = True
    if interrupted:
        raise asyncio.CancelledError()


async def _open_while_connected(request, service, token, request_key, body):
    """Do not leave a pre-header provider stream running after disconnect."""
    work = asyncio.create_task(_open(service, token, request_key, body))
    disconnected = asyncio.create_task(_disconnect(request))
    try:
        done, _ = await asyncio.wait((work, disconnected), return_when=asyncio.FIRST_COMPLETED)
        if disconnected in done:
            raise HTTPException(499, 'Request disconnected')
        # Retain ownership until the last suspension before handing the
        # iterator to the response. Cancellation here must close work.result().
        await _cancel(disconnected)
        return work.result()
    except BaseException:
        try:
            await _cancel(work)
        finally:
            try:
                if not work.cancelled() and work.exception() is None:
                    iterator, _ = work.result()
                    await _close(iterator)
            finally:
                if not disconnected.done():
                    await _cancel(disconnected)
        raise


async def _stream(first, iterator):
    pending = None
    try:
        yield first
        while True:
            # ASGI owns cancellation of the consumer, not repeated cancellation
            # of the provider while it is acknowledging transport shutdown.
            pending = asyncio.create_task(anext(iterator))
            try:
                chunk = await asyncio.shield(pending)
            except StopAsyncIteration:
                break
            pending = None
            if type(chunk) is not bytes:
                raise RuntimeError('Inference service emitted an invalid stream chunk.')
            yield chunk
    except asyncio.CancelledError:
        raise
    except Exception:
        # Once headers have been sent, terminate the response rather than emit
        # a false completion event or any provider diagnostic.
        raise RuntimeError('Inference stream failed.') from None
    finally:
        if pending is not None:
            await _cancel(pending)


class _InferenceResponse(StreamingResponse):
    """Own cleanup even if sending response headers fails before iteration."""
    def __init__(self, first, iterator, admission, owner_id, **kwargs):
        super().__init__(_stream(first, iterator), **kwargs)
        self._provider_iterator = iterator
        self._admission = admission
        self._owner_id = owner_id

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            try:
                try:
                    await _close(self.body_iterator)
                finally:
                    await _close(self._provider_iterator)
            except asyncio.CancelledError:
                raise
            except Exception:
                raise RuntimeError('Inference stream cleanup failed.') from None
            finally:
                await _release(self._admission, self._owner_id)


def add_inference_routes(app, *, anthropic=None, gemini=None, openrouter=None, token_from_request):
    """Mount only the fixed provider routes selected by trusted startup code."""
    admission = _Admission()
    if anthropic is not None:
        _add(app, '/v1/messages', anthropic, token_from_request, admission, query=())
    if gemini is not None:
        _add(app, '/v1beta/models/gemini-3.8-flash:streamGenerateContent', gemini,
             token_from_request, admission, query=(('alt', 'sse'),))
    if openrouter is not None:
        _add(app, '/v1/chat/completions', openrouter, token_from_request, admission, query=())


def _add(app, path, service, token_from_request, admission, *, query):
    async def generate(request: Request):
        token = token_from_request(request)
        # Deliberately before checking headers that could encourage body reads.
        try:
            lease = await service._authorize(token)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise _http_error(error) from None
        try:
            admitted = await admission.acquire(lease.owner_id)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise _http_error(error) from None
        if not admitted:
            raise HTTPException(429, 'Inference capacity is unavailable')
        if tuple(request.query_params.multi_items()) != query:
            await _release(admission, lease.owner_id)
            raise HTTPException(400, 'Inference route parameters are unsupported')
        try:
            request_key = _request_key(request)
            body = await _json_body(request)
            iterator, first = await _open_while_connected(request, service, token, request_key, body)
            try:
                headers = dict(_STREAM_HEADERS)
                headers['X-Gateway-Request-Key'] = request_key
                return _InferenceResponse(first, iterator, admission, lease.owner_id, status_code=200,
                                          media_type='text/event-stream', headers=headers)
            except BaseException:
                await _close(iterator)
                raise
        except asyncio.CancelledError:
            await _release(admission, lease.owner_id)
            raise
        except HTTPException:
            await _release(admission, lease.owner_id)
            raise
        except Exception as error:
            await _release(admission, lease.owner_id)
            raise _http_error(error) from None

    app.post(path)(generate)
