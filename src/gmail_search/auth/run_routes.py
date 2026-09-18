"""Invited browser chat, replay and Stop using the full worker controller.

Mount only in the invited app composition, never beside the legacy analyze
routes. A disconnected stream leaves the owned run available for replay; Stop
uses the explicit cancel route. Conversation claiming must atomically verify
ownership in the same store used by the trusted answer persistence callback.
"""
import asyncio
import json
import re
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from gmail_search.gateway.browser_runs import event_frame
from gmail_search.gateway.event_http import _settled_call
from .identity_store import IdentityDenied
from .public import SESSION_COOKIE
from .result_routes import _ResultRoute


# The browser's deep-backend names, mapped to the guest runtime each one boots.
BACKENDS = {None: 'pi', 'pi': 'pi', 'claude_code': 'claude'}


def _model_hint(value):
    """The picker always sends a model; the gateway pins its own, so accept and drop it."""
    return value is None or (type(value) is str and 0 < len(value) <= 128)


def _frame(kind, payload):
    return f'event: {kind}\ndata: '+json.dumps(payload,ensure_ascii=False)+'\n\n'


def _conversation(value):
    if type(value) is not str or not re.fullmatch(r'[A-Za-z0-9_-]{1,256}',value):
        raise HTTPException(400,'Conversation required')
    return value


def _object(pairs):
    result = {}
    for key,value in pairs:
        if key in result:
            raise ValueError()
        result[key] = value
    return result


def create_run_router(identities, runs, *, origin, claim_conversation):
    parsed = urlsplit(origin)
    if (parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password
            or parsed.path or parsed.query or parsed.fragment or not callable(claim_conversation)):
        raise ValueError('Fixed HTTPS origin and trusted conversation claim required')
    router = APIRouter(route_class=_ResultRoute)

    async def session(request):
        if request.headers.get('authorization') or request.headers.get('x-user-id'):
            raise IdentityDenied()
        account = await _settled_call(identities.read_session,request.cookies.get(SESSION_COOKIE,''))
        if account is None:
            raise IdentityDenied()
        return account

    async def recheck(request, account):
        current = await session(request)
        if (current.owner_id,current.generation) != (account.owner_id,account.generation):
            raise IdentityDenied()

    async def body(request, account):
        if request.headers.getlist('origin') != [origin]:
            raise HTTPException(403,'Same-origin request required')
        if request.query_params:
            raise HTTPException(400,'Unexpected request parameters')
        if request.headers.get('content-type','').split(';')[0].strip().lower() != 'application/json':
            raise HTTPException(415,'JSON required')
        content = bytearray()
        try:
            async with asyncio.timeout(2):
                async for chunk in request.stream():
                    if len(content)+len(chunk) > 32768:
                        raise HTTPException(413,'Request exceeds its byte limit')
                    content.extend(chunk)
        except TimeoutError:
            raise HTTPException(408,'Request deadline exceeded') from None
        try:
            value = json.loads(content,object_pairs_hook=_object)
            if type(value) is not dict:
                raise ValueError()
        except (ValueError,UnicodeError,RecursionError):
            raise HTTPException(400,'Invalid request') from None
        await recheck(request,account)
        return value

    async def snapshot(request, account, conversation, run, after):
        result = await _settled_call(runs.snapshot,account.owner_id,conversation,run,after=after)
        await recheck(request,account)
        return result

    def response(request, account, conversation, run, initial, after, *, admitted=False):
        published = False
        async def stream():
            batch, cursor = initial, after
            try:
                await recheck(request,account)
                yield _frame('session',{'session_id':run,'conversation_id':conversation,'supports_cancel':True})
                while True:
                    for row in batch['events']:
                        await recheck(request,account)
                        cursor = row['seq']
                        yield event_frame(row)
                    # A terminal snapshot can still have more than one page.
                    if len(batch['events']) < 100:
                        if batch['state'] == 'completed':
                            await recheck(request,account)
                            yield _frame('final',{'agent':'agent','payload':{'text':batch['answer']}})
                            yield _frame('persist_ok',{'payload':{'session_id':run}})
                            return
                        if batch['state'] in ('failed','cancelled','stopping'):
                            reason = {'failed':'The run failed. Please try again.',
                                'cancelled':'Run stopped.',
                                'stopping':'Worker cleanup is pending. Please retry Stop.'}[batch['state']]
                            yield _frame('error',{'payload':{'message':reason,'state':batch['state']}})
                            return
                        if await request.is_disconnected():
                            return
                        await asyncio.sleep(runs.poll_seconds)
                    batch = await snapshot(request,account,conversation,run,cursor)
            except (IdentityDenied,PermissionError):
                # Headers may already be sent; close without any more private data.
                return
        class RunResponse(StreamingResponse):
            async def __call__(self, scope, receive, send):
                async def publish(message):
                    nonlocal published
                    await send(message)
                    if message['type']=='http.response.body' and message.get('body'):
                        published = True
                try:
                    await super().__call__(scope,receive,publish)
                finally:
                    # Also covers cancellation before the body iterator starts.
                    # Replay never owns the original run's admission cleanup.
                    if admitted and not published:
                        await runs.abandon_unpublished(run)
        return RunResponse(stream(),media_type='text/event-stream',headers={
            'X-Accel-Buffering':'no','X-GMS-Transcript-Owner':'server'})

    @router.post('/api/agent/analyze')
    async def analyze(request: Request):
        account = await session(request)
        value = await body(request,account)
        if (set(value)-{'question','conversation_id','backend','model'}
                or value.get('backend') not in BACKENDS or not _model_hint(value.get('model'))):
            raise HTTPException(400,'Unsupported run parameters')
        runtime = BACKENDS[value.get('backend')]
        conversation = _conversation(value.get('conversation_id'))
        question = value.get('question')
        try:
            if (type(question) is not str or not question.strip() or '\x00' in question
                    or len(question.encode('utf-8')) > 16384):
                raise ValueError()
        except (ValueError,UnicodeError):
            raise HTTPException(400,'Question must contain at most 16 KiB of text') from None
        claimed = await _settled_call(claim_conversation,account.owner_id,conversation)
        if claimed is not True:
            raise PermissionError()
        await recheck(request,account)
        run = await runs.start(account.owner_id,conversation,question,runtime)
        try:
            initial = await snapshot(request,account,conversation,run,0)
        except BaseException:
            # Admission happened but no run identifier reached this browser.
            await runs.abandon_unpublished(run)
            raise
        return response(request,account,conversation,run,initial,0,admitted=True)

    @router.get('/api/agent/analyze/{run}/events')
    async def replay(run: str, request: Request):
        account = await session(request)
        pairs = list(request.query_params.multi_items())
        params = dict(pairs)
        if len(params) != len(pairs) or set(params)-{'conversation_id','after'}:
            raise HTTPException(400,'Invalid replay parameters')
        conversation = _conversation(params.get('conversation_id'))
        after = params.get('after','0')
        if not re.fullmatch(r'[0-9]{1,18}',after):
            raise HTTPException(400,'Invalid replay cursor')
        initial = await snapshot(request,account,conversation,run,int(after))
        return response(request,account,conversation,run,initial,int(after))

    @router.post('/api/agent/analyze/{run}/cancel')
    async def cancel(run: str, request: Request):
        account = await session(request)
        value = await body(request,account)
        if set(value) != {'conversation_id'}:
            raise HTTPException(400,'Conversation required')
        conversation = _conversation(value['conversation_id'])
        await snapshot(request,account,conversation,run,0)
        try:
            await runs.cancel(account.owner_id,conversation,run)
        except PermissionError:
            result = await snapshot(request,account,conversation,run,0)
            if result['state'] != 'stopping':
                raise
        else:
            result = await snapshot(request,account,conversation,run,0)
        return JSONResponse({'state':result['state']},status_code=202 if result['state']=='stopping' else 200)

    return router
