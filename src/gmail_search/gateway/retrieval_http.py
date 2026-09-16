"""Optional run-authenticated typed retrieval routes."""
import asyncio

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse


async def _disconnect(request):
    # Called only after the bounded request body has been fully consumed.
    while True:
        if (await request.receive())['type'] == 'http.disconnect':
            return


async def run_while_connected(request, operation, *, before_publish=None, publication_deadline=None):
    """Drain owned transport work before checking permission to publish.

    Callers returning private results supply a fresh authorization callback.
    The optional absolute deadline includes transport cleanup. Neither check
    promises revocation can recall bytes already sent to a client.
    This boundary covers owned cleanup before response construction; ASGI
    middleware and socket sends run afterward and cannot be made atomic with
    capability revocation by this helper.
    """
    work = asyncio.create_task(operation)
    disconnected = asyncio.create_task(_disconnect(request))
    try:
        done, _ = await asyncio.wait((work, disconnected), return_when=asyncio.FIRST_COMPLETED)
        if disconnected in done:
            await disconnected
            lost_connection = True
            value = None
        else:
            lost_connection = False
            value = await work
    finally:
        async def cleanup():
            work.cancel()
            disconnected.cancel()
            await asyncio.gather(work, disconnected, return_exceptions=True)
        # A second HTTP-task cancellation must not re-cancel the service while
        # it is closing a database connection or stopping/reaping its VM.
        draining = asyncio.create_task(cleanup())
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
    if lost_connection:
        return JSONResponse({'detail': 'Request disconnected'}, status_code=499)
    if before_publish is not None:
        await before_publish()
    if publication_deadline is not None and asyncio.get_running_loop().time() >= publication_deadline:
        raise TimeoutError('Response deadline exceeded')
    return JSONResponse(value)


def add_retrieval_routes(app, service, token_from_request, read_json):
    @app.post('/v1/thread')
    async def thread(request: Request):
        token = token_from_request(request)
        await service.authorize(token)
        value = await read_json(request)
        allowed = {'thread_id', 'message_offset', 'message_limit', 'body_offset', 'body_limit', 'attachment_after_id', 'attachment_limit'}
        if set(value) - allowed or 'thread_id' not in value:
            raise HTTPException(400, 'Invalid thread request')
        return await run_while_connected(request, service.thread(token, **value),
                                         before_publish=lambda: service.authorize(token),
                                         publication_deadline=asyncio.get_running_loop().time()+30)
