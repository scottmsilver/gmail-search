"""Optional worker event submission; replay remains a trusted browser operation."""
import asyncio

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse


async def _settled_call(function, *args, **kwargs):
    """Drain the short SQLite operation even if HTTP cancellation repeats."""
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
        except Exception:
            break
    if cancelled:
        if not task.cancelled():
            task.exception()
        raise asyncio.CancelledError()
    return task.result()


def add_event_routes(app, events, token_from_request, json_body):
    @app.post('/v1/events')
    async def append(request: Request):
        token = token_from_request(request)
        await _settled_call(events.capabilities.authorize, token, audience='events', operation='append')
        if request.query_params:
            raise HTTPException(400, 'Event submission accepts no extra parameters')
        event = await json_body(request)
        # append rechecks the capability and fence inside the write transaction.
        # Guest usage events are display data; they never settle provider costs.
        sequence = await _settled_call(events.append, token, event)
        return JSONResponse({'seq': sequence})
