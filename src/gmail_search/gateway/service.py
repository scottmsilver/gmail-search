"""Run-scoped analytical adapter; no caller-selectable owner or database login.

The embedding gateway must expose only this adapter to agent traffic. SQLite
authorization runs off the event loop. Both token revocation and caller
cancellation propagate to QueryGateway, which discards its database connection.
"""
import asyncio
from collections.abc import Mapping

from .capabilities import Capabilities
from .database import QueryGateway, QueryResult
from .schema import ANALYTICAL_SCHEMA


async def _drain(awaitable):
    """Keep registry and database work owned through repeated cancellation."""
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


class RunQueryService:
    def __init__(self, capabilities: Capabilities, gateway: QueryGateway):
        self.capabilities = capabilities
        self.gateway = gateway

    async def _authorize(self, token, operation):
        return await _drain(asyncio.to_thread(self.capabilities.authorize, token, audience='sql', operation=operation))

    async def _watch(self, token):
        while True:
            await asyncio.sleep(.1)
            await self._authorize(token, 'query')

    async def query(self, token, query: str) -> QueryResult:
        lease = await self._authorize(token, 'query')
        run = asyncio.create_task(self.gateway.query(lease.owner_id, query))
        watch = asyncio.create_task(self._watch(token))
        try:
            done, _ = await asyncio.wait((run, watch), return_when=asyncio.FIRST_COMPLETED)
            if watch in done:
                await watch
            result = await run
        finally:
            for task in (run, watch):
                if not task.done() and not task.cancelling():
                    task.cancel()
            # Do not return revoked results or abandon a live database operation.
            await _drain(asyncio.gather(run, watch, return_exceptions=True))
        # Watcher cleanup can outlive the query. Reauthorize after all owned
        # work drains, with no later suspension before publishing the result.
        await self._authorize(token, 'query')
        return result

    async def schema(self, token) -> Mapping[str, Mapping[str, str]]:
        await self._authorize(token, 'schema')
        return ANALYTICAL_SCHEMA
