"""Optional invited-browser artifact and event reads; no legacy owner fallback.

A qualified release may mount this router beside invited auth. It is not mounted
by existing public startup. IDs select objects only within the authenticated
owner and conversation. Worker capabilities never authenticate browser reads.
"""
import re

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from fastapi.routing import APIRoute

from gmail_search.gateway.event_http import _settled_call
from .identity_store import IdentityDenied
from .public import SESSION_COOKIE


class _ResultRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()
        async def guarded(request):
            try:
                response = await handler(request)
            except IdentityDenied:
                response = JSONResponse({'detail':'Sign-in required'},status_code=401)
            except PermissionError:
                response = JSONResponse({'detail':'Result unavailable'},status_code=404)
            except HTTPException as error:
                response = JSONResponse({'detail':error.detail},status_code=error.status_code)
            except (OSError, RuntimeError):
                response = JSONResponse({'detail':'Result service unavailable'},status_code=503)
            response.headers['Cache-Control'] = 'private, no-store'
            response.headers['Referrer-Policy'] = 'no-referrer'
            response.headers['X-Content-Type-Options'] = 'nosniff'
            return response
        return guarded


def create_result_router(identities, artifacts, events):
    if artifacts.registry is not events.registry:
        raise ValueError('Result stores must share the trusted run registry')
    router = APIRouter(route_class=_ResultRoute)

    async def session(request):
        if request.headers.get('authorization') or request.headers.get('x-user-id'):
            raise IdentityDenied()
        token = request.cookies.get(SESSION_COOKIE, '')
        account = await _settled_call(identities.read_session,token)
        if account is None:
            raise IdentityDenied()
        return account

    def parameters(request, allowed):
        pairs = list(request.query_params.multi_items())
        values = dict(pairs)
        if len(values) != len(pairs) or set(values) - allowed:
            raise HTTPException(400,'Invalid result parameters')
        conversation = values.get('conversation_id')
        if type(conversation) is not str or not re.fullmatch(r'[A-Za-z0-9_-]{1,256}',conversation):
            raise HTTPException(400,'Conversation required')
        return values

    async def recheck(request, account):
        current = await session(request)
        if current.owner_id != account.owner_id or current.generation != account.generation:
            raise IdentityDenied()

    @router.get('/api/agent-artifacts/{artifact_id}')
    async def artifact(artifact_id: str, request: Request):
        account = await session(request)
        params = parameters(request, {'conversation_id'})
        def read():
            item = artifacts.metadata(account.owner_id,params['conversation_id'],artifact_id)
            data = artifacts.read(account.owner_id,params['conversation_id'],artifact_id)
            return item,data
        item,data = await _settled_call(read)
        await recheck(request,account)
        return Response(data,headers=item.download_headers)

    @router.get('/api/agent-events/{run_id}')
    async def replay(run_id: str, request: Request):
        account = await session(request)
        params = parameters(request, {'conversation_id','after','limit'})
        after, limit = params.get('after','0'), params.get('limit','100')
        if (not re.fullmatch(r'[0-9]{1,18}',after) or not re.fullmatch(r'[0-9]{1,3}',limit)
                or not 1 <= int(limit) <= 100):
            raise HTTPException(400,'Invalid replay cursor or limit')
        result = await _settled_call(events.read,account.owner_id,params['conversation_id'],run_id,
                                     after=int(after),limit=int(limit))
        await recheck(request,account)
        return JSONResponse({'events':result,'next_cursor':result[-1]['seq'] if result else int(after)})

    return router
