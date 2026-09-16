"""Invited browser conversation CRUD on the owner-scoped transcript store."""
import asyncio
import json
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from gmail_search.gateway.event_http import _settled_call
from .identity_store import IdentityDenied
from .public import SESSION_COOKIE
from .result_routes import _ResultRoute
from .run_routes import _conversation, _object


def create_conversation_router(identities, conversations, *, origin):
    parsed = urlsplit(origin)
    if (parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password
            or parsed.path or parsed.query or parsed.fragment):
        raise ValueError('Fixed HTTPS origin required')
    router = APIRouter(route_class=_ResultRoute)

    async def session(request):
        if request.headers.get('authorization') or request.headers.get('x-user-id'):
            raise IdentityDenied()
        account = await _settled_call(identities.read_session, request.cookies.get(SESSION_COOKIE, ''))
        if account is None:
            raise IdentityDenied()
        return account

    async def recheck(request, account):
        current = await session(request)
        if (current.owner_id, current.generation) != (account.owner_id, account.generation):
            raise IdentityDenied()

    def parameters(request):
        if request.query_params:
            raise HTTPException(400, 'Unexpected request parameters')

    async def mutation(request, account, *, json_body):
        parameters(request)
        if request.headers.getlist('origin') != [origin]:
            raise HTTPException(403, 'Same-origin request required')
        if json_body and request.headers.get('content-type', '').split(';')[0].strip().lower() != 'application/json':
            raise HTTPException(415, 'JSON required')
        content = bytearray()
        try:
            async with asyncio.timeout(2):
                async for chunk in request.stream():
                    if len(content) + len(chunk) > (262144 if json_body else 0):
                        raise HTTPException(413, 'Request exceeds its byte limit')
                    content.extend(chunk)
        except TimeoutError:
            raise HTTPException(408, 'Request deadline exceeded') from None
        value = None
        if json_body:
            try:
                value = json.loads(content, object_pairs_hook=_object)
                if type(value) is not dict or set(value) - {'title', 'messages'}:
                    raise ValueError()
            except (ValueError, UnicodeError, RecursionError):
                raise HTTPException(400, 'Invalid conversation') from None
        await recheck(request, account)
        return value

    @router.get('/api/conversations')
    async def listing(request: Request):
        account = await session(request)
        parameters(request)
        result = await _settled_call(conversations.list, account.owner_id, limit=100)
        await recheck(request, account)
        return JSONResponse({'conversations': result})

    @router.get('/api/conversations/{conversation}')
    async def get(conversation: str, request: Request):
        account = await session(request)
        parameters(request)
        result = await _settled_call(conversations.get, account.owner_id, _conversation(conversation))
        await recheck(request, account)
        if result is None:
            raise PermissionError()
        return JSONResponse(result)

    @router.put('/api/conversations/{conversation}')
    async def save(conversation: str, request: Request):
        account = await session(request)
        conversation = _conversation(conversation)
        value = await mutation(request, account, json_body=True)
        if await _settled_call(conversations.save, account.owner_id, conversation, value) is not True:
            raise PermissionError()
        await recheck(request, account)
        return JSONResponse({'ok': True})

    @router.delete('/api/conversations/{conversation}')
    async def delete(conversation: str, request: Request):
        account = await session(request)
        conversation = _conversation(conversation)
        await mutation(request, account, json_body=False)
        if await _settled_call(conversations.delete, account.owner_id, conversation) is not True:
            raise PermissionError()
        await recheck(request, account)
        return JSONResponse({'ok': True})

    return router
