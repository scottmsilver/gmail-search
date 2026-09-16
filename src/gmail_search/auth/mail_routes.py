"""Invited browser citations; caller input never chooses a mailbox or SQL."""
import re

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from gmail_search.gateway.browser_mail import AmbiguousThread
from gmail_search.gateway.event_http import _settled_call
from .identity_store import IdentityDenied
from .public import SESSION_COOKIE
from .result_routes import _ResultRoute


def create_mail_router(identities, mail):
    router=APIRouter(route_class=_ResultRoute)

    async def session(request):
        if request.headers.get('authorization') or request.headers.get('x-user-id'):
            raise IdentityDenied()
        account=await _settled_call(identities.read_session,request.cookies.get(SESSION_COOKIE,''))
        if account is None:
            raise IdentityDenied()
        return account

    async def recheck(request,account):
        current=await session(request)
        if (current.owner_id,current.generation)!=(account.owner_id,account.generation):
            raise IdentityDenied()

    def parameters(request,allowed):
        pairs=list(request.query_params.multi_items());value=dict(pairs)
        if len(pairs)!=len(value) or set(value)-allowed:
            raise HTTPException(400,'Invalid citation parameters')
        return value

    @router.get('/api/thread_lookup')
    async def lookup(request: Request):
        account=await session(request)
        value=parameters(request,{'cite_ref'})
        try:
            result=await mail.lookup(account.owner_id,value.get('cite_ref'))
        except AmbiguousThread:
            raise HTTPException(409,'Citation is ambiguous') from None
        except ValueError:
            raise HTTPException(400,'Invalid citation') from None
        await recheck(request,account)
        return JSONResponse(result)

    @router.get('/api/thread/{thread_id}')
    async def thread(thread_id: str, request: Request):
        account=await session(request)
        value=parameters(request,{'message_offset','body_offset'})
        offsets={key:value.get(key,'0') for key in ('message_offset','body_offset')}
        if any(not re.fullmatch('[0-9]{1,10}',item) for item in offsets.values()):
            raise HTTPException(400,'Invalid thread page')
        try:
            result=await mail.thread(account.owner_id,thread_id,**{key:int(item) for key,item in offsets.items()})
        except ValueError:
            raise HTTPException(400,'Invalid thread page') from None
        await recheck(request,account)
        return JSONResponse(result)

    return router
