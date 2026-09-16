"""Invited browser citations; caller input never chooses a mailbox or SQL."""
import asyncio
import re

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response

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

    # Canonical decimal only: no leading zeros, no sign, no whitespace, and
    # inside int64 so the id cannot overflow the column it selects on. `0` is
    # excluded because attachment ids start at 1 — accepting it would make a
    # typo look like a real lookup rather than a rejected one.
    ATTACHMENT_ID = re.compile('[1-9][0-9]{0,18}')

    def _deadline():
        # Matches the bound `BrowserMail.thread` already uses for its manifest
        # read. The reader refuses a non-finite or absent deadline outright, so
        # this is required rather than a nicety.
        return asyncio.get_running_loop().time() + 10

    def attachment_id(raw):
        if not ATTACHMENT_ID.fullmatch(raw) or int(raw) > 9223372036854775807:
            raise HTTPException(400,'Invalid attachment')
        return int(raw)

    async def attachment_call(request, read, allowed=frozenset()):
        """Read one attachment for the session owner, then re-check the session.

        The re-check is the point. An attachment read can take long enough for a
        revocation to land while it is in flight, and returning the bytes anyway
        would serve private mail to a session that is no longer entitled to it.
        `check_active` lets the service abort mid-read; the re-check afterwards
        covers the window between the service finishing and us responding.
        """
        account = await session(request)
        parameters(request, set(allowed))

        async def active():
            current = await _settled_call(identities.read_session,
                                          request.cookies.get(SESSION_COOKIE,''))
            if current is None or (current.owner_id,current.generation) != (account.owner_id,account.generation):
                raise IdentityDenied()
            return True

        result = await read(account, active)
        await recheck(request, account)
        return result

    @router.get('/api/attachment/{raw_id}/meta')
    async def attachment_meta(raw_id: str, request: Request):
        identifier = attachment_id(raw_id)

        async def read(account, active):
            return await mail.attachment_metadata(account.owner_id, identifier,
                                                  deadline=_deadline(), check_active=active)

        return JSONResponse(await attachment_call(request, read))

    @router.get('/api/attachment/{raw_id}')
    async def attachment_download(raw_id: str, request: Request):
        identifier = attachment_id(raw_id)
        # No ranged reads. Partial responses would mean handing out bytes before
        # the post-read session re-check, and the reader has no ranged path.
        if request.headers.get('range') is not None:
            raise HTTPException(400,'Ranged attachment reads are not supported')

        async def read(account, active):
            return await mail.attachment_download(account.owner_id, identifier,
                                                  deadline=_deadline(), check_active=active)

        item = await attachment_call(request, read)
        return Response(item.data, headers=item.download_headers)

    return router
