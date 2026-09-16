"""Complete staged browser onboarding router; deliberately not mounted by default.

A fully qualified release factory may mount these routes instead of the old auth
router. It must keep all mailbox APIs on owner-scoped gateways, wire revocation
workers, and provide a trusted idempotent provisioning callback. Current public
startup and its owner-only guard remain unchanged.
"""
import inspect
import asyncio
import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.routing import APIRoute
from starlette.responses import JSONResponse, RedirectResponse
from starlette.concurrency import run_in_threadpool

from . import public
from .gmail_consent import GmailConsent
from .identity_store import IdentityDenied, IdentityStore, VerifiedGoogleIdentity
from .invited_broker import BoundGmailBroker, BrokerUnavailable

GMAIL_STATE_COOKIE = '__Host-gms_gmail_state'


def _private(response):
    response.headers['Cache-Control'] = 'private, no-store'
    response.headers['Referrer-Policy'] = 'no-referrer'
    return response


class _PrivateRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()
        async def guarded(request):
            try:
                return _private(await handler(request))
            except IdentityDenied:
                return _private(JSONResponse({'detail': 'Authentication or account binding unavailable'}, status_code=401))
            except BrokerUnavailable:
                return _private(JSONResponse({'detail': 'Gmail connection service unavailable'}, status_code=503))
            except HTTPException as error:
                return _private(JSONResponse({'detail': error.detail}, status_code=error.status_code))
        return guarded


def create_invited_auth_router(identities: IdentityStore, consent: GmailConsent, broker: BoundGmailBroker, *, provision_account):
    """Provision callback(account, verified_claims) must return exactly True.

    It must create/verify the exact server-owned users.id and fixed database
    readers, reject pre-existing conflicting email/subject bindings, and never
    copy another mailbox. Import known old owner IDs explicitly before inviting;
    this router never guesses an existing mailbox owner from an email.
    """
    config = public.validate_public_auth_config()
    if config is None or config.secret == broker.signing_secret or consent.identities is not identities:
        raise RuntimeError('An explicit isolated public identity registration is required')
    router = APIRouter(route_class=_PrivateRoute)

    def session(request):
        token = request.cookies.get(public.SESSION_COOKIE, '')
        account = identities.read_session(token)
        if account is None:
            raise IdentityDenied()
        return token, account

    def query(request, expected):
        if sorted(key for key, _ in request.query_params.multi_items()) != sorted(expected):
            raise HTTPException(400, 'Unexpected request parameters')

    async def mutation(request):
        if request.headers.getlist('origin') != [config.origin]:
            raise HTTPException(403, 'Same-origin request required')
        query(request, [])
        # Reject unauthenticated callers before pulling any request bytes.
        session(request)
        body = bytearray()
        try:
            async with asyncio.timeout(2):
                async for chunk in request.stream():
                    if len(body) + len(chunk) > 1024:
                        raise HTTPException(413, 'Request body exceeds its byte limit')
                    body.extend(chunk)
        except TimeoutError:
            raise HTTPException(408, 'Request body deadline exceeded') from None
        if body.strip():
            try:
                if json.loads(body) != {}:
                    raise ValueError()
            except (ValueError, UnicodeDecodeError):
                raise HTTPException(400, 'Request does not accept account fields') from None

    def cookie(response, name, value, max_age):
        response.set_cookie(name, value, max_age=max_age, secure=True, httponly=True, samesite='lax', path='/')

    def clear(response, name):
        response.delete_cookie(name, secure=True, httponly=True, samesite='lax', path='/')

    @router.get('/api/auth/login')
    async def login(request: Request):
        query(request, ['return_url'] if 'return_url' in request.query_params else [])
        return public.start_login(request.query_params.get('return_url', '/'))

    @router.get('/api/auth/callback')
    async def callback(request: Request):
        query(request, ['silver_oauth'])
        claims, destination = public.consume_handoff(request.query_params['silver_oauth'], request.cookies.get(public.NONCE_COOKIE, ''))
        verified = VerifiedGoogleIdentity(claims['email'], claims['sub'], claims['email_verified'])
        account = identities.prepare_admission(verified)
        try:
            ready = await run_in_threadpool(provision_account, account, verified)
            if inspect.isawaitable(ready):
                ready = await ready
            if ready is not True:
                raise IdentityDenied()
        except Exception:
            raise HTTPException(503, 'Account provisioning unavailable') from None
        identities.mark_provisioned(account.owner_id, generation=account.generation)
        token = identities.admit(verified)
        response = RedirectResponse(destination, status_code=303)
        cookie(response, public.SESSION_COOKIE, token, public.SESSION_TTL)
        clear(response, public.NONCE_COOKIE)
        return response

    @router.get('/api/auth/me')
    async def me(request: Request):
        query(request, [])
        _, account = session(request)
        return JSONResponse({'multi_tenant': True, 'user': {'id': account.owner_id, 'email': account.email}})

    @router.post('/api/auth/logout')
    async def logout(request: Request):
        await mutation(request)
        token, _ = session(request)
        identities.revoke_session(token)
        response = JSONResponse({'signed_out': True})
        clear(response, public.SESSION_COOKIE)
        clear(response, GMAIL_STATE_COOKIE)
        return response

    @router.post('/api/auth/connect-gmail')
    async def connect(request: Request):
        await mutation(request)
        token, _ = session(request)
        state = consent.begin(token)
        destination = await run_in_threadpool(broker.start_consent, state)
        response = RedirectResponse(destination, status_code=303)
        cookie(response, GMAIL_STATE_COOKIE, state.secret, 600)
        return response

    @router.get('/api/auth/gmail-callback')
    async def gmail_callback(request: Request):
        query(request, ['gmail_consent'])
        token, _ = session(request)
        grant = consent.consume_broker_handoff(request.cookies.get(GMAIL_STATE_COOKIE, ''), session=token,
            token=request.query_params['gmail_consent'], broker_origin=broker.origin, signing_secret=broker.signing_secret)
        consent.validate_grant(grant)
        response = RedirectResponse('/settings', status_code=303)
        clear(response, GMAIL_STATE_COOKIE)
        return response

    @router.get('/api/auth/gmail-status')
    async def gmail_status(request: Request):
        query(request, [])
        token, _ = session(request)
        return JSONResponse({'multi_tenant': True, 'connect_method': 'POST', **consent.connection_status(token)})

    @router.post('/api/auth/disconnect-gmail')
    async def disconnect(request: Request):
        await mutation(request)
        token, _ = session(request)
        cleanup = consent.disconnect(token)
        if cleanup is not None:
            await run_in_threadpool(broker.disconnect, cleanup)
            consent.complete_cleanup(cleanup)
        response = JSONResponse({'connected': False})
        clear(response, GMAIL_STATE_COOKIE)
        return response

    return router
