"""Dedicated invited-browser service composition.

Construct with trusted services, then serve on a private loopback listener behind
Next.js. Run one API process per Registry. This module never imports the legacy
server, installs a database schema, or provisions users during startup.
"""
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import Headers
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import JSONResponse

from .auth.conversation_routes import create_conversation_router
from .auth.invited_routes import create_invited_auth_router
from .auth.mail_routes import create_mail_router
from .auth.public import validate_public_auth_config
from .auth.result_routes import create_result_router
from .auth.run_routes import create_run_router


class InvitedBoundary:
    def __init__(self, app, *, origin):
        self.app, self.origin = app, origin

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        headers = Headers(scope=scope)
        async def private_send(message):
            if message['type'] == 'http.response.start':
                message['headers'] = [(k,v) for k,v in message.get('headers', [])
                    if k.lower() not in (b'cache-control', b'referrer-policy', b'x-content-type-options')]
                message['headers'].extend([(b'cache-control', b'private, no-store'),
                    (b'referrer-policy', b'no-referrer'), (b'x-content-type-options', b'nosniff')])
            await send(message)
        if headers.get('authorization') or headers.get('x-user-id'):
            return await JSONResponse({'detail':'Sign-in required'}, status_code=401)(scope, receive, private_send)
        if scope['method'] not in ('GET', 'HEAD', 'OPTIONS') and headers.getlist('origin') != [self.origin]:
            return await JSONResponse({'detail':'Same-origin request required'}, status_code=403)(scope, receive, private_send)
        await self.app(scope, receive, private_send)


def create_invited_app(*, identities, consent, broker, provision_account, runs, conversations, artifacts, mail=None,
                       startup_prepared=False, runtimes=None):
    """Assemble browser routes using a single trusted identity/registry.

    Only the owning launcher may set startup_prepared, after recovery and Gmail
    cleanup have completed before either of its listeners is exposed.
    """
    if type(startup_prepared) is not bool:
        raise ValueError('Trusted startup preparation flag required')
    config = validate_public_auth_config()
    if config is None or consent.identities is not identities or artifacts.registry is not runs.registry:
        raise ValueError('Explicit invited identity and shared result registry required')
    # `runtimes`: the guest runtimes this deployment can launch; the picker and
    # the run route offer only those.
    auth = create_invited_auth_router(identities, consent, broker, provision_account=provision_account, runtimes=runtimes)

    @asynccontextmanager
    async def lifespan(app):
        try:
            if not startup_prepared:
                await runs.recover()
                await run_in_threadpool(broker.drain_cleanup, consent)
            yield
        finally:
            await runs.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.include_router(auth)
    if mail is not None:
        app.include_router(create_mail_router(identities, mail))
    app.include_router(create_conversation_router(identities, conversations, origin=config.origin))
    app.include_router(create_run_router(identities, runs, origin=config.origin, claim_conversation=conversations.claim,
                                         runtimes=runtimes))
    app.include_router(create_result_router(identities, artifacts, runs.events))
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=[urlsplit(config.origin).hostname, '127.0.0.1', 'localhost'])
    app.add_middleware(InvitedBoundary, origin=config.origin)
    return app
