"""Public backend admission policy; one process, no trust in forwarded identity."""
from __future__ import annotations

import asyncio
import re
import time
from collections import deque

from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse

from .public import public_enabled

MAX_BODY = 1024 * 1024
_PUBLIC = {'/healthz', '/api/auth/login', '/api/auth/callback', '/api/auth/whoami'}
_READ = {
    '/api/auth/me', '/api/auth/gmail-status', '/api/search', '/api/query', '/api/find_facts',
    '/api/inbox', '/api/priority-inbox', '/api/status', '/api/users/me/sync-status',
    '/api/thread_lookup', '/api/conversations', '/api/conversations/live',
}
_READ_PATTERN = re.compile(r'^/api/(?:thread/[^/]+|message/[^/]+|attachment/[^/]+(?:/(?:meta|raw|text))?|artifact/\d+|conversations/[^/]+|agent/analyze/[^/]+/events)$')
_CONVERSATION = re.compile(r'^/api/conversations/[^/]+$')


def request_user(request: Request) -> str:
    from .session import require_user_id
    # Authentication resolves public sessions against the current allowlist and DB.
    return require_user_id(request)


def allowed(path: str, method: str) -> bool:
    if method == 'GET':
        return path in _PUBLIC or path in _READ or bool(_READ_PATTERN.fullmatch(path))
    if method == 'POST':
        return path in {'/api/auth/logout', '/api/agent/analyze'}
    return method in {'PUT', 'DELETE'} and bool(_CONVERSATION.fullmatch(path))


class PublicBoundaryMiddleware:
    def __init__(self, app):
        self.app = app
        self.arrivals: dict[str, deque[float]] = {}
        self.anonymous_active = 0
        self.active = 0
        self.analysis: dict[str, int] = {}

    async def __call__(self, scope, receive, send):
        if not public_enabled() or scope['type'] != 'http':
            return await self.app(scope, receive, send)

        async def deny(status, detail):
            response = JSONResponse({'detail': detail}, status_code=status, headers={
                'Cache-Control': 'private, no-store', 'Referrer-Policy': 'no-referrer',
                'X-Content-Type-Options': 'nosniff',
            })
            await response(scope, receive, send)

        path, method = scope['path'], scope['method']
        if not allowed(path, method):
            return await deny(404, 'Not found')
        req = Request(scope)
        uid = None
        if path not in _PUBLIC:
            try:
                uid = await asyncio.to_thread(request_user, req)
            except HTTPException as exc:
                return await deny(exc.status_code, 'Authentication required')
            if not uid:
                return await deny(401, 'Authentication required')
        # Reserve authenticated capacity; rejected anonymous probes cannot starve
        # existing sessions or the internal retrieval calls needed by a chat.
        key = uid or '<anonymous>'
        now = time.monotonic()
        for expired in [key for key, queue in self.arrivals.items() if not queue or queue[-1] <= now - 60]:
            del self.arrivals[expired]
        if key not in self.arrivals and len(self.arrivals) >= 4096:
            return await deny(429, 'Request capacity reached')
        arrivals = self.arrivals.setdefault(key, deque())
        while arrivals and arrivals[0] <= now - 60:
            arrivals.popleft()
        if len(arrivals) >= (600 if uid else 120) or (self.active >= 32 if uid else self.anonymous_active >= 8):
            return await deny(429, 'Request capacity reached; retry shortly')
        arrivals.append(now)
        if uid:
            self.active += 1
        else:
            self.anonymous_active += 1
        analysis_user = None
        try:
            if path == '/api/agent/analyze':
                if sum(self.analysis.values()) >= 4 or self.analysis.get(uid, 0) >= 2:
                    return await deny(429, 'Chat capacity reached; wait for the current answer')
                self.analysis[uid] = self.analysis.get(uid, 0) + 1
                analysis_user = uid
            # Buffer with a hard byte cap even for chunked requests; never trust Content-Length.
            body = bytearray()
            if method in {'POST', 'PUT', 'DELETE'}:
                try:
                    async with asyncio.timeout(15):
                        while True:
                            event = await receive()
                            if event['type'] == 'http.disconnect':
                                return
                            body.extend(event.get('body', b''))
                            if len(body) > MAX_BODY:
                                return await deny(413, 'Request too large')
                            if not event.get('more_body', False):
                                break
                except TimeoutError:
                    return await deny(408, 'Request body timeout')
                sent = False
                async def buffered_receive():
                    nonlocal sent
                    if not sent:
                        sent = True
                        return {'type': 'http.request', 'body': bytes(body), 'more_body': False}
                    return await receive()
                await self.app(scope, buffered_receive, send)
            else:
                await self.app(scope, receive, send)
        finally:
            if uid:
                self.active -= 1
            else:
                self.anonymous_active -= 1
            if analysis_user is not None:
                self.analysis[analysis_user] -= 1
                if not self.analysis[analysis_user]:
                    del self.analysis[analysis_user]
