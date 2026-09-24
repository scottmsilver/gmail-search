"""Worker-facing SQL routes; controller/administration APIs are never mounted.

This factory is not wired to the public app. Deployment must bind it to the
qualified worker relay, not expose it as an unauthenticated Internet service.
"""
import asyncio
from decimal import Decimal
import json
import logging
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .analytics import QueryRejected, TextScanRejected
from .service import RunQueryService

timing_logger = logging.getLogger('gmail_search.gateway.timing')
# Calls at or above this are logged as warnings (the tool deadline is 5 s).
SLOW_CALL_MS = 2000

_HEADERS = {'Cache-Control': 'private, no-store', 'X-Content-Type-Options': 'nosniff'}
_MAX_BODY = 32768


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def _token(request):
    authorization = request.headers.getlist('authorization')
    if request.headers.get('x-user-id') or request.headers.get('cookie'):
        raise HTTPException(400, 'Only run capability authentication is supported')
    if len(authorization) != 1 or not authorization[0].startswith('Bearer '):
        raise HTTPException(401, 'Run capability required')
    return authorization[0][7:]


async def _json_body(request, limit=_MAX_BODY):
    size, chunks = 0, []
    try:
        async with asyncio.timeout(3):
            async for chunk in request.stream():
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, 'Query request exceeds its byte limit')
                chunks.append(chunk)
    except TimeoutError:
        raise HTTPException(408, 'Query request timed out') from None
    if request.headers.get('content-type', '').split(';', 1)[0] != 'application/json':
        raise HTTPException(400, 'JSON query required')
    try:
        value = json.loads(b''.join(chunks), object_pairs_hook=_object)
        if type(value) is not dict:
            raise ValueError()
        return value
    except (ValueError, RecursionError, UnicodeError):
        raise HTTPException(400, 'Invalid analytical query request') from None


async def _body(request):
    value = await _json_body(request)
    if set(value) != {'query'} or type(value['query']) is not str:
        raise HTTPException(400, 'Invalid analytical query request')
    return value['query']


def _log_timing(request, response, started):
    """Route, status and wall time only; never arguments or results."""
    elapsed_ms = (time.perf_counter() - started) * 1000
    log = timing_logger.warning if elapsed_ms >= SLOW_CALL_MS else timing_logger.info
    log('gateway %s %s %d %.0fms', request.method, request.url.path, response.status_code, elapsed_ms)


def create_gateway_app(service: RunQueryService, *, artifacts=None, retrieval=None, search=None, facts=None, metadata=None, attachment_reads=None, raw_attachments=None, attachments=None, anthropic=None, gemini=None, openrouter=None, events=None, judge=None) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware('http')
    async def secure_errors(request, call_next):
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except PermissionError:
            response = JSONResponse({'detail':'Run access denied'}, status_code=403)
        except TextScanRejected as error:
            response = JSONResponse({'detail':str(error)}, status_code=400)
        except QueryRejected:
            response = JSONResponse({'detail':'Unsupported analytical query'}, status_code=400)
        except TimeoutError:
            response = JSONResponse({'detail':'Query deadline exceeded'}, status_code=504)
        except RuntimeError:
            response = JSONResponse({'detail':'Query service unavailable'}, status_code=503)
        for name, value in _HEADERS.items():
            response.headers[name] = value
        _log_timing(request, response, started)
        return response

    @app.get('/v1/schema')
    async def schema(request: Request):
        value = await service.schema(_token(request))
        return JSONResponse({table: dict(columns) for table, columns in value.items()})

    @app.post('/v1/sql')
    async def query(request: Request):
        token = _token(request)
        # Authenticate before accepting or parsing a potentially chunked body.
        await service._authorize(token, 'query')
        text = await _body(request)
        from .retrieval_http import run_while_connected
        async def execute():
            result = await service.query(token, text)
            # Decimal strings preserve aggregate precision across JSON clients.
            rows = [[str(value) if isinstance(value, Decimal) else value for value in row] for row in result.rows]
            return {'columns': result.columns, 'rows': rows, 'complete': result.complete, 'scope': 'query_result'}
        return await run_while_connected(request, execute(),
                                         before_publish=lambda: service._authorize(token, 'query'))

    if artifacts is not None:
        from .artifact_http import add_artifact_routes
        add_artifact_routes(app, artifacts, _token)

    if retrieval is not None:
        from .retrieval_http import add_retrieval_routes
        add_retrieval_routes(app, retrieval, _token, _json_body)

    if search is not None:
        from .search_http import add_search_routes
        add_search_routes(app, search, _token, _json_body)

    if facts is not None:
        from .facts_http import add_facts_routes
        add_facts_routes(app, facts, _token, _json_body)
    if judge is not None:
        from .judge_http import add_judge_routes
        add_judge_routes(app, judge, _token, _json_body)

    if metadata is not None:
        from .metadata_http import add_metadata_routes
        add_metadata_routes(app, metadata, _token, _json_body)

    if attachment_reads is not None:
        from .attachment_read_http import add_attachment_read_routes
        add_attachment_read_routes(app, attachment_reads, _token, _json_body)

    if attachments is not None:
        from .attachment_http import add_attachment_routes
        add_attachment_routes(app, attachments, _token, _json_body)

    if anthropic is not None or gemini is not None or openrouter is not None:
        from .inference_http import add_inference_routes
        add_inference_routes(app, anthropic=anthropic, gemini=gemini, openrouter=openrouter, token_from_request=_token)

    if events is not None:
        from .event_http import add_event_routes
        # An event may be as large as the event store accepts; the guest bounds
        # tool displays to that, not to the query body limit.
        add_event_routes(app, events, _token,
                         lambda request: _json_body(request, limit=events.max_event_bytes))

    # Raw payload admission must enclose application-boundary sends, outside
    # BaseHTTPMiddleware's buffered response channel. Keep this registration last.
    if raw_attachments is not None:
        from .attachment_raw_http import add_raw_attachment_middleware
        add_raw_attachment_middleware(app, raw_attachments)

    return app
