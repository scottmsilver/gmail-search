"""Optional private metadata, stored-text and attachment inventory routes."""
import asyncio

from fastapi import HTTPException, Request

from .retrieval_http import run_while_connected


def add_attachment_read_routes(app, service, token_from_request, read_json):
    async def read(request, operation, method, allowed, required):
        token = token_from_request(request)
        await service.authorize(token, operation)
        value = await read_json(request)
        if set(value)-allowed or required not in value:
            raise HTTPException(400, 'Invalid attachment read request')
        try:
            return await run_while_connected(
                request, method(token, **value),
                before_publish=lambda: service.authorize(token, operation),
                publication_deadline=asyncio.get_running_loop().time()+30,
            )
        except (ValueError, TypeError):
            raise HTTPException(400, 'Invalid attachment read request') from None

    @app.post('/v1/attachment/meta')
    async def metadata(request: Request):
        return await read(request, 'meta', service.describe, {'attachment_id'}, 'attachment_id')

    @app.post('/v1/attachment/text')
    async def text(request: Request):
        return await read(request, 'text', service.text_page, {'attachment_id', 'offset', 'limit'}, 'attachment_id')

    @app.post('/v1/attachment/list')
    async def manifest(request: Request):
        return await read(request, 'meta', service.list_for_thread,
                          {'thread_id', 'after_attachment_id', 'limit'}, 'thread_id')
