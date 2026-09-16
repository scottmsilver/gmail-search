"""Run-authenticated attachment parsing; output bytes are never host-decoded."""
import base64

from fastapi import HTTPException, Request

from .retrieval_http import run_while_connected


def add_attachment_routes(app, service, token_from_request, read_json):
    async def parse(token, options):
        result = await service.parse(token, **options)
        return {'text': result.text, 'truncated': result.truncated,
                'pages': [{'number': page.number, 'width': page.width, 'height': page.height,
                           'png': base64.b64encode(page.png).decode('ascii')} for page in result.pages]}

    @app.post('/v1/attachment/parse')
    async def attachment(request: Request):
        token = token_from_request(request)
        await service._authorize(token)
        value = await read_json(request)
        if (set(value) - {'attachment_id', 'dpi', 'pages'} or 'attachment_id' not in value
                or ('pages' in value and (type(value['pages']) is not list or len(value['pages']) > 8))):
            raise HTTPException(400, 'Invalid attachment request')
        if 'pages' in value:
            value['pages'] = tuple(value['pages'])
        return await run_while_connected(request, parse(token, value),
                                         before_publish=lambda: service._authorize(token))
