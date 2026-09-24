"""Optional private route for capability-bound structured mail filters."""

from fastapi import HTTPException, Request

from .retrieval_http import run_while_connected
from .tool_deadline import publication_deadline


def add_metadata_routes(app, service, token_from_request, read_json):
    @app.post('/v1/query-emails')
    async def query_emails(request: Request):
        token = token_from_request(request)
        await service.authorize(token)
        value = await read_json(request)
        allowed = {'sender', 'subject_contains', 'date_from', 'date_to', 'label', 'has_attachment', 'order_by', 'limit'}
        if set(value)-allowed:
            raise HTTPException(400, 'Invalid metadata request')
        try:
            return await run_while_connected(
                request, service.query_emails(token, **value),
                before_publish=lambda: service.authorize(token),
                publication_deadline=publication_deadline(),
            )
        except (ValueError, TypeError):
            raise HTTPException(400, 'Invalid metadata request') from None
