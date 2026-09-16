"""Optional facts route for the private capability-bound worker gateway."""
import asyncio

from fastapi import HTTPException, Request

from .retrieval_http import run_while_connected


def add_facts_routes(app, service, token_from_request, read_json):
    @app.post('/v1/find-facts')
    async def find_facts(request: Request):
        token = token_from_request(request)
        await service._authorize(token)
        value = await read_json(request)
        if set(value)-{'query', 'exhaustive', 'k'} or 'query' not in value:
            raise HTTPException(400, 'Invalid facts request')
        try:
            return await run_while_connected(
                request, service.find_facts(token, **value),
                before_publish=lambda: service._authorize(token),
                publication_deadline=asyncio.get_running_loop().time()+30,
            )
        except (ValueError, TypeError):
            raise HTTPException(400, 'Invalid facts request') from None
