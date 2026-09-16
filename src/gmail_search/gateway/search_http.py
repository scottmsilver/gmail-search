"""Optional capability-bound ranked search route for the private worker gateway."""
import asyncio

from fastapi import HTTPException, Request

from .retrieval_http import run_while_connected


def add_search_routes(app,service,token_from_request,read_json):
    @app.post('/v1/search')
    async def search(request: Request):
        token=token_from_request(request)
        await service.authorize(token)
        value=await read_json(request)
        allowed={'query','top_k','date_from','date_to','detail','max_matches'}
        if set(value)-allowed or 'query' not in value:
            raise HTTPException(400,'Invalid search request')
        try:
            return await run_while_connected(
                request, service.search(token, **value),
                before_publish=lambda: service.authorize(token),
                publication_deadline=asyncio.get_running_loop().time()+30,
            )
        except (ValueError,TypeError):
            raise HTTPException(400,'Invalid search request') from None
