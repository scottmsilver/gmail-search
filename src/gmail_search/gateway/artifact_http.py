"""Optional worker-only artifact byte routes; public browser routes are separate."""
import asyncio

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, Response


def add_artifact_routes(app, store, token_from_request):
    @app.post('/v1/artifacts')
    async def upload(request: Request):
        token = token_from_request(request)
        await asyncio.to_thread(store._authorize, token)
        params = list(request.query_params.multi_items())
        if len(params) != 1 or params[0][0] != 'filename':
            raise HTTPException(400, 'One artifact filename is required')
        if request.headers.get('content-type', '').split(';', 1)[0] != 'application/octet-stream':
            raise HTTPException(400, 'Raw artifact bytes required')
        item = await store.publish(token, request.stream(), filename=params[0][1])
        return JSONResponse({'id':item.id, 'filename':item.filename, 'size':item.size}, status_code=201)

    @app.get('/v1/artifacts/{artifact_id}')
    async def download(artifact_id: str, request: Request):
        if request.query_params:
            raise HTTPException(400, 'Artifact lookup accepts no extra parameters')
        token = token_from_request(request)
        def read():
            lease = store.capabilities.authorize(token, audience='artifact', operation='artifact.read')
            item = store.metadata(lease.owner_id, lease.conversation_id, artifact_id)
            data = store.read(lease.owner_id, lease.conversation_id, artifact_id)
            store.capabilities.authorize(token, audience='artifact', operation='artifact.read')
            return item, data
        item, data = await asyncio.to_thread(read)
        return Response(data, headers=item.download_headers)
