"""Public route policy never reaches protected handlers without authorization."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

@pytest.fixture(autouse=True)
def _isolated_pg_schema():
    yield None

@pytest.fixture
def public_env(monkeypatch):
    for k,v in dict(GMAIL_MULTI_TENANT='1',GMS_PUBLIC_ORIGIN='https://gms.example',GMS_PUBLIC_ALLOWED_EMAILS='owner@example.com',GMS_IDENTITY_BROKER_URL='https://identity.example',GMS_IDENTITY_HANDOFF_SECRET='i'*40,GMS_SESSION_SECRET='s'*40).items(): monkeypatch.setenv(k,v)


def make_client(monkeypatch):
    from gmail_search.auth.boundary import PublicBoundaryMiddleware
    app=FastAPI()
    app.add_middleware(PublicBoundaryMiddleware)
    @app.api_route('/{path:path}',methods=['GET','POST','PUT','DELETE'])
    async def handler(path:str): return {'handler':True}
    return TestClient(app,base_url='https://gms.example')

@pytest.mark.parametrize('path',['/api/sql','/api/sql_schema','/api/admin/users','/api/jobs/backfill','/api/conversations/abc/debug','/api/conversations/abc/workspace/tree','/docs','/openapi.json'])
def test_public_denies_unlisted_routes(public_env,monkeypatch,path):
    assert make_client(monkeypatch).get(path).status_code==404

@pytest.mark.parametrize('path',['/api/search','/api/thread/abc','/api/conversations','/api/agent/analyze/abc/events'])
def test_public_auth_before_handler(public_env,monkeypatch,path):
    assert make_client(monkeypatch).get(path).status_code==401


def test_public_handoff_route_allows_anonymous(public_env,monkeypatch):
    assert make_client(monkeypatch).get('/api/auth/callback').status_code==200


def test_public_rejects_oversize_before_handler(public_env,monkeypatch):
    from gmail_search.auth import boundary
    monkeypatch.setattr(boundary,'request_user',lambda request:'owner')
    r=make_client(monkeypatch).put('/api/conversations/abc',content=b'x'*(boundary.MAX_BODY+1))
    assert r.status_code==413


def test_private_boundary_is_inactive(monkeypatch):
    monkeypatch.delenv('GMS_PUBLIC_ORIGIN',raising=False)
    assert make_client(monkeypatch).get('/api/admin/users').status_code==200


def test_anonymous_flood_does_not_exhaust_signed_in_capacity(public_env,monkeypatch):
    from gmail_search.auth import boundary
    c=make_client(monkeypatch)
    for _ in range(125): c.get('/api/auth/login')
    assert c.get('/api/auth/login').status_code==429
    monkeypatch.setattr(boundary,'request_user',lambda req:'owner')
    assert c.get('/api/search').status_code==200


# ── battles ──────────────────────────────────────────────────────────────────

def test_battle_endpoints_authenticate_rather_than_404(public_env, monkeypatch):
    """Battles are available to owners the server reports as capable.

    This is the server-side twin of `web/lib/publicBoundary.ts`. Both are
    default-deny allowlists and both had to learn about battles: the browser one
    404'd the vote, and fixing only that moved the 404 here, because POST was
    permitted for exactly `/api/auth/logout` and `/api/agent/analyze`.

    Reaching the endpoint is not permission to battle -- the chat route still
    gates that on capability, and both endpoints remain per-owner scoped. What
    must change is the failure: authentication, not "no such route".
    """
    client = make_client(monkeypatch)
    assert client.post('/api/battle/vote', json={}, headers={'origin': 'https://gms.example'}).status_code == 401
    assert client.get('/api/battle/stats').status_code == 401


@pytest.mark.parametrize('method,path', [
    ('get', '/api/battle/vote'),
    ('post', '/api/battle/stats'),
])
def test_battle_endpoints_still_refuse_the_wrong_method(public_env, monkeypatch, method, path):
    client = make_client(monkeypatch)
    call = getattr(client, method)
    response = call(path, **({'json': {}, 'headers': {'origin': 'https://gms.example'}} if method == 'post' else {}))
    assert response.status_code == 404
