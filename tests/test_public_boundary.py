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
