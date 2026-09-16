"""Worker byte transfer routes bind all objects to a live run capability."""
import pytest
from fastapi.testclient import TestClient

from gmail_search.gateway.artifacts import ArtifactStore
from gmail_search.gateway.capabilities import Capabilities
from gmail_search.gateway.http import create_gateway_app
from gmail_search.gateway.registry import Registry


@pytest.fixture
def client(tmp_path):
    registry = Registry(tmp_path/'registry', is_active=lambda owner: owner in {'alice','bob'})
    caps = Capabilities(registry)
    root = tmp_path/'objects'
    root.mkdir(mode=0o700)
    store = ArtifactStore(root, caps, max_object_bytes=8)
    tokens = {}
    for owner in ('alice','bob'):
        run = registry.start_run(owner, 'conversation', request_key='request')
        tokens[owner] = caps.issue(run.run_id, audience='artifact', operations={'artifact.commit','artifact.read'}).secret
    with TestClient(create_gateway_app(None, artifacts=store)) as http:
        yield http, tokens, caps


def auth(token):
    return {'Authorization':'Bearer '+token, 'Content-Type':'application/octet-stream'}


def test_upload_and_download_are_run_and_owner_scoped(client):
    http, tokens, _ = client
    result = http.post('/v1/artifacts?filename=report.html', content=b'<b>x</b>', headers=auth(tokens['alice']))
    assert result.status_code == 201
    artifact_id = result.json()['id']
    own = http.get('/v1/artifacts/'+artifact_id, headers=auth(tokens['alice']))
    assert own.status_code == 200
    assert own.content == b'<b>x</b>'
    assert own.headers['content-disposition'].startswith('attachment;')
    assert own.headers['content-type'] == 'application/octet-stream'
    assert own.headers['cache-control'] == 'private, no-store'
    assert http.get('/v1/artifacts/'+artifact_id, headers=auth(tokens['bob'])).status_code == 403


def test_upload_rejects_owner_selection_oversize_traversal_and_revocation(client):
    http, tokens, caps = client
    assert http.post('/v1/artifacts?filename=x', content=b'x').status_code == 401
    for query in ('filename=x&user_id=bob','filename=../secret','filename=x&filename=y'):
        assert http.post('/v1/artifacts?'+query, content=b'x', headers=auth(tokens['alice'])).status_code in (400,403)
    assert http.post('/v1/artifacts?filename=x', content=b'123456789', headers=auth(tokens['alice'])).status_code == 403
    caps.revoke(tokens['alice'])
    assert http.post('/v1/artifacts?filename=x', content=b'x', headers=auth(tokens['alice'])).status_code == 403
