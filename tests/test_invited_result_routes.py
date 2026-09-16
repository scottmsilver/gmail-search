"""Browser result access uses invited sessions, never guest tokens or owner hints."""
import asyncio
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from gmail_search.auth.identity_store import IdentityStore, VerifiedGoogleIdentity
from gmail_search.auth.public import SESSION_COOKIE
from gmail_search.gateway.artifacts import ArtifactStore
from gmail_search.gateway.capabilities import Capabilities
from gmail_search.gateway.events import Events
from gmail_search.gateway.registry import Registry


@pytest.fixture
def state(tmp_path):
    from gmail_search.auth.result_routes import create_result_router
    identities = IdentityStore(tmp_path/'identities')
    accounts, sessions = {}, {}
    for name in ('alice','bob'):
        accounts[name] = identities.invite(name+'@example.test')
        identities.mark_provisioned(accounts[name].owner_id)
        sessions[name] = identities.admit(VerifiedGoogleIdentity(name+'@example.test','google-'+name,True))
    registry = Registry(tmp_path/'registry', is_active=identities.is_active)
    caps = Capabilities(registry)
    root = tmp_path/'artifacts'
    root.mkdir(mode=0o700)
    artifacts, events = ArtifactStore(root,caps), Events(caps)
    run = registry.start_run(accounts['alice'].owner_id,'conversation',request_key='request')
    token = caps.issue(run.run_id,audience='artifact',operations={'artifact.commit'}).secret
    async def data():
        yield b'<script>synthetic</script>'
    item = asyncio.run(artifacts.publish(token,data(),filename='report.html'))
    event_token = caps.issue(run.run_id,audience='events',operations={'append'}).secret
    events.append(event_token,{'type':'text','text':'synthetic private output'})
    app = FastAPI()
    app.include_router(create_result_router(identities,artifacts,events))
    with TestClient(app,base_url='https://gms.example.test') as client:
        yield client, identities, accounts, sessions, item, run, token


def auth(session):
    return {'Cookie': SESSION_COOKIE+'='+session}


def test_browser_artifacts_enforce_owner_and_inert_download_headers(state):
    client, _, _, sessions, item, _, token = state
    path = '/api/agent-artifacts/'+item.id+'?conversation_id=conversation'
    assert client.get(path).status_code == 401
    assert client.get(path,headers={'Authorization':'Bearer '+token}).status_code == 401
    assert client.get(path,headers=auth(sessions['bob'])).status_code == 404
    response = client.get(path,headers=auth(sessions['alice']))
    assert response.status_code == 200 and response.content == b'<script>synthetic</script>'
    assert response.headers['content-type'] == 'application/octet-stream'
    assert response.headers['content-disposition'].startswith('attachment;')
    assert response.headers['content-security-policy'] == "sandbox; default-src 'none'"
    assert response.headers['cache-control'] == 'private, no-store'
    assert client.get(path+'&owner_id=bob',headers=auth(sessions['alice'])).status_code == 400


def test_browser_event_replay_and_session_revocation(state):
    client, identities, _, sessions, _, run, _ = state
    path = '/api/agent-events/'+run.run_id+'?conversation_id=conversation'
    assert client.get(path,headers=auth(sessions['bob'])).status_code == 404
    response = client.get(path,headers=auth(sessions['alice']))
    assert response.status_code == 200
    assert response.json() == {'events':[{'seq':1,'event':{'type':'text','text':'synthetic private output'}}], 'next_cursor':1}
    assert client.get(path+'&after=1',headers=auth(sessions['alice'])).json()['events'] == []
    for query in ('&limit=101','&after=-1','&after=1&after=2'):
        assert client.get(path+query,headers=auth(sessions['alice'])).status_code == 400
    identities.revoke_session(sessions['alice'])
    result = client.get(path,headers=auth(sessions['alice']))
    assert result.status_code == 401
    assert 'synthetic private' not in result.text
    assert result.headers['cache-control'] == 'private, no-store'


def test_browser_revocation_during_artifact_read_suppresses_bytes(state, monkeypatch):
    client, identities, _, sessions, item, _, _ = state
    # Replace only the storage read to revoke after ownership checked and bytes read.
    original = ArtifactStore.read
    def read(store, *args):
        data = original(store,*args)
        identities.revoke_session(sessions['alice'])
        return data
    monkeypatch.setattr(ArtifactStore,'read',read)
    result = client.get('/api/agent-artifacts/'+item.id+'?conversation_id=conversation',headers=auth(sessions['alice']))
    assert result.status_code == 401 and b'<script>' not in result.content
