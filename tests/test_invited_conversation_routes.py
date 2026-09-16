"""Browser conversation boundary with real invited sessions."""
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gmail_search.auth.identity_store import IdentityStore, VerifiedGoogleIdentity
from gmail_search.auth.public import SESSION_COOKIE
from gmail_search.gateway.registry import AccessDenied


@pytest.fixture
def state(tmp_path):
    from gmail_search.auth.conversation_routes import create_conversation_router
    identities = IdentityStore(tmp_path/'identities')
    tokens, owners = {}, {}
    for name in ('alice', 'bob'):
        account = identities.invite(name+'@example.test')
        identities.mark_provisioned(account.owner_id)
        owners[name] = account.owner_id
        tokens[name] = identities.admit(VerifiedGoogleIdentity(name+'@example.test', name, True))
    class Store:
        def __init__(self):
            self.rows = {}
            self.after_read = lambda: None
        def list(self, owner, limit=100):
            return [v for (o, _), v in self.rows.items() if o == owner][:limit]
        def get(self, owner, conversation):
            result = self.rows.get((owner, conversation))
            self.after_read()
            return result
        def save(self, owner, conversation, payload):
            if any(c == conversation and o != owner for o, c in self.rows):
                raise AccessDenied()
            self.rows[owner, conversation] = {'id': conversation, **payload}
            return True
        def delete(self, owner, conversation):
            if self.rows.pop((owner, conversation), None) is None:
                raise AccessDenied()
            return True
    store = Store()
    app = FastAPI()
    app.include_router(create_conversation_router(identities, store, origin='https://gms.example.test'))
    with TestClient(app, base_url='https://gms.example.test') as client:
        yield SimpleNamespace(client=client, store=store, identities=identities, tokens=tokens, owners=owners)


def headers(s, name='alice'):
    return {'Cookie': SESSION_COOKIE+'='+s.tokens[name], 'Origin': 'https://gms.example.test'}


def test_owned_browser_crud_and_foreign_denial(state):
    s = state
    path = '/api/conversations/conversation'
    payload = {'title': 'Receipts', 'messages': [{'role': 'user', 'content': 'Summarize receipts'}]}
    assert s.client.put(path, headers=headers(s), json=payload).status_code == 200
    response = s.client.get(path, headers=headers(s))
    assert response.json() == {'id': 'conversation', **payload}
    assert response.headers['cache-control'] == 'private, no-store'
    assert len(s.client.get('/api/conversations', headers=headers(s)).json()['conversations']) == 1
    assert s.client.get('/api/conversations', headers=headers(s, 'bob')).json() == {'conversations': []}
    assert s.client.get(path, headers=headers(s, 'bob')).status_code == 404
    assert s.client.put(path, headers=headers(s, 'bob'), json=payload).status_code == 404
    assert s.client.delete(path, headers=headers(s, 'bob')).status_code == 404
    assert s.client.delete(path, headers=headers(s)).status_code == 200
    assert s.client.get(path, headers=headers(s)).status_code == 404


def test_rejects_spoofing_origin_and_unbounded_input(state):
    s = state
    path = '/api/conversations/conversation'
    assert s.client.get(path).status_code == 401
    for key in ('Authorization', 'X-User-ID'):
        assert s.client.get(path, headers={**headers(s), key: 'bob'}).status_code == 401
    assert s.client.put(path, headers={**headers(s), 'Origin': 'https://evil.test'}, json={}).status_code == 403
    assert s.client.delete(path, headers={**headers(s), 'Origin': 'https://evil.test'}).status_code == 403
    for raw in ('{"title":"a","title":"b"}', '[]', '{"owner_id":"bob"}'):
        assert s.client.put(path, headers={**headers(s), 'Content-Type': 'application/json'}, content=raw).status_code == 400
    assert s.client.put(path, headers=headers(s), content='{}').status_code == 415
    assert s.client.put(path, headers=headers(s), json={'title': 'a'*262144}).status_code == 413
    assert s.client.get(path+'?owner_id=bob', headers=headers(s)).status_code == 400
    assert s.client.get('/api/conversations?limit=99999', headers=headers(s)).status_code == 400
    assert not s.store.rows


def test_revocation_during_read_does_not_publish_content(state):
    s = state
    s.store.rows[s.owners['alice'], 'conversation'] = {'id': 'conversation', 'title': 'PRIVATE'}
    s.store.after_read = lambda: s.identities.revoke_session(s.tokens['alice'])
    response = s.client.get('/api/conversations/conversation', headers=headers(s))
    assert response.status_code == 401
    assert 'PRIVATE' not in response.text
