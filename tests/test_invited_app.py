"""Compose the browser API without any legacy mailbox or admin routes."""
import threading
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from test_invited_run_routes import app_state, headers, frames  # noqa: F401
from gmail_search.auth.gmail_consent import GmailConsent
from gmail_search.auth.invited_broker import BoundGmailBroker
from gmail_search.gateway.artifacts import ArtifactStore


def _lifespan_app(state, tmp_path, monkeypatch, broker):
    from gmail_search.invited_app import create_invited_app
    for key, value in {
        'GMAIL_MULTI_TENANT':'1', 'GMS_PUBLIC_ORIGIN':'https://gms.example.test',
        'GMS_PUBLIC_ALLOWED_EMAILS':'original@example.test', 'GMS_IDENTITY_BROKER_URL':'https://identity.example.test',
        'GMS_IDENTITY_HANDOFF_SECRET':'i'*48, 'GMS_SESSION_SECRET':'s'*48,
    }.items():
        monkeypatch.setenv(key, value)
    root = tmp_path/'lifespan-artifacts'
    root.mkdir(mode=0o700)
    artifacts = ArtifactStore(root, state.runs.events.capabilities)
    conversations = SimpleNamespace(claim=lambda owner, conversation: True,
        list=lambda owner, limit: [], get=lambda owner, conversation: None,
        save=lambda *args: True, delete=lambda *args: True)
    consent = GmailConsent(state.identities)
    return create_invited_app(identities=state.identities, consent=consent,
        broker=broker, provision_account=lambda *args: True, runs=state.runs,
        conversations=conversations, artifacts=artifacts), consent


def test_lifespan_recovers_then_drains_cleanup_off_loop_before_admission(app_state, tmp_path, monkeypatch):  # noqa: F811
    events = []

    async def recover():
        events.append(('recover', threading.get_ident()))

    async def close():
        events.append(('close', threading.get_ident()))

    def drain_cleanup(consent):
        events.append(('drain', threading.get_ident()))

    monkeypatch.setattr(app_state.runs, 'recover', recover)
    monkeypatch.setattr(app_state.runs, 'close', close)
    broker = SimpleNamespace(origin='https://gmail-broker.example.test',
        signing_secret='h'*48, drain_cleanup=drain_cleanup)
    app, _ = _lifespan_app(app_state, tmp_path, monkeypatch, broker)

    with TestClient(app, base_url='https://gms.example.test'):
        assert [name for name, _ in events] == ['recover', 'drain']
        assert events[0][1] != events[1][1]
    assert [name for name, _ in events] == ['recover', 'drain', 'close']


def test_lifespan_closes_recovered_runs_when_cleanup_drain_fails(app_state, tmp_path, monkeypatch):  # noqa: F811
    events = []

    async def recover():
        events.append('recover')

    async def close():
        events.append('close')

    def drain_cleanup(consent):
        events.append('drain')
        raise RuntimeError('cleanup unavailable')

    monkeypatch.setattr(app_state.runs, 'recover', recover)
    monkeypatch.setattr(app_state.runs, 'close', close)
    broker = SimpleNamespace(origin='https://gmail-broker.example.test',
        signing_secret='h'*48, drain_cleanup=drain_cleanup)
    app, _ = _lifespan_app(app_state, tmp_path, monkeypatch, broker)

    with pytest.raises(RuntimeError, match='cleanup unavailable'):
        with TestClient(app, base_url='https://gms.example.test'):
            pass
    assert events == ['recover', 'drain', 'close']


def test_composed_app_identity_chat_results_and_no_legacy_routes(app_state, tmp_path, monkeypatch):  # noqa: F811
    from gmail_search.invited_app import create_invited_app
    s = app_state
    for key, value in {
        'GMAIL_MULTI_TENANT':'1', 'GMS_PUBLIC_ORIGIN':'https://gms.example.test',
        'GMS_PUBLIC_ALLOWED_EMAILS':'original@example.test', 'GMS_IDENTITY_BROKER_URL':'https://identity.example.test',
        'GMS_IDENTITY_HANDOFF_SECRET':'i'*48, 'GMS_SESSION_SECRET':'s'*48,
    }.items():
        monkeypatch.setenv(key, value)
    root = tmp_path/'artifacts'
    root.mkdir(mode=0o700)
    artifacts = ArtifactStore(root, s.runs.events.capabilities)
    conversations = SimpleNamespace(claim=lambda owner, conversation: True,
        list=lambda owner, limit: [], get=lambda owner, conversation: None,
        save=lambda *args: True, delete=lambda *args: True)
    with httpx.Client(trust_env=False, transport=httpx.MockTransport(lambda req: httpx.Response(503))) as http:
        broker = BoundGmailBroker('https://gmail-broker.example.test', bearer='b'*48,
            signing_secret='h'*48, client=http)
        app = create_invited_app(identities=s.identities, consent=GmailConsent(s.identities),
            broker=broker, provision_account=lambda *args: True, runs=s.runs,
            conversations=conversations, artifacts=artifacts)
        with TestClient(app, base_url='https://gms.example.test') as client:
            assert client.get('/api/auth/me').status_code == 401
            assert client.get('/api/auth/me', headers=headers(s)).json()['user']['id'] == s.accounts['alice'].owner_id
            assert client.get('/api/conversations', headers=headers(s)).json() == {'conversations': []}
            answer = client.post('/api/agent/analyze', headers=headers(s), json={
                'question':'summarize receipts', 'conversation_id':'conversation'})
            assert answer.status_code == 200
            run = frames(answer)[0][1]['session_id']
            assert frames(answer)[-1][0] == 'persist_ok'
            assert client.get(f'/api/agent-events/{run}?conversation_id=conversation', headers=headers(s)).status_code == 200
            assert client.get(f'/api/agent-events/{run}?conversation_id=conversation', headers=headers(s,'bob')).status_code == 404
            for path in ('/api/sql', '/api/admin/users', '/docs', '/openapi.json'):
                assert client.get(path, headers=headers(s)).status_code == 404
            assert client.get('/api/auth/me', headers={**headers(s),'X-User-ID':'bob'}).status_code == 401
            assert client.get('/api/auth/me', headers={**headers(s),'host':'evil.test'}).status_code == 400
            assert not s.backend.handles
        assert s.runs._closed
