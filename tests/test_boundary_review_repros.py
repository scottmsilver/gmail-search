"""Regressions for authenticated input and durable ingestion resource limits."""
# ruff: noqa: F811 -- imported pytest fixtures are injected by name
import asyncio
import zipfile

import pytest
from starlette.requests import Request

from test_invited_routes import setup, login  # noqa: F401
from test_invitation_lifecycle import admit, lifecycle  # noqa: F401
from gmail_search.auth.gmail_consent import GmailConsent
from gmail_search.auth.identity_store import IdentityDenied
from gmail_search.extract.archive import extract_zip, MAX_FILES


def test_repeated_consent_has_durable_rolling_attempt_quota(lifecycle):
    _, session = admit(lifecycle)
    store, _, _, now = lifecycle
    consent = GmailConsent(store)
    for _ in range(10):
        consent.begin(session)
    with pytest.raises(IdentityDenied):
        GmailConsent(store).begin(session)
    assert len(consent.pending_cleanup()) == 9
    now[0] += 599
    with pytest.raises(IdentityDenied):
        consent.begin(session)
    now[0] += 1
    consent.begin(session)
    assert len(consent.pending_cleanup()) == 10


def test_cleanup_backlog_blocks_begin_but_never_disconnect(lifecycle):
    account, session = admit(lifecycle)
    store, _, _, _ = lifecycle
    consent = GmailConsent(store)
    consent.begin(session)
    # Model an unavailable broker with durable pending cleanup obligations.
    with store._transaction() as db:
        db.executemany('INSERT INTO gmail_credential_cleanup VALUES(?,?,?,?,?)',
            [(account.owner_id, account.email, 'google-alice', account.generation, i)
             for i in range(100)])
        prior = db.execute('SELECT credential_generation FROM gmail_connections').fetchone()[0]
    with pytest.raises(IdentityDenied):
        consent.begin(session)
    with store._transaction() as db:
        assert db.execute('SELECT credential_generation FROM gmail_connections').fetchone()[0] == prior
    assert consent.disconnect(session) is not None


@pytest.mark.parametrize('path', ['connect-gmail', 'disconnect-gmail', 'logout'])
def test_unauthenticated_mutation_never_reads_body(setup, monkeypatch, path):
    _, _, _, client, _, _, _ = setup
    async def forbidden_stream(self):
        raise AssertionError('Unauthenticated body was pulled')
        yield b''
    monkeypatch.setattr(Request, 'stream', forbidden_stream)
    response = client.post('/api/auth/' + path,
        headers={'origin': 'https://gms.example.test'}, content=b' ' * 2048)
    assert response.status_code == 401


def test_authenticated_mutation_rejects_oversized_stream(setup):
    store, _, _, client, _, posted, _ = setup
    store.invite('alice@example.test')
    login(client)
    response = client.post('/api/auth/connect-gmail',
        headers={'origin': 'https://gms.example.test'}, content=b' ' * 1025)
    assert response.status_code == 413
    assert posted == []


def test_authenticated_mutation_has_body_deadline(setup, monkeypatch):
    store, _, _, client, _, posted, _ = setup
    store.invite('alice@example.test')
    login(client)
    async def stalled_stream(self):
        await asyncio.sleep(3)
        yield b'{}'
    monkeypatch.setattr(Request, 'stream', stalled_stream)
    response = client.post('/api/auth/connect-gmail', headers={'origin': 'https://gms.example.test'})
    assert response.status_code == 408
    assert posted == []


def test_archive_file_limit_counts_unsupported_members_without_writing(tmp_path):
    archive = tmp_path / 'fixture.zip'
    with zipfile.ZipFile(archive, 'w') as output:
        for i in range(MAX_FILES + 5):
            output.writestr(f'{i}.unsupported', b'synthetic')
        output.writestr('past-limit.txt', b'must not parse')
    assert extract_zip(archive, {}) is None
    assert list((tmp_path / 'fixture_zip').iterdir()) == []
