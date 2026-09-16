"""Staged invitations use synthetic identities and local operational metadata only."""
from pathlib import Path
import sqlite3

import pytest

from gmail_search.auth.identity_store import IdentityDenied, IdentityStore, VerifiedGoogleIdentity
from gmail_search.auth.invitations import Invitations
from gmail_search.gateway.capabilities import Capabilities
from gmail_search.gateway.registry import AccessDenied, Registry


@pytest.fixture
def lifecycle(tmp_path):
    tmp_path.chmod(0o700)
    now = [1000.0]
    clock = lambda: now[0]
    identities = IdentityStore(tmp_path / 'identities.sqlite', clock=clock)
    registry = Registry(tmp_path / 'runs.sqlite', is_active=identities.is_active, clock=clock)
    service = Invitations(identities, registry, administrators={'admin'})
    return identities, registry, service, now


def identity(email='alice@example.test', subject='google-alice', verified=True):
    return VerifiedGoogleIdentity(email=email, subject=subject, email_verified=verified)


def admit(lifecycle, email='alice@example.test', subject='google-alice'):
    store, _, service, _ = lifecycle
    account = service.invite('admin', email)
    store.mark_provisioned(account.owner_id)
    token = store.admit(identity(email, subject))
    return account, token


def test_invite_creates_distinct_stable_empty_accounts_and_requires_provisioning(lifecycle):
    store, _, service, _ = lifecycle
    alice = service.invite('admin', 'Alice@EXAMPLE.test')
    assert service.invite('admin', ' alice@example.test ').owner_id == alice.owner_id
    bob = service.invite('admin', 'bob@example.test')
    assert alice.owner_id != bob.owner_id
    with pytest.raises(IdentityDenied):
        store.admit(identity())
    store.mark_provisioned(alice.owner_id)
    token = store.admit(identity())
    assert store.read_session(token).owner_id == alice.owner_id
    assert store.is_active(bob.owner_id) is False
    with sqlite3.connect(store.path) as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert 'messages' not in tables


@pytest.mark.parametrize('claims', [
    identity(verified=False), identity(verified='true'), identity(subject=''),
    identity(email='alice+other@example.test'), identity(email='a.lice@example.test'),
])
def test_only_exact_verified_google_identity_is_admitted(lifecycle, claims):
    store, _, service, _ = lifecycle
    account = service.invite('admin', 'alice@example.test')
    store.mark_provisioned(account.owner_id)
    with pytest.raises(IdentityDenied):
        store.admit(claims)


def test_provider_subject_and_email_binding_cannot_be_reassigned(lifecycle):
    account, token = admit(lifecycle)
    store, _, service, _ = lifecycle
    bob = service.invite('admin', 'bob@example.test')
    store.mark_provisioned(bob.owner_id)
    with pytest.raises(IdentityDenied):
        store.admit(identity(subject='different-google-account'))
    with pytest.raises(IdentityDenied):
        store.admit(identity(email='bob@example.test'))
    assert store.read_session(token).owner_id == account.owner_id


def test_admin_admission_has_no_mailbox_bypass(lifecycle):
    store, _, service, _ = lifecycle
    with pytest.raises(IdentityDenied):
        service.invite('untrusted', 'alice@example.test')
    with pytest.raises(IdentityDenied):
        store.admit(identity('admin@example.test', 'admin'))


def test_revoke_invalidates_sessions_runs_and_reinvitation_never_revives_them(lifecycle):
    account, token = admit(lifecycle)
    store, registry, service, _ = lifecycle
    lease = registry.start_run(account.owner_id, 'conversation', request_key='request')
    capabilities = Capabilities(registry)
    capability = capabilities.issue(lease.run_id, audience='mail', operations={'mail.read'})
    revoked = service.revoke('admin', account.email)
    assert revoked == (lease.run_id,)
    assert store.read_session(token) is None
    with pytest.raises(AccessDenied):
        capabilities.authorize(capability.secret, audience='mail', operation='mail.read')
    assert service.invite('admin', account.email).owner_id == account.owner_id
    store.mark_provisioned(account.owner_id)
    replacement = store.admit(identity())
    assert store.read_session(replacement).owner_id == account.owner_id
    assert store.read_session(token) is None
    with pytest.raises(AccessDenied):
        capabilities.authorize(capability.secret, audience='mail', operation='mail.read')


def test_failed_cancellation_is_durable_and_blocks_reactivation(lifecycle, monkeypatch):
    account, token = admit(lifecycle)
    store, registry, service, _ = lifecycle
    run = registry.start_run(account.owner_id, 'conversation', request_key='request')
    real_cancel = registry.cancel_owner
    monkeypatch.setattr(registry, 'cancel_owner', lambda owner: (_ for _ in ()).throw(RuntimeError('offline')))
    with pytest.raises(IdentityDenied):
        service.revoke('admin', account.email)
    assert store.read_session(token) is None
    with pytest.raises(AccessDenied):
        registry.heartbeat(run.run_id)
    with pytest.raises(IdentityDenied):
        service.invite('admin', account.email)
    reopened = IdentityStore(store.path, clock=store.clock)
    assert tuple(item.owner_id for item in reopened.pending_revocations()) == (account.owner_id,)
    monkeypatch.setattr(registry, 'cancel_owner', real_cancel)
    service = Invitations(reopened, registry, administrators={'admin'})
    assert service.drain_revocations() == (run.run_id,)
    assert reopened.pending_revocations() == ()
    assert service.invite('admin', account.email).owner_id == account.owner_id


def test_sessions_are_hashed_expiring_and_survive_restart(lifecycle):
    account, token = admit(lifecycle)
    store, _, _, now = lifecycle
    assert token.encode() not in Path(store.path).read_bytes()
    reopened = IdentityStore(store.path, clock=store.clock)
    assert reopened.read_session(token).owner_id == account.owner_id
    assert reopened.read_session('forged') is None
    now[0] += 3601
    assert reopened.read_session(token) is None


def test_stale_revocation_completion_cannot_clear_a_newer_revocation(lifecycle):
    account, _ = admit(lifecycle)
    store, _, service, _ = lifecycle
    first = store.revoke(account.email)
    store.complete_revocation(first)
    service.invite('admin', account.email)
    second = store.revoke(account.email)
    store.complete_revocation(first)
    assert store.pending_revocations() == (second,)
    with pytest.raises(IdentityDenied):
        service.invite('admin', account.email)


def test_revoking_one_invitee_preserves_other_sessions_and_capabilities(lifecycle):
    alice, alice_session = admit(lifecycle)
    bob, bob_session = admit(lifecycle, 'bob@example.test', 'google-bob')
    store, registry, service, _ = lifecycle
    lease = registry.start_run(bob.owner_id, 'conversation', request_key='request')
    caps = Capabilities(registry)
    capability = caps.issue(lease.run_id, audience='mail', operations={'mail.read'})
    with pytest.raises(IdentityDenied):
        service.revoke(alice.owner_id, bob.email)
    service.revoke('admin', alice.email)
    assert store.read_session(alice_session) is None
    assert store.read_session(bob_session).owner_id == bob.owner_id
    assert caps.authorize(capability.secret, audience='mail', operation='mail.read').owner_id == bob.owner_id
    assert store.is_active('foreign-owner-id') is False


def test_logout_revokes_only_the_selected_browser_session(lifecycle):
    account, token = admit(lifecycle)
    store, _, _, _ = lifecycle
    second = store.admit(identity())
    store.revoke_session(token)
    assert store.read_session(token) is None
    assert store.read_session(second).owner_id == account.owner_id
