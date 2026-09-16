"""Account-bound consent uses synthetic claims; it never exchanges Google codes."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sqlite3

import pytest

from gmail_search.auth.gmail_consent import GmailConsent
from gmail_search.auth.identity_store import IdentityDenied
from test_invitation_lifecycle import admit as admit, identity as identity, lifecycle as lifecycle


def test_consent_returns_only_the_expected_owner_and_consumes_once(lifecycle):
    account, session = admit(lifecycle)
    store, _, _, _ = lifecycle
    consent = GmailConsent(store)
    pending = consent.begin(session)
    assert pending.secret.encode() not in Path(store.path).read_bytes()
    grant = consent.consume(pending.secret, session=session, claims=identity())
    assert (grant.owner_id, grant.email, grant.google_subject, grant.generation) == (
        account.owner_id, account.email, 'google-alice', account.generation,
    )
    consent.validate_grant(grant)
    with pytest.raises(IdentityDenied):
        consent.consume(pending.secret, session=session, claims=identity())
    with sqlite3.connect(store.path) as db:
        columns = [row[1] for row in db.execute('PRAGMA table_info(gmail_consent_states)')]
    assert not {'access_token', 'refresh_token', 'credentials'} & set(columns)


@pytest.mark.parametrize('claims', [identity(email='bob@example.test', subject='google-bob'),
    identity(subject='other-google-account'), identity(verified=False)])
def test_wrong_account_callback_never_reaches_credential_storage(lifecycle, claims):
    _, session = admit(lifecycle)
    consent = GmailConsent(lifecycle[0])
    pending = consent.begin(session)
    stored = []
    with pytest.raises(IdentityDenied):
        grant = consent.consume(pending.secret, session=session, claims=claims)
        stored.append(grant)  # Trusted credential adapter runs only after validation.
    assert stored == []
    with pytest.raises(IdentityDenied):
        consent.consume(pending.secret, session=session, claims=identity())


def test_consent_rejects_other_browser_and_expired_sessions(lifecycle):
    _, alice_session = admit(lifecycle)
    _, bob_session = admit(lifecycle, 'bob@example.test', 'google-bob')
    consent = GmailConsent(lifecycle[0])
    pending = consent.begin(alice_session)
    with pytest.raises(IdentityDenied):
        consent.consume(pending.secret, session=bob_session, claims=identity())
    with pytest.raises(IdentityDenied):
        consent.begin('forged-session')


def test_consent_expires_and_is_not_revived_by_reinvitation(lifecycle):
    account, session = admit(lifecycle)
    store, _, service, now = lifecycle
    consent = GmailConsent(store)
    expired = consent.begin(session, ttl=10)
    now[0] += 11
    with pytest.raises(IdentityDenied):
        consent.consume(expired.secret, session=session, claims=identity())
    pending = consent.begin(session)
    service.revoke('admin', account.email)
    service.invite('admin', account.email)
    store.mark_provisioned(account.owner_id)
    new_session = store.admit(identity())
    with pytest.raises(IdentityDenied):
        consent.consume(pending.secret, session=new_session, claims=identity())


def test_consent_is_durable_and_concurrent_callbacks_have_one_winner(lifecycle):
    _, session = admit(lifecycle)
    store = lifecycle[0]
    state = GmailConsent(store).begin(session)
    consent = GmailConsent(store)
    def finish():
        try:
            return consent.consume(state.secret, session=session, claims=identity())
        except IdentityDenied:
            return None
    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _: finish(), range(2)))
    assert sum(result is not None for result in outcomes) == 1


def test_grant_must_be_revalidated_after_revocation_before_credential_use(lifecycle):
    account, session = admit(lifecycle)
    consent = GmailConsent(lifecycle[0])
    pending = consent.begin(session)
    grant = consent.consume(pending.secret, session=session, claims=identity())
    lifecycle[2].revoke('admin', account.email)
    with pytest.raises(IdentityDenied):
        consent.validate_grant(grant)


def test_grant_expires_without_credential_storage(lifecycle):
    _, session = admit(lifecycle)
    consent = GmailConsent(lifecycle[0])
    pending = consent.begin(session)
    grant = consent.consume(pending.secret, session=session, claims=identity())
    lifecycle[3][0] += 61
    with pytest.raises(IdentityDenied):
        consent.validate_grant(grant)


def test_signed_broker_handoff_is_bound_to_app_owner_generation_and_state(lifecycle):
    import jwt
    account, session = admit(lifecycle)
    consent = GmailConsent(lifecycle[0])
    broker = 'https://gmail-broker.example.test'
    secret = 'h' * 48
    def handoff(state, **extra):
        return jwt.encode(dict(iss=broker, aud='gmail-search:gmail-consent', sub='google-alice',
            email=account.email, email_verified=True, owner_id=account.owner_id,
            invitation_generation=account.generation, credential_generation=pending.credential_generation, nonce=state, iat=1000, exp=1060, jti='unique-result', **extra), secret, algorithm='HS256')
    pending = consent.begin(session)
    grant = consent.consume_broker_handoff(pending.secret, session=session, token=handoff(pending.secret),
                                          broker_origin=broker, signing_secret=secret)
    assert grant.owner_id == account.owner_id
    for field, value in [('owner_id', 'foreign-owner'), ('invitation_generation', 999), ('credential_generation', 999), ('aud', 'wezterm-web'),
                         ('sub', 'other-google'), ('iss', 'https://evil.test'), ('exp', 999)]:
        pending = consent.begin(session)
        claims = jwt.decode(handoff(pending.secret), options={'verify_signature': False})
        claims[field] = value
        with pytest.raises(IdentityDenied):
            consent.consume_broker_handoff(pending.secret, session=session,
                token=jwt.encode(claims, secret, algorithm='HS256'), broker_origin=broker, signing_secret=secret)


def test_disconnect_reconnect_advances_credentials_without_logging_out(lifecycle):
    account, session = admit(lifecycle)
    consent = GmailConsent(lifecycle[0])
    first = consent.begin(session)
    old = consent.consume(first.secret, session=session, claims=identity())
    cleanup = consent.disconnect(session)
    assert cleanup.credential_generation == old.credential_generation
    assert lifecycle[0].read_session(session).owner_id == account.owner_id
    with pytest.raises(IdentityDenied):
        consent.validate_grant(old)
    second = consent.begin(session)
    new = consent.consume(second.secret, session=session, claims=identity())
    assert new.credential_generation > old.credential_generation
    assert new.generation == old.generation
    consent.validate_grant(new)


def test_new_consent_invalidates_older_pending_callbacks(lifecycle):
    _, session = admit(lifecycle)
    consent = GmailConsent(lifecycle[0])
    first, second = consent.begin(session), consent.begin(session)
    assert second.credential_generation > first.credential_generation
    with pytest.raises(IdentityDenied):
        consent.consume(first.secret, session=session, claims=identity())
    consent.validate_grant(consent.consume(second.secret, session=session, claims=identity()))


def test_invitation_revoke_enqueues_old_credentials_and_reinvite_cannot_revive(lifecycle):
    account, session = admit(lifecycle)
    store, _, service, _ = lifecycle
    consent = GmailConsent(store)
    state = consent.begin(session)
    old = consent.consume(state.secret, session=session, claims=identity())
    service.revoke('admin', account.email)
    assert any(task.owner_id == account.owner_id and task.credential_generation == old.credential_generation
               for task in consent.pending_cleanup())
    service.invite('admin', account.email)
    store.mark_provisioned(account.owner_id)
    session = store.admit(identity())
    new_state = consent.begin(session)
    new = consent.consume(new_state.secret, session=session, claims=identity())
    assert new.credential_generation > old.credential_generation
    assert new.generation > old.generation
    with pytest.raises(IdentityDenied):
        consent.validate_grant(old)
    consent.validate_grant(new)


def test_long_lived_credential_use_rechecks_both_current_generations(lifecycle):
    _, session = admit(lifecycle)
    consent = GmailConsent(lifecycle[0])
    state = consent.begin(session)
    grant = consent.consume(state.secret, session=session, claims=identity())
    assert consent.credential_is_active(grant.owner_id, grant.generation, grant.credential_generation)
    lifecycle[3][0] += 61  # The short-lived storage grant expired; connected account remains usable.
    assert consent.credential_is_active(grant.owner_id, grant.generation, grant.credential_generation)
    consent.disconnect(session)
    assert not consent.credential_is_active(grant.owner_id, grant.generation, grant.credential_generation)
    assert not consent.credential_is_active('foreign-owner', grant.generation, grant.credential_generation)
