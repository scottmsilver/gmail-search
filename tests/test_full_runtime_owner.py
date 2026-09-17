"""Allowlisted owners may use the unrestricted runtimes on the public origin.

The public deployment exists because arbitrary execution is not cleared for
shared use: `agents/service.py` short-circuits every run to the bounded Gemini
retrieval loop before the requested backend is even read. That is the right
default and stays the default.

This narrows it by identity rather than by deployment. An address in
`GMS_FULL_RUNTIME_EMAILS` gets the private app's runtimes -- Pi with Gemini and
OpenRouter, model choice, battles -- on the public origin. Everyone else keeps
the bounded loop. The list is a strict subset of the admission allowlist, so it
can only narrow, never widen, who gets in.
"""
from __future__ import annotations

import pytest

from gmail_search.auth import public as public_auth


@pytest.fixture
def public_env(monkeypatch):
    monkeypatch.setenv('GMS_PUBLIC_ORIGIN', 'https://gms.example.com')
    monkeypatch.setenv('GMS_IDENTITY_BROKER_URL', 'https://broker.example.com')
    monkeypatch.setenv('GMAIL_MULTI_TENANT', '1')
    monkeypatch.setenv('GMS_IDENTITY_HANDOFF_SECRET', 'h' * 48)
    monkeypatch.setenv('GMS_SESSION_SECRET', 's' * 48)
    monkeypatch.setenv('GMS_PUBLIC_ALLOWED_EMAILS', 'owner@example.com,guest@example.com')
    monkeypatch.setenv('WEB_CONCURRENCY', '1')
    monkeypatch.delenv('BROKER_HANDOFF_SECRET', raising=False)
    return monkeypatch


def test_no_owner_has_full_runtime_by_default(public_env):
    public_env.delenv('GMS_FULL_RUNTIME_EMAILS', raising=False)
    assert public_auth.full_runtime_emails() == frozenset()


def test_a_listed_owner_is_recognised(public_env):
    public_env.setenv('GMS_FULL_RUNTIME_EMAILS', 'Owner@Example.com')
    assert public_auth.full_runtime_emails() == frozenset({'owner@example.com'})


def test_the_list_cannot_widen_admission(public_env):
    """An address that cannot sign in must not be grantable extra power; that
    would turn a capability setting into an admission setting."""
    public_env.setenv('GMS_FULL_RUNTIME_EMAILS', 'stranger@example.com')
    with pytest.raises(RuntimeError, match='subset'):
        public_auth.full_runtime_emails()


def test_it_is_inert_outside_the_public_deployment(public_env):
    """On the private app there is no short-circuit to relax."""
    public_env.delenv('GMS_PUBLIC_ORIGIN')
    public_env.setenv('GMS_FULL_RUNTIME_EMAILS', 'owner@example.com')
    assert public_auth.full_runtime_emails() == frozenset()


@pytest.mark.parametrize('email,expected', [
    ('owner@example.com', True), ('OWNER@EXAMPLE.COM', True),
    ('guest@example.com', False), ('', False), (None, False),
])
def test_membership_check(public_env, email, expected):
    public_env.setenv('GMS_FULL_RUNTIME_EMAILS', 'owner@example.com')
    assert public_auth.has_full_runtime(email) is expected


# ── the routing gate ─────────────────────────────────────────────────────────

def test_the_service_consults_the_allowlist_before_short_circuiting():
    """The bounded loop is chosen at `agents/service.py` before the requested
    backend is read. If the exemption were applied anywhere later, a listed
    owner would still be forced into the retrieval loop."""
    from pathlib import Path

    source = (Path(__file__).parents[1] / 'src/gmail_search/agents/service.py').read_text()
    gate = source[source.index('if public_enabled()'):]
    gate = gate[:gate.index('\n\n')]
    assert '_owner_has_full_runtime' in gate, \
        'the public short-circuit must consult the full-runtime allowlist'


def test_owner_lookup_is_server_derived_not_request_supplied():
    """The exemption keys off an email resolved from the users table for the
    authenticated owner id. A request-supplied email would let any caller claim
    the exemption."""
    from pathlib import Path

    source = (Path(__file__).parents[1] / 'src/gmail_search/agents/service.py').read_text()
    body = source[source.index('def _owner_has_full_runtime'):]
    body = body[:body.index('\n\n\n')]
    assert 'SELECT email FROM users WHERE id' in body
    assert 'has_full_runtime' in body


def test_auth_me_reports_the_capability():
    """The frontend decided what to show from a build-time origin flag, so one
    build served the private app and another the public one, and no user could
    differ from their deployment. The server reports it per user instead."""
    from pathlib import Path

    source = (Path(__file__).parents[1] / 'src/gmail_search/auth/routes.py').read_text()
    body = source[source.index('"multi_tenant": True'):]
    body = body[:body.index('    @app.post')]
    assert '"capabilities"' in body and 'full_runtime' in body
    assert 'has_full_runtime' in body
    # The private app is unrestricted, so it must report the capability too.
    assert 'not public_auth.public_enabled() or' in body
