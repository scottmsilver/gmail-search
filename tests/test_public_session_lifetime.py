"""Public sessions should outlive a coffee break.

The public deployment pinned SESSION_TTL to one hour and never extended it:
`read_session` checked the expiry but did not move it, so an actively-used
session still ended exactly an hour after sign-in. The private app already had
`GMS_SESSION_TTL_DAYS` (default 30) and the public path simply ignored it.

Sessions still live in process memory, so a restart ends them regardless. These
tests cover the lifetime, not the durability.
"""
from __future__ import annotations

import pytest

from gmail_search.auth import public as public_auth


@pytest.fixture
def public_env(monkeypatch):
    for key, value in dict(
        GMAIL_MULTI_TENANT="1", GMS_PUBLIC_ORIGIN="https://gms.example",
        GMS_PUBLIC_ALLOWED_EMAILS="owner@example.com",
        GMS_IDENTITY_BROKER_URL="https://identity.example",
        GMS_IDENTITY_HANDOFF_SECRET="i" * 40, GMS_SESSION_SECRET="s" * 40,
    ).items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("BROKER_HANDOFF_SECRET", raising=False)
    public_auth._sessions.clear()
    yield monkeypatch
    public_auth._sessions.clear()


def test_public_sessions_default_to_thirty_days(public_env):
    public_env.delenv("GMS_SESSION_TTL_DAYS", raising=False)
    assert public_auth.session_ttl_seconds() == 30 * 86400


def test_the_lifetime_is_configurable(public_env):
    public_env.setenv("GMS_SESSION_TTL_DAYS", "7")
    assert public_auth.session_ttl_seconds() == 7 * 86400


@pytest.mark.parametrize("bad", ["0", "-1", "nonsense", ""])
def test_a_nonsense_lifetime_falls_back_rather_than_crashing(public_env, bad):
    public_env.setenv("GMS_SESSION_TTL_DAYS", bad)
    assert public_auth.session_ttl_seconds() == 30 * 86400


def test_reading_a_session_extends_it(public_env, monkeypatch):
    """Sliding renewal: the old behaviour expired an active session mid-use."""
    clock = [1_000_000.0]
    monkeypatch.setattr(public_auth.time, "time", lambda: clock[0])
    token = public_auth.issue_session({"email": "owner@example.com"})
    first = public_auth._sessions[public_auth._digest(token)][0]

    clock[0] += 86400  # a day later, still inside the window
    assert public_auth.read_session(token) is not None
    assert public_auth._sessions[public_auth._digest(token)][0] > first


def test_an_abandoned_session_still_expires(public_env, monkeypatch):
    clock = [1_000_000.0]
    monkeypatch.setattr(public_auth.time, "time", lambda: clock[0])
    token = public_auth.issue_session({"email": "owner@example.com"})
    clock[0] += 31 * 86400
    assert public_auth.read_session(token) is None
