"""Persistence tests for the OAuth provider state (option 1 of the
restart-reauth fix): clients, auth codes, and token pairs survive an MCP
server restart by living in Postgres instead of process memory.

Key security property pinned here: the DB never holds raw bearer
secrets — token/code rows are keyed by SHA-256 and the stored JSON has
the secret field blanked, so a DB dump can't be replayed against /mcp.
"""

from __future__ import annotations

import asyncio

import pytest
from mcp.shared.auth import OAuthClientInformationFull

from gmail_search.agents import mcp_oauth

_BASE = "https://gmail-mcp.example.com"
_OWNER = "scottmsilver@gmail.com"
_SECRET = "s" * 48


@pytest.fixture
def oauth_env(monkeypatch):
    monkeypatch.setenv("GMAIL_MCP_PUBLIC_URL", _BASE)
    monkeypatch.setenv("GMAIL_MCP_TRANSPORT_SECRET", _SECRET)
    monkeypatch.setenv("BROKER_HANDOFF_SECRET", _SECRET)
    monkeypatch.setenv("GMAIL_MCP_OAUTH_OWNER_EMAIL", _OWNER)


@pytest.fixture
def pg_store(db_backend):
    """A PgOAuthStore against the isolated test schema."""
    if db_backend is None:  # pragma: no cover - guarded by db_backend skip
        pytest.skip("Postgres not reachable")
    from gmail_search.agents.mcp_oauth_store import PgOAuthStore

    return PgOAuthStore


def _provider(store):
    return mcp_oauth.GatedBrokerOAuthProvider(
        broker_start_url="https://auth.example.com/start",
        callback_url=f"{_BASE}/oauth/callback",
        owner=_OWNER,
        store=store,
    )


def _client_info(client_id="c1"):
    return OAuthClientInformationFull(
        client_id=client_id,
        redirect_uris=["https://claude.ai/api/mcp/auth_callback"],
        scope="mcp",
    )


def _signed_state(nonce="n1"):
    return mcp_oauth._encode_state(
        {
            "client_id": "c1",
            "redirect_uri": "https://claude.ai/api/mcp/auth_callback",
            "redirect_uri_provided_explicitly": True,
            "code_challenge": "abc",
            "scopes": ["mcp"],
            "client_state": "s",
            "resource": f"{_BASE}/mcp",
            "nonce": nonce,
        }
    )


def _code_from_redirect(redirect: str) -> str:
    import urllib.parse

    qs = urllib.parse.parse_qs(urllib.parse.urlparse(redirect).query)
    return qs["code"][0]


def test_tokens_survive_provider_restart(oauth_env, pg_store):
    """The whole point: a token pair issued by provider A is honored by
    provider B (fresh process state, same DB) — no re-auth after restart."""

    async def drive():
        a = _provider(pg_store())
        info = _client_info()
        await a.register_client(info)
        redirect = a.resume_after_broker(state_token=_signed_state(), email=_OWNER)
        code = _code_from_redirect(redirect)
        loaded = await a.load_authorization_code(info, code)
        pair = await a.exchange_authorization_code(info, loaded)

        b = _provider(pg_store())  # simulated restart
        # Client registration survived (claude.ai skips re-DCR).
        assert (await b.get_client("c1")) is not None
        # Access token still valid.
        tok = await b.load_access_token(pair.access_token)
        assert tok is not None and tok.client_id == "c1"
        # Refresh rotation works across the restart AND preserves the
        # RFC 8707 resource audience.
        old_refresh = await b.load_refresh_token(info, pair.refresh_token)
        assert old_refresh is not None
        rotated = await b.exchange_refresh_token(info, old_refresh, ["mcp"])
        c = _provider(pg_store())
        new_access = await c.load_access_token(rotated.access_token)
        assert new_access is not None
        assert new_access.resource == f"{_BASE}/mcp"
        # Rotation revoked the old pair everywhere.
        assert (await c.load_access_token(pair.access_token)) is None
        assert (await c.load_refresh_token(info, pair.refresh_token)) is None

    asyncio.run(drive())


def test_db_never_stores_raw_secrets(oauth_env, pg_store, db_backend):
    """Token/code rows are keyed by SHA-256 and the stored JSON contains
    no raw secret material."""
    from gmail_search.store.db import get_connection

    async def drive():
        a = _provider(pg_store())
        info = _client_info()
        await a.register_client(info)
        redirect = a.resume_after_broker(state_token=_signed_state(), email=_OWNER)
        code = _code_from_redirect(redirect)
        loaded = await a.load_authorization_code(info, code)
        pair = await a.exchange_authorization_code(info, loaded)
        return code, pair

    code, pair = asyncio.run(drive())

    conn = get_connection(db_backend["db_path"])
    rows = conn.execute("SELECT kind, key, value::text AS v FROM mcp_oauth_state").fetchall()
    conn.close()
    assert rows, "expected persisted oauth rows"
    for row in rows:
        for secret in (code, pair.access_token, pair.refresh_token):
            assert row["key"] != secret
            assert secret not in row["v"]


def test_auth_code_single_use_across_restart(oauth_env, pg_store):
    """A consumed authorization code cannot be replayed against a fresh
    provider instance (invalid_grant, not a second token pair)."""

    async def drive():
        a = _provider(pg_store())
        info = _client_info()
        await a.register_client(info)
        redirect = a.resume_after_broker(state_token=_signed_state(), email=_OWNER)
        code = _code_from_redirect(redirect)
        loaded = await a.load_authorization_code(info, code)
        await a.exchange_authorization_code(info, loaded)

        b = _provider(pg_store())
        assert (await b.load_authorization_code(info, code)) is None
        with pytest.raises(mcp_oauth.TokenError):
            await b.exchange_authorization_code(info, loaded)

    asyncio.run(drive())


def test_broker_state_single_use_across_restart(oauth_env, pg_store):
    """The signed broker state (nonce) is single-use even across a
    restart — replaying it against a fresh provider mints nothing."""
    a = _provider(pg_store())
    state = _signed_state(nonce="replay-me")
    assert a.resume_after_broker(state_token=state, email=_OWNER)
    b = _provider(pg_store())
    assert b.resume_after_broker(state_token=state, email=_OWNER) is None


def test_expired_access_token_rejected_and_purged(oauth_env, pg_store, monkeypatch):
    """An access token past expires_at is refused after restart."""

    async def drive():
        a = _provider(pg_store())
        info = _client_info()
        await a.register_client(info)
        monkeypatch.setattr(mcp_oauth, "_ACCESS_TOKEN_TTL_SECONDS", -10)
        pair = a._issue_token_pair("c1", ["mcp"], None)
        assert pair.access_token
        b = _provider(pg_store())
        assert (await b.load_access_token(pair.access_token)) is None

    asyncio.run(drive())


def test_revocation_cascades_in_db(oauth_env, pg_store):
    """Revoking the refresh token kills the paired access token across
    instances (and vice versa is covered by rotation above)."""

    async def drive():
        a = _provider(pg_store())
        info = _client_info()
        await a.register_client(info)
        pair = a._issue_token_pair("c1", ["mcp"], None)
        refresh = await a.load_refresh_token(info, pair.refresh_token)
        await a.revoke_token(refresh)
        b = _provider(pg_store())
        assert (await b.load_access_token(pair.access_token)) is None
        assert (await b.load_refresh_token(info, pair.refresh_token)) is None

    asyncio.run(drive())


def test_memory_store_remains_default(oauth_env):
    """No store argument → in-memory dicts, exactly the v1 behavior the
    existing test_mcp_oauth.py suite pins."""
    p = mcp_oauth.GatedBrokerOAuthProvider(
        broker_start_url="https://auth.example.com/start",
        callback_url=f"{_BASE}/oauth/callback",
        owner=_OWNER,
    )
    assert p.access_tokens == {}
    pair = p._issue_token_pair("c1", ["mcp"], None)
    assert pair.access_token in p.access_tokens


def test_build_default_store_requires_opt_in(monkeypatch):
    """Without GMAIL_MCP_OAUTH_PERSIST the default store is memory —
    tests and dev setups must never write OAuth rows to DB_DSN."""
    from gmail_search.agents.mcp_oauth_store import MemoryOAuthStore, build_default_store

    monkeypatch.delenv("GMAIL_MCP_OAUTH_PERSIST", raising=False)
    assert isinstance(build_default_store(), MemoryOAuthStore)


def test_refresh_rotation_single_use_under_race(oauth_env, pg_store):
    """Codex-High: two callers who both loaded the same refresh token
    (concurrent /token requests) must not BOTH mint pairs — the second
    exchange fails invalid_grant because the consume is atomic."""

    async def drive():
        a = _provider(pg_store())
        info = _client_info()
        await a.register_client(info)
        pair = a._issue_token_pair("c1", ["mcp"], None)
        loaded1 = await a.load_refresh_token(info, pair.refresh_token)
        loaded2 = await a.load_refresh_token(info, pair.refresh_token)
        assert loaded1 is not None and loaded2 is not None
        await a.exchange_refresh_token(info, loaded1, ["mcp"])
        with pytest.raises(mcp_oauth.TokenError):
            await a.exchange_refresh_token(info, loaded2, ["mcp"])

    asyncio.run(drive())


def test_expired_access_does_not_kill_refresh(oauth_env, pg_store, monkeypatch):
    """Presenting an EXPIRED access token must not revoke the paired
    refresh token — access expiry is routine (1h TTL) and the refresh
    grant must survive it. (v1's in-memory cascade on expiry silently
    killed the refresh pair, itself forcing re-auths.)"""

    async def drive():
        a = _provider(pg_store())
        info = _client_info()
        await a.register_client(info)
        monkeypatch.setattr(mcp_oauth, "_ACCESS_TOKEN_TTL_SECONDS", -10)
        pair = a._issue_token_pair("c1", ["mcp"], None)
        assert (await a.load_access_token(pair.access_token)) is None
        refreshed = await a.load_refresh_token(info, pair.refresh_token)
        assert refreshed is not None, "refresh token must survive access expiry"

    asyncio.run(drive())


def test_expired_access_does_not_kill_refresh_memory(oauth_env, monkeypatch):
    """Same expiry semantics in the memory store (v1 divergence fixed)."""
    from gmail_search.agents.mcp_oauth_store import MemoryOAuthStore

    async def drive():
        a = _provider(MemoryOAuthStore())
        info = _client_info()
        await a.register_client(info)
        monkeypatch.setattr(mcp_oauth, "_ACCESS_TOKEN_TTL_SECONDS", -10)
        pair = a._issue_token_pair("c1", ["mcp"], None)
        assert (await a.load_access_token(pair.access_token)) is None
        assert (await a.load_refresh_token(info, pair.refresh_token)) is not None

    asyncio.run(drive())


def test_persistence_flag_on_fails_closed(monkeypatch):
    """Codex-Low: with GMAIL_MCP_OAUTH_PERSIST explicitly enabled, a DB
    failure must fail startup loudly, not silently degrade to memory."""
    import gmail_search.agents.mcp_oauth_store as store_mod

    monkeypatch.setenv("GMAIL_MCP_OAUTH_PERSIST", "1")

    class _Boom:
        def __init__(self):
            raise RuntimeError("db unreachable")

    monkeypatch.setattr(store_mod, "PgOAuthStore", _Boom)
    with pytest.raises(RuntimeError):
        store_mod.build_default_store()
