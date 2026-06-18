"""Network-free tests for the OAuth 2.1 layer (mcp_oauth.py).

We verify the security-relevant slice we own:
  - flag-off path is byte-for-byte unchanged (no auth routes mounted)
  - RFC 8414 / RFC 9728 metadata endpoints return correct JSON
  - DCR (/register) round-trips a client
  - /authorize without a broker handoff does NOT issue a code (it 302s to the
    broker instead)
  - /oauth/callback rejects a forged handoff, a stale (expired) handoff, and a
    non-allowlisted email — issuing no code in any case
  - state tampering / replay is rejected
  - the owner gate is taken from the verified handoff, not request params

No uvicorn, no real broker, no network. The broker handoff JWT is minted with
the same secret the broker uses via `issue_handoff_jwt_for_test`.
"""

from __future__ import annotations

import time
import urllib.parse

import jwt
import pytest
from starlette.testclient import TestClient

from gmail_search.agents import mcp_oauth
from gmail_search.agents import mcp_tools_server as mts

_SECRET = "x" * 40  # >= 32 bytes
_BASE = "https://gmail-mcp.oursilverfamily.com"
_OWNER = "scottmsilver@gmail.com"


@pytest.fixture(autouse=True)
def _clear_session_registry():
    mts._SESSIONS.clear()
    yield
    mts._SESSIONS.clear()


@pytest.fixture
def oauth_env(monkeypatch):
    """Configure the env so the OAuth layer builds against a fake public URL
    and broker, with strong secrets."""
    monkeypatch.setenv("GMAIL_MCP_OAUTH_ENABLED", "1")
    monkeypatch.setenv("GMAIL_MCP_PUBLIC_URL", _BASE)
    monkeypatch.setenv("GMAIL_MCP_TRANSPORT_SECRET", _SECRET)
    monkeypatch.setenv("BROKER_HANDOFF_SECRET", _SECRET)
    monkeypatch.setenv("SILVER_OAUTH_BROKER_URL", "https://auth.oursilverfamily.com")
    monkeypatch.setenv("GMAIL_MCP_OAUTH_OWNER_EMAIL", _OWNER)
    yield


def _client(monkeypatch):
    app = mts.build_app(host="127.0.0.1", port=0)
    return TestClient(app.streamable_http_app()), app


# ── flag OFF: unchanged ────────────────────────────────────────────


def test_flag_off_mounts_no_auth_routes(monkeypatch):
    monkeypatch.delenv("GMAIL_MCP_OAUTH_ENABLED", raising=False)
    client, _ = _client(monkeypatch)
    # No AS metadata, no PR metadata, no /authorize, no /register.
    assert client.get("/.well-known/oauth-authorization-server").status_code == 404
    assert client.get("/.well-known/oauth-protected-resource/mcp").status_code == 404
    assert client.get("/authorize").status_code == 404
    assert client.post("/register", json={}).status_code == 404


def test_is_oauth_enabled_truthiness(monkeypatch):
    for v in ("1", "true", "YES", "on"):
        monkeypatch.setenv("GMAIL_MCP_OAUTH_ENABLED", v)
        assert mcp_oauth.is_oauth_enabled()
    for v in ("0", "false", "", "no"):
        monkeypatch.setenv("GMAIL_MCP_OAUTH_ENABLED", v)
        assert not mcp_oauth.is_oauth_enabled()


# ── RFC 8414 AS metadata ───────────────────────────────────────────


def test_authorization_server_metadata(oauth_env, monkeypatch):
    client, _ = _client(monkeypatch)
    r = client.get("/.well-known/oauth-authorization-server")
    assert r.status_code == 200
    meta = r.json()
    assert meta["issuer"].rstrip("/") == _BASE
    assert meta["authorization_endpoint"] == f"{_BASE}/authorize"
    assert meta["token_endpoint"] == f"{_BASE}/token"
    assert meta["registration_endpoint"] == f"{_BASE}/register"
    assert meta["revocation_endpoint"] == f"{_BASE}/revoke"
    assert meta["response_types_supported"] == ["code"]
    assert "S256" in meta["code_challenge_methods_supported"]
    assert "authorization_code" in meta["grant_types_supported"]
    assert "refresh_token" in meta["grant_types_supported"]


# ── RFC 9728 protected-resource metadata ───────────────────────────


def test_protected_resource_metadata(oauth_env, monkeypatch):
    client, _ = _client(monkeypatch)
    r = client.get("/.well-known/oauth-protected-resource/mcp")
    assert r.status_code == 200
    meta = r.json()
    assert meta["resource"].rstrip("/") == f"{_BASE}/mcp"
    assert meta["authorization_servers"]
    assert meta["authorization_servers"][0].rstrip("/") == _BASE
    # No secret material leaks into discovery metadata.
    body = r.text.lower()
    assert _SECRET not in body
    assert "secret" not in body


def test_metadata_endpoints_no_secret_material(oauth_env, monkeypatch):
    client, _ = _client(monkeypatch)
    for path in (
        "/.well-known/oauth-authorization-server",
        "/.well-known/oauth-protected-resource/mcp",
    ):
        assert _SECRET not in client.get(path).text


# ── DCR (RFC 7591) ─────────────────────────────────────────────────


def _register_client(client, redirect_uri="https://claude.ai/api/mcp/auth_callback"):
    r = client.post(
        "/register",
        json={
            "redirect_uris": [redirect_uri],
            "token_endpoint_auth_method": "none",
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "scope": "mcp",
        },
    )
    return r


def test_dynamic_client_registration(oauth_env, monkeypatch):
    client, _ = _client(monkeypatch)
    r = _register_client(client)
    assert r.status_code in (200, 201)
    body = r.json()
    assert body["client_id"]
    assert "https://claude.ai/api/mcp/auth_callback" in body["redirect_uris"]


# ── /authorize gates to the broker, issues no code ─────────────────


def test_authorize_redirects_to_broker_without_issuing_code(oauth_env, monkeypatch):
    client, app = _client(monkeypatch)
    reg = _register_client(client).json()
    provider = app._auth_server_provider
    assert len(provider.auth_codes) == 0

    params = {
        "response_type": "code",
        "client_id": reg["client_id"],
        "redirect_uri": "https://claude.ai/api/mcp/auth_callback",
        "code_challenge": "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM",
        "code_challenge_method": "S256",
        "state": "client-csrf-123",
        "scope": "mcp",
    }
    r = client.get("/authorize", params=params, follow_redirects=False)
    assert r.status_code in (302, 307)
    location = r.headers["location"]
    # Redirected to the broker, NOT to the client redirect_uri with a code.
    assert location.startswith("https://auth.oursilverfamily.com/start?")
    parsed = urllib.parse.urlparse(location)
    q = urllib.parse.parse_qs(parsed.query)
    assert "return_url" in q
    return_url = q["return_url"][0]
    assert return_url.startswith(f"{_BASE}/oauth/callback")
    # CRITICAL: no authorization code was minted.
    assert len(provider.auth_codes) == 0


# ── helpers to drive a full authorize -> callback round-trip ───────


def _broker_state_from_authorize(client, reg, challenge="E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"):
    params = {
        "response_type": "code",
        "client_id": reg["client_id"],
        "redirect_uri": "https://claude.ai/api/mcp/auth_callback",
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": "client-csrf-123",
        "scope": "mcp",
    }
    r = client.get("/authorize", params=params, follow_redirects=False)
    location = r.headers["location"]
    return_url = urllib.parse.parse_qs(urllib.parse.urlparse(location).query)["return_url"][0]
    broker_state = urllib.parse.parse_qs(urllib.parse.urlparse(return_url).query)["broker_state"][0]
    return broker_state


def _handoff(email, ttl=60):
    now = int(time.time())
    return jwt.encode({"email": email, "iat": now, "exp": now + ttl}, _SECRET, algorithm="HS256")


# ── /oauth/callback rejections ─────────────────────────────────────


def test_callback_rejects_non_allowlisted_email_no_code(oauth_env, monkeypatch):
    client, app = _client(monkeypatch)
    reg = _register_client(client).json()
    provider = app._auth_server_provider
    broker_state = _broker_state_from_authorize(client, reg)

    r = client.get(
        "/oauth/callback",
        params={"broker_state": broker_state, "silver_oauth": _handoff("attacker@evil.com")},
        follow_redirects=False,
    )
    assert r.status_code == 403
    assert len(provider.auth_codes) == 0


def test_callback_email_comes_from_handoff_not_query_param(oauth_env, monkeypatch):
    """A query ?email=owner can't override the handoff's attacker email."""
    client, app = _client(monkeypatch)
    reg = _register_client(client).json()
    provider = app._auth_server_provider
    broker_state = _broker_state_from_authorize(client, reg)

    r = client.get(
        "/oauth/callback",
        params={
            "broker_state": broker_state,
            "silver_oauth": _handoff("attacker@evil.com"),
            "email": _OWNER,
        },
        follow_redirects=False,
    )
    assert r.status_code == 403
    assert len(provider.auth_codes) == 0


def test_callback_rejects_forged_handoff(oauth_env, monkeypatch):
    client, app = _client(monkeypatch)
    reg = _register_client(client).json()
    provider = app._auth_server_provider
    broker_state = _broker_state_from_authorize(client, reg)

    forged = jwt.encode(
        {"email": _OWNER, "iat": int(time.time()), "exp": int(time.time()) + 60},
        "wrong-secret-but-long-enough-aaaaaaaaaaaa",
        algorithm="HS256",
    )
    r = client.get(
        "/oauth/callback",
        params={"broker_state": broker_state, "silver_oauth": forged},
        follow_redirects=False,
    )
    assert r.status_code == 403
    assert len(provider.auth_codes) == 0


def test_callback_rejects_stale_handoff(oauth_env, monkeypatch):
    client, app = _client(monkeypatch)
    reg = _register_client(client).json()
    provider = app._auth_server_provider
    broker_state = _broker_state_from_authorize(client, reg)

    stale = _handoff(_OWNER, ttl=-10)  # already expired
    r = client.get(
        "/oauth/callback",
        params={"broker_state": broker_state, "silver_oauth": stale},
        follow_redirects=False,
    )
    assert r.status_code == 403
    assert len(provider.auth_codes) == 0


def test_callback_rejects_handoff_without_exp(oauth_env, monkeypatch):
    client, app = _client(monkeypatch)
    reg = _register_client(client).json()
    provider = app._auth_server_provider
    broker_state = _broker_state_from_authorize(client, reg)

    no_exp = jwt.encode({"email": _OWNER, "iat": int(time.time())}, _SECRET, algorithm="HS256")
    r = client.get(
        "/oauth/callback",
        params={"broker_state": broker_state, "silver_oauth": no_exp},
        follow_redirects=False,
    )
    assert r.status_code == 403
    assert len(provider.auth_codes) == 0


def test_callback_rejects_tampered_state(oauth_env, monkeypatch):
    client, app = _client(monkeypatch)
    reg = _register_client(client).json()
    provider = app._auth_server_provider
    broker_state = _broker_state_from_authorize(client, reg)
    tampered = broker_state[:-3] + ("AAA" if not broker_state.endswith("AAA") else "BBB")

    r = client.get(
        "/oauth/callback",
        params={"broker_state": tampered, "silver_oauth": _handoff(_OWNER)},
        follow_redirects=False,
    )
    assert r.status_code == 400
    assert len(provider.auth_codes) == 0


def test_callback_missing_params(oauth_env, monkeypatch):
    client, _ = _client(monkeypatch)
    assert client.get("/oauth/callback").status_code == 400
    assert client.get("/oauth/callback", params={"silver_oauth": "x"}).status_code == 400


# ── happy path: owner completes the gate, code is minted ───────────


def test_callback_owner_mints_code_and_redirects_to_client(oauth_env, monkeypatch):
    client, app = _client(monkeypatch)
    reg = _register_client(client).json()
    provider = app._auth_server_provider
    broker_state = _broker_state_from_authorize(client, reg)

    r = client.get(
        "/oauth/callback",
        params={"broker_state": broker_state, "silver_oauth": _handoff(_OWNER)},
        follow_redirects=False,
    )
    assert r.status_code == 302
    location = r.headers["location"]
    # Redirect goes to the client's registered redirect_uri (from signed
    # state), carrying code + the client's original state.
    assert location.startswith("https://claude.ai/api/mcp/auth_callback?")
    q = urllib.parse.parse_qs(urllib.parse.urlparse(location).query)
    assert q["state"] == ["client-csrf-123"]
    assert q["code"]
    assert len(provider.auth_codes) == 1
    # The stored code carries the PKCE challenge from /authorize.
    code = q["code"][0]
    assert provider.auth_codes[code].code_challenge == "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"


def test_callback_state_is_single_use(oauth_env, monkeypatch):
    client, app = _client(monkeypatch)
    reg = _register_client(client).json()
    provider = app._auth_server_provider
    broker_state = _broker_state_from_authorize(client, reg)

    first = client.get(
        "/oauth/callback",
        params={"broker_state": broker_state, "silver_oauth": _handoff(_OWNER)},
        follow_redirects=False,
    )
    assert first.status_code == 302
    # Replaying the same broker_state must not mint a second code.
    second = client.get(
        "/oauth/callback",
        params={"broker_state": broker_state, "silver_oauth": _handoff(_OWNER)},
        follow_redirects=False,
    )
    assert second.status_code == 400
    assert len(provider.auth_codes) == 1


# ── provider unit: code single-use + PKCE persistence ──────────────


def test_provider_authorization_code_single_use(oauth_env):
    import asyncio

    from mcp.shared.auth import OAuthClientInformationFull

    provider = mcp_oauth.GatedBrokerOAuthProvider(
        broker_start_url="https://auth.oursilverfamily.com/start",
        callback_url=f"{_BASE}/oauth/callback",
        owner=_OWNER,
    )
    client_info = OAuthClientInformationFull(
        client_id="c1",
        redirect_uris=["https://claude.ai/api/mcp/auth_callback"],
        scope="mcp",
    )

    async def drive():
        await provider.register_client(client_info)
        # Manually build a signed state for c1.
        state = mcp_oauth._encode_state(
            {
                "client_id": "c1",
                "redirect_uri": "https://claude.ai/api/mcp/auth_callback",
                "redirect_uri_provided_explicitly": True,
                "code_challenge": "abc",
                "scopes": ["mcp"],
                "client_state": "s",
                "resource": f"{_BASE}/mcp",
                "nonce": "n1",
            }
        )
        redirect = provider.resume_after_broker(state_token=state, email=_OWNER)
        assert redirect
        code_val = next(iter(provider.auth_codes))
        loaded = await provider.load_authorization_code(client_info, code_val)
        assert loaded is not None
        token = await provider.exchange_authorization_code(client_info, loaded)
        assert token.access_token
        assert token.refresh_token
        # Second exchange of the same (now consumed) code fails.
        with pytest.raises(mcp_oauth.TokenError):
            await provider.exchange_authorization_code(client_info, loaded)

    asyncio.run(drive())


def test_provider_refresh_token_rotates(oauth_env):
    import asyncio

    from mcp.shared.auth import OAuthClientInformationFull

    provider = mcp_oauth.GatedBrokerOAuthProvider(
        broker_start_url="https://auth.oursilverfamily.com/start",
        callback_url=f"{_BASE}/oauth/callback",
        owner=_OWNER,
    )
    client_info = OAuthClientInformationFull(
        client_id="c1", redirect_uris=["https://claude.ai/api/mcp/auth_callback"], scope="mcp"
    )

    async def drive():
        await provider.register_client(client_info)
        first = provider._issue_token_pair("c1", ["mcp"], f"{_BASE}/mcp")
        old_refresh = provider.refresh_tokens[first.refresh_token]
        rotated = await provider.exchange_refresh_token(client_info, old_refresh, ["mcp"])
        # Old refresh token is gone (rotation), new one is present.
        assert first.refresh_token not in provider.refresh_tokens
        assert rotated.refresh_token in provider.refresh_tokens
        assert rotated.access_token in provider.access_tokens
        # RFC 8707 resource audience is preserved across rotation (not dropped).
        assert provider.access_tokens[rotated.access_token].resource == f"{_BASE}/mcp"

    asyncio.run(drive())
