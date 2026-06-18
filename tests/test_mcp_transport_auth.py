"""Tests for the token-identity transport-auth model on the MCP server.

Three slices, all hermetic (no real DB, no live server):

  1. **Mint endpoint** (`POST /admin/transport-tokens`) — admin-gated,
     email→user_id resolved SERVER-SIDE, caller-supplied user_id ignored,
     unknown email 404, missing email 400, secret-unavailable 503, token
     carries an expiry.
  2. **`/mcp` middleware** — valid token sets identity + passes through,
     expired/wrong-aud/missing rejected (401 when the enforce flag is on,
     pass-through when off), `/admin/*` unaffected.
  3. **Tool scoping** — transport identity auto-materializes an ephemeral
     session and OVERRIDES any pre-registered session's user_id (a token
     for user A can't read user B's data even via a B-registered
     session_id). Without transport identity the register_session path is
     unchanged.

The users-table lookup (`_resolve_user_id_by_email`) and the underlying
tool impls are monkeypatched, matching the style of
`test_mcp_tools_server.py`.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import time

import jwt
import pytest
from gmail_search.agents import mcp_tools_server as mts
from starlette.testclient import TestClient

# >=32 bytes so the transport-secret gate accepts it.
_SECRET = "transport-secret-not-for-prod-padded-32+"
_ADMIN = "admin-tok-xyz"


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    """Isolate the module-level registries + the transport contextvar
    around every test, and pin a known signing secret + admin token."""
    mts._SESSIONS.clear()
    mts._TRANSPORT_SESSIONS.clear()
    mts._SESSION_CALLS.clear()
    monkeypatch.setenv("GMAIL_MCP_TRANSPORT_SECRET", _SECRET)
    monkeypatch.setenv("GMAIL_MCP_ADMIN_TOKEN", _ADMIN)
    monkeypatch.delenv("GMAIL_MCP_REQUIRE_TRANSPORT_AUTH", raising=False)
    monkeypatch.delenv("GMAIL_MCP_TRANSPORT_ALLOW_SHARED_SECRET", raising=False)
    # Default: any email resolves to a deterministic user_id unless a
    # test overrides this.
    monkeypatch.setattr(
        mts,
        "_resolve_user_id_by_email",
        lambda email: {"a@x.com": "u_aaa", "b@x.com": "u_bbb"}.get(email.lower()),
    )
    # Ensure no transport identity bleeds in from a prior test.
    tok = mts._transport_user_id.set(None)
    yield
    mts._transport_user_id.reset(tok)
    mts._SESSIONS.clear()
    mts._TRANSPORT_SESSIONS.clear()
    mts._SESSION_CALLS.clear()


# ── 1. Mint endpoint ───────────────────────────────────────────────


def _mint_client():
    app = mts.build_app(host="127.0.0.1", port=0)
    # /admin/* is loopback-gated; the real host-local callers report a
    # loopback client, so make the TestClient do the same (its default is
    # the synthetic "testclient" host).
    return TestClient(app.streamable_http_app(), client=("127.0.0.1", 50000))


def test_mint_requires_admin_token():
    c = _mint_client()
    # No token
    r = c.post("/admin/transport-tokens", json={"email": "a@x.com"})
    assert r.status_code == 401
    # Wrong token
    r = c.post(
        "/admin/transport-tokens",
        headers={"Authorization": "Bearer nope"},
        json={"email": "a@x.com"},
    )
    assert r.status_code == 401


def test_mint_resolves_user_id_server_side_and_ignores_caller_supplied():
    c = _mint_client()
    r = c.post(
        "/admin/transport-tokens",
        headers={"Authorization": f"Bearer {_ADMIN}"},
        # Attacker tries to inject a foreign user_id — must be ignored.
        json={"email": "a@x.com", "user_id": "u_attacker"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["user_id"] == "u_aaa"
    assert "u_attacker" not in body["token"]
    # Decode the token: uid is the server-resolved id, aud is pinned.
    claims = jwt.decode(body["token"], _SECRET, algorithms=["HS256"], audience="mcp-transport")
    assert claims["uid"] == "u_aaa"
    assert claims["email"] == "a@x.com"
    assert claims["aud"] == "mcp-transport"
    assert body["expires_at"] == claims["exp"]
    assert claims["exp"] > time.time()


def test_mint_unknown_email_404():
    c = _mint_client()
    r = c.post(
        "/admin/transport-tokens",
        headers={"Authorization": f"Bearer {_ADMIN}"},
        json={"email": "ghost@x.com"},
    )
    assert r.status_code == 404


def test_mint_missing_email_400():
    c = _mint_client()
    r = c.post(
        "/admin/transport-tokens",
        headers={"Authorization": f"Bearer {_ADMIN}"},
        json={},
    )
    assert r.status_code == 400


def test_mint_503_when_secret_unavailable(monkeypatch):
    monkeypatch.delenv("GMAIL_MCP_TRANSPORT_SECRET", raising=False)
    monkeypatch.delenv("GMS_SESSION_SECRET", raising=False)
    c = _mint_client()
    r = c.post(
        "/admin/transport-tokens",
        headers={"Authorization": f"Bearer {_ADMIN}"},
        json={"email": "a@x.com"},
    )
    assert r.status_code == 503


def test_mint_honors_custom_ttl():
    c = _mint_client()
    before = int(time.time())
    r = c.post(
        "/admin/transport-tokens",
        headers={"Authorization": f"Bearer {_ADMIN}"},
        json={"email": "a@x.com", "ttl_seconds": 60},
    )
    assert r.status_code == 200
    exp = r.json()["expires_at"]
    assert before + 55 <= exp <= before + 70


def test_shared_secret_fallback_requires_opt_in(monkeypatch):
    """GMS_SESSION_SECRET is only used as the transport secret when the
    operator explicitly opts in — otherwise minting is disabled (503)."""
    monkeypatch.delenv("GMAIL_MCP_TRANSPORT_SECRET", raising=False)
    monkeypatch.setenv("GMS_SESSION_SECRET", "session-secret-padded-to-32-bytes-yyyy")
    c = _mint_client()
    # Opt-in off → 503
    r = c.post(
        "/admin/transport-tokens",
        headers={"Authorization": f"Bearer {_ADMIN}"},
        json={"email": "a@x.com"},
    )
    assert r.status_code == 503
    # Opt-in on → mints
    monkeypatch.setenv("GMAIL_MCP_TRANSPORT_ALLOW_SHARED_SECRET", "1")
    r = c.post(
        "/admin/transport-tokens",
        headers={"Authorization": f"Bearer {_ADMIN}"},
        json={"email": "a@x.com"},
    )
    assert r.status_code == 200


# ── 2. /mcp middleware ──────────────────────────────────────────────
#
# A pass-through (allowed) /mcp request reaches FastMCP's transport,
# which rejects a bare request that lacks a negotiated MCP session —
# but with a non-401 status. So "not 401" is our pass-through signal;
# "401 with our JSON" is the rejection signal.


def _valid_token(uid="u_aaa", email="a@x.com", ttl=3600):
    tok, _ = mts.mint_transport_token(user_id=uid, email=email, ttl_seconds=ttl)
    return tok


def _expired_token():
    now = int(time.time())
    return jwt.encode(
        {"uid": "u_aaa", "email": "a@x.com", "aud": "mcp-transport", "iat": now - 100, "exp": now - 10},
        _SECRET,
        algorithm="HS256",
    )


def _wrong_aud_token():
    now = int(time.time())
    return jwt.encode(
        {"uid": "u_aaa", "email": "a@x.com", "aud": "something-else", "iat": now, "exp": now + 3600},
        _SECRET,
        algorithm="HS256",
    )


def _mcp_post(client, headers=None):
    h = {"Accept": "application/json, text/event-stream"}
    if headers:
        h.update(headers)
    return client.post("/mcp", json={"jsonrpc": "2.0", "method": "ping", "id": 1}, headers=h)


def test_middleware_valid_token_passes_through():
    asgi = mts.build_asgi_app(host="127.0.0.1", port=0)
    with TestClient(asgi) as c:
        r = _mcp_post(c, {"Authorization": f"Bearer {_valid_token()}"})
        assert r.status_code != 401


def test_middleware_missing_token_passthrough_when_flag_off():
    asgi = mts.build_asgi_app(host="127.0.0.1", port=0)
    with TestClient(asgi) as c:
        r = _mcp_post(c)
        assert r.status_code != 401


def test_middleware_missing_token_401_when_flag_on(monkeypatch):
    monkeypatch.setenv("GMAIL_MCP_REQUIRE_TRANSPORT_AUTH", "1")
    asgi = mts.build_asgi_app(host="127.0.0.1", port=0)
    with TestClient(asgi) as c:
        r = _mcp_post(c)
        assert r.status_code == 401
        assert "error" in r.json()


def test_middleware_expired_token_401_when_flag_on(monkeypatch):
    monkeypatch.setenv("GMAIL_MCP_REQUIRE_TRANSPORT_AUTH", "1")
    asgi = mts.build_asgi_app(host="127.0.0.1", port=0)
    with TestClient(asgi) as c:
        r = _mcp_post(c, {"Authorization": f"Bearer {_expired_token()}"})
        assert r.status_code == 401


def test_middleware_expired_token_passthrough_when_flag_off():
    asgi = mts.build_asgi_app(host="127.0.0.1", port=0)
    with TestClient(asgi) as c:
        r = _mcp_post(c, {"Authorization": f"Bearer {_expired_token()}"})
        assert r.status_code != 401


def test_middleware_wrong_aud_rejected_when_flag_on(monkeypatch):
    monkeypatch.setenv("GMAIL_MCP_REQUIRE_TRANSPORT_AUTH", "1")
    asgi = mts.build_asgi_app(host="127.0.0.1", port=0)
    with TestClient(asgi) as c:
        r = _mcp_post(c, {"Authorization": f"Bearer {_wrong_aud_token()}"})
        assert r.status_code == 401


def test_middleware_empty_uid_token_rejected_when_flag_on(monkeypatch):
    """A validly-signed token carrying no `uid` claim carries no usable
    identity — it must be treated as an auth failure, not scoped to an
    empty user_id."""
    monkeypatch.setenv("GMAIL_MCP_REQUIRE_TRANSPORT_AUTH", "1")
    now = int(time.time())
    no_uid = jwt.encode(
        {"email": "a@x.com", "aud": "mcp-transport", "iat": now, "exp": now + 3600},
        _SECRET,
        algorithm="HS256",
    )
    asgi = mts.build_asgi_app(host="127.0.0.1", port=0)
    with TestClient(asgi) as c:
        r = _mcp_post(c, {"Authorization": f"Bearer {no_uid}"})
        assert r.status_code == 401


def test_middleware_admin_routes_unaffected():
    """/admin/* must not be subject to the /mcp 401 even with the flag
    on — they keep their own admin-token gate."""
    asgi = mts.build_asgi_app(host="127.0.0.1", port=0)
    import os

    os.environ["GMAIL_MCP_REQUIRE_TRANSPORT_AUTH"] = "1"
    try:
        with TestClient(asgi, client=("127.0.0.1", 50000)) as c:
            # No transport token, but valid admin token → admin route works.
            r = c.get("/admin/calls/sess-x", headers={"Authorization": f"Bearer {_ADMIN}"})
            assert r.status_code == 200
            assert r.json() == {"calls": []}
    finally:
        os.environ.pop("GMAIL_MCP_REQUIRE_TRANSPORT_AUTH", None)


def test_middleware_sets_and_resets_identity():
    """The verified token's uid is visible inside the request and reset
    after. We assert it by routing through a tool while the contextvar
    is set (covered fully in scoping tests); here we check the verify
    helper + contextvar wiring directly."""
    claims = mts.verify_transport_token(_valid_token(uid="u_zzz"))
    assert claims is not None and claims["uid"] == "u_zzz"
    assert mts.verify_transport_token(_expired_token()) is None
    assert mts.verify_transport_token(_wrong_aud_token()) is None


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def test_verify_rejects_alg_none_token():
    """An `alg=none` (unsigned) token with an otherwise-valid payload
    must be rejected — algorithm pinning to HS256 means an attacker
    can't strip the signature and have it accepted."""
    now = int(time.time())
    header = _b64url(json.dumps({"alg": "none", "typ": "JWT"}).encode())
    payload = _b64url(
        json.dumps({"uid": "u_aaa", "email": "a@x.com", "aud": "mcp-transport", "iat": now, "exp": now + 3600}).encode()
    )
    # Unsigned token: header.payload. (empty signature segment)
    forged = f"{header}.{payload}."
    assert mts.verify_transport_token(forged) is None


def test_verify_rejects_rs256_confusion_token():
    """A token whose header claims an asymmetric alg (RS256) must be
    rejected by the HS256 pin — defends against the classic
    RS256-public-key-as-HS256-secret confusion attack."""
    now = int(time.time())
    # Hand-forge the classic confusion token: header says RS256, but the
    # signature is an HMAC-SHA256 over signing-input using the transport
    # secret as the HMAC key (i.e. an attacker who treats the verifier's
    # HMAC secret as if it were an RS256 public key). PyJWT validates the
    # header `alg` against our pinned `algorithms=["HS256"]` list and
    # refuses RS256 outright — so the forged signature never matters.
    header = _b64url(json.dumps({"alg": "RS256", "typ": "JWT"}).encode())
    payload = _b64url(
        json.dumps({"uid": "u_aaa", "email": "a@x.com", "aud": "mcp-transport", "iat": now, "exp": now + 3600}).encode()
    )
    signing_input = f"{header}.{payload}".encode("ascii")
    sig = _b64url(hmac.new(_SECRET.encode("utf-8"), signing_input, hashlib.sha256).digest())
    forged = f"{header}.{payload}.{sig}"
    assert mts.verify_transport_token(forged) is None


def test_verify_rejects_token_missing_exp():
    """A token correctly signed with the REAL transport secret but with
    no `exp` claim must be rejected (options require exp) — otherwise it
    would be a non-expiring credential."""
    now = int(time.time())
    no_exp = jwt.encode(
        {"uid": "u_aaa", "email": "a@x.com", "aud": "mcp-transport", "iat": now},
        _SECRET,
        algorithm="HS256",
    )
    assert mts.verify_transport_token(no_exp) is None


# ── 3. Tool scoping prefers transport identity ──────────────────────


def test_transport_identity_auto_materializes_unregistered_session(monkeypatch):
    """With transport identity set, a tool call against a session_id
    that was NEVER registered must succeed (auto-materialized) and scope
    to the token's user_id."""
    captured = {}

    async def fake_batch(searches, *, user_id=None):
        captured["user_id"] = user_id
        return {"results": []}

    monkeypatch.setattr(mts, "_search_emails_batch_impl", fake_batch)
    monkeypatch.setenv("DB_DSN", "postgres://server-side")

    tok = mts._transport_user_id.set("u_aaa")
    try:
        out = asyncio.run(mts._tool_search_emails_batch("brand-new-sess", searches=[{"query": "x"}]))
    finally:
        mts._transport_user_id.reset(tok)

    assert out == {"results": []}
    assert captured["user_id"] == "u_aaa"
    # The ephemeral context lives in the transport-scoped store keyed by
    # (uid, session_id) — NOT in `_SESSIONS` (which the orchestrator
    # owns) — and got the server-side DSN, not caller input.
    assert "brand-new-sess" not in mts._SESSIONS
    eph = mts._TRANSPORT_SESSIONS[("u_aaa", "brand-new-sess")]
    assert eph.db_dsn == "postgres://server-side"
    assert eph.user_id == "u_aaa"


def test_transport_identity_overrides_registered_session(monkeypatch):
    """Adversarial: a session_id pre-registered to user B must NOT let a
    token for user A read B's data. The transport identity overrides the
    registered user_id."""
    captured = {}

    async def fake_batch(searches, *, user_id=None):
        captured["user_id"] = user_id
        return {"results": []}

    monkeypatch.setattr(mts, "_search_emails_batch_impl", fake_batch)

    # B registered this session out-of-band (orchestrator path).
    mts.register_session("victim-sess", evidence_records=None, db_dsn=None, user_id="u_bbb")

    # A's transport token is presented for the SAME session_id.
    tok = mts._transport_user_id.set("u_aaa")
    try:
        asyncio.run(mts._tool_search_emails_batch("victim-sess", searches=[{"query": "x"}]))
    finally:
        mts._transport_user_id.reset(tok)

    # The call scoped to A (the token), NOT B (the registered context).
    assert captured["user_id"] == "u_aaa"
    # SECURITY: the override is request-local — B's stored registration
    # is NOT mutated. (See test_transport_override_does_not_poison_*
    # for the full cross-request invariant.)
    assert mts._SESSIONS["victim-sess"].user_id == "u_bbb"


def test_transport_override_does_not_poison_registered_session(monkeypatch):
    """REGRESSION (codex finding): a transport caller (owner A) resolving
    a session_id that the in-process orchestrator registered for owner B
    must NOT repin the stored context. The override has to be
    request-local — otherwise A poisons B's session and B's subsequent
    legacy/orchestrator calls inherit A's identity (cross-tenant bleed
    via shared mutable state)."""
    # B registered "S" out-of-band.
    mts.register_session("S", evidence_records=None, db_dsn=None, user_id="u_B")

    # A resolves "S" under transport identity → scoped to A.
    tok = mts._transport_user_id.set("u_A")
    try:
        ctx_a = mts._resolve_ctx("S")
        assert ctx_a.user_id == "u_A"
    finally:
        mts._transport_user_id.reset(tok)

    # The stored registration is untouched...
    assert mts._SESSIONS["S"].user_id == "u_B"
    # ...and a subsequent NO-transport resolve still returns B's context.
    ctx_b = mts._resolve_ctx("S")
    assert ctx_b is mts._SESSIONS["S"]
    assert ctx_b.user_id == "u_B"


def test_unregister_session_drops_transport_scoped_context():
    """unregister_session must also evict the transport-scoped ephemeral
    context for that session_id (pins the cleanup loop). Resolve a
    session under a transport identity to create the
    `_TRANSPORT_SESSIONS[(uid, session_id)]` entry, then unregister and
    assert it's gone."""
    tok = mts._transport_user_id.set("u_aaa")
    try:
        mts._resolve_ctx("eph-sess")
    finally:
        mts._transport_user_id.reset(tok)
    assert ("u_aaa", "eph-sess") in mts._TRANSPORT_SESSIONS

    mts.unregister_session("eph-sess")
    assert ("u_aaa", "eph-sess") not in mts._TRANSPORT_SESSIONS


def test_no_transport_identity_preserves_register_session_path(monkeypatch):
    """Without a transport identity, behavior is exactly as before:
    an unregistered session_id raises, and a registered one uses its own
    user_id verbatim."""
    captured = {}

    async def fake_batch(searches, *, user_id=None):
        captured["user_id"] = user_id
        return {"results": []}

    monkeypatch.setattr(mts, "_search_emails_batch_impl", fake_batch)

    # Unregistered → still raises (no auto-materialize without a token).
    with pytest.raises(RuntimeError, match="not registered"):
        asyncio.run(mts._tool_search_emails_batch("nope-sess", searches=[{"query": "x"}]))

    # Registered → uses the registered user_id.
    mts.register_session("orch-sess", evidence_records=None, db_dsn=None, user_id="u_bbb")
    asyncio.run(mts._tool_search_emails_batch("orch-sess", searches=[{"query": "x"}]))
    assert captured["user_id"] == "u_bbb"


# ── 4. Service tokens (mcp-service audience) ────────────────────────
#
# A service token authenticates a TRUSTED server-side MCP client
# (claudebox) to satisfy /mcp enforcement, but carries NO tenant —
# scoping stays on the REGISTERED session (the orchestrator calls
# register_session(user_id=...) before each run). Untrusted VM clients
# still use tenant-bound `mcp-transport` tokens; they cannot obtain a
# service token (cc-web only mints transport tokens).


def _service_token(ttl=3600):
    return mts.mint_service_token(ttl_seconds=ttl)


def test_mint_service_token_shape():
    """A service token carries aud=mcp-service, an exp, an iat, and NO
    tenant claims (uid/email)."""
    tok = mts.mint_service_token(ttl_seconds=60)
    claims = jwt.decode(tok, _SECRET, algorithms=["HS256"], audience="mcp-service")
    assert claims["aud"] == "mcp-service"
    assert "uid" not in claims
    assert "email" not in claims
    assert claims["exp"] > time.time()
    assert "iat" in claims


def test_mint_service_token_default_ttl_is_long():
    """The default TTL is long (a static server-side client credential),
    distinct from the short transport TTL."""
    before = int(time.time())
    tok = mts.mint_service_token()
    claims = jwt.decode(tok, _SECRET, algorithms=["HS256"], audience="mcp-service")
    # Default is 30 days; assert it's comfortably more than a day.
    assert claims["exp"] > before + 86400 * 2


def test_mint_service_token_raises_when_secret_unavailable(monkeypatch):
    monkeypatch.delenv("GMAIL_MCP_TRANSPORT_SECRET", raising=False)
    monkeypatch.delenv("GMS_SESSION_SECRET", raising=False)
    with pytest.raises(RuntimeError):
        mts.mint_service_token()


# ── verify_token accepts both audiences ─────────────────────────────


def test_verify_token_accepts_transport_aud():
    claims = mts.verify_token(_valid_token(uid="u_zzz"))
    assert claims is not None
    assert claims["aud"] == "mcp-transport"
    assert claims["uid"] == "u_zzz"


def test_verify_token_accepts_service_aud():
    claims = mts.verify_token(_service_token())
    assert claims is not None
    assert claims["aud"] == "mcp-service"
    assert "uid" not in claims


def test_verify_token_rejects_unknown_aud():
    """A validly-signed token with a 3rd/unknown audience is rejected —
    only the two known audiences are accepted."""
    now = int(time.time())
    other = jwt.encode(
        {"aud": "some-other-aud", "iat": now, "exp": now + 3600},
        _SECRET,
        algorithm="HS256",
    )
    assert mts.verify_token(other) is None


def test_verify_token_rejects_alg_none():
    now = int(time.time())
    header = _b64url(json.dumps({"alg": "none", "typ": "JWT"}).encode())
    payload = _b64url(json.dumps({"aud": "mcp-service", "iat": now, "exp": now + 3600}).encode())
    forged = f"{header}.{payload}."
    assert mts.verify_token(forged) is None


def test_verify_token_rejects_wrong_secret():
    now = int(time.time())
    bad = jwt.encode(
        {"aud": "mcp-service", "iat": now, "exp": now + 3600},
        "a-different-secret-padded-to-32-bytes!!",
        algorithm="HS256",
    )
    assert mts.verify_token(bad) is None


def test_verify_token_rejects_expired_service_token():
    now = int(time.time())
    expired = jwt.encode(
        {"aud": "mcp-service", "iat": now - 100, "exp": now - 10},
        _SECRET,
        algorithm="HS256",
    )
    assert mts.verify_token(expired) is None


def test_verify_token_rejects_missing_exp():
    now = int(time.time())
    no_exp = jwt.encode({"aud": "mcp-service", "iat": now}, _SECRET, algorithm="HS256")
    assert mts.verify_token(no_exp) is None


def test_verify_token_rejects_missing_aud():
    now = int(time.time())
    no_aud = jwt.encode({"iat": now, "exp": now + 3600}, _SECRET, algorithm="HS256")
    assert mts.verify_token(no_aud) is None


# ── middleware with a service token ─────────────────────────────────


def test_middleware_service_token_passes_through():
    asgi = mts.build_asgi_app(host="127.0.0.1", port=0)
    with TestClient(asgi) as c:
        r = _mcp_post(c, {"Authorization": f"Bearer {_service_token()}"})
        assert r.status_code != 401


def test_middleware_service_token_200_when_flag_on(monkeypatch):
    monkeypatch.setenv("GMAIL_MCP_REQUIRE_TRANSPORT_AUTH", "1")
    asgi = mts.build_asgi_app(host="127.0.0.1", port=0)
    with TestClient(asgi) as c:
        # Service token under enforcement → passes (not 401).
        r = _mcp_post(c, {"Authorization": f"Bearer {_service_token()}"})
        assert r.status_code != 401
        # No token → 401.
        r = _mcp_post(c)
        assert r.status_code == 401


def test_service_token_does_not_set_transport_identity(monkeypatch):
    """A service token authenticates but carries NO tenant: it must NOT
    set `_transport_user_id`. We assert this end-to-end by routing a tool
    call through the middleware with a service token and a REGISTERED
    session — the call scopes to the REGISTERED user_id, not a transport
    one, and the transport contextvar is unset inside the tool."""
    seen = {}

    async def fake_batch(searches, *, user_id=None):
        seen["user_id"] = user_id
        seen["transport_uid"] = mts._transport_user_id.get()
        return {"results": []}

    monkeypatch.setattr(mts, "_search_emails_batch_impl", fake_batch)

    # The orchestrator registered this session for a specific tenant.
    mts.register_session("svc-sess", evidence_records=None, db_dsn=None, user_id="u_bbb")

    # Drive the middleware directly with a service token so the aud
    # branch runs, then invoke the tool from inside the request.
    service_tok = _service_token()

    async def app(scope, receive, send):
        # Inside the request: the service token must NOT have set identity.
        await mts._tool_search_emails_batch("svc-sess", searches=[{"query": "x"}])
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    mw = mts._TransportAuthMiddleware(app)
    scope = {
        "type": "http",
        "path": "/mcp",
        "headers": [(b"authorization", f"Bearer {service_tok}".encode("latin-1"))],
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    sent = []

    async def send(msg):
        sent.append(msg)

    monkeypatch.setenv("GMAIL_MCP_REQUIRE_TRANSPORT_AUTH", "1")
    asyncio.run(mw(scope, receive, send))

    # Scoped to the REGISTERED tenant, not a transport identity.
    assert seen["user_id"] == "u_bbb"
    # No transport identity was set by the service token.
    assert seen["transport_uid"] is None
    # The registered session is untouched (service tokens don't poison it).
    assert mts._SESSIONS["svc-sess"].user_id == "u_bbb"


def test_service_token_does_not_auto_materialize_unregistered_session(monkeypatch):
    """A service token carries no tenant, so an UNregistered session_id
    still raises "not registered" — unlike a transport token, it does NOT
    auto-materialize an ephemeral tenant-scoped context."""
    raised = {}

    async def app(scope, receive, send):
        try:
            await mts._tool_search_emails_batch("ghost-sess", searches=[{"query": "x"}])
        except RuntimeError as exc:
            raised["err"] = str(exc)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    mw = mts._TransportAuthMiddleware(app)
    scope = {
        "type": "http",
        "path": "/mcp",
        "headers": [(b"authorization", f"Bearer {_service_token()}".encode("latin-1"))],
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(msg):
        pass

    asyncio.run(mw(scope, receive, send))
    assert "not registered" in raised.get("err", "")
    # And nothing got materialized in the transport store.
    assert not any(k[1] == "ghost-sess" for k in mts._TRANSPORT_SESSIONS)


# ── /admin/service-tokens endpoint ──────────────────────────────────


def test_service_token_endpoint_requires_admin():
    c = _mint_client()
    r = c.post("/admin/service-tokens", json={})
    assert r.status_code == 401
    r = c.post("/admin/service-tokens", headers={"Authorization": "Bearer nope"}, json={})
    assert r.status_code == 401


def test_service_token_endpoint_mints():
    c = _mint_client()
    r = c.post(
        "/admin/service-tokens",
        headers={"Authorization": f"Bearer {_ADMIN}"},
        json={},
    )
    assert r.status_code == 200
    body = r.json()
    claims = jwt.decode(body["token"], _SECRET, algorithms=["HS256"], audience="mcp-service")
    assert claims["aud"] == "mcp-service"
    assert "uid" not in claims
    assert body["expires_at"] == claims["exp"]


def test_service_token_endpoint_honors_custom_ttl():
    c = _mint_client()
    before = int(time.time())
    r = c.post(
        "/admin/service-tokens",
        headers={"Authorization": f"Bearer {_ADMIN}"},
        json={"ttl_seconds": 60},
    )
    assert r.status_code == 200
    exp = r.json()["expires_at"]
    assert before + 55 <= exp <= before + 70


def test_service_token_endpoint_503_when_secret_unavailable(monkeypatch):
    monkeypatch.delenv("GMAIL_MCP_TRANSPORT_SECRET", raising=False)
    monkeypatch.delenv("GMS_SESSION_SECRET", raising=False)
    c = _mint_client()
    r = c.post(
        "/admin/service-tokens",
        headers={"Authorization": f"Bearer {_ADMIN}"},
        json={},
    )
    assert r.status_code == 503


# ── 5. /admin/* loopback restriction ────────────────────────────────
#
# Once the firewall opens :7878 to the bhatti VM subnet, a remote VM must
# not be able to reach any /admin/* route. All legit admin callers are
# host-local. We unit-test the guard against a fake request (Starlette's
# TestClient always reports a loopback client) and assert the 403-before-
# 401 ordering so a remote caller never learns whether its token is valid.


class _FakeClient:
    def __init__(self, host):
        self.host = host


class _FakeRequest:
    """Minimal stand-in exposing the attributes the admin guard reads:
    a `.client` with `.host`, and `.headers` for the token check."""

    def __init__(self, *, client_host, auth=None):
        self.client = _FakeClient(client_host) if client_host is not None else None
        self.headers = {}
        if auth is not None:
            self.headers["authorization"] = auth


def test_admin_guard_rejects_non_loopback_before_token_check():
    """A non-loopback caller with a VALID admin token is still rejected —
    with 403, BEFORE the token is even consulted (no leak of validity)."""
    req = _FakeRequest(client_host="10.0.1.5", auth=f"Bearer {_ADMIN}")
    resp = mts._admin_guard(req)
    assert resp is not None
    assert resp.status_code == 403


def test_admin_guard_rejects_when_client_is_none():
    """If we can't prove the client is loopback (request.client is None),
    treat it as non-loopback → 403."""
    req = _FakeRequest(client_host=None, auth=f"Bearer {_ADMIN}")
    resp = mts._admin_guard(req)
    assert resp is not None
    assert resp.status_code == 403


def test_admin_guard_loopback_valid_token_allowed():
    req = _FakeRequest(client_host="127.0.0.1", auth=f"Bearer {_ADMIN}")
    assert mts._admin_guard(req) is None
    # IPv6 loopback too.
    req6 = _FakeRequest(client_host="::1", auth=f"Bearer {_ADMIN}")
    assert mts._admin_guard(req6) is None


def test_admin_guard_loopback_bad_token_401():
    """Loopback passes the address gate, then the token gate rejects a
    wrong token with 401."""
    req = _FakeRequest(client_host="127.0.0.1", auth="Bearer nope")
    resp = mts._admin_guard(req)
    assert resp is not None
    assert resp.status_code == 401


def test_admin_guard_allow_remote_escape_hatch(monkeypatch):
    """With GMAIL_MCP_ADMIN_ALLOW_REMOTE=1, a non-loopback caller with a
    valid token is allowed (for deployments that legitimately need remote
    admin)."""
    monkeypatch.setenv("GMAIL_MCP_ADMIN_ALLOW_REMOTE", "1")
    req = _FakeRequest(client_host="10.0.1.5", auth=f"Bearer {_ADMIN}")
    assert mts._admin_guard(req) is None
    # But a bad token under the escape hatch still fails (401).
    bad = _FakeRequest(client_host="10.0.1.5", auth="Bearer nope")
    resp = mts._admin_guard(bad)
    assert resp is not None
    assert resp.status_code == 401


def test_admin_route_loopback_valid_token_still_works():
    """End-to-end via TestClient (reports a loopback client): a valid
    admin token on /admin/transport-tokens works (200) — the loopback
    gate doesn't break the normal host-local path."""
    c = _mint_client()
    r = c.post(
        "/admin/transport-tokens",
        headers={"Authorization": f"Bearer {_ADMIN}"},
        json={"email": "a@x.com"},
    )
    assert r.status_code == 200
