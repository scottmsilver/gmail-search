"""Tests for the per-turn `mcp-session` token identity.

A session token binds ONE already-registered session_id (`sid`) + its
owner (`uid`) to a single /mcp bearer token, for MCP clients (the
pi-mcp-adapter extension) that can't inject a `session_id` tool
argument. Mirrors the style of `test_mcp_transport_auth.py`.

Three slices:
  1. mint/verify primitives (roundtrip, wrong audience).
  2. `_resolve_ctx` under a session token (happy path, session_id
     mismatch, owner mismatch, unregistered sid).
  3. the `/admin/session-tokens` admin route.
"""

from __future__ import annotations

import asyncio
import time

import jwt
import pytest
from starlette.testclient import TestClient

from gmail_search.agents import mcp_tools_server as mts

_SECRET = "transport-secret-not-for-prod-padded-32+"
_ADMIN = "admin-tok-xyz"


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    mts._SESSIONS.clear()
    mts._TRANSPORT_SESSIONS.clear()
    mts._SESSION_CALLS.clear()
    monkeypatch.setenv("GMAIL_MCP_TRANSPORT_SECRET", _SECRET)
    monkeypatch.setenv("GMAIL_MCP_ADMIN_TOKEN", _ADMIN)
    monkeypatch.delenv("GMAIL_MCP_SESSION_TOKEN_TTL", raising=False)
    uid_tok = mts._transport_user_id.set(None)
    sid_tok = mts._transport_session_id.set(None)
    yield
    mts._transport_user_id.reset(uid_tok)
    mts._transport_session_id.reset(sid_tok)
    mts._SESSIONS.clear()
    mts._TRANSPORT_SESSIONS.clear()
    mts._SESSION_CALLS.clear()


# ── 1. mint / verify ────────────────────────────────────────────────


def test_mint_session_token_roundtrip():
    tok, exp = mts.mint_session_token(session_id="sess-1", user_id="u_aaa", ttl_seconds=60)
    claims = mts.verify_session_token(tok)
    assert claims is not None
    assert claims["sid"] == "sess-1"
    assert claims["uid"] == "u_aaa"
    assert claims["aud"] == "mcp-session"
    assert claims["exp"] == exp
    assert exp > time.time()


def test_verify_session_token_rejects_wrong_audience():
    """A transport token (aud=mcp-transport) must not verify as a
    session token, even though it's signed with the same secret."""
    transport_tok, _ = mts.mint_transport_token(user_id="u_aaa", email="a@x.com", ttl_seconds=60)
    assert mts.verify_session_token(transport_tok) is None

    service_tok = mts.mint_service_token(ttl_seconds=60)
    assert mts.verify_session_token(service_tok) is None


def test_verify_session_token_rejects_hand_forged_wrong_aud():
    now = int(time.time())
    forged = jwt.encode(
        {"sid": "s1", "uid": "u_aaa", "aud": "something-else", "iat": now, "exp": now + 60},
        _SECRET,
        algorithm="HS256",
    )
    assert mts.verify_session_token(forged) is None


def test_verify_session_token_rejects_missing_exp():
    now = int(time.time())
    no_exp = jwt.encode({"sid": "s1", "uid": "u_aaa", "aud": "mcp-session", "iat": now}, _SECRET, algorithm="HS256")
    assert mts.verify_session_token(no_exp) is None


def test_mint_session_token_raises_when_secret_unavailable(monkeypatch):
    monkeypatch.delenv("GMAIL_MCP_TRANSPORT_SECRET", raising=False)
    monkeypatch.delenv("GMS_SESSION_SECRET", raising=False)
    with pytest.raises(RuntimeError):
        mts.mint_session_token(session_id="s1", user_id="u_aaa", ttl_seconds=60)


def test_verify_session_token_none_when_secret_unavailable(monkeypatch):
    """Minted while the secret was live; the secret is then pulled
    before verification — `_transport_secret` is re-resolved on every
    call, so a rotated-away secret makes a previously-valid token
    unverifiable rather than caching stale trust."""
    tok, _ = mts.mint_session_token(session_id="s1", user_id="u_aaa", ttl_seconds=60)
    monkeypatch.delenv("GMAIL_MCP_TRANSPORT_SECRET", raising=False)
    monkeypatch.delenv("GMS_SESSION_SECRET", raising=False)
    assert mts.verify_session_token(tok) is None


# ── 2. _resolve_ctx under a session token ──────────────────────────


def _bind_session_identity(uid: str, sid: str):
    uid_tok = mts._transport_user_id.set(uid)
    sid_tok = mts._transport_session_id.set(sid)
    return uid_tok, sid_tok


def _unbind(tokens):
    uid_tok, sid_tok = tokens
    mts._transport_user_id.reset(uid_tok)
    mts._transport_session_id.reset(sid_tok)


def test_resolve_ctx_session_token_returns_registered_context():
    mts.register_session("turn-1", evidence_records=None, db_dsn=None, user_id="u_aaa", workspace="ws-1")
    tokens = _bind_session_identity("u_aaa", "turn-1")
    try:
        ctx = mts._resolve_ctx("")  # session_id arg empty — bound by token
    finally:
        _unbind(tokens)
    assert ctx is mts._SESSIONS["turn-1"]
    assert ctx.workspace == "ws-1"


def test_resolve_ctx_session_token_accepts_matching_explicit_session_id():
    mts.register_session("turn-2", evidence_records=None, db_dsn=None, user_id="u_aaa")
    tokens = _bind_session_identity("u_aaa", "turn-2")
    try:
        ctx = mts._resolve_ctx("turn-2")
    finally:
        _unbind(tokens)
    assert ctx is mts._SESSIONS["turn-2"]


def test_resolve_ctx_session_token_rejects_mismatched_session_id_arg():
    mts.register_session("turn-3", evidence_records=None, db_dsn=None, user_id="u_aaa")
    tokens = _bind_session_identity("u_aaa", "turn-3")
    try:
        with pytest.raises(RuntimeError, match="session mismatch"):
            mts._resolve_ctx("some-other-session")
    finally:
        _unbind(tokens)


def test_resolve_ctx_session_token_rejects_owner_mismatch():
    """A session token minted for A's session must never resolve if the
    registered session's owner has since diverged from the token's uid —
    defence-in-depth even though /admin/session-tokens only mints
    against the session's own registered owner."""
    mts.register_session("turn-4", evidence_records=None, db_dsn=None, user_id="u_bbb")
    tokens = _bind_session_identity("u_aaa", "turn-4")
    try:
        with pytest.raises(RuntimeError, match="owner mismatch"):
            mts._resolve_ctx("")
    finally:
        _unbind(tokens)


def test_resolve_ctx_session_token_unregistered_sid_raises():
    tokens = _bind_session_identity("u_aaa", "never-registered-turn")
    try:
        with pytest.raises(RuntimeError, match="not registered"):
            mts._resolve_ctx("")
    finally:
        _unbind(tokens)


# ── 2b. call-log + publish keyed by the BOUND session, not the raw arg ─
#
# pi-mcp-adapter can't inject a `session_id` tool argument, so every
# call it makes passes session_id="". Before this fix, `_record_call`
# and `_publish_one` were keyed off that raw "" argument instead of the
# token's bound sid — every pi-turn call log landed in
# `_SESSION_CALLS[""]` (shared across turns) and every FK write against
# `agent_events`/`agent_artifacts` for session_id="" silently failed.


def test_tool_call_records_under_bound_session_not_empty_arg(monkeypatch):
    """A session-token caller passes session_id=""; `_record_call` must
    land in `_SESSION_CALLS[<bound sid>]`, retrievable via
    `get_session_calls(<bound sid>)` — never under the empty string."""
    mts.register_session("turn-10", evidence_records=None, db_dsn=None, user_id="u_aaa")
    tokens = _bind_session_identity("u_aaa", "turn-10")

    async def fake_search(searches, *, user_id):
        return {"results": []}

    monkeypatch.setattr(mts, "_search_emails_batch_impl", fake_search)
    try:
        response = asyncio.run(mts._tool_search_emails_batch("", searches=[{"query": "x"}]))
    finally:
        _unbind(tokens)

    assert response == {"results": []}
    assert not mts._SESSION_CALLS.get("")
    calls = mts.get_session_calls("turn-10")
    assert len(calls) == 1 and calls[0]["name"] == "search_emails_batch"


def test_tool_call_records_under_bound_session_with_explicit_matching_arg(monkeypatch):
    """Same as above but the caller ALSO passes the matching session_id
    explicitly — behaviour must be identical (this is the defence-in-
    depth path `_resolve_session_token_ctx` already validates)."""
    mts.register_session("turn-11", evidence_records=None, db_dsn=None, user_id="u_aaa")
    tokens = _bind_session_identity("u_aaa", "turn-11")

    async def fake_search(searches, *, user_id):
        return {"results": []}

    monkeypatch.setattr(mts, "_search_emails_batch_impl", fake_search)
    try:
        response = asyncio.run(mts._tool_search_emails_batch("turn-11", searches=[{"query": "x"}]))
    finally:
        _unbind(tokens)

    assert response == {"results": []}
    calls = mts.get_session_calls("turn-11")
    assert len(calls) == 1 and calls[0]["name"] == "search_emails_batch"


def test_publish_artifact_batch_publishes_under_bound_session(monkeypatch):
    """The publish path must be invoked with session_id=<bound sid>,
    not the raw (empty) tool argument — else `save_artifact` writes
    against a session_id with no `agent_sessions` row and the FK
    violation drops the artifact."""
    mts.register_session("turn-12", evidence_records=None, db_dsn=None, user_id="u_aaa", workspace="ws-12")
    tokens = _bind_session_identity("u_aaa", "turn-12")

    captured: dict = {}

    def fake_publish_one(*, ctx, session_id, path, name, mime_type):
        captured["session_id"] = session_id
        return {"id": 1, "name": name or "f", "mime_type": mime_type or "text/plain", "size": 0}

    monkeypatch.setattr(mts, "_publish_one", fake_publish_one)
    try:
        response = asyncio.run(mts._tool_publish_artifact_batch("", items=[{"path": "plot.png"}]))
    finally:
        _unbind(tokens)

    assert captured["session_id"] == "turn-12"
    assert response["results"][0]["result"]["id"] == 1
    calls = mts.get_session_calls("turn-12")
    assert len(calls) == 1 and calls[0]["name"] == "publish_artifact_batch"


# ── 3. /admin/session-tokens route ─────────────────────────────────


def _admin_client():
    app = mts.build_app(host="127.0.0.1", port=0)
    return TestClient(app.streamable_http_app(), client=("127.0.0.1", 50000))


def test_admin_mints_for_registered_session():
    mts.register_session("turn-5", evidence_records=None, db_dsn=None, user_id="u_aaa")
    c = _admin_client()
    r = c.post(
        "/admin/session-tokens",
        headers={"Authorization": f"Bearer {_ADMIN}"},
        json={"session_id": "turn-5"},
    )
    assert r.status_code == 200
    body = r.json()
    claims = jwt.decode(body["token"], _SECRET, algorithms=["HS256"], audience="mcp-session")
    assert claims["sid"] == "turn-5"
    assert claims["uid"] == "u_aaa"
    assert body["expires_at"] == claims["exp"]


def test_admin_honors_custom_ttl():
    mts.register_session("turn-6", evidence_records=None, db_dsn=None, user_id="u_aaa")
    c = _admin_client()
    before = int(time.time())
    r = c.post(
        "/admin/session-tokens",
        headers={"Authorization": f"Bearer {_ADMIN}"},
        json={"session_id": "turn-6", "ttl_seconds": 60},
    )
    assert r.status_code == 200
    exp = r.json()["expires_at"]
    assert before + 55 <= exp <= before + 70


def test_admin_refuses_unregistered_session():
    c = _admin_client()
    r = c.post(
        "/admin/session-tokens",
        headers={"Authorization": f"Bearer {_ADMIN}"},
        json={"session_id": "ghost-turn"},
    )
    assert r.status_code == 404


def test_admin_refuses_session_with_no_owner():
    mts.register_session("turn-7", evidence_records=None, db_dsn=None, user_id=None)
    c = _admin_client()
    r = c.post(
        "/admin/session-tokens",
        headers={"Authorization": f"Bearer {_ADMIN}"},
        json={"session_id": "turn-7"},
    )
    assert r.status_code == 400


def test_admin_requires_admin_token():
    mts.register_session("turn-8", evidence_records=None, db_dsn=None, user_id="u_aaa")
    c = _admin_client()
    r = c.post("/admin/session-tokens", json={"session_id": "turn-8"})
    assert r.status_code == 401


def test_admin_requires_session_id():
    c = _admin_client()
    r = c.post("/admin/session-tokens", headers={"Authorization": f"Bearer {_ADMIN}"}, json={})
    assert r.status_code == 400


# ── 4. Middleware wiring: a minted session token sets both contextvars ─


def _mcp_post_with(client, token):
    return client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "method": "ping", "id": 1},
        headers={"Accept": "application/json, text/event-stream", "Authorization": f"Bearer {token}"},
    )


def test_middleware_session_token_sets_both_contextvars(monkeypatch):
    mts.register_session("turn-9", evidence_records=None, db_dsn=None, user_id="u_aaa")
    tok, _ = mts.mint_session_token(session_id="turn-9", user_id="u_aaa", ttl_seconds=60)
    seen = {}

    async def app(scope, receive, send):
        seen["uid"] = mts._transport_user_id.get()
        seen["sid"] = mts._transport_session_id.get()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    mw = mts._TransportAuthMiddleware(app)
    scope = {"type": "http", "path": "/mcp", "headers": [(b"authorization", f"Bearer {tok}".encode("latin-1"))]}

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    sent = []

    async def send(msg):
        sent.append(msg)

    import asyncio

    asyncio.run(mw(scope, receive, send))
    assert seen["uid"] == "u_aaa"
    assert seen["sid"] == "turn-9"
    # Reset after the request.
    assert mts._transport_user_id.get() is None
    assert mts._transport_session_id.get() is None


def test_middleware_session_token_missing_sid_is_rejected(monkeypatch):
    """A hand-forged session token with no `sid` claim carries no usable
    identity and must be treated as unauthenticated, not silently
    upgraded to a tenant-only scope."""
    monkeypatch.setenv("GMAIL_MCP_REQUIRE_TRANSPORT_AUTH", "1")
    now = int(time.time())
    no_sid = jwt.encode(
        {"uid": "u_aaa", "aud": "mcp-session", "iat": now, "exp": now + 60},
        _SECRET,
        algorithm="HS256",
    )
    asgi = mts.build_asgi_app(host="127.0.0.1", port=0)
    with TestClient(asgi) as c:
        r = _mcp_post_with(c, no_sid)
        assert r.status_code == 401


# ── 5. _resolve_server_db_dsn fallback ──────────────────────────────


def test_resolve_server_db_dsn_returns_db_dsn_when_set(monkeypatch):
    """When DB_DSN env var is set, _resolve_server_db_dsn returns it."""
    monkeypatch.setenv("DB_DSN", "postgresql://user:pass@host/db")
    monkeypatch.delenv("GMAIL_DB_DSN", raising=False)
    assert mts._resolve_server_db_dsn() == "postgresql://user:pass@host/db"


def test_resolve_server_db_dsn_returns_gmail_db_dsn_when_db_dsn_unset(monkeypatch):
    """When DB_DSN is unset but GMAIL_DB_DSN is set, return GMAIL_DB_DSN."""
    monkeypatch.delenv("DB_DSN", raising=False)
    monkeypatch.setenv("GMAIL_DB_DSN", "postgresql://alt:alt@althost/altdb")
    assert mts._resolve_server_db_dsn() == "postgresql://alt:alt@althost/altdb"


def test_resolve_server_db_dsn_falls_back_to_pg_dsn_when_both_unset(monkeypatch):
    """When both DB_DSN and GMAIL_DB_DSN are unset, fall back to _pg_dsn
    (the docker-compose default)."""
    monkeypatch.delenv("DB_DSN", raising=False)
    monkeypatch.delenv("GMAIL_DB_DSN", raising=False)
    from gmail_search.store.db import _pg_dsn

    assert mts._resolve_server_db_dsn() == _pg_dsn()
