"""Phase 1 auth tests — broker handoff + session cookie + allowlist.

Covers the failure modes the plan called out:
  * `require_user` defaults to "no-op returns None" when the env flag
    is off (legacy single-pool callers must keep working).
  * When the flag is on, every error path (no cookie, bad cookie,
    expired, deleted user, uninvited) must 401/403 — no silent success.
  * /api/auth/callback gates on both the broker handoff signature AND
    the email allowlist (env var union'd with `invited_emails` table).

The handoff flow is exercised end-to-end without spinning up the real
silver-oauth broker by signing a synthetic handoff JWT with the same
HMAC secret the verifier expects.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gmail_search.auth.routes import register_auth_routes
from gmail_search.auth.session import (
    is_email_allowed,
    is_multi_tenant_enabled,
    issue_handoff_jwt_for_test,
    make_session_cookie,
)
from gmail_search.store.db import get_connection, init_db

# Real-length secrets so the >=32-byte gate doesn't reject our test
# fixtures the way it would in production.
_TEST_SESSION_SECRET = "test-session-secret-not-for-prod-padded-to-32+"
_TEST_HANDOFF_SECRET = "test-handoff-secret-not-for-prod-padded-to-32+"
_BROKER_URL = "https://broker.test.example.com"


@pytest.fixture
def app(db_backend, monkeypatch) -> FastAPI:
    """Minimal FastAPI app with the auth routes wired in. Adds
    `testserver` (Starlette TestClient's default Host) to the host
    allowlist so most tests don't have to think about H1 — the
    dedicated host-allowlist tests below clear/override this."""
    monkeypatch.setenv("GMS_SESSION_SECRET", _TEST_SESSION_SECRET)
    monkeypatch.setenv("BROKER_HANDOFF_SECRET", _TEST_HANDOFF_SECRET)
    monkeypatch.setenv("SILVER_OAUTH_BROKER_URL", _BROKER_URL)
    monkeypatch.setenv("GMS_ALLOWED_HOSTS", "testserver")
    init_db(db_backend["db_path"])
    app = FastAPI()
    app.state.data_dir = db_backend["db_path"].parent
    register_auth_routes(app, db_backend["db_path"])
    return app


@pytest.fixture
def client(app) -> TestClient:
    # follow_redirects=False so login/callback 302s are inspectable.
    return TestClient(app, follow_redirects=False)


@pytest.fixture
def invited_email(db_backend) -> str:
    email = "alice@example.com"
    conn = get_connection(db_backend["db_path"])
    try:
        conn.execute("INSERT INTO invited_emails (email) VALUES (%s)", (email,))
        conn.commit()
    finally:
        conn.close()
    return email


def _sign_in(client: TestClient, *, user_id: str = "u_test_alice", email: str = "alice@example.com") -> None:
    """Set the cookie a successfully-signed-in user would carry.
    Mutates the client's cookie jar (the modern Starlette pattern —
    per-request `cookies=` was deprecated). Tests that need different
    identities across calls should `client.cookies.clear()` first."""
    client.cookies.set("gms_session", make_session_cookie(user_id=user_id, email=email))


@pytest.fixture
def valid_state(client) -> str:
    """Pre-set a valid state cookie + return its value so callback
    tests can build a `&state=…` URL. Mirrors what /login would do
    in a real round-trip."""
    state = "test-state-token-xyz123"
    client.cookies.set("gms_oauth_state", state)
    return state


# ─── flag-off (single-pool) regression ───────────────────────────────────


def test_flag_off_is_legacy_default(monkeypatch):
    monkeypatch.delenv("GMAIL_MULTI_TENANT", raising=False)
    assert is_multi_tenant_enabled() is False


def test_flag_off_me_returns_null(client, monkeypatch):
    monkeypatch.delenv("GMAIL_MULTI_TENANT", raising=False)
    r = client.get("/api/auth/me")
    assert r.status_code == 200
    assert r.json() == {"multi_tenant": False, "user": None}


def test_flag_off_login_404(client, monkeypatch):
    """Login is a noop when multi-tenant is off — should not silently
    redirect to the broker."""
    monkeypatch.delenv("GMAIL_MULTI_TENANT", raising=False)
    r = client.get("/api/auth/login")
    assert r.status_code == 404


def test_whoami_always_public(client, monkeypatch):
    monkeypatch.delenv("GMAIL_MULTI_TENANT", raising=False)
    r = client.get("/api/auth/whoami")
    assert r.status_code == 200
    assert r.json() == {"multi_tenant_enabled": False}


# ─── flag-on /me cookie verification ─────────────────────────────────────


def test_flag_on_no_cookie_401(client, monkeypatch):
    monkeypatch.setenv("GMAIL_MULTI_TENANT", "1")
    r = client.get("/api/auth/me")
    assert r.status_code == 401
    assert "signed in" in r.json()["detail"].lower()


def test_flag_on_garbage_cookie_401(client, monkeypatch):
    monkeypatch.setenv("GMAIL_MULTI_TENANT", "1")
    client.cookies.set("gms_session", "not-a-jwt")
    r = client.get("/api/auth/me")
    assert r.status_code == 401


def test_flag_on_wrong_secret_cookie_401(client, monkeypatch):
    monkeypatch.setenv("GMAIL_MULTI_TENANT", "1")
    monkeypatch.setenv("GMS_SESSION_SECRET", "different-secret-but-still-32-bytes")
    bad = make_session_cookie(user_id="u_x", email="x@example.com")
    monkeypatch.setenv("GMS_SESSION_SECRET", _TEST_SESSION_SECRET)
    client.cookies.set("gms_session", bad)
    r = client.get("/api/auth/me")
    assert r.status_code == 401


def test_flag_on_session_for_deleted_user_401(client, monkeypatch):
    """Cookie verifies cryptographically but the underlying users row
    was deleted (e.g. CLI delete-user). Force a fresh sign-in instead
    of pretending the user still exists."""
    monkeypatch.setenv("GMAIL_MULTI_TENANT", "1")
    monkeypatch.setenv("GMS_ALLOWED_EMAILS", "ghost@example.com")
    _sign_in(client, user_id="u_does_not_exist", email="ghost@example.com")
    r = client.get("/api/auth/me")
    assert r.status_code == 401
    assert "deleted" in r.json()["detail"].lower()


def test_flag_on_uninvited_after_signin_403(client, db_backend, monkeypatch, invited_email):
    """Re-checks allowlist on every request. Uninviting an email (delete
    from invited_emails OR remove from env var) should kick the user
    out without waiting for cookie expiry."""
    monkeypatch.setenv("GMAIL_MULTI_TENANT", "1")
    from gmail_search.auth.routes import _upsert_user

    user = _upsert_user(db_backend["db_path"], email=invited_email, name="Alice", picture=None)
    conn = get_connection(db_backend["db_path"])
    try:
        conn.execute("DELETE FROM invited_emails WHERE email = %s", (invited_email,))
        conn.commit()
    finally:
        conn.close()
    _sign_in(client, user_id=user["id"], email=invited_email)
    r = client.get("/api/auth/me")
    assert r.status_code == 403
    assert "no longer allowed" in r.json()["detail"].lower()


def test_flag_on_signed_in_user_200(client, db_backend, monkeypatch, invited_email):
    monkeypatch.setenv("GMAIL_MULTI_TENANT", "1")
    from gmail_search.auth.routes import _upsert_user

    user = _upsert_user(db_backend["db_path"], email=invited_email, name="Alice", picture=None)
    _sign_in(client, user_id=user["id"], email=invited_email)
    r = client.get("/api/auth/me")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["multi_tenant"] is True
    assert body["user"]["email"] == invited_email
    assert body["user"]["id"] == user["id"]
    assert body["user"]["name"] == "Alice"


# ─── /api/auth/login → broker redirect ──────────────────────────────────


def test_login_redirects_to_broker_with_state(client, monkeypatch):
    monkeypatch.setenv("GMAIL_MULTI_TENANT", "1")
    r = client.get("/api/auth/login?return_url=%2Finbox")
    assert r.status_code in (302, 307)
    location = r.headers["location"]
    assert location.startswith(f"{_BROKER_URL}/start?")
    assert "scope=openid%2Cprofile" in location
    assert "%2Fapi%2Fauth%2Fcallback" in location
    # /inbox is inside the inner callback URL, so it's double-encoded.
    assert "inbox" in location
    # State token is passed as &state= inside the inner callback URL,
    # so it shows up double-URL-encoded too. Just confirm 'state' is
    # present somewhere in the chain.
    assert "state" in location.lower()
    # And the state cookie is set on us.
    assert "gms_oauth_state" in r.headers.get("set-cookie", "")


def test_login_open_redirect_clamped_to_root(client, monkeypatch):
    """An external return_url must NOT survive past sign-in. Phishing
    via /api/auth/login?return_url=https://evil.example would bounce
    a victim through real broker auth and dump them on evil.example
    with a logged-in cookie."""
    monkeypatch.setenv("GMAIL_MULTI_TENANT", "1")
    r = client.get("/api/auth/login?return_url=https://evil.example/")
    assert r.status_code in (302, 307)
    location = r.headers["location"]
    # The inner callback's return= should now be %2F, not the external URL.
    assert "evil.example" not in location


# ─── /api/auth/callback — handoff JWT + state binding ───────────────────


def test_callback_missing_token_400(client, monkeypatch):
    monkeypatch.setenv("GMAIL_MULTI_TENANT", "1")
    r = client.get("/api/auth/callback")
    assert r.status_code == 400


def test_callback_no_state_cookie_400(client, monkeypatch):
    """A request that arrives at /callback without ever having gone
    through /login (no state cookie) must be rejected — that's how
    the login-CSRF / session-swap attack lands."""
    monkeypatch.setenv("GMAIL_MULTI_TENANT", "1")
    handoff = issue_handoff_jwt_for_test(email="alice@example.com")
    r = client.get(f"/api/auth/callback?silver_oauth={handoff}&state=anything")
    assert r.status_code == 400
    assert "state" in r.json()["detail"].lower()


def test_callback_state_mismatch_400(client, monkeypatch, valid_state):
    """Cookie says one state, query says another → 400. Closes the
    forced-cookie variant of the same attack."""
    monkeypatch.setenv("GMAIL_MULTI_TENANT", "1")
    handoff = issue_handoff_jwt_for_test(email="alice@example.com")
    r = client.get(f"/api/auth/callback?silver_oauth={handoff}&state=different-from-cookie")
    assert r.status_code == 400


def test_callback_garbage_token_401(client, monkeypatch, valid_state):
    monkeypatch.setenv("GMAIL_MULTI_TENANT", "1")
    r = client.get(f"/api/auth/callback?silver_oauth=not-a-jwt&state={valid_state}")
    assert r.status_code == 401


def test_callback_handoff_signed_with_wrong_secret_401(client, monkeypatch, valid_state):
    monkeypatch.setenv("GMAIL_MULTI_TENANT", "1")
    monkeypatch.setenv("BROKER_HANDOFF_SECRET", "wrong-handoff-secret-but-32-bytes-yes")
    bad_handoff = issue_handoff_jwt_for_test(email="alice@example.com")
    monkeypatch.setenv("BROKER_HANDOFF_SECRET", _TEST_HANDOFF_SECRET)
    r = client.get(f"/api/auth/callback?silver_oauth={bad_handoff}&state={valid_state}")
    assert r.status_code == 401


def test_callback_expired_handoff_401(client, monkeypatch, valid_state):
    monkeypatch.setenv("GMAIL_MULTI_TENANT", "1")
    expired = issue_handoff_jwt_for_test(email="alice@example.com", ttl_seconds=-30)
    r = client.get(f"/api/auth/callback?silver_oauth={expired}&state={valid_state}")
    assert r.status_code == 401


def test_callback_email_not_in_allowlist_403(client, monkeypatch, valid_state):
    monkeypatch.setenv("GMAIL_MULTI_TENANT", "1")
    monkeypatch.setenv("GMS_ALLOWED_EMAILS", "")
    handoff = issue_handoff_jwt_for_test(email="stranger@example.com")
    r = client.get(f"/api/auth/callback?silver_oauth={handoff}&state={valid_state}")
    assert r.status_code == 403
    assert "not allowed" in r.json()["detail"].lower()


def test_callback_invited_via_db_table_provisions_user(client, db_backend, monkeypatch, invited_email, valid_state):
    monkeypatch.setenv("GMAIL_MULTI_TENANT", "1")
    handoff = issue_handoff_jwt_for_test(email=invited_email, name="Alice")
    r = client.get(f"/api/auth/callback?silver_oauth={handoff}&state={valid_state}&return=/inbox")
    assert r.status_code in (302, 307)
    assert r.headers["location"] == "/inbox"
    cookie_header = r.headers.get("set-cookie", "")
    assert "gms_session=" in cookie_header
    conn = get_connection(db_backend["db_path"])
    try:
        row = conn.execute("SELECT email, name FROM users WHERE email = %s", (invited_email,)).fetchone()
    finally:
        conn.close()
    assert row is not None
    assert row["email"] == invited_email
    assert row["name"] == "Alice"


def test_callback_open_redirect_clamped(client, monkeypatch, invited_email, valid_state):
    """`return=` from the broker round-trip must also be clamped — an
    attacker could tamper with the inner URL's return= mid-flight."""
    monkeypatch.setenv("GMAIL_MULTI_TENANT", "1")
    handoff = issue_handoff_jwt_for_test(email=invited_email)
    r = client.get(f"/api/auth/callback?silver_oauth={handoff}&state={valid_state}" f"&return=https://evil.example/x")
    assert r.status_code in (302, 307)
    assert r.headers["location"] == "/"


def test_callback_invited_via_env_var_provisions_user(client, db_backend, monkeypatch, valid_state):
    """Env var allowlist works without any invited_emails row — the two
    sources are union'd."""
    monkeypatch.setenv("GMAIL_MULTI_TENANT", "1")
    monkeypatch.setenv("GMS_ALLOWED_EMAILS", "envuser@example.com")
    handoff = issue_handoff_jwt_for_test(email="envuser@example.com")
    r = client.get(f"/api/auth/callback?silver_oauth={handoff}&state={valid_state}")
    assert r.status_code in (302, 307)
    conn = get_connection(db_backend["db_path"])
    try:
        row = conn.execute("SELECT email FROM users WHERE email = %s", ("envuser@example.com",)).fetchone()
    finally:
        conn.close()
    assert row is not None


def test_callback_then_me_round_trips(client, db_backend, monkeypatch, invited_email, valid_state):
    """End-to-end: callback sets cookie → next /me call sees it."""
    monkeypatch.setenv("GMAIL_MULTI_TENANT", "1")
    handoff = issue_handoff_jwt_for_test(email=invited_email, name="Alice")
    cb = client.get(f"/api/auth/callback?silver_oauth={handoff}&state={valid_state}")
    assert cb.status_code in (302, 307)
    me = client.get("/api/auth/me")
    assert me.status_code == 200, me.text
    assert me.json()["user"]["email"] == invited_email


def test_callback_email_normalized_lowercase(client, db_backend, monkeypatch, valid_state):
    """Mixed-case email in the handoff JWT must match an allowlist
    entry stored lowercase."""
    monkeypatch.setenv("GMAIL_MULTI_TENANT", "1")
    monkeypatch.setenv("GMS_ALLOWED_EMAILS", "mixedcase@example.com")
    handoff = issue_handoff_jwt_for_test(email="MixedCase@Example.com")
    r = client.get(f"/api/auth/callback?silver_oauth={handoff}&state={valid_state}")
    assert r.status_code in (302, 307)


def test_callback_email_with_whitespace_normalizes(client, db_backend, monkeypatch, valid_state):
    """Belt-and-suspenders: a handoff carrying ' alice@example.com '
    should not bypass an 'alice@example.com' allowlist entry. Pre-fix,
    is_email_allowed stripped but _upsert_user did not, so the row
    would store a leading-space email."""
    monkeypatch.setenv("GMAIL_MULTI_TENANT", "1")
    monkeypatch.setenv("GMS_ALLOWED_EMAILS", "spaceuser@example.com")
    handoff = issue_handoff_jwt_for_test(email=" SpaceUser@Example.com ")
    r = client.get(f"/api/auth/callback?silver_oauth={handoff}&state={valid_state}")
    assert r.status_code in (302, 307)
    conn = get_connection(db_backend["db_path"])
    try:
        row = conn.execute("SELECT email FROM users WHERE email = %s", ("spaceuser@example.com",)).fetchone()
    finally:
        conn.close()
    assert row is not None


# ─── /api/auth/logout ───────────────────────────────────────────────────


def test_logout_clears_cookie(client, monkeypatch):
    monkeypatch.setenv("GMAIL_MULTI_TENANT", "1")
    r = client.post("/api/auth/logout")
    assert r.status_code == 200
    cookie_header = r.headers.get("set-cookie", "")
    assert "gms_session=" in cookie_header
    assert ("Max-Age=0" in cookie_header) or ("expires" in cookie_header.lower())


# ─── allowlist helper ───────────────────────────────────────────────────


def test_allowlist_union(db_backend, monkeypatch):
    """is_email_allowed should treat env var and table as union."""
    monkeypatch.setenv("GMS_ALLOWED_EMAILS", "envonly@example.com")
    # No `app` fixture here, so init_db ourselves before touching
    # invited_emails.
    init_db(db_backend["db_path"])
    conn = get_connection(db_backend["db_path"])
    try:
        conn.execute("INSERT INTO invited_emails (email) VALUES ('dbonly@example.com')")
        conn.commit()
    finally:
        conn.close()
    db_path = db_backend["db_path"]
    assert is_email_allowed(db_path, "envonly@example.com") is True
    assert is_email_allowed(db_path, "dbonly@example.com") is True
    assert is_email_allowed(db_path, "stranger@example.com") is False
    assert is_email_allowed(db_path, "EnvOnly@Example.com") is True


# ─── secret-strength gate (defense-in-depth) ────────────────────────────


def test_short_session_secret_rejected(client, monkeypatch):
    monkeypatch.setenv("GMAIL_MULTI_TENANT", "1")
    monkeypatch.setenv("GMS_SESSION_SECRET", "x" * 31)
    client.cookies.set("gms_session", "anything")
    r = client.get("/api/auth/me")
    assert r.status_code == 500
    assert "32" in r.json()["detail"]


def test_short_handoff_secret_rejected(client, monkeypatch, valid_state):
    monkeypatch.setenv("GMAIL_MULTI_TENANT", "1")
    monkeypatch.setenv("BROKER_HANDOFF_SECRET", "x" * 31)
    r = client.get(f"/api/auth/callback?silver_oauth=anything&state={valid_state}")
    assert r.status_code == 500
    assert "32" in r.json()["detail"]


# ─── Host allowlist (codex H1) ──────────────────────────────────────────


def test_login_rejects_unknown_host(client, monkeypatch):
    """A forged Host header (e.g. via misconfigured upstream) must NOT
    end up in the broker callback URL. State-cookie binding bounds the
    blast radius, but failing closed here is cheaper than relying on it."""
    monkeypatch.setenv("GMAIL_MULTI_TENANT", "1")
    monkeypatch.delenv("GMS_ALLOWED_HOSTS", raising=False)
    r = client.get("/api/auth/login", headers={"Host": "evil.example"})
    assert r.status_code == 400
    assert "allowlist" in r.json()["detail"].lower() or "host" in r.json()["detail"].lower()


def test_login_accepts_default_dev_host(client, monkeypatch):
    monkeypatch.setenv("GMAIL_MULTI_TENANT", "1")
    monkeypatch.delenv("GMS_ALLOWED_HOSTS", raising=False)
    # TestClient defaults to "testserver" as Host, which is NOT in the
    # default allowlist — confirm it gets rejected, then add it via env.
    r = client.get("/api/auth/login")
    assert r.status_code == 400
    monkeypatch.setenv("GMS_ALLOWED_HOSTS", "testserver")
    r = client.get("/api/auth/login")
    assert r.status_code in (302, 307)


def test_login_x_forwarded_host_in_allowlist(client, monkeypatch):
    """The proxy chain forwards X-Forwarded-Host. Allowlist must be
    consulted against THAT, not the underlying Host header."""
    monkeypatch.setenv("GMAIL_MULTI_TENANT", "1")
    monkeypatch.setenv("GMS_ALLOWED_HOSTS", "gms.i.oursilverfamily.com")
    r = client.get(
        "/api/auth/login",
        headers={"Host": "127.0.0.1:8090", "X-Forwarded-Host": "gms.i.oursilverfamily.com"},
    )
    assert r.status_code in (302, 307)
    assert "gms.i.oursilverfamily.com" in r.headers["location"]


# ─── Return-URL auth-loop guard (codex L2) ──────────────────────────────


def test_safe_return_url_rejects_auth_paths():
    """A return_url pointing back into /api/auth/* would let sign-in
    chain into immediate sign-out, infinite re-auth loops, or callback
    re-trigger with stale params."""
    from gmail_search.auth.session import safe_relative_return_url

    assert safe_relative_return_url("/api/auth/login") == "/"
    assert safe_relative_return_url("/api/auth/logout") == "/"
    assert safe_relative_return_url("/api/auth/callback?silver_oauth=foo") == "/"
    # Non-auth paths still survive.
    assert safe_relative_return_url("/inbox") == "/inbox"
    assert safe_relative_return_url("/search?q=hi") == "/search?q=hi"
    # External + protocol-relative still rejected.
    assert safe_relative_return_url("https://evil.example") == "/"
    assert safe_relative_return_url("//evil.example") == "/"


# ─── ON CONFLICT race fix (codex M1) ────────────────────────────────────


def test_upsert_user_idempotent_concurrent_first_login(db_backend, monkeypatch):
    """Two concurrent first-logins for the same email used to 500 on
    UNIQUE violation (find-then-insert had a race). The ON CONFLICT
    upsert should produce the SAME row both times."""
    monkeypatch.setenv("GMAIL_MULTI_TENANT", "1")
    init_db(db_backend["db_path"])
    from gmail_search.auth.routes import _upsert_user

    a = _upsert_user(db_backend["db_path"], email="alice@example.com", name="Alice", picture=None)
    b = _upsert_user(db_backend["db_path"], email="alice@example.com", name="Alice 2", picture="pic.jpg")
    assert a["id"] == b["id"]
    # When the new sign-in carries a non-null name, it wins (COALESCE
    # is `EXCLUDED.name OR users.name` — new wins unless null). This
    # matches the prior find-then-update behavior so a Google profile
    # rename propagates on next sign-in.
    assert b["name"] == "Alice 2"
    # And the picture, which was null on the first call, gets set.
    conn = get_connection(db_backend["db_path"])
    try:
        row = conn.execute("SELECT avatar_url FROM users WHERE id = %s", (a["id"],)).fetchone()
    finally:
        conn.close()
    assert row["avatar_url"] == "pic.jpg"


def test_upsert_user_no_synthetic_google_sub(db_backend, monkeypatch):
    """The schema now allows google_sub IS NULL; we should NOT be
    fabricating one (the prior `broker:{email}` workaround leaked the
    email into google_sub and confused future migrations)."""
    monkeypatch.setenv("GMAIL_MULTI_TENANT", "1")
    init_db(db_backend["db_path"])
    from gmail_search.auth.routes import _upsert_user

    _upsert_user(db_backend["db_path"], email="bob@example.com", name=None, picture=None)
    conn = get_connection(db_backend["db_path"])
    try:
        row = conn.execute("SELECT google_sub FROM users WHERE email = %s", ("bob@example.com",)).fetchone()
    finally:
        conn.close()
    assert row is not None
    assert row["google_sub"] is None
