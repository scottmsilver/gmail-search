"""Public-mode contract tests; no mail database or live broker required."""

import time
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gmail_search.auth import routes, session


@pytest.fixture(autouse=True)
def _pg_isolation():
    """These tests stub user storage and must never open a database."""
    yield None


@pytest.fixture
def public_env(monkeypatch):
    for name, value in {
        "GMAIL_MULTI_TENANT": "1",
        "GMS_PUBLIC_ORIGIN": "https://gms.example",
        "GMS_PUBLIC_ALLOWED_EMAILS": "owner@example.com",
        "GMS_IDENTITY_BROKER_URL": "https://identity.example",
        "GMS_IDENTITY_HANDOFF_SECRET": "i" * 40,
        "GMS_SESSION_SECRET": "s" * 40,
    }.items():
        monkeypatch.setenv(name, value)


def client(monkeypatch):
    app = FastAPI()
    routes.register_auth_routes(app, Path("/unused"))
    monkeypatch.setattr(routes, "_existing_public_user", lambda db, email: {"id": "owner", "email": email})
    monkeypatch.setattr(routes, "_upsert_user", lambda *a, **kw: {"id": "owner", "email": kw["email"]})
    monkeypatch.setattr(
        session,
        "_lookup_user_by_id",
        lambda *a: {"id": "owner", "email": "owner@example.com", "name": None},
    )
    return TestClient(app, base_url="https://gms.example")


def claims(nonce):
    now = int(time.time())
    return dict(
        iss="https://identity.example",
        aud="gmail-search",
        sub="google-owner",
        email="owner@example.com",
        email_verified=True,
        iat=now,
        exp=now + 60,
        nonce=nonce,
        jti="unique-" + nonce,
    )


def begin(c):
    response = c.get(
        "/api/auth/login?return_url=/mail",
        follow_redirects=False,
        headers={"x-forwarded-host": "evil.example"},
    )
    assert response.status_code in (302, 307)
    location = urlsplit(response.headers["location"])
    assert location.path == "/identity/start"
    params = parse_qs(location.query)
    assert params["app"] == ["gmail-search"]
    assert params["return_url"] == ["https://gms.example/api/auth/callback"]
    assert "scope" not in params
    assert "__Host-gms_oauth_nonce=" in response.headers["set-cookie"]
    return params["nonce"][0]


def callback(c, payload):
    return c.get(
        "/api/auth/callback",
        params={"silver_oauth": jwt.encode(payload, "i" * 40, algorithm="HS256")},
        follow_redirects=False,
    )


def test_public_allowlist_never_inherits_private_invites(public_env, monkeypatch):
    monkeypatch.setenv("GMS_ALLOWED_EMAILS", "invited@example.com")
    assert not session.is_email_allowed(Path("/unused"), "invited@example.com")


def test_identity_login_replay_and_revocation(public_env, monkeypatch):
    c = client(monkeypatch)
    payload = claims(begin(c))
    response = callback(c, payload)
    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/mail"
    assert "__Host-gms_session=" in response.headers["set-cookie"]
    cookie = c.cookies.get("__Host-gms_session")
    assert c.get("/api/auth/me").status_code == 200
    assert callback(c, payload).status_code == 401
    assert c.post("/api/auth/logout", headers={"origin": "https://gms.example"}).status_code == 200
    c.cookies.set("__Host-gms_session", cookie)
    assert c.get("/api/auth/me").status_code == 401


@pytest.mark.parametrize(
    "claim",
    ["iss", "aud", "sub", "email", "email_verified", "iat", "exp", "nonce", "jti"],
)
def test_missing_claim_rejected(public_env, monkeypatch, claim):
    c = client(monkeypatch)
    payload = claims(begin(c))
    del payload[claim]
    assert callback(c, payload).status_code == 401


@pytest.mark.parametrize(
    ("claim", "value"),
    [
        ("aud", "wezterm-web"),
        ("iss", "https://other.example"),
        ("sub", 3),
        ("email_verified", "true"),
        ("iat", True),
        ("exp", "123"),
        ("nonce", "wrong"),
        ("jti", ""),
        ("exp", int(time.time()) + 3600),
        ("iat", int(time.time()) + 20),
    ],
)
def test_invalid_claim_rejected(public_env, monkeypatch, claim, value):
    c = client(monkeypatch)
    payload = claims(begin(c))
    payload[claim] = value
    assert callback(c, payload).status_code == 401


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("GMAIL_MULTI_TENANT", "0"),
        ("GMS_PUBLIC_ALLOWED_EMAILS", ""),
        ("GMS_PUBLIC_ORIGIN", "http://gms.example"),
        ("GMS_PUBLIC_ORIGIN", "https://gms.example/"),
        ("GMS_IDENTITY_BROKER_URL", "https://identity.example/path"),
        ("GMS_IDENTITY_HANDOFF_SECRET", ""),
        ("GMS_IDENTITY_HANDOFF_SECRET", "s" * 40),
    ],
)
def test_invalid_public_config_fails(public_env, monkeypatch, name, value):
    from gmail_search.auth.public import validate_public_auth_config

    monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError):
        validate_public_auth_config()


def test_public_mutations_require_exact_origin(public_env):
    from gmail_search.auth.public import PublicAuthMiddleware

    app = FastAPI()
    app.add_middleware(PublicAuthMiddleware)

    @app.post("/api/mutate")
    def mutate():
        return {"ok": True}

    c = TestClient(app)
    for headers in ({}, {"origin": "https://sibling.example"}, {"origin": "null"}):
        assert c.post("/api/mutate", headers=headers).status_code == 403
    assert c.post("/api/mutate", headers={"origin": "https://gms.example"}).status_code == 200


def test_nonce_browser_binding_and_restart_fail_closed(public_env, monkeypatch):
    from gmail_search.auth import public

    c = client(monkeypatch)
    payload = claims(begin(c))
    other_browser = client(monkeypatch)
    assert callback(other_browser, payload).status_code == 401
    public._nonces.clear()  # Simulate process restart: no pending server state survives.
    assert callback(c, payload).status_code == 401
    payload = claims(begin(c))
    assert callback(c, payload).status_code in (302, 307)
    assert c.get("/api/auth/me").status_code == 200
    public._sessions.clear()
    assert c.get("/api/auth/me").status_code == 401


def test_public_cookies_secure_and_no_legacy_fallback(public_env, monkeypatch):
    monkeypatch.setenv("BROKER_HANDOFF_SECRET", "b" * 40)
    c = client(monkeypatch)
    payload = claims(begin(c))
    response = callback(c, payload)
    for cookie in response.headers.get_list("set-cookie"):
        assert "Secure" in cookie and "HttpOnly" in cookie and "Path=/" in cookie
        assert "Domain=" not in cookie
    legacy = jwt.encode(
        {
            "email": "owner@example.com",
            "iat": int(time.time()),
            "exp": int(time.time()) + 60,
        },
        "b" * 40,
        algorithm="HS256",
    )
    assert session.verify_handoff_jwt(legacy) is None


def test_origin_exception_requires_valid_service_credential(public_env, monkeypatch):
    from gmail_search.auth.public import PublicAuthMiddleware

    monkeypatch.setenv("GMAIL_MCP_ADMIN_TOKEN", "internal-secret")
    app = FastAPI()
    app.add_middleware(PublicAuthMiddleware)

    @app.post("/api/mutate")
    def mutate():
        return {"ok": True}

    c = TestClient(app)
    assert (
        c.post(
            "/api/mutate",
            headers={"authorization": "Bearer forged", "x-user-id": "owner"},
        ).status_code
        == 403
    )
    assert (
        c.post(
            "/api/mutate",
            headers={"authorization": "Bearer internal-secret", "x-user-id": "owner"},
        ).status_code
        == 200
    )


@pytest.mark.parametrize("target", ["/\t/evil.example", "/\r/evil.example", "/" + "a" * 4096])
def test_public_return_target_cannot_smuggle_authority_or_exhaust_store(public_env, monkeypatch, target):
    c = client(monkeypatch)
    response = c.get("/api/auth/login", params={"return_url": target}, follow_redirects=False)
    nonce = parse_qs(urlsplit(response.headers["location"]).query)["nonce"][0]
    response = callback(c, claims(nonce))
    assert response.headers["location"] == "/"


def test_public_session_expires_server_side(public_env, monkeypatch):
    from gmail_search.auth import public

    now = time.time()
    token = public.issue_session({"uid": "owner"})
    assert public.read_session(token) == {"uid": "owner"}
    monkeypatch.setattr(public.time, "time", lambda: now + public.SESSION_TTL + 1)
    assert public.read_session(token) is None
