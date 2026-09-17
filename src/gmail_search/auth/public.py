"""Opt-in public identity contract. Requires one application worker.

State is bounded, process-local and protected by a lock. Restarting loses all
pending nonces and sessions (fail closed). Multi-worker deployment requires a
shared atomic store before activation; no broker registration is provisioned here.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import threading
import time
from dataclasses import dataclass
from urllib.parse import urlencode, urlsplit

import jwt
from fastapi import HTTPException
from starlette.datastructures import Headers
from starlette.responses import JSONResponse, RedirectResponse

SESSION_COOKIE = "__Host-gms_session"
NONCE_COOKIE = "__Host-gms_oauth_nonce"
SESSION_TTL = 3600
NONCE_TTL = 600
MAX_ENTRIES = 4096
_lock = threading.Lock()
_nonces: dict[str, tuple[float, str]] = {}
_replays: dict[str, float] = {}
_sessions: dict[str, tuple[float, dict]] = {}


@dataclass(frozen=True)
class PublicConfig:
    origin: str
    broker: str
    secret: str
    emails: frozenset[str]


def public_enabled() -> bool:
    # An explicitly empty setting is a configuration error, never a private fallback.
    return "GMS_PUBLIC_ORIGIN" in os.environ


def _origin(name: str) -> str:
    raw = os.environ.get(name, "")
    try:
        parsed = urlsplit(raw)
        valid = (
            parsed.scheme == "https"
            and parsed.hostname
            and not parsed.username
            and not parsed.password
            and not parsed.path
            and not parsed.query
            and not parsed.fragment
            and raw == f"https://{parsed.netloc}"
            and parsed.port in (None, 443)
            and not any(c.isspace() for c in raw)
            and "\\" not in raw
        )
    except ValueError:
        valid = False
    if not valid:
        raise RuntimeError(f"{name} must be an exact HTTPS origin without path or trailing slash")
    return raw


def validate_public_auth_config() -> PublicConfig | None:
    if not public_enabled():
        return None
    if os.environ.get("GMAIL_MULTI_TENANT") != "1":
        raise RuntimeError("Public deployment requires GMAIL_MULTI_TENANT=1")
    origin, broker = _origin("GMS_PUBLIC_ORIGIN"), _origin("GMS_IDENTITY_BROKER_URL")
    secret = os.environ.get("GMS_IDENTITY_HANDOFF_SECRET", "")
    session = os.environ.get("GMS_SESSION_SECRET", "")
    if len(secret.encode()) < 32 or len(session.encode()) < 32:
        raise RuntimeError("Public deployment requires independent identity and session secrets of at least 32 bytes")
    if secret == session or secret == os.environ.get("BROKER_HANDOFF_SECRET"):
        raise RuntimeError("GMS_IDENTITY_HANDOFF_SECRET must be independent of legacy and session secrets")
    emails = frozenset(
        e.strip().lower() for e in os.environ.get("GMS_PUBLIC_ALLOWED_EMAILS", "").split(",") if e.strip()
    )
    if not emails or any(e.count("@") != 1 or any(c.isspace() for c in e) for e in emails):
        raise RuntimeError("Public deployment requires an explicit GMS_PUBLIC_ALLOWED_EMAILS allowlist")
    if os.environ.get("WEB_CONCURRENCY", "1") != "1":
        raise RuntimeError("Public authentication currently requires a single application worker")
    return PublicConfig(origin, broker, secret, emails)


def full_runtime_emails() -> frozenset[str]:
    """Owners cleared for the unrestricted agent runtimes on the public origin.

    The public deployment routes every run to a bounded retrieval loop before the
    requested backend is read (`agents/service.py`), because arbitrary execution
    is not cleared for shared use. That stays the default.

    An address listed here is exempt: it gets the private app's runtimes -- Pi
    with Gemini and OpenRouter, model choice, battles -- on the public origin.
    The setting is a capability, never an admission: it must name addresses that
    are already admitted, so it can only narrow who gets the extra power.

    Empty by default, and inert on the private app, where there is no
    short-circuit to relax.
    """
    configured = frozenset(
        e.strip().lower() for e in os.environ.get("GMS_FULL_RUNTIME_EMAILS", "").split(",") if e.strip()
    )
    if not configured:
        return frozenset()
    config = validate_public_auth_config()
    if config is None:
        return frozenset()
    if not configured <= config.emails:
        raise RuntimeError(
            "GMS_FULL_RUNTIME_EMAILS must be a subset of GMS_PUBLIC_ALLOWED_EMAILS; "
            "it grants capability to admitted users and cannot admit anyone"
        )
    return configured


def has_full_runtime(email: str | None) -> bool:
    """True when `email` may use the unrestricted runtimes on this deployment."""
    if not isinstance(email, str) or not email:
        return False
    return email.strip().lower() in full_runtime_emails()


def _prune() -> None:
    now = time.time()
    for mapping in (_nonces, _sessions):
        for key in [key for key, value in mapping.items() if value[0] <= now]:
            del mapping[key]
    for key in [key for key, expiry in _replays.items() if expiry <= now]:
        del _replays[key]


def start_login(return_url: str):
    from .session import safe_relative_return_url

    config = validate_public_auth_config()
    if len(return_url) > 2048 or any(ord(c) < 32 or c == "\\" for c in return_url):
        return_url = "/"
    nonce = secrets.token_urlsafe(32)
    with _lock:
        _prune()
        if len(_nonces) >= MAX_ENTRIES:
            raise HTTPException(503, "login capacity reached")
        _nonces[nonce] = (time.time() + NONCE_TTL, safe_relative_return_url(return_url))
    params = urlencode(
        dict(
            app="gmail-search",
            return_url=config.origin + "/api/auth/callback",
            nonce=nonce,
        )
    )
    response = RedirectResponse(config.broker + "/identity/start?" + params)
    response.set_cookie(
        NONCE_COOKIE,
        nonce,
        max_age=NONCE_TTL,
        secure=True,
        httponly=True,
        samesite="lax",
        path="/",
    )
    response.headers["Cache-Control"] = "private, no-store"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


def consume_handoff(token: str, nonce: str) -> tuple[dict, str]:
    config = validate_public_auth_config()
    try:
        payload = jwt.decode(
            token,
            config.secret,
            algorithms=["HS256"],
            issuer=config.broker,
            audience="gmail-search",
            options={
                "require": [
                    "iss",
                    "aud",
                    "sub",
                    "email",
                    "email_verified",
                    "iat",
                    "exp",
                    "nonce",
                    "jti",
                ],
                "strict_aud": True,
            },
        )
        if any(
            type(payload[k]) is not str or not payload[k] or len(payload[k]) > 2048
            for k in ("iss", "aud", "sub", "email", "nonce", "jti")
        ):
            raise ValueError()
        if type(payload["iat"]) is not int or type(payload["exp"]) is not int:
            raise ValueError()
        now = time.time()
        if not (payload["iat"] <= now < payload["exp"] and 0 < payload["exp"] - payload["iat"] <= 60):
            raise ValueError()
        if payload["email_verified"] is not True or not nonce or not hmac.compare_digest(payload["nonce"], nonce):
            raise ValueError()
        with _lock:
            _prune()
            pending = _nonces.pop(nonce, None)
            if pending is None or payload["jti"] in _replays or len(_replays) >= MAX_ENTRIES:
                raise ValueError()
            _replays[payload["jti"]] = payload["exp"]
        return payload, pending[1]
    except (jwt.PyJWTError, ValueError, TypeError):
        raise HTTPException(401, "identity handoff invalid or expired") from None


def issue_session(payload: dict) -> str:
    token = secrets.token_urlsafe(32)
    with _lock:
        _prune()
        if len(_sessions) >= MAX_ENTRIES:
            raise HTTPException(503, "session capacity reached")
        _sessions[hashlib.sha256(token.encode()).hexdigest()] = (
            time.time() + SESSION_TTL,
            dict(payload),
        )
    return token


def read_session(token: str) -> dict | None:
    with _lock:
        _prune()
        entry = _sessions.get(hashlib.sha256(token.encode()).hexdigest())
        return dict(entry[1]) if entry else None


def revoke_session(token: str) -> None:
    with _lock:
        _sessions.pop(hashlib.sha256(token.encode()).hexdigest(), None)


def trusted_service(headers) -> bool:
    expected = os.environ.get("GMAIL_MCP_ADMIN_TOKEN", "")
    raw = headers.get("authorization", "")
    return bool(
        expected
        and headers.get("x-user-id")
        and raw.lower().startswith("bearer ")
        and hmac.compare_digest(raw[7:].strip(), expected)
    )


class PublicAuthMiddleware:
    """Exact Origin check for browser mutations, plus private response headers."""

    def __init__(self, app):
        self.app = app
        validate_public_auth_config()

    async def __call__(self, scope, receive, send):
        config = validate_public_auth_config()
        if config is None or scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = Headers(scope=scope)
        if scope["method"] not in ("GET", "HEAD", "OPTIONS") and not trusted_service(headers):
            origins = headers.getlist("origin")
            if origins != [config.origin]:
                response = JSONResponse(
                    {"detail": "same-origin request required"},
                    status_code=403,
                    headers={"Cache-Control": "private, no-store"},
                )
                return await response(scope, receive, send)

        async def secure_send(message):
            if message["type"] == "http.response.start":
                message["headers"] = [
                    (k, v)
                    for k, v in message.get("headers", [])
                    if k.lower() not in (b"cache-control", b"referrer-policy")
                ]
                message["headers"].extend(
                    [
                        (b"cache-control", b"private, no-store"),
                        (b"referrer-policy", b"no-referrer"),
                    ]
                )
            await send(message)

        await self.app(scope, receive, secure_send)
