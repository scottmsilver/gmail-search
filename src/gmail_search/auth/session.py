"""Session-cookie auth via the silver-oauth broker.

Pivoted from a NextAuth-issued-Bearer-JWT model after realizing the
silver-oauth broker pattern (see ../silver-oauth/docs/consumer-integration.md)
already solves the per-app-Google-OAuth-client problem we were
re-inventing. The broker holds the only Google OAuth client for the
whole homelab and hands us a short-lived signed JWT proving the user's
email; we exchange that for our own session cookie and we're done.

Sign-in flow (mirrors cchost):
  1. Frontend hits /api/auth/login?return_url=/whatever
  2. We 302 to the broker's /start endpoint with our /api/auth/callback
     as the broker's `return_url` and `scope=openid,profile`.
  3. Broker runs Google OAuth, then 302s to our callback with
     `?silver_oauth=<jwt>` (HMAC-signed handoff, 60-second TTL).
  4. We verify the JWT with `BROKER_HANDOFF_SECRET`, check the email
     against the allowlist, upsert the `users` row, set our own session
     cookie, 302 to the original return_url.
  5. Subsequent requests carry the cookie; `require_user` validates it.

The whole surface is gated on `GMAIL_MULTI_TENANT == "1"`. Off (the
Phase 0 default), `require_user` is a no-op and every endpoint behaves
like the legacy single-pool path.

Env (all required when the flag is on):
  SILVER_OAUTH_BROKER_URL     base URL of the broker, no trailing slash
  BROKER_HANDOFF_SECRET       HMAC key shared with the broker (>=32 bytes)
  GMS_SESSION_SECRET          HMAC key for our session cookie  (>=32 bytes)
                               (separate from BROKER_HANDOFF_SECRET so
                                leaking one doesn't grant the other)
  GMS_ALLOWED_EMAILS          comma-separated email allowlist (union'd
                               with the `invited_emails` DB table)
  GMS_SESSION_TTL_DAYS        cookie lifetime in days (default 30)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import jwt
from fastapi import HTTPException, Request, Response, status

logger = logging.getLogger(__name__)

SESSION_COOKIE = "gms_session"
HANDOFF_QUERY_PARAM = "silver_oauth"

# Algorithm pinning — accepting a list is how libraries get tricked into
# alg=none / RS256-as-HS256 confusion attacks.
_ALGORITHM = "HS256"
_MIN_SECRET_BYTES = 32


class AuthError(HTTPException):
    """401 with a stable shape so the frontend can distinguish from 500s."""

    def __init__(self, detail: str, *, status_code: int = status.HTTP_401_UNAUTHORIZED):
        super().__init__(status_code=status_code, detail=detail)


@dataclass(frozen=True)
class User:
    """Resolved-from-cookie identity. `id` is our internal user_id (the
    FK target Phases 2/3 will scope rows by). `email` is what the
    allowlist matches on. `name`/`picture` are best-effort UI hints."""

    id: str
    email: str
    name: Optional[str] = None
    picture: Optional[str] = None


# ─── env / config ────────────────────────────────────────────────────────


def is_multi_tenant_enabled() -> bool:
    return os.environ.get("GMAIL_MULTI_TENANT") == "1"


def _require_secret(name: str) -> str:
    """Fetch a >=32-byte HMAC secret. Below that, HS256 forgery is
    practical, so we fail closed rather than silently sign with a
    guessable key."""
    raw = os.environ.get(name)
    if not raw:
        raise AuthError(f"{name} is not configured on the server", status_code=500)
    if len(raw.encode("utf-8")) < _MIN_SECRET_BYTES:
        raise AuthError(
            f"{name} must be at least {_MIN_SECRET_BYTES} bytes " "(generate with `openssl rand -base64 32`)",
            status_code=500,
        )
    return raw


def _session_secret() -> str:
    return _require_secret("GMS_SESSION_SECRET")


def _handoff_secret() -> str:
    return _require_secret("BROKER_HANDOFF_SECRET")


def broker_url() -> str:
    raw = os.environ.get("SILVER_OAUTH_BROKER_URL")
    if not raw:
        raise AuthError("SILVER_OAUTH_BROKER_URL is not configured", status_code=500)
    return raw.rstrip("/")


def _session_ttl_seconds() -> int:
    return int(os.environ.get("GMS_SESSION_TTL_DAYS", "30")) * 86400


def normalize_email(email: str) -> str:
    """Single source of truth for canonicalizing emails. Used by every
    write site (CLI invite, callback upsert, allowlist check) so a
    leading-whitespace email can't pass the allowlist and then store
    a different value in `users`."""
    return email.strip().lower()


def env_allowed_emails() -> set[str]:
    """Static env-var allowlist. Union'd with the `invited_emails` DB
    table — either source is sufficient for sign-in."""
    raw = os.environ.get("GMS_ALLOWED_EMAILS", "")
    return {normalize_email(e) for e in raw.split(",") if e.strip()}


def is_email_allowed(db_path: Path, email: str) -> bool:
    """Check both the env allowlist and the `invited_emails` table.
    Env-var membership is checked first (cheaper, no DB hit)."""
    normalized = normalize_email(email)
    if normalized in env_allowed_emails():
        return True
    from gmail_search.store.db import get_connection

    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT 1 FROM invited_emails WHERE email = %s", (normalized,)).fetchone()
        return row is not None
    finally:
        conn.close()


def safe_relative_return_url(url: str) -> str:
    """Restrict `return_url` to local paths so a forged
    `/api/auth/login?return_url=https://evil.example` can't redirect
    a victim off-site after sign-in.

    Rejects:
      - empty or non-`/`-prefixed URLs (absolute / scheme-relative)
      - protocol-relative `//evil.com`
      - back-slash smuggling `/\\evil.com`
      - `/api/auth/*` paths so sign-in can't chain into immediate
        sign-out, infinite-loop a re-auth, or re-trigger a callback
        with stale params.
    Anything rejected falls back to `/`."""
    if not url or not url.startswith("/"):
        return "/"
    if url.startswith("//") or url.startswith("/\\"):
        return "/"
    # Strip query/fragment before path-prefix checks so an attacker
    # can't smuggle `/api/auth/login?evil` past the auth-loop guard.
    path = url.split("?", 1)[0].split("#", 1)[0]
    if path.startswith("/api/auth/"):
        return "/"
    return url


# ─── JWT helpers ─────────────────────────────────────────────────────────


def make_session_cookie(*, user_id: str, email: str, name: Optional[str] = None, picture: Optional[str] = None) -> str:
    """Sign our own session cookie. Independent of the broker handoff —
    the handoff JWT lives only long enough to be exchanged for this."""
    now = int(time.time())
    payload: dict[str, object] = {
        "uid": user_id,
        "email": email,
        "iat": now,
        "exp": now + _session_ttl_seconds(),
    }
    if name:
        payload["name"] = name
    if picture:
        payload["picture"] = picture
    return jwt.encode(payload, _session_secret(), algorithm=_ALGORITHM)


def _verify_session_cookie(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, _session_secret(), algorithms=[_ALGORITHM])
    except jwt.PyJWTError as exc:
        logger.debug("session cookie verification failed: %s", exc)
        return None


def verify_handoff_jwt(token: str) -> Optional[dict]:
    """Verify the broker's handoff JWT. Short-TTL (~60s) HS256 token
    proving the holder went through Google OAuth and the broker says
    this email is theirs. Returning None lets the caller 401 without
    leaking which check failed (sig vs expiry vs missing claim)."""
    try:
        return jwt.decode(token, _handoff_secret(), algorithms=[_ALGORITHM])
    except jwt.PyJWTError as exc:
        logger.warning("handoff JWT verification failed: %s", exc)
        return None


# ─── cookie I/O on Response objects ──────────────────────────────────────


def set_session_cookie(
    response: Response,
    request: Request,
    *,
    user_id: str,
    email: str,
    name: Optional[str] = None,
    picture: Optional[str] = None,
) -> None:
    """`secure=True` only when the request came in over HTTPS — local
    dev on http://localhost:3000 needs `secure=False` or the browser
    drops the cookie."""
    is_https = request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"
    response.set_cookie(
        SESSION_COOKIE,
        make_session_cookie(user_id=user_id, email=email, name=name, picture=picture),
        max_age=_session_ttl_seconds(),
        httponly=True,
        secure=is_https,
        samesite="lax",
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(SESSION_COOKIE, path="/")


# ─── user lookup / dependency ────────────────────────────────────────────


def _resolve_db_path(request: Request) -> Path:
    data_dir = getattr(request.app.state, "data_dir", None)
    if data_dir is None:
        data_dir = Path("data")
    return Path(data_dir) / "gmail_search.db"


def _lookup_user_by_id(db_path: Path, user_id: str) -> Optional[dict]:
    from gmail_search.store.db import get_connection

    conn = get_connection(db_path)
    try:
        return conn.execute("SELECT id, email, name FROM users WHERE id = %s", (user_id,)).fetchone()
    finally:
        conn.close()


def require_user(request: Request) -> Optional[User]:
    """FastAPI dependency: returns the verified `User` for this request
    or raises 401.

    When `GMAIL_MULTI_TENANT != "1"`: returns None unconditionally.
    Existing handlers should treat None as "system / sole user".
    """
    if not is_multi_tenant_enabled():
        return None

    cookie = request.cookies.get(SESSION_COOKIE)
    if not cookie:
        raise AuthError("not signed in")
    payload = _verify_session_cookie(cookie)
    if not payload:
        raise AuthError("session invalid or expired")

    email = str(payload.get("email") or "")
    user_id = str(payload.get("uid") or "")
    if not email or not user_id:
        raise AuthError("session missing required claims")

    # Re-check the allowlist on every request — uninvite (CLI delete or
    # env var change) takes effect immediately, no need to wait for
    # cookie expiry.
    db_path = _resolve_db_path(request)
    if not is_email_allowed(db_path, email):
        raise AuthError("email no longer allowed", status_code=403)

    row = _lookup_user_by_id(db_path, user_id)
    if not row:
        # The cookie verifies but the underlying users row was deleted
        # (e.g., `gmail-search delete-user`). Force a fresh sign-in.
        raise AuthError("session refers to a deleted user")
    return User(
        id=row["id"],
        email=row["email"],
        name=row["name"],
        picture=payload.get("picture"),
    )


def issue_handoff_jwt_for_test(*, email: str, name: Optional[str] = None, ttl_seconds: int = 60) -> str:
    """Issue a handoff JWT compatible with `verify_handoff_jwt`. Tests
    use this to drive the /api/auth/callback flow without spinning up
    the actual broker."""
    now = int(time.time())
    payload: dict[str, object] = {"email": email, "iat": now, "exp": now + ttl_seconds}
    if name:
        payload["name"] = name
    return jwt.encode(payload, _handoff_secret(), algorithm=_ALGORITHM)
