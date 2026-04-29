"""FastAPI routes for the silver-oauth broker sign-in flow.

Five endpoints, all under `/api/auth`:
  * `GET /api/auth/login?return_url=…` — kick off sign-in. Sets a
    short-lived state cookie, 302s to the broker's `/start` with our
    `/api/auth/callback?return=…&state=…` as the broker's return_url.
    Public.
  * `GET /api/auth/callback?silver_oauth=…&return=…&state=…` — broker
    redirects here with a 60-second handoff JWT. We verify both the
    handoff JWT AND the state cookie matches the state query param,
    check the email allowlist, upsert the `users` row, set our
    session cookie, and 302 to the original return target.
  * `GET /api/auth/me` — return the current user (or 401 if not
    signed in / `null` user when the multi-tenant flag is off).
  * `POST /api/auth/logout` — clear the session cookie.
  * `GET /api/auth/whoami` — public diagnostic. Reports the multi-tenant
    flag posture without ever reading the cookie.

The state cookie closes the login-CSRF / session-swap window where an
attacker's own valid handoff JWT could be force-fed to a victim
within the broker's 60-second TTL — without the cookie the victim's
browser can't satisfy the equality check.

When `GMAIL_MULTI_TENANT != "1"`:
  * login/callback/logout still register but are dead code.
  * me returns `{multi_tenant: false, user: null}` — the legacy
    "single-pool, no auth" probe answer the frontend expects.
"""

from __future__ import annotations

import hmac
import logging
import os
import secrets
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote, urlencode

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, RedirectResponse

from gmail_search.auth.session import (
    User,
    broker_url,
    clear_session_cookie,
    is_email_allowed,
    is_multi_tenant_enabled,
    normalize_email,
    require_user,
    safe_relative_return_url,
    set_session_cookie,
    verify_handoff_jwt,
)

logger = logging.getLogger(__name__)

HANDOFF_QUERY_PARAM = "silver_oauth"
STATE_COOKIE = "gms_oauth_state"
STATE_QUERY_PARAM = "state"
STATE_TTL_SECONDS = 600

# Defense-in-depth host allowlist. The broker callback URL we hand to
# silver-oauth is built from the request's `Host` (or `X-Forwarded-Host`
# from a trusted proxy). If a forged Host slipped through some upstream,
# the broker would 302 the victim to an attacker domain. The state-cookie
# binding limits the actual blast radius (the attacker can't replay the
# resulting handoff JWT against us without our cookie), but we'd rather
# fail the request than rely on the cookie defense alone.
#
# Defaults cover local dev. Production hosts must be added via the
# `GMS_ALLOWED_HOSTS` env var (comma-separated, lowercase, host:port
# matching exactly what the browser sees in the URL bar — e.g.
# `gms.i.oursilverfamily.com`).
_DEFAULT_ALLOWED_HOSTS: frozenset[str] = frozenset(
    {
        "localhost:3000",
        "localhost:8090",
        "127.0.0.1:3000",
        "127.0.0.1:8090",
    }
)


def _allowed_hosts() -> set[str]:
    raw = os.environ.get("GMS_ALLOWED_HOSTS", "")
    explicit = {h.strip().lower() for h in raw.split(",") if h.strip()}
    return set(_DEFAULT_ALLOWED_HOSTS) | explicit


def _new_user_id() -> str:
    """Opaque per-user id. Not the email — emails change (rename, alias)
    and exposing them as FK targets leaks PII into log lines."""
    return f"u_{secrets.token_urlsafe(12)}"


def _upsert_user(db_path: Path, *, email: str, name: Optional[str], picture: Optional[str]) -> dict[str, Any]:
    """Race-safe find-or-create on `users` keyed on canonicalized email.

    Single-statement upsert via `INSERT … ON CONFLICT (email) DO UPDATE`
    so two concurrent first-logins (e.g. user clicks sign-in twice or
    two browser tabs land on /callback at once) can't both pass an
    existence check and then both INSERT — the previous find-then-insert
    pattern would have 500'd on the UNIQUE violation.

    `google_sub` is left NULL: the silver-oauth broker doesn't expose
    Google's `sub` claim and we don't fabricate one. The schema makes
    the column nullable + UNIQUE — see pg_schema.sql for the rationale.
    """
    from gmail_search.store.db import get_connection

    normalized = normalize_email(email)
    user_id = _new_user_id()
    conn = get_connection(db_path)
    try:
        # COALESCE on UPDATE so a missing `name`/`avatar_url` from a
        # later sign-in doesn't blank out values from an earlier one.
        # The `xmax = 0` trick distinguishes INSERT (0) from UPDATE
        # (non-zero) so we can log provisioning vs. a return visit.
        row = conn.execute(
            "INSERT INTO users (id, email, name, avatar_url, last_login_at) "
            "VALUES (%s, %s, %s, %s, NOW()) "
            "ON CONFLICT (email) DO UPDATE SET "
            "  name = COALESCE(EXCLUDED.name, users.name), "
            "  avatar_url = COALESCE(EXCLUDED.avatar_url, users.avatar_url), "
            "  last_login_at = NOW() "
            "RETURNING id, email, name, (xmax = 0) AS was_inserted",
            (user_id, normalized, name, picture),
        ).fetchone()
        conn.commit()
        if row["was_inserted"]:
            logger.info("provisioned new user: %s (%s)", normalized, row["id"])
        return {"id": row["id"], "email": row["email"], "name": row["name"]}
    finally:
        conn.close()


def _set_state_cookie(response: RedirectResponse, request: Request, value: str) -> None:
    """Mirrors the session cookie's settings (HttpOnly, conditional
    Secure, SameSite=lax) but lives only `STATE_TTL_SECONDS`. SameSite
    must NOT be 'strict' or the cookie won't be sent on the broker's
    cross-site redirect back to /api/auth/callback."""
    is_https = request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"
    response.set_cookie(
        STATE_COOKIE,
        value,
        max_age=STATE_TTL_SECONDS,
        httponly=True,
        secure=is_https,
        samesite="lax",
        path="/",
    )


def _clear_state_cookie(response) -> None:
    response.delete_cookie(STATE_COOKIE, path="/")


def register_auth_routes(app: FastAPI, db_path: Path) -> None:

    @app.get("/api/auth/login")
    async def login(request: Request, return_url: str = "/"):
        """Bounce the user to the broker. Issues a state cookie + state
        query param so /callback can prove this completion belongs to
        the same browser that started the flow."""
        if not is_multi_tenant_enabled():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="multi-tenant auth is disabled (GMAIL_MULTI_TENANT != 1)",
            )
        # Restrict return_url to local paths — open-redirect prevention.
        # An external return_url would let an attacker phish through a
        # successful sign-in.
        safe_return = safe_relative_return_url(return_url)
        # Build our own callback URL using the request's host so
        # localhost / prod / LAN-name all work without per-host config.
        # When the request arrived via a proxy (Next.js dev server,
        # Caddy in prod), prefer X-Forwarded-Host — `Host` would be
        # the proxy's upstream (e.g. 127.0.0.1:8090) and the browser
        # would then visit the wrong origin on the broker round-trip,
        # which would drop both the state cookie (different origin)
        # and the session cookie (set on the wrong domain).
        host = (request.headers.get("x-forwarded-host") or request.headers.get("host", "")).lower()
        # Defense-in-depth: a forged Host header from an upstream proxy
        # would land in the broker callback URL and ship the victim to
        # an attacker domain after sign-in. State-cookie binding limits
        # the damage (the resulting handoff JWT can't be replayed
        # against us without the cookie), but fail closed anyway.
        if host not in _allowed_hosts():
            logger.warning("rejected /api/auth/login from non-allowlisted host: %r", host)
            raise HTTPException(
                status_code=400,
                detail=f"host {host!r} is not in GMS_ALLOWED_HOSTS",
            )
        scheme = "https" if request.headers.get("x-forwarded-proto") == "https" else request.url.scheme
        state = secrets.token_urlsafe(24)
        our_callback = (
            f"{scheme}://{host}/api/auth/callback"
            f"?return={quote(safe_return, safe='')}"
            f"&{STATE_QUERY_PARAM}={quote(state, safe='')}"
        )
        qs = urlencode({"return_url": our_callback, "scope": "openid,profile"})
        response = RedirectResponse(url=f"{broker_url()}/start?{qs}")
        _set_state_cookie(response, request, state)
        return response

    @app.get("/api/auth/callback")
    async def callback(request: Request):
        """Verify the broker's handoff JWT + the state cookie/query
        match, check allowlist, provision the user, set session cookie,
        redirect to the original return_url."""
        if not is_multi_tenant_enabled():
            raise HTTPException(status_code=404, detail="multi-tenant auth is disabled")

        # Open-redirect prevention applies on the way out too — the
        # `return` param flowed through the broker, anyone could have
        # tampered with it.
        return_target = safe_relative_return_url(request.query_params.get("return") or "/")
        token = request.query_params.get(HANDOFF_QUERY_PARAM, "")
        if not token:
            raise HTTPException(status_code=400, detail="missing silver_oauth handoff token")

        # State binding: cookie + query param must agree (constant-time
        # compare). Without this, an attacker's valid 60s handoff JWT
        # could be force-fed to a victim's browser via a crafted URL,
        # logging the victim into the attacker's account (session swap).
        cookie_state = request.cookies.get(STATE_COOKIE) or ""
        query_state = request.query_params.get(STATE_QUERY_PARAM) or ""
        if not cookie_state or not query_state or not hmac.compare_digest(cookie_state, query_state):
            raise HTTPException(
                status_code=400,
                detail="login state mismatch — restart sign-in",
            )

        payload = verify_handoff_jwt(token)
        if not payload:
            raise HTTPException(status_code=401, detail="handoff token invalid or expired")

        email = normalize_email(str(payload.get("email") or ""))
        if not email:
            raise HTTPException(status_code=401, detail="handoff token missing email")

        if not is_email_allowed(db_path, email):
            logger.info("rejected sign-in for non-allowlisted email: %s", email)
            raise HTTPException(
                status_code=403,
                detail=f"{email} is not allowed to use this app",
            )

        user = _upsert_user(
            db_path,
            email=email,
            name=payload.get("name"),
            picture=payload.get("picture"),
        )

        response = RedirectResponse(url=return_target)
        # Clear the one-time state cookie now that we've consumed it.
        _clear_state_cookie(response)
        set_session_cookie(
            response,
            request,
            user_id=user["id"],
            email=user["email"],
            name=user.get("name"),
            picture=payload.get("picture"),
        )
        return response

    @app.get("/api/auth/me")
    async def me(user: Optional[User] = Depends(require_user)) -> dict[str, Any]:
        """Auth-gated identity probe. With multi-tenant ON, returns 401
        unless the cookie verifies. With it OFF, returns the legacy
        `null` shape so the frontend can detect 'no auth required
        here, single-pool mode'."""
        if user is None:
            return {"multi_tenant": False, "user": None}
        return {
            "multi_tenant": True,
            "user": {
                "id": user.id,
                "email": user.email,
                "name": user.name,
                "picture": user.picture,
            },
        }

    @app.post("/api/auth/logout")
    async def logout() -> JSONResponse:
        response = JSONResponse({"ok": True})
        clear_session_cookie(response)
        return response

    @app.get("/api/auth/whoami")
    async def whoami() -> dict[str, Any]:
        """Public diagnostic. Reports auth posture without ever reading
        the cookie. Lets ops confirm the env flag is active."""
        return {"multi_tenant_enabled": is_multi_tenant_enabled()}
