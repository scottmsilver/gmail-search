"""OAuth 2.1 authorization-server layer for the gmail-search MCP server.

Why this exists: Claude's custom-connector flow speaks the MCP authorization
spec (OAuth 2.1 + RFC 7591 DCR + RFC 8414 AS-metadata + RFC 9728 PR-metadata +
PKCE S256 + RFC 8707 resource indicators). The low-level `mcp.server.fastmcp`
SDK already mounts every required route (`.well-known/*`, `/authorize`,
`/token`, `/register`, `/revoke`) and wraps `/mcp` in `RequireAuthMiddleware`
the moment you pass `auth_server_provider=` + `auth=AuthSettings(...)`. The only
thing the SDK leaves to us is the actual authorization decision in
`authorize()` — which is exactly where we interpose the broker-gated Google
login and gate to a single owner email.

Design:
  - `GatedBrokerOAuthProvider` duck-types the SDK
    `OAuthAuthorizationServerProvider` protocol. Clients/codes/tokens live in
    in-memory dicts (v1; lost on restart — Claude silently re-runs DCR + the
    OAuth dance, which is a clean 401 -> re-auth, never a 500).
  - `authorize()` does NOT mint a code. It stashes the OAuth params under an
    integrity-protected `state` token (HS256, signed with a server secret) and
    returns the broker `/start?return_url=...` URL. The SDK turns that string
    into a 302.
  - The `/oauth/callback` handler (registered as an unauthenticated
    `custom_route` on the FastMCP app) verifies the broker handoff JWT
    *strictly* (require exp), enforces the owner-email gate, decodes the signed
    state, and only THEN mints + stores the authorization code and 302s back to
    the client's `redirect_uri` with `code` + the client's original `state`.

PKCE: we persist `params.code_challenge` on the AuthorizationCode; the SDK
TokenHandler recomputes BASE64URL(SHA256(verifier)) and compares. redirect_uri
is replayed byte-identically from the stored code, so the SDK's /token
redirect_uri check passes automatically.
"""

from __future__ import annotations

import logging
import os
import secrets
import time
import urllib.parse
from typing import Any, Optional

from mcp.server.auth.provider import (  # noqa: F401
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    AuthorizeError,
    RefreshToken,
    TokenError,
    construct_redirect_uri,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken  # noqa: F401

logger = logging.getLogger(__name__)

# Default single-owner gate. Overridable via env so the value is not hardcoded
# as the only path, but the production default is the sole authorized user.
_DEFAULT_OWNER_EMAIL = "scottmsilver@gmail.com"

_STATE_ALGORITHM = "HS256"
_STATE_AUD = "mcp-oauth-state"
_STATE_TTL_SECONDS = 600  # 10 minutes; broker round-trip is much shorter.

_AUTH_CODE_TTL_SECONDS = 5 * 60
_ACCESS_TOKEN_TTL_SECONDS = 60 * 60


def owner_email() -> str:
    """The single email permitted to complete the OAuth flow. Normalized."""
    raw = os.environ.get("GMAIL_MCP_OAUTH_OWNER_EMAIL", _DEFAULT_OWNER_EMAIL)
    return raw.strip().lower()


def _public_base_url() -> str:
    """The bare public https origin (no trailing slash, no path). Required as
    the OAuth issuer + protected-resource base. Pulled from env — never
    hardcoded (CLAUDE.md rule)."""
    raw = os.environ.get("GMAIL_MCP_PUBLIC_URL")
    if not raw:
        raise RuntimeError(
            "GMAIL_MCP_PUBLIC_URL must be set when GMAIL_MCP_OAUTH_ENABLED is on "
            "(e.g. https://gmail-mcp.oursilverfamily.com)"
        )
    return raw.rstrip("/")


def issuer_url() -> str:
    """OAuth issuer / AS base = the bare public origin."""
    return _public_base_url()


def resource_url() -> str:
    """Canonical MCP resource (RFC 8707/9728) = origin + the /mcp path."""
    from gmail_search.agents.mcp_tools_server import DEFAULT_PATH

    return _public_base_url() + DEFAULT_PATH


def callback_url() -> str:
    """Where the broker redirects back to after Google login."""
    return _public_base_url() + "/oauth/callback"


def _state_secret() -> str:
    """Secret used to sign our OAuth `state`. We reuse the transport secret
    (its own >=32-byte gate is enforced where minted); a dedicated
    GMAIL_MCP_OAUTH_STATE_SECRET takes precedence if set, to keep the blast
    radius narrow."""
    raw = os.environ.get("GMAIL_MCP_OAUTH_STATE_SECRET") or os.environ.get("GMAIL_MCP_TRANSPORT_SECRET")
    if not raw:
        raise RuntimeError(
            "GMAIL_MCP_OAUTH_STATE_SECRET (or GMAIL_MCP_TRANSPORT_SECRET) must be " "set to sign OAuth state"
        )
    if len(raw.encode("utf-8")) < 32:
        raise RuntimeError("OAuth state secret must be at least 32 bytes")
    return raw


def _encode_state(payload: dict[str, Any]) -> str:
    """Sign the OAuth round-trip state as a short-TTL HS256 JWT so a tampered
    or forged state is rejected on return from the broker."""
    import jwt as _jwt

    now = int(time.time())
    claims = dict(payload)
    claims.update({"iat": now, "exp": now + _STATE_TTL_SECONDS, "aud": _STATE_AUD})
    return _jwt.encode(claims, _state_secret(), algorithm=_STATE_ALGORITHM)


def _decode_state(token: str) -> Optional[dict[str, Any]]:
    """Verify a state token: pinned alg, required exp, pinned aud. Returns None
    on any failure (caller fails closed without leaking which check failed)."""
    import jwt as _jwt

    try:
        return _jwt.decode(
            token,
            _state_secret(),
            algorithms=[_STATE_ALGORITHM],
            audience=_STATE_AUD,
            options={"require": ["exp", "aud"]},
        )
    except _jwt.PyJWTError as exc:
        logger.warning("OAuth state verification failed: %s", exc)
        return None


def _verify_broker_handoff_strict(handoff: str) -> Optional[dict[str, Any]]:
    """Strictly verify the broker handoff JWT: pinned HS256, require exp.

    `gmail_search.auth.session.verify_handoff_jwt` does not require exp, so we
    decode here with options={'require':['exp']} against the same secret to
    fail closed on a token that omits expiry."""
    import jwt as _jwt

    from gmail_search.auth.session import _handoff_secret

    try:
        return _jwt.decode(
            handoff,
            _handoff_secret(),
            algorithms=["HS256"],
            options={"require": ["exp"]},
        )
    except _jwt.PyJWTError as exc:
        logger.warning("broker handoff verification failed: %s", exc)
        return None


class GatedBrokerOAuthProvider:
    """OAuth 2.1 AS that gates `/authorize` behind the broker's Google
    login and a single-owner email allowlist. Duck-types the SDK
    `OAuthAuthorizationServerProvider` protocol.

    State lives in a pluggable store: `PgOAuthStore` in production so
    clients/codes/tokens survive service restarts (no forced re-auth),
    `MemoryOAuthStore` (the default) preserving the v1 in-process
    behavior for tests and DB-less deployments."""

    def __init__(self, *, broker_start_url: str, callback_url: str, owner: str, store=None):
        from gmail_search.agents.mcp_oauth_store import MemoryOAuthStore

        self.broker_start_url = broker_start_url.rstrip("/")
        self.callback_url = callback_url
        self.owner = owner.strip().lower()
        self._store = store if store is not None else MemoryOAuthStore()

    # Back-compat views for the memory store's dicts (tests inspect
    # these). PG-backed providers have no dict views — nothing should
    # reach into provider state in production.
    @property
    def clients(self):
        return self._store.clients

    @property
    def auth_codes(self):
        return self._store.auth_codes

    @property
    def access_tokens(self):
        return self._store.access_tokens

    @property
    def refresh_tokens(self):
        return self._store.refresh_tokens

    # ── DCR (RFC 7591) ─────────────────────────────────────────────

    async def get_client(self, client_id: str) -> Optional[OAuthClientInformationFull]:
        return self._store.get_client(client_id)

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        self._store.put_client(client_info)

    # ── /authorize — the gate ──────────────────────────────────────

    async def authorize(self, client: OAuthClientInformationFull, params: AuthorizationParams) -> str:
        """Do NOT mint a code. Stash the (already-validated) OAuth params in a
        signed state and redirect the user-agent to the broker. The SDK
        AuthorizationHandler has already validated redirect_uri + scopes
        against the registered client before calling us."""
        state_token = _encode_state(
            {
                "client_id": client.client_id,
                "redirect_uri": str(params.redirect_uri),
                "redirect_uri_provided_explicitly": params.redirect_uri_provided_explicitly,
                "code_challenge": params.code_challenge,
                "scopes": params.scopes or [],
                "client_state": params.state,
                "resource": params.resource,
                "nonce": secrets.token_urlsafe(16),
            }
        )
        return_url = f"{self.callback_url}?broker_state={urllib.parse.quote(state_token)}"
        query = urllib.parse.urlencode({"return_url": return_url, "scope": "openid,profile,email"})
        return f"{self.broker_start_url}?{query}"

    def resume_after_broker(self, *, state_token: str, email: str) -> Optional[str]:
        """Called from /oauth/callback AFTER the handoff JWT is verified and the
        owner-email gate passes. Decode + single-use-check the state, mint an
        authorization code bound to (client_id, redirect_uri, code_challenge,
        resource), and return the client redirect URL with code + client state.

        Returns None if the state is invalid/replayed (caller -> error)."""
        claims = _decode_state(state_token)
        if not claims:
            return None
        # Single-use: a state (and the handoff bound to it) can mint exactly one
        # code. nonce identifies the state instance; consume_nonce is atomic so
        # the guard holds across concurrent callbacks AND service restarts.
        nonce = str(claims.get("nonce") or "")
        if not nonce or not self._store.consume_nonce(nonce, _STATE_TTL_SECONDS):
            return None

        # Defense-in-depth: the email is the verified handoff email, never a
        # request param. Re-check the gate here too.
        if email.strip().lower() != self.owner:
            return None

        code_value = f"ac_{secrets.token_hex(16)}"
        self._store.put_auth_code(
            AuthorizationCode(
                code=code_value,
                client_id=str(claims["client_id"]),
                redirect_uri=claims["redirect_uri"],
                redirect_uri_provided_explicitly=bool(claims["redirect_uri_provided_explicitly"]),
                scopes=list(claims.get("scopes") or []),
                expires_at=time.time() + _AUTH_CODE_TTL_SECONDS,
                code_challenge=str(claims["code_challenge"]),
                resource=claims.get("resource"),
            )
        )
        return construct_redirect_uri(
            str(claims["redirect_uri"]),
            code=code_value,
            state=claims.get("client_state"),
        )

    # ── /token — authorization_code grant ──────────────────────────

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> Optional[AuthorizationCode]:
        code = self._store.get_auth_code(authorization_code)
        if not code:
            return None
        if code.client_id != client.client_id:
            return None
        if code.expires_at < time.time():
            self._store.delete_auth_code(authorization_code)
            return None
        return code

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode
    ) -> OAuthToken:
        # Single-use: consume_auth_code is an atomic delete, so a replayed
        # /token (or a concurrent duplicate) fails invalid_grant.
        if not self._store.consume_auth_code(authorization_code.code):
            raise TokenError("invalid_grant", "authorization code not found or already used")
        return self._issue_token_pair(client.client_id, authorization_code.scopes, authorization_code.resource)

    # ── /token — refresh_token grant (rotation, OAuth 2.1 MUST) ─────

    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> Optional[RefreshToken]:
        tok = self._store.get_refresh_token(refresh_token)
        if not tok:
            return None
        if tok.client_id != client.client_id:
            return None
        if tok.expires_at is not None and tok.expires_at < time.time():
            self._store.revoke_refresh(tok.token)
            return None
        return tok

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        if not set(scopes).issubset(set(refresh_token.scopes)):
            raise TokenError("invalid_scope", "requested scopes exceed those granted")
        # Rotate atomically: consume_refresh claims the row exactly once
        # (returning the preserved RFC 8707 resource audience), so two
        # concurrent refreshes of the same token can't both mint pairs.
        consumed, resource = self._store.consume_refresh(refresh_token.token)
        if not consumed:
            raise TokenError("invalid_grant", "refresh token not found or already rotated")
        return self._issue_token_pair(client.client_id, scopes or refresh_token.scopes, resource)

    # ── token verification (TokenVerifier path for /mcp) ────────────

    async def load_access_token(self, token: str) -> Optional[AccessToken]:
        tok = self._store.get_access_token(token)
        if tok is not None:
            if tok.expires_at is not None and tok.expires_at < time.time():
                # Expiry is routine — drop ONLY the access token. The
                # paired refresh token must survive so the client rotates
                # instead of re-authing. (v1 cascaded the whole pair here,
                # which itself forced spurious re-auths.)
                self._store.discard_access(tok.token)
                return None
            return tok
        # Fallback: accept the existing transport/service JWTs as valid
        # access tokens for THIS resource, so local + trusted server-side
        # clients keep working through the SDK's bearer auth (which calls
        # load_access_token) when OAuth is enabled. Inline import avoids an
        # import cycle with mcp_tools_server.
        from gmail_search.agents.mcp_tools_server import _SERVICE_AUD, _TRANSPORT_AUD
        from gmail_search.agents.mcp_tools_server import verify_token as _verify_transport_jwt

        claims = _verify_transport_jwt(token)
        if claims is None:
            return None
        aud = claims.get("aud")
        if aud not in (_TRANSPORT_AUD, _SERVICE_AUD):
            return None
        # client_id encodes the caller identity; a transport token carries
        # uid, a service token is tenantless.
        client_id = str(claims.get("uid") or "") if aud == _TRANSPORT_AUD else "mcp-service"
        exp = claims.get("exp")
        return AccessToken(
            token=token,
            client_id=client_id or "mcp-transport",
            scopes=["mcp"],
            expires_at=int(exp) if isinstance(exp, (int, float)) else None,
            resource=None,
        )

    async def verify_token(self, token: str) -> Optional[AccessToken]:
        return await self.load_access_token(token)

    # ── revocation (RFC 7009) ──────────────────────────────────────

    async def revoke_token(self, token: Any) -> None:
        if isinstance(token, AccessToken):
            self._store.revoke_access(token.token)
        elif isinstance(token, RefreshToken):
            self._store.revoke_refresh(token.token)

    # ── internals ──────────────────────────────────────────────────

    def _issue_token_pair(self, client_id: str, scopes: list[str], resource: Optional[str]) -> OAuthToken:
        access_value = f"at_{secrets.token_hex(32)}"
        refresh_value = f"rt_{secrets.token_hex(32)}"
        access = AccessToken(
            token=access_value,
            client_id=client_id,
            scopes=list(scopes),
            expires_at=int(time.time() + _ACCESS_TOKEN_TTL_SECONDS),
            resource=resource,
        )
        refresh = RefreshToken(
            token=refresh_value,
            client_id=client_id,
            scopes=list(scopes),
            expires_at=None,
        )
        self._store.put_token_pair(access, refresh, resource)
        return OAuthToken(
            access_token=access_value,
            token_type="Bearer",
            expires_in=_ACCESS_TOKEN_TTL_SECONDS,
            refresh_token=refresh_value,
            scope=" ".join(scopes),
        )


def is_oauth_enabled() -> bool:
    """OAuth layer is opt-in. Default OFF so existing behavior is unchanged."""
    return os.environ.get("GMAIL_MCP_OAUTH_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")


def build_auth_settings():
    """Construct the SDK AuthSettings for the OAuth-enabled FastMCP app.
    Returns (AuthSettings, GatedBrokerOAuthProvider)."""
    from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions, RevocationOptions

    from gmail_search.agents.mcp_oauth_store import build_default_store
    from gmail_search.auth.session import broker_url

    provider = GatedBrokerOAuthProvider(
        broker_start_url=broker_url() + "/start",
        callback_url=callback_url(),
        owner=owner_email(),
        # PG-backed when the DB is reachable, so clients/tokens survive
        # service restarts; memory fallback logs a loud warning.
        store=build_default_store(),
    )
    settings = AuthSettings(
        issuer_url=issuer_url(),
        resource_server_url=resource_url(),
        client_registration_options=ClientRegistrationOptions(
            enabled=True, valid_scopes=["mcp"], default_scopes=["mcp"]
        ),
        revocation_options=RevocationOptions(enabled=True),
        required_scopes=[],
    )
    return settings, provider


def register_oauth_callback_route(app, provider: GatedBrokerOAuthProvider) -> None:
    """Mount the unauthenticated /oauth/callback landing route. The handoff JWT
    is the auth here, so this route is intentionally outside RequireAuthMiddleware.

    Fails CLOSED (no code issued) on any verification failure and returns
    generic errors so it can't be used as an oracle."""
    from starlette.requests import Request
    from starlette.responses import JSONResponse, RedirectResponse

    @app.custom_route("/oauth/callback", methods=["GET"])
    async def oauth_callback(request: Request):  # noqa: ANN001
        handoff = request.query_params.get("silver_oauth")
        broker_state = request.query_params.get("broker_state")
        if not handoff or not broker_state:
            return JSONResponse({"error": "invalid_request"}, status_code=400)

        claims = _verify_broker_handoff_strict(handoff)
        if not claims:
            return JSONResponse({"error": "access_denied"}, status_code=403)

        # The email comes ONLY from the verified handoff JWT — never a request
        # param — and is normalized before the gate check.
        from gmail_search.auth.session import normalize_email

        email = normalize_email(str(claims.get("email") or ""))
        if not email or email != provider.owner:
            return JSONResponse({"error": "access_denied"}, status_code=403)

        redirect = provider.resume_after_broker(state_token=broker_state, email=email)
        if not redirect:
            return JSONResponse({"error": "invalid_request"}, status_code=400)
        return RedirectResponse(redirect, status_code=302)
