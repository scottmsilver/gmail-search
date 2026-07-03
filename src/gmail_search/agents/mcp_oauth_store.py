"""Durable backing stores for the MCP OAuth provider state.

Why: `GatedBrokerOAuthProvider` v1 kept clients/codes/tokens in process
memory, so every `gmail-search-mcp.service` restart invalidated claude.ai's
tokens and forced a full re-auth. `PgOAuthStore` moves that state into
Postgres so restarts are invisible to connected clients.

Security model: the DB never holds raw bearer secrets. Token/code rows are
keyed by SHA-256(secret) and the serialized model has its secret field
blanked; the raw value is re-attached from the presented credential at load
time. Client registrations are stored as-is (client_secret must remain
retrievable for the SDK's client authentication).

`MemoryOAuthStore` preserves the v1 in-process behavior byte-for-byte (raw
keys, plain dicts) for tests and for deployments without a reachable DB.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Optional

from mcp.server.auth.provider import AccessToken, AuthorizationCode, RefreshToken
from mcp.shared.auth import OAuthClientInformationFull

logger = logging.getLogger(__name__)

_KIND_CLIENT = "client"
_KIND_CODE = "code"
_KIND_ACCESS = "access"
_KIND_REFRESH = "refresh"
_KIND_NONCE = "nonce"

_DDL = """
CREATE TABLE IF NOT EXISTS mcp_oauth_state (
    kind TEXT NOT NULL,
    key TEXT NOT NULL,
    value JSONB NOT NULL,
    expires_at DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (kind, key)
);
CREATE INDEX IF NOT EXISTS mcp_oauth_state_expires_idx
    ON mcp_oauth_state (expires_at) WHERE expires_at IS NOT NULL;
"""


def _sha(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def _dump_without(model, secret_field: str, extra: dict | None = None) -> dict:
    """Serialize a pydantic model with its secret field blanked, plus
    bookkeeping extras (pair hashes, resource)."""
    data = model.model_dump(mode="json")
    data[secret_field] = ""
    if extra:
        data.update(extra)
    return data


def _load_model(model_cls, secret_field: str, raw_secret: str, value: dict):
    """Rebuild a pydantic model from a stored row, re-attaching the raw
    secret the caller presented and dropping bookkeeping extras."""
    fields = {k: v for k, v in value.items() if k in model_cls.model_fields}
    fields[secret_field] = raw_secret
    return model_cls(**fields)


class MemoryOAuthStore:
    """Process-local store — v1 semantics, state lost on restart."""

    def __init__(self) -> None:
        self.clients: dict[str, OAuthClientInformationFull] = {}
        self.auth_codes: dict[str, AuthorizationCode] = {}
        self.access_tokens: dict[str, AccessToken] = {}
        self.refresh_tokens: dict[str, RefreshToken] = {}
        self._access_to_refresh: dict[str, str] = {}
        self._refresh_to_access: dict[str, str] = {}
        self._refresh_resource: dict[str, Optional[str]] = {}
        self._consumed_nonces: set[str] = set()

    def get_client(self, client_id: str) -> Optional[OAuthClientInformationFull]:
        return self.clients.get(client_id)

    def put_client(self, info: OAuthClientInformationFull) -> None:
        self.clients[info.client_id] = info

    def put_auth_code(self, code: AuthorizationCode) -> None:
        self.auth_codes[code.code] = code

    def get_auth_code(self, code: str) -> Optional[AuthorizationCode]:
        return self.auth_codes.get(code)

    def delete_auth_code(self, code: str) -> None:
        self.auth_codes.pop(code, None)

    def consume_auth_code(self, code: str) -> bool:
        return self.auth_codes.pop(code, None) is not None

    def put_token_pair(self, access: AccessToken, refresh: RefreshToken, resource: Optional[str]) -> None:
        self.access_tokens[access.token] = access
        self.refresh_tokens[refresh.token] = refresh
        self._access_to_refresh[access.token] = refresh.token
        self._refresh_to_access[refresh.token] = access.token
        self._refresh_resource[refresh.token] = resource

    def get_access_token(self, token: str) -> Optional[AccessToken]:
        return self.access_tokens.get(token)

    def get_refresh_token(self, token: str) -> Optional[RefreshToken]:
        return self.refresh_tokens.get(token)

    def get_refresh_resource(self, token: str) -> Optional[str]:
        return self._refresh_resource.get(token)

    def revoke_access(self, token: str) -> None:
        self.access_tokens.pop(token, None)
        paired = self._access_to_refresh.pop(token, None)
        if paired:
            self.refresh_tokens.pop(paired, None)
            self._refresh_to_access.pop(paired, None)
            self._refresh_resource.pop(paired, None)

    def revoke_refresh(self, token: str) -> None:
        self.refresh_tokens.pop(token, None)
        self._refresh_resource.pop(token, None)
        paired = self._refresh_to_access.pop(token, None)
        if paired:
            self.access_tokens.pop(paired, None)
            self._access_to_refresh.pop(paired, None)

    def consume_refresh(self, token: str) -> tuple[bool, Optional[str]]:
        """Atomically claim a refresh token for rotation: exactly one
        caller gets (True, resource); everyone else (False, None). The
        old pair is revoked as part of the claim."""
        if self.refresh_tokens.pop(token, None) is None:
            return (False, None)
        resource = self._refresh_resource.pop(token, None)
        paired = self._refresh_to_access.pop(token, None)
        if paired:
            self.access_tokens.pop(paired, None)
            self._access_to_refresh.pop(paired, None)
        return (True, resource)

    def discard_access(self, token: str) -> None:
        """Drop ONLY the access token (expiry path). The paired refresh
        token must survive — access expiry is routine and the refresh
        grant is what lets the client recover without a re-auth."""
        self.access_tokens.pop(token, None)
        self._access_to_refresh.pop(token, None)

    def consume_nonce(
        self, nonce: str, ttl_seconds: float
    ) -> bool:  # noqa: ARG002 — memory set never expires (process-lifetime only)
        if nonce in self._consumed_nonces:
            return False
        self._consumed_nonces.add(nonce)
        return True


class PgOAuthStore:
    """Postgres write-through store. Opens a short-lived connection per
    operation (same pattern as `_persist_call_to_events`) so concurrent
    async handlers never share connection state. Raises on construction
    if the DB is unreachable — the caller decides the fallback."""

    def __init__(self, conn_factory=None) -> None:
        if conn_factory is None:
            from gmail_search.store.db import get_connection

            conn_factory = lambda: get_connection(None)  # noqa: E731 — db_path arg is ignored by the PG shim
        self._connect = conn_factory
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Idempotent DDL at construction: the MCP service can restart
        before serve's init_db has applied a schema that includes this
        table, so it must be self-sufficient."""
        conn = self._connect()
        try:
            conn.execute(_DDL)
            conn.commit()
        finally:
            conn.close()

    # ── row primitives ─────────────────────────────────────────────

    def _put(self, kind: str, key: str, value: dict, expires_at: Optional[float]) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO mcp_oauth_state (kind, key, value, expires_at) "
                "VALUES (%s, %s, %s::jsonb, %s) "
                "ON CONFLICT (kind, key) DO UPDATE "
                "SET value = EXCLUDED.value, expires_at = EXCLUDED.expires_at",
                (kind, key, json.dumps(value), expires_at),
            )
            # Opportunistic purge: the table is tiny, expired rows are
            # pure noise, and doing it on the write path avoids a cron.
            conn.execute(
                "DELETE FROM mcp_oauth_state WHERE expires_at IS NOT NULL AND expires_at < %s",
                (time.time(),),
            )
            conn.commit()
        finally:
            conn.close()

    def _get(self, kind: str, key: str) -> Optional[dict]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT value, expires_at FROM mcp_oauth_state WHERE kind = %s AND key = %s",
                (kind, key),
            ).fetchone()
            if row is None:
                return None
            expires_at = row["expires_at"]
            if expires_at is not None and expires_at < time.time():
                conn.execute("DELETE FROM mcp_oauth_state WHERE kind = %s AND key = %s", (kind, key))
                conn.commit()
                return None
            value = row["value"]
            return value if isinstance(value, dict) else json.loads(value)
        finally:
            conn.close()

    def _delete(self, kind: str, key: str) -> bool:
        conn = self._connect()
        try:
            rows = conn.execute(
                "DELETE FROM mcp_oauth_state WHERE kind = %s AND key = %s RETURNING 1",
                (kind, key),
            ).fetchall()
            conn.commit()
            return bool(rows)
        finally:
            conn.close()

    # ── store interface ────────────────────────────────────────────

    def get_client(self, client_id: str) -> Optional[OAuthClientInformationFull]:
        value = self._get(_KIND_CLIENT, client_id)
        return OAuthClientInformationFull(**value) if value else None

    def put_client(self, info: OAuthClientInformationFull) -> None:
        self._put(_KIND_CLIENT, info.client_id, info.model_dump(mode="json"), None)

    def put_auth_code(self, code: AuthorizationCode) -> None:
        self._put(_KIND_CODE, _sha(code.code), _dump_without(code, "code"), code.expires_at)

    def get_auth_code(self, code: str) -> Optional[AuthorizationCode]:
        value = self._get(_KIND_CODE, _sha(code))
        return _load_model(AuthorizationCode, "code", code, value) if value else None

    def delete_auth_code(self, code: str) -> None:
        self._delete(_KIND_CODE, _sha(code))

    def consume_auth_code(self, code: str) -> bool:
        # Atomic single-use: DELETE ... RETURNING means exactly one
        # concurrent /token exchange can win a given code.
        return self._delete(_KIND_CODE, _sha(code))

    def put_token_pair(self, access: AccessToken, refresh: RefreshToken, resource: Optional[str]) -> None:
        access_sha, refresh_sha = _sha(access.token), _sha(refresh.token)
        self._put(
            _KIND_ACCESS,
            access_sha,
            _dump_without(access, "token", {"refresh_sha": refresh_sha}),
            float(access.expires_at) if access.expires_at is not None else None,
        )
        self._put(
            _KIND_REFRESH,
            refresh_sha,
            _dump_without(refresh, "token", {"access_sha": access_sha, "resource": resource}),
            float(refresh.expires_at) if refresh.expires_at is not None else None,
        )

    def get_access_token(self, token: str) -> Optional[AccessToken]:
        value = self._get(_KIND_ACCESS, _sha(token))
        return _load_model(AccessToken, "token", token, value) if value else None

    def get_refresh_token(self, token: str) -> Optional[RefreshToken]:
        value = self._get(_KIND_REFRESH, _sha(token))
        return _load_model(RefreshToken, "token", token, value) if value else None

    def get_refresh_resource(self, token: str) -> Optional[str]:
        value = self._get(_KIND_REFRESH, _sha(token))
        return value.get("resource") if value else None

    def revoke_access(self, token: str) -> None:
        self._revoke_by_hash(access_sha=_sha(token))

    def revoke_refresh(self, token: str) -> None:
        self._revoke_by_hash(refresh_sha=_sha(token))

    def consume_refresh(self, token: str) -> tuple[bool, Optional[str]]:
        """Atomically claim a refresh token for rotation. DELETE ...
        RETURNING means exactly one of N concurrent /token refreshes
        wins the row; losers get (False, None) → invalid_grant. The
        winner's old access token is revoked as part of the claim."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "DELETE FROM mcp_oauth_state WHERE kind = %s AND key = %s RETURNING value",
                (_KIND_REFRESH, _sha(token)),
            ).fetchall()
            conn.commit()
        finally:
            conn.close()
        if not rows:
            return (False, None)
        value = rows[0]["value"]
        if not isinstance(value, dict):
            value = json.loads(value)
        if value.get("access_sha"):
            self._delete(_KIND_ACCESS, value["access_sha"])
        return (True, value.get("resource"))

    def discard_access(self, token: str) -> None:
        """Drop ONLY the access row (expiry path); the paired refresh
        row survives so the client can rotate instead of re-authing."""
        self._delete(_KIND_ACCESS, _sha(token))

    def _revoke_by_hash(self, *, access_sha: Optional[str] = None, refresh_sha: Optional[str] = None) -> None:
        """Delete a token row and its pair. Pairing is stored as hashes
        inside each row, so cascade needs no raw secrets."""
        if access_sha:
            value = self._get(_KIND_ACCESS, access_sha)
            self._delete(_KIND_ACCESS, access_sha)
            if value and value.get("refresh_sha"):
                self._delete(_KIND_REFRESH, value["refresh_sha"])
        if refresh_sha:
            value = self._get(_KIND_REFRESH, refresh_sha)
            self._delete(_KIND_REFRESH, refresh_sha)
            if value and value.get("access_sha"):
                self._delete(_KIND_ACCESS, value["access_sha"])

    def consume_nonce(self, nonce: str, ttl_seconds: float) -> bool:
        """Atomic first-use check: INSERT ... ON CONFLICT DO NOTHING —
        exactly one caller ever sees True for a given nonce."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "INSERT INTO mcp_oauth_state (kind, key, value, expires_at) "
                "VALUES (%s, %s, %s::jsonb, %s) "
                "ON CONFLICT (kind, key) DO NOTHING RETURNING 1",
                (_KIND_NONCE, _sha(nonce), "{}", time.time() + ttl_seconds),
            ).fetchall()
            conn.commit()
            return bool(rows)
        finally:
            conn.close()


def is_persistence_enabled() -> bool:
    """Opt-in via GMAIL_MCP_OAUTH_PERSIST (mirrors GMAIL_MCP_OAUTH_ENABLED's
    style). Default OFF so tests and dev setups never write OAuth rows to
    whatever DB_DSN happens to point at."""
    import os

    return os.environ.get("GMAIL_MCP_OAUTH_PERSIST", "").strip().lower() in ("1", "true", "yes", "on")


def build_default_store() -> Any:
    """PG-backed when GMAIL_MCP_OAUTH_PERSIST is enabled, else memory
    (v1 behavior: restart forces re-auth).

    Fail-closed: when persistence is explicitly enabled, a DB failure
    raises and stops startup. Silently degrading to memory would defeat
    the persistence guarantee the operator asked for — same philosophy
    as the OAuth owner-uid startup gate in mcp_tools_server."""
    if not is_persistence_enabled():
        return MemoryOAuthStore()
    return PgOAuthStore()
