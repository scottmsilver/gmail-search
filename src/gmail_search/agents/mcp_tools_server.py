"""Streamable-HTTP MCP server exposing the existing retrieval +
analysis tools to MCP clients (claudebox / Claude Desktop / etc.)
without re-implementing them.

Why this exists: the same five tools the ADK agents use
(`search_emails`, `query_emails`, `get_thread`, `sql_query`,
`run_code`) need to be reachable from non-ADK MCP clients running
multiple turns concurrently against a single MCP endpoint. We wrap
the existing async functions + sandbox primitives in a FastMCP app
and add per-session state binding so concurrent calls don't bleed
into each other.

Per-session state model:
  - The orchestrator calls `register_session(session_id, evidence,
    db_dsn)` BEFORE issuing any LLM turn that may invoke a tool, and
    `unregister_session(session_id)` after. The registry keeps a lazy
    `db_conn` opened on first artifact-persist.
  - Every tool takes a `session_id` argument; the LLM is told (in the
    tool description) to pass the session_id from its system prompt
    on every call. Approach (b) from the spec — header plumbing
    avoided.
  - Unknown session_id => RuntimeError. That's a programming bug
    upstream, not user input — fail loud rather than silently make
    something up.

Run as: `python -m gmail_search.agents.mcp_tools_server`. Port via
`GMAIL_MCP_TOOLS_PORT` (default 7878). Endpoint at `/mcp`.
"""

from __future__ import annotations

import contextvars
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gmail_search.agents.session import save_artifact
from gmail_search.agents.tools import describe_schema as _describe_schema_impl
from gmail_search.agents.tools import get_attachment_batch as _get_attachment_batch_impl
from gmail_search.agents.tools import get_thread_batch as _get_thread_batch_impl
from gmail_search.agents.tools import query_emails_batch as _query_emails_batch_impl
from gmail_search.agents.tools import search_emails_batch as _search_emails_batch_impl
from gmail_search.agents.tools import sql_query_batch as _sql_query_batch_impl
from gmail_search.trace import (  # noqa: F401  (used in tool wrappers + _log_tool_call; keep against import-stripper)
    current_trace_id,
    new_trace_id,
    set_trace_id,
)

logger = logging.getLogger(__name__)


# Bind to 0.0.0.0 by default. The colocated claudebox container reaches
# us via `host.docker.internal` (mapped through `extra_hosts:
# host-gateway` in `deploy/claudebox/docker-compose.yml`). On Linux,
# `host-gateway` resolves to the docker bridge gateway IP (typically
# `172.17.0.1`), NOT loopback — so a 127.0.0.1 bind is unreachable from
# the container. 0.0.0.0 with a required bearer token is the safe path;
# `_assert_safe_bind_combination` refuses to start with auto-gen token
# + 0.0.0.0 unless `GMAIL_MCP_ALLOW_INSECURE=1` is explicit.
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 7878
DEFAULT_PATH = "/mcp"

# Per-session call-log cap. The side channel exists to bypass
# claudebox's tool_result truncation, but unbounded growth turns it
# into a memory leak / DoS vector if a caller loops a tool. Real turns
# are dozens of calls, not thousands — this ceiling is high enough not
# to trip on legitimate work but low enough to bound RAM.
_MAX_CALLS_PER_SESSION = 1000

# Path the auto-generated admin token is written to (mode 0600) so
# the orchestrator + tests can read it without us logging the secret.
# Resolved relative to CWD because the daemon's working directory is
# the project root in dev/test setups.
_AUTO_TOKEN_FILE = "scripts/mcp_admin_token"


# ── Session registry ───────────────────────────────────────────────


@dataclass
class SessionContext:
    """Per-turn state bound to a session_id. The DB connection is
    lazy: we don't pay the connection cost on sessions that never
    persist an artifact (search-only turns).

    `conversation_id`, when set, threads down into `run_code` so the
    sandbox uses a per-conversation persistent /work mount instead of
    a fresh tmpdir.

    `workspace`, when set, names the claudebox per-turn workspace
    (e.g. `deep-<session_id>`). `publish_artifact` resolves
    model-supplied paths under this workspace's host bind-mount."""

    evidence_records: list[dict] | dict | None
    db_dsn: str | None
    conversation_id: str | None = None
    workspace: str | None = None
    # `user_id`, when set, scopes every tool call this session makes
    # to that user's data. The runtime adapter passes the deep-mode
    # turn's user_id at register_session time; tools forward it as
    # `X-User-Id` to the FastAPI side, where `require_user_id` honors
    # it iff the request also carries the MCP admin token. None ==
    # legacy single-pool behavior (resolves to bootstrap user).
    user_id: str | None = None
    _db_conn: Any = field(default=None, repr=False)

    def get_db_conn(self):
        """Open a psycopg connection on first use. We don't open
        eagerly because most retrieval-only sessions never need it,
        and a no-op session shouldn't burn a PG slot."""
        if self._db_conn is None:
            self._db_conn = _open_db_conn(self.db_dsn)
        return self._db_conn

    def close(self) -> None:
        """Drop the lazy DB connection if we ever opened one. Caller
        invokes this from `unregister_session`. Safe to call when no
        connection was ever opened."""
        if self._db_conn is not None:
            try:
                self._db_conn.close()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"closing session db_conn failed: {e}")
            self._db_conn = None


_SESSIONS: dict[str, SessionContext] = {}

# Ephemeral, transport-scoped session contexts for /mcp callers. Keyed
# by `(transport_uid, session_id)` so a transport caller can NEVER
# collide with — or overwrite — a plain `session_id` registration in
# `_SESSIONS` (which the in-process orchestrator owns). This keeps the
# transport-override identity strictly request/owner-local: a token for
# owner A can't poison owner B's registered session. Entries are cheap
# (no db_conn opened until an artifact is persisted) and are wiped by
# `unregister_session` alongside the registered entry.
_TRANSPORT_SESSIONS: dict[tuple[str, str], SessionContext] = {}

# Streamable-HTTP session-riding guard: mcp-session-id → the first
# authenticated uid seen on it (see _TransportAuthMiddleware). Bounded;
# a reset just re-binds live sessions on their next request.
_TRANSPORT_SESSION_OWNERS: dict[str, str] = {}
_MAX_TRANSPORT_SESSION_OWNERS = 10_000


# ── Side-channel: per-session structured tool-call log ─────────────
#
# Why this exists: claudebox stringifies + truncates tool_result
# content to 2000 chars (`/tmp/docker-claudebox/jsonpipe.py:_assemble`).
# The orchestrator's downstream walkers (`_cite_refs_from_tool_calls`,
# `_artifact_ids_from_tool_calls`) need the FULL structured response.
# We already have it here — the MCP server produced it — so we record
# it keyed by session_id and let `claudebox_invoke()` fetch it via an
# admin HTTP endpoint, bypassing the truncation.

_SESSION_CALLS: dict[str, list[dict]] = {}

# Track which sessions have already had the cap-warning emitted so we
# don't spam the log on every dropped call after the cap is hit.
_SESSION_CAP_WARNED: set[str] = set()


def _transport_ids() -> tuple[str, str]:
    """Return (transport_session_id, jsonrpc_request_id) for the MCP
    request currently being served, or ('-', '-') outside one.

    Instrumentation for the 2026-07-03 cross-wiring reports: two claude.ai
    chats of the same user intermittently saw each other's tool responses.
    Our handler-level logs pair correctly, so the crossing is downstream —
    either the SDK's streamable-HTTP response routing or claude.ai's
    connector demux. Logging which TRANSPORT carried each call (plus the
    JSON-RPC id, to expose id collisions on a shared transport) is the
    evidence that tells those apart. Must never raise: it runs on the
    logging path of every tool call."""
    try:
        from mcp.server.lowlevel.server import request_ctx

        ctx = request_ctx.get()
    except Exception:  # noqa: BLE001 — LookupError outside a request; anything else, degrade
        return ("-", "-")
    transport = "-"
    try:
        headers = getattr(ctx.request, "headers", None)
        if headers is not None:
            transport = headers.get("mcp-session-id") or "-"
    except Exception:  # noqa: BLE001
        pass
    return (transport, str(ctx.request_id))


def _log_tool_call(session_id: str, name: str, args: dict, response: dict) -> None:
    """Emit one concise, on-disk log line per MCP tool call: tool name, a
    truncated arg summary (the caller's own queries — not secrets), and the
    outcome (ok / N-items-with-K-errors / ERROR). With the service's stdout
    redirected to data/mcp.log, this is the human-readable MCP call log."""
    import json as _json

    try:
        argstr = _json.dumps(args, default=str)
    except Exception:  # noqa: BLE001
        argstr = str(args)
    if len(argstr) > 400:
        argstr = argstr[:400] + "…"

    if isinstance(response, dict) and "error" in response:
        outcome = f"ERROR: {str(response['error'])[:200]}"
    elif isinstance(response, dict) and isinstance(response.get("results"), list):
        results = response["results"]
        errs = sum(
            1 for r in results if isinstance(r, dict) and isinstance(r.get("result"), dict) and "error" in r["result"]
        )
        outcome = f"ok ({len(results)} items" + (f", {errs} errors)" if errs else ")")
    else:
        outcome = "ok"

    # trace_id is added to the record by gmail_search.trace.TraceIdFilter (shows
    # as [trace_id] in text logs and a trace_id field in JSON). The extra fields
    # carry the FULL session_id (the visible line truncates to 12 chars) + the
    # structured event so JSON logs are queryable by tool/session/outcome.
    transport_sid, rpc_id = _transport_ids()
    logger.info(
        "MCP CALL tool=%s session=%s transport=%s rpc=%s trace=%s args=%s -> %s",
        name,
        (session_id or "")[:12],
        transport_sid,
        rpc_id,
        (current_trace_id() or "-")[:12],
        argstr,
        outcome,
        extra={
            "event": "mcp_tool_call",
            "tool": name,
            "session_id": session_id or "",
            "transport_session_id": transport_sid,
            "rpc_id": rpc_id,
            "outcome": outcome,
        },
    )


def _record_call(session_id: str, name: str, args: dict, response: dict) -> None:
    """Append one tool call to two stores:

    1. **In-memory side-channel** (kept for backwards compat with
       `/admin/calls/<session_id>` and the orchestrator's post-turn
       drain). Bounded by `_MAX_CALLS_PER_SESSION` so a runaway tool
       loop can't exhaust process memory.
    2. **`agent_events` table** — durable, survives orchestrator
       crashes between tool execution and post-turn drain (which the
       in-memory log alone can't, see codex review of the original
       per-conversation plan). This is the source of truth the debug
       pane reads from. Best-effort: a failed DB write logs and
       continues — the in-memory log is the fallback."""
    import time as _time

    _log_tool_call(session_id, name, args, response)

    bucket = _SESSION_CALLS.setdefault(session_id, [])
    over_cap = len(bucket) >= _MAX_CALLS_PER_SESSION
    if over_cap:
        if session_id not in _SESSION_CAP_WARNED:
            _SESSION_CAP_WARNED.add(session_id)
            logger.warning(
                "session %s reached _MAX_CALLS_PER_SESSION=%d; in-memory log "
                "is full but DB persistence still records every call",
                session_id,
                _MAX_CALLS_PER_SESSION,
            )
    else:
        entry = {"name": name, "args": dict(args), "response": response, "ts": _time.time()}
        bucket.append(entry)
    _persist_call_to_events(session_id, name, args, response)


def _persist_call_to_events(session_id: str, name: str, args: dict, response: dict) -> None:
    """Write a single tool call to `agent_events` as a `mcp_tool_call_full`
    event. Opens a short-lived psycopg connection per call so concurrent
    parallel-tool_use writes for the same session don't share connection
    state (httpx async runtime can interleave multiple `_record_call`s
    on the event loop). The seq race is handled by retrying the INSERT
    a few times — append_event computes seq from MAX+1 and the
    UNIQUE (session_id, seq) constraint catches collisions cleanly."""
    dsn = _resolve_server_db_dsn()
    if not dsn:
        return
    conn = None
    try:
        from gmail_search.agents.session import append_event

        conn = _open_db_conn(dsn)
        payload = {"name": name, "args": dict(args), "response": response}
        for attempt in range(5):
            try:
                append_event(
                    conn,
                    session_id=session_id,
                    agent_name="mcp",
                    kind="mcp_tool_call_full",
                    payload=payload,
                )
                return
            except Exception as exc:  # noqa: BLE001
                # UniqueViolation on (session_id, seq) means a concurrent
                # writer claimed our seq — recompute and retry. ForeignKey
                # violation means the agent_sessions row doesn't exist
                # yet (e.g. an admin-test session that didn't go through
                # create_session); skip silently.
                import psycopg

                if isinstance(exc, psycopg.errors.ForeignKeyViolation):
                    return
                if isinstance(exc, psycopg.errors.UniqueViolation) and attempt < 4:
                    conn.rollback()
                    continue
                raise
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "persist call to agent_events failed for session=%s name=%s: %s",
            session_id,
            name,
            exc,
        )
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass


def get_session_calls(session_id: str) -> list[dict]:
    """Return a copy of the recorded calls for a session. Empty list
    if the session never ran a tool (or was never registered)."""
    return list(_SESSION_CALLS.get(session_id, []))


def clear_session_calls(session_id: str) -> None:
    """Drop the call log for a session. Idempotent."""
    _SESSION_CALLS.pop(session_id, None)
    _SESSION_CAP_WARNED.discard(session_id)


def _open_db_conn(db_dsn: str | None):
    """Open a psycopg connection for artifact persistence. We import
    psycopg lazily so test paths that mock `save_artifact` don't pay
    the import cost."""
    if not db_dsn:
        raise RuntimeError("session has no db_dsn — cannot persist artifacts")
    import psycopg
    from psycopg.rows import dict_row

    return psycopg.connect(db_dsn, row_factory=dict_row, autocommit=False)


def _resolve_server_db_dsn() -> str | None:
    """Server-side DSN resolution. The admin endpoint never trusts a
    caller-supplied value (see `post_session`); instead we read the
    same env var the rest of the codebase uses (`DB_DSN`, mirroring
    `gmail_search.store.db._pg_dsn`). Returns None if unset so
    artifact-free sessions still work."""
    return os.environ.get("DB_DSN") or os.environ.get("GMAIL_DB_DSN") or None


# ── Transport-token identity (per-owner /mcp auth) ─────────────────
#
# The bhatti microVM presents `Authorization: Bearer <transport token>`
# to /mcp. The token is a per-owner, expiring JWT (HS256) minted by the
# admin endpoint below; the middleware verifies it and stashes the
# token's user_id here for the duration of the request. The tool
# wrappers prefer this identity over any caller-supplied session, so a
# VM agent can never widen its scope to another tenant.

_TRANSPORT_AUD = "mcp-transport"
# A service token authenticates a TRUSTED server-side MCP client
# (claudebox, the shared deep-analysis sandbox) to satisfy /mcp
# enforcement. Unlike a transport token it carries NO tenant — per-run
# scoping stays on the REGISTERED session (the orchestrator calls
# register_session(user_id=...) before each run). cc-web only mints
# transport tokens, so an untrusted VM can never obtain a service token.
_SERVICE_AUD = "mcp-service"
_TRANSPORT_ALGORITHM = "HS256"
_TRANSPORT_MIN_SECRET_BYTES = 32
_DEFAULT_TRANSPORT_TTL_SECONDS = 86400
# Service tokens are static server-side client credentials, so they get a
# long default lifetime (30 days) rather than the short transport TTL.
_DEFAULT_SERVICE_TTL_SECONDS = 86400 * 30

# Per-request transport identity. Unset (None) means no transport token
# was presented — behavior falls back to the legacy register_session
# path so the in-process deep-analysis orchestrator keeps working.
_transport_user_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("_transport_user_id", default=None)


def _transport_secret() -> str | None:
    """Resolve the signing secret for transport tokens, or None when
    transport auth is not configured.

    Prefers the dedicated `GMAIL_MCP_TRANSPORT_SECRET` (must be >=32
    bytes). Falls back to `GMS_SESSION_SECRET` ONLY when the operator
    explicitly opts in via `GMAIL_MCP_TRANSPORT_ALLOW_SHARED_SECRET=1`
    — sharing the cookie-signing key with the transport audience is a
    blast-radius expansion, so it's never silent. Returns None (rather
    than raising) so the mint endpoint can 503 and the middleware can
    treat the server as "no auth configured" without crashing."""
    secret = os.environ.get("GMAIL_MCP_TRANSPORT_SECRET")
    if not secret and os.environ.get("GMAIL_MCP_TRANSPORT_ALLOW_SHARED_SECRET") == "1":
        secret = os.environ.get("GMS_SESSION_SECRET")
    if not secret:
        return None
    if len(secret.encode("utf-8")) < _TRANSPORT_MIN_SECRET_BYTES:
        logger.warning(
            "transport secret is under %d bytes; refusing to use it " "(HS256 forgery is practical below that)",
            _TRANSPORT_MIN_SECRET_BYTES,
        )
        return None
    return secret


def _transport_ttl_seconds() -> int:
    raw = os.environ.get("GMAIL_MCP_TRANSPORT_TTL_SECONDS")
    if not raw:
        return _DEFAULT_TRANSPORT_TTL_SECONDS
    try:
        return int(raw)
    except ValueError:
        logger.warning("invalid GMAIL_MCP_TRANSPORT_TTL_SECONDS=%r; using default", raw)
        return _DEFAULT_TRANSPORT_TTL_SECONDS


def mint_transport_token(*, user_id: str, email: str, ttl_seconds: int | None = None) -> tuple[str, int]:
    """Mint a signed per-owner transport token. Returns `(token, exp)`.

    The `uid` claim is resolved SERVER-SIDE by the caller (the mint
    endpoint) from the email — never trusted from request input. Raises
    RuntimeError if the signing secret is unavailable so the endpoint
    can map that to a 503."""
    import jwt

    secret = _transport_secret()
    if secret is None:
        raise RuntimeError("transport signing secret unavailable")
    now = int(time.time())
    exp = now + (ttl_seconds if ttl_seconds is not None else _transport_ttl_seconds())
    payload = {
        "uid": user_id,
        "email": email,
        "aud": _TRANSPORT_AUD,
        "iat": now,
        "exp": exp,
    }
    token = jwt.encode(payload, secret, algorithm=_TRANSPORT_ALGORITHM)
    return token, exp


def _service_ttl_seconds() -> int:
    raw = os.environ.get("GMAIL_MCP_SERVICE_TTL_SECONDS")
    if not raw:
        return _DEFAULT_SERVICE_TTL_SECONDS
    try:
        return int(raw)
    except ValueError:
        logger.warning("invalid GMAIL_MCP_SERVICE_TTL_SECONDS=%r; using default", raw)
        return _DEFAULT_SERVICE_TTL_SECONDS


def mint_service_token(ttl_seconds: int | None = None) -> str:
    """Mint a signed SERVICE token for a trusted server-side MCP client
    (claudebox). Signed with the SAME transport secret, audience
    `mcp-service`, with `iat` + `exp` but NO tenant claims (uid/email) —
    scoping for claudebox runs stays on the registered session.

    Raises RuntimeError if the signing secret is unavailable so the
    admin endpoint can map that to a 503."""
    import jwt

    secret = _transport_secret()
    if secret is None:
        raise RuntimeError("transport signing secret unavailable")
    now = int(time.time())
    exp = now + (ttl_seconds if ttl_seconds is not None else _service_ttl_seconds())
    payload = {
        "aud": _SERVICE_AUD,
        "iat": now,
        "exp": exp,
    }
    return jwt.encode(payload, secret, algorithm=_TRANSPORT_ALGORITHM)


def _resolve_user_id_by_email(email: str) -> str | None:
    """Map an owner email to its internal user_id via the users table.
    Opens a short-lived connection (the DSN is server-side config, never
    caller input). Returns None if no such user exists. Kept thin so
    tests can monkeypatch it without a real DB."""
    from gmail_search.auth.write_user import get_user_id_by_email
    from gmail_search.store.db import get_connection

    conn = get_connection(None)
    try:
        return get_user_id_by_email(conn, email)
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass


def verify_token(token: str) -> dict | None:
    """Verify a token's signature + audience + expiry against the
    transport secret, accepting BOTH the `mcp-transport` (tenant-bound,
    untrusted VM) and `mcp-service` (tenantless, trusted server-side
    client) audiences. Returns the decoded claims (including `aud`) on
    success, None on any failure (bad sig, unknown/missing aud, expired,
    missing exp, or no secret configured) so callers can decide
    401-vs-passthrough without leaking which check failed.

    PyJWT accepts a list for `audience` and verifies the token's `aud`
    is one of them — a 3rd/unknown audience is rejected."""
    import jwt

    secret = _transport_secret()
    if secret is None:
        return None
    try:
        return jwt.decode(
            token,
            secret,
            algorithms=[_TRANSPORT_ALGORITHM],
            audience=[_TRANSPORT_AUD, _SERVICE_AUD],
            # Require exp + aud to be present: a signed token lacking
            # either is rejected rather than treated as non-expiring /
            # unaudienced. Defence-in-depth matching the spec's "enforce
            # exp" — PyJWT only validates exp/aud when the claim exists.
            options={"require": ["exp", "aud"]},
        )
    except jwt.PyJWTError as exc:
        logger.debug("token verification failed: %s", exc)
        return None


def verify_transport_token(token: str) -> dict | None:
    """Verify a token and require the `mcp-transport` audience. Returns
    the claims only for a tenant-bound transport token; a service token
    (or any other aud) yields None. Kept for callers/tests that want the
    transport audience specifically."""
    claims = verify_token(token)
    if claims is None or claims.get("aud") != _TRANSPORT_AUD:
        return None
    return claims


def register_session(
    session_id: str,
    *,
    evidence_records: list[dict] | dict | None,
    db_dsn: str | None,
    conversation_id: str | None = None,
    workspace: str | None = None,
    user_id: str | None = None,
) -> None:
    """Bind a session_id to the evidence + DSN this turn's tool calls
    will see. MUST be called before the LLM is allowed to invoke any
    tool with this session_id. Re-registering an existing id replaces
    the prior context (and closes its db_conn) — handy for reusing
    ids across turns in tests.

    `conversation_id`, when set, opts the session into per-conversation
    persistent sandbox scratch (/work survives across run_code calls
    AND across turns).

    `workspace`, when set, names the claudebox per-turn workspace dir
    (e.g. `deep-<session_id>`). `publish_artifact` uses it to resolve
    Claude-supplied file paths.

    `user_id`, when set, scopes every tool call this session makes to
    that user's data. The orchestrator passes the deep-mode turn's
    user from the request context."""
    _assert_session_id(session_id)
    if session_id in _SESSIONS:
        _SESSIONS[session_id].close()
    _SESSIONS[session_id] = SessionContext(
        evidence_records=evidence_records,
        db_dsn=db_dsn,
        conversation_id=conversation_id,
        workspace=workspace,
        user_id=user_id,
    )


def unregister_session(session_id: str) -> None:
    """Drop a session from the registry and close any lazy db_conn.
    Idempotent — unknown ids are a no-op so cleanup paths never
    raise on double-shutdown. Also wipes the side-channel call log
    for the session — the runtime adapter must have fetched it
    before calling unregister."""
    ctx = _SESSIONS.pop(session_id, None)
    if ctx is not None:
        ctx.close()
    # Also drop any transport-scoped ephemeral contexts for this
    # session_id (across all transport uids), closing their lazy
    # db_conns so we don't leak PG slots.
    for key in [k for k in _TRANSPORT_SESSIONS if k[1] == session_id]:
        _TRANSPORT_SESSIONS.pop(key).close()
    clear_session_calls(session_id)


def _assert_session_id(session_id: str) -> None:
    """A non-empty session_id is non-negotiable. Empty / None means
    the LLM didn't read its system prompt, or the orchestrator
    forgot to call `register_session` — either way a programming
    bug, not user input."""
    if not session_id or not isinstance(session_id, str):
        raise ValueError("session_id must be a non-empty string")


def _get_session(session_id: str) -> SessionContext:
    """Lookup with a clear error message. We don't auto-create
    sessions on the fly — that would let a leaked LLM call get a
    blank context and silently succeed against the wrong evidence."""
    _assert_session_id(session_id)
    ctx = _SESSIONS.get(session_id)
    if ctx is None:
        raise RuntimeError(
            f"session_id {session_id!r} not registered — caller must " f"call register_session() before invoking tools"
        )
    return ctx


def _resolve_ctx(session_id: str) -> SessionContext:
    """Resolve the SessionContext a tool call should run against,
    preferring the transport identity when one is present.

    Two regimes:

    1. **Transport identity set** (a verified per-owner /mcp token):
       the token's `uid` is the effective user_id, FULL STOP. We return
       a TRANSPORT-LOCAL SessionContext scoped to that uid — keyed by
       `(uid, session_id)` in `_TRANSPORT_SESSIONS`, NEVER by mutating or
       reading the user_id off a `_SESSIONS[session_id]` entry. This is
       the security-critical part: a token for owner A must not be able
       to repin (poison) a session that the in-process orchestrator
       registered for owner B — that would bleed A's identity into B's
       subsequent legacy calls via shared mutable state. The transport
       context is cached per `(uid, session_id)` only so repeat calls in
       the same request reuse one (lazy) db_conn; it can't collide with
       a plain `session_id` registration.

    2. **Transport identity unset** (legacy / orchestrator path):
       behavior is exactly as before — `_get_session` requires a
       prior register_session and uses its user_id verbatim. The
       registered `_SESSIONS` entry is never touched by the transport
       path, so its user_id is whatever register_session stored."""
    transport_uid = _transport_user_id.get()
    if not transport_uid:
        # No transport identity (None == in-process orchestrator path;
        # empty is never set by the middleware but is treated the same).
        return _get_session(session_id)

    _assert_session_id(session_id)
    key = (transport_uid, session_id)
    ctx = _TRANSPORT_SESSIONS.get(key)
    if ctx is None:
        ctx = SessionContext(
            evidence_records=None,
            db_dsn=_resolve_server_db_dsn(),
            user_id=transport_uid,
        )
        _TRANSPORT_SESSIONS[key] = ctx
    return ctx


# ── Tool implementations (re-routes) ───────────────────────────────


async def _tool_search_emails_batch(session_id: str, searches: list[dict]) -> dict:
    """Re-route to `search_emails_batch`. Threads `user_id` from the
    SessionContext into the impl so the resulting HTTP call to
    /api/* sets the X-User-Id header — FastAPI's `require_user_id`
    honors it iff the request also carries the MCP admin token,
    keeping the per-user gate intact even from the MCP path."""
    set_trace_id(new_trace_id())  # fresh trace id per tool call; propagates to /api/* + logs
    ctx = _resolve_ctx(session_id)
    args = {"searches": searches}
    response = await _search_emails_batch_impl(searches, user_id=ctx.user_id)
    _record_call(session_id, "search_emails_batch", args, response)
    return response


async def _tool_query_emails_batch(session_id: str, filters: list[dict]) -> dict:
    set_trace_id(new_trace_id())  # fresh trace id per tool call; propagates to /api/* + logs
    ctx = _resolve_ctx(session_id)
    args = {"filters": filters}
    response = await _query_emails_batch_impl(filters, user_id=ctx.user_id)
    _record_call(session_id, "query_emails_batch", args, response)
    return response


async def _tool_get_thread_batch(session_id: str, thread_ids: list[str]) -> dict:
    set_trace_id(new_trace_id())  # fresh trace id per tool call; propagates to /api/* + logs
    ctx = _resolve_ctx(session_id)
    args = {"thread_ids": thread_ids}
    response = await _get_thread_batch_impl(thread_ids, user_id=ctx.user_id)
    _record_call(session_id, "get_thread_batch", args, response)
    return response


async def _tool_sql_query_batch(session_id: str, queries: list[str]) -> dict:
    set_trace_id(new_trace_id())  # fresh trace id per tool call; propagates to /api/* + logs
    ctx = _resolve_ctx(session_id)
    args = {"queries": queries}
    response = await _sql_query_batch_impl(queries, user_id=ctx.user_id)
    _record_call(session_id, "sql_query_batch", args, response)
    return response


async def _tool_find_facts(session_id: str, query: str, exhaustive: bool = True, k: int = 200) -> dict:
    """Re-route to `find_facts`. Threads `user_id` from the
    SessionContext so the /api/find_facts call is scoped to the
    session's user via the X-User-Id + admin-token pair."""
    from gmail_search.agents.tools import find_facts as _find_facts_impl

    set_trace_id(new_trace_id())  # fresh trace id per tool call; propagates to /api/* + logs
    ctx = _resolve_ctx(session_id)
    args = {"query": query, "exhaustive": exhaustive, "k": k}
    response = await _find_facts_impl(query, exhaustive=exhaustive, k=k, user_id=ctx.user_id)
    _record_call(session_id, "find_facts", args, response)
    return response


async def _tool_describe_schema(session_id: str) -> dict:  # noqa: D401
    """Re-route to `describe_schema`. Threads `user_id` so the schema
    response is prepended with a scoping preamble pinning the active
    user — the LLM uses that to write correct WHERE user_id = ... clauses.
    """
    set_trace_id(new_trace_id())  # fresh trace id per tool call; propagates to /api/* + logs
    ctx = _resolve_ctx(session_id)
    response = await _describe_schema_impl(user_id=ctx.user_id)
    _record_call(session_id, "describe_schema", {}, response)
    return response


def _rewrite_blob_urls(response: dict) -> None:
    """Turn serve's per-attachment `blob_token` into an absolute, model-fetchable
    `fetch_url` on this MCP host's public /attachment route. Mutates in place."""
    base = os.environ.get("GMAIL_MCP_PUBLIC_URL", "").rstrip("/")
    if not base:
        return
    for item in response.get("results", []) or []:
        r = item.get("result")
        if isinstance(r, dict) and r.get("blob_token"):
            r["fetch_url"] = f"{base}/attachment?t={r.pop('blob_token')}"
            r["fetch_url_note"] = "Signed link, expires in ~15 min — fetch directly (no auth, no session needed)."


async def _tool_get_attachment_batch(session_id: str, items: list[dict]) -> dict:
    set_trace_id(new_trace_id())  # fresh trace id per tool call; propagates to /api/* + logs
    ctx = _resolve_ctx(session_id)
    args = {"items": items}
    response = await _get_attachment_batch_impl(items, user_id=ctx.user_id)
    _rewrite_blob_urls(response)
    _record_call(session_id, "get_attachment_batch", args, response)
    return response


# Cap matches the agent_artifacts.data BYTEA column's 10MB limit.
MAX_PUBLISH_BYTES = 10 * 1024 * 1024

# Roots Claude can publish from. Both are bound from the host into the
# claudebox container; we read them on the host filesystem.
_PUBLISH_WORKSPACE_ROOT = "deploy/claudebox/workspaces"
_PUBLISH_SCRATCH_ROOT = "data/agent_scratch"


def _strip_known_path_prefix(path: str, *, workspace: str | None) -> str:
    """Strip the in-container absolute prefix the model might use so
    the rest of the resolver can treat the path as relative.

    `/workspaces/<workspace>/foo.png` -> `foo.png`
    `/work/foo.png`                   -> `foo.png`
    `foo.png`                         -> `foo.png`
    """
    if path.startswith("/workspaces/") and workspace:
        prefix = f"/workspaces/{workspace}/"
        if path.startswith(prefix):
            return path[len(prefix) :]
    if path.startswith("/work/"):
        return path[len("/work/") :]
    return path.lstrip("/")


def _resolve_publish_source(
    path: str,
    *,
    workspace: str | None,
    conversation_id: str | None,
) -> Path:
    """Resolve a model-supplied path to a host filesystem Path under
    one of the known mount roots. Rejects path traversal."""
    rel = _strip_known_path_prefix(path, workspace=workspace)
    if not rel:
        raise FileNotFoundError(f"empty path after prefix-strip: {path!r}")

    candidates: list[Path] = []
    if workspace:
        candidates.append(Path(_PUBLISH_WORKSPACE_ROOT) / workspace)
    if conversation_id:
        candidates.append(Path(_PUBLISH_SCRATCH_ROOT) / conversation_id)
    if not candidates:
        raise RuntimeError("session has neither workspace nor conversation_id; cannot publish")

    for root in candidates:
        try:
            root_resolved = root.resolve()
            file_path = (root / rel).resolve()
            file_path.relative_to(root_resolved)
        except (ValueError, OSError):
            continue
        if file_path.is_file():
            return file_path
    raise FileNotFoundError(f"no such file under publishable roots: {path!r}")


def _sniff_mime(path: Path) -> str:
    import mimetypes

    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


def _publish_one(
    *,
    ctx: SessionContext,
    session_id: str,
    path: str,
    name: str,
    mime_type: str,
) -> dict:
    """Publish a single file as a user-visible artifact. Synchronous
    because the underlying file read + save_artifact are sync; the
    batch wrapper sequences these on the event loop thread (no
    benefit to gather() — the bottleneck is the DB write, not I/O)."""
    try:
        file_path = _resolve_publish_source(
            path,
            workspace=ctx.workspace,
            conversation_id=ctx.conversation_id,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        return {"error": str(exc)}
    size = file_path.stat().st_size
    if size > MAX_PUBLISH_BYTES:
        return {
            "error": f"file too large: {size} bytes; max {MAX_PUBLISH_BYTES}. Save a smaller version (downsample, lower DPI, or summarize)."
        }
    data = file_path.read_bytes()
    final_name = name or file_path.name
    final_mime = mime_type or _sniff_mime(file_path)
    art_id = save_artifact(
        ctx.get_db_conn(),
        session_id=session_id,
        name=final_name,
        mime_type=final_mime,
        data=data,
    )
    return {"id": art_id, "name": final_name, "mime_type": final_mime, "size": size}


async def _tool_publish_artifact_batch(session_id: str, items: list[dict]) -> dict:
    """Publish many files as user-visible artifacts in ONE call.
    Each item is `{path, name?, mime_type?}`. Per-item errors land
    in that entry's `result` as `{error: ...}`; the batch as a
    whole still succeeds so the agent can publish every other file
    even if one is missing."""
    set_trace_id(new_trace_id())  # fresh trace id per tool call; propagates to /api/* + logs
    ctx = _resolve_ctx(session_id)
    args = {"items": items}
    if not isinstance(items, list) or not items:
        response = {"error": "items must be a non-empty list of dicts"}
        _record_call(session_id, "publish_artifact_batch", args, response)
        return response
    from gmail_search.agents.tools import BATCH_MAX_ITEMS as _CAP

    if len(items) > _CAP:
        response = {"error": f"items cap is {_CAP}; got {len(items)}. Split into multiple batches."}
        _record_call(session_id, "publish_artifact_batch", args, response)
        return response
    results = []
    for it in items:
        result = _publish_one(
            ctx=ctx,
            session_id=session_id,
            path=it.get("path", ""),
            name=it.get("name", ""),
            mime_type=it.get("mime_type", ""),
        )
        results.append({"input": it, "result": result})
    response = {"results": results}
    _record_call(session_id, "publish_artifact_batch", args, response)
    return response


# ── FastMCP app construction ───────────────────────────────────────


# Tool descriptions injected into the FastMCP `tool()` decorator.
# Kept as module-level constants so the LLM-facing prose is easy to
# audit / edit without diving into the function bodies.
SESSION_PARAM_NOTE = (
    "ALWAYS pass the `session_id` from your system prompt as the "
    "first argument. The server uses it to bind the call to the "
    "right evidence + database context."
)

SEARCH_BATCH_DESC = (
    "Run many semantic searches concurrently in ONE call. `searches` "
    "is a list (1-100) of `{query, date_from?, date_to?, top_k?, "
    "detail?, max_matches?}` dicts; `top_k` defaults to 10. Use for "
    'relevance lookups ("what did we decide about X", "find messages '
    'mentioning Y"). Returns `{results: [{input, result}, ...]}` '
    "aligned with input. Per-search errors land in that entry's "
    "`result` as `{error: ...}`.\n\n"
    "`detail` controls per-match payload — pick the cheapest that "
    "answers the question, since larger levels cost far more tokens:\n"
    '  - "refs": ONE LINE per thread ({thread_id, subject, date_last, '
    "from, score}), no matches array — use for fan-out inventory "
    "questions across many queries where you only need to know which "
    "threads exist.\n"
    '  - "snippet" (default): matched-text snippet only — best for '
    "inventories / counting / locating threads.\n"
    '  - "summary": + a one-line LLM summary per matched message.\n'
    '  - "full": + the WHOLE email body per match — use when you '
    "must read the matches; avoids N get_thread calls but is large, "
    "so keep top_k modest.\n\n"
    "`max_matches` caps matching messages returned per thread "
    "(top-scoring first; default 3, 0 = unlimited). A bitten cap is "
    "reported per thread as `matches_truncated`.\n\n"
    "ALWAYS use this for multi-angle investigations — fan out across "
    "phrasings/date-windows in ONE call. Even a single search goes "
    f"through this tool (pass a one-item list). {SESSION_PARAM_NOTE}"
)

QUERY_BATCH_DESC = (
    "Run many structured-metadata filters concurrently in ONE call. "
    "`filters` is a list (1-100) of dicts; each accepts any of "
    "`{sender, subject_contains, date_from, date_to, label, "
    "has_attachment, order_by, limit}`. Use when you know WHICH "
    "fields to filter on (no relevance ranking). Returns "
    f"`{{results: [{{input, result}}, ...]}}` aligned with input. {SESSION_PARAM_NOTE}"
)

THREAD_BATCH_DESC = (
    "Fetch many threads concurrently in ONE call. `thread_ids` is a "
    "list (1-100). Use this whenever you need ≥2 threads — never "
    "split into multiple sequential single-thread calls. Each "
    "thread's payload is the full message bodies + attachment "
    "manifest (bodies clipped to 20k chars). Returns "
    "`{results: [{thread_id, result}, ...]}` aligned with input. "
    "Per-thread errors land in that entry's `result` as "
    f"`{{error: ...}}`; the batch as a whole still succeeds. {SESSION_PARAM_NOTE}"
)

SQL_QUERY_BATCH_DESC = (
    "Run many read-only SELECTs concurrently in ONE call against the "
    "messages DB (Postgres + ParadeDB). `queries` is a list (1-100). "
    "Each query is independently gated: SELECT/WITH only, no DDL/DML, "
    "500-row + 10s per-query timeout. Returns "
    "`{results: [{query, result}, ...]}` aligned with input; per-query "
    "errors land in that entry's `result` as `{error: ...}` — the "
    "batch as a whole still succeeds.\n\n"
    "Even a single query goes through this tool — pass a one-item "
    "list. Use multi-item batches for multi-angle investigations: "
    "different senders, keywords, time windows, year-buckets, ticket "
    "numbers. Same wall clock for 1 query or 20.\n\n"
    "Call `describe_schema` first if unsure about column names. "
    "Common gotchas: `from_addr` (not `sender`), `body_text` (not "
    "`body`), `id` (not `message_id`), no `snippet` column.\n\n"
    "REQUIRED — BM25 for free-text. The server REJECTS `LIKE`/`ILIKE` "
    "on these columns (forces seq scan, ~50x slower):\n"
    "  messages: subject, body_text, from_addr, to_addr\n"
    "  attachments: filename, extracted_text\n"
    "Use the `@@@` operator with the row PK (`id`) instead. Tantivy "
    "syntax: `field:term`, combinable with AND / OR / NOT and parens. "
    "Translation table:\n"
    "  WHERE subject ILIKE '%credit%'\n"
    "    →  WHERE id @@@ 'subject:credit'\n"
    "  WHERE from_addr LIKE '%delta%' AND subject LIKE '%cancel%'\n"
    "    →  WHERE id @@@ 'from_addr:delta AND subject:cancel'\n"
    "  WHERE body_text LIKE '%refund issued%'  (phrase)\n"
    "    →  WHERE id @@@ 'body_text:\"refund issued\"'\n"
    "Add `ORDER BY paradedb.score(id) DESC` for relevance. Escape "
    "hatch: any query containing `@@@` skips the LIKE check, so a "
    "BM25 prefilter + LIKE refinement is allowed (use only when BM25 "
    "truly cannot express the predicate). Cells > 8000 chars are "
    f"clipped. {SESSION_PARAM_NOTE}"
)

FIND_FACTS_DESC = (
    "ENUMERATE every instance of an entity/attribute across the whole "
    "mailbox in ONE call — instead of issuing many `search_emails_batch` "
    "reformulations and hoping you covered them all. Use for exhaustive "
    '"list ALL my X" questions: all my license plates, all my account '
    "numbers, every flight booked, every address lived at. It runs "
    "hybrid (semantic ∪ keyword) retrieval over pre-extracted atomic "
    "facts ('propositions') mined from the mailbox, so bare identifiers "
    "(plates, VINs, codes) that embeddings rank near-zero are still "
    "recalled via the BM25 half. `query` is the entity/attribute to "
    "enumerate; `exhaustive` (default true) returns the full candidate "
    "set; `k` (default 200) caps the count. Returns "
    "`{facts: [{fact, message_id, thread_id, cosine, bm25}, ...]}`. Each "
    "fact carries a `message_id` back-pointer — use `get_thread_batch` "
    "with it to cite/verify the source before reporting. If propositions "
    f"haven't been backfilled yet, `facts` is `[]`. {SESSION_PARAM_NOTE}"
)

DESCRIBE_SCHEMA_DESC = (
    "Return markdown documentation for every table the `sql_query` "
    "tool can read. Call this before writing a non-trivial sql_query "
    "if you're unsure about column names. Cheap, no parameters beyond "
    f"session_id. {SESSION_PARAM_NOTE}"
)

ATTACHMENT_BATCH_DESC = (
    "Fetch many email attachments concurrently in ONE call. `items` "
    "is a list (1-100) of `{attachment_id, mode?, inline?}` dicts. `mode` is "
    "one of: 'meta' (filename/mime/size only — cheap), 'text' "
    "(extracted text from PDFs/docx/OCR — the usual choice, default), "
    "'rendered_pages' (PDF pages as base64 PNGs — heavy, only when "
    "text is empty/unhelpful), 'raw' (ORIGINAL BYTES: base64 inline for "
    "files <=1MB by default, PLUS a signed fetch_url (expires ~15 min) "
    "you can GET directly with no auth for any size; inline:false for "
    "url-only). `attachment_id` comes from a thread's "
    "`attachments[].id`. Returns `{results: [{input, result}, ...]}` "
    f"aligned with input. Even a single attachment goes through this tool. {SESSION_PARAM_NOTE}"
)

PUBLISH_ARTIFACT_BATCH_DESC = (
    "Register many files as part of the answer the user sees, in ONE "
    "call. `items` is a list (1-100) of `{path, name?, mime_type?}` "
    "dicts. Anything you produce that should appear in the user's "
    "answer must be published first — files you write to disk are "
    "invisible by default. `path` may be relative ('plot.png') or "
    "absolute ('/workspaces/<your-workspace>/plot.png' or "
    "'/work/plot.png'). Optional: `name` for the citation chip, "
    "`mime_type` if auto-detect is wrong. Returns "
    "`{results: [{input, result: {id, name, mime_type, size} | {error}}, ...]}`; "
    "cite each `id` as `[art:<id>]` in your answer. Files >10MB are "
    f"rejected per-item. {SESSION_PARAM_NOTE}"
)


def _resolve_admin_token() -> tuple[str, bool]:
    """The admin token gates the side-channel + session-management
    endpoints. Caller can pin it via `GMAIL_MCP_ADMIN_TOKEN` so the
    runtime adapter can be configured to match; otherwise we
    auto-generate one and stash it in the env for the rest of the
    process. Returns `(token, was_auto_generated)` so the entrypoint
    can write the auto-gen value to a 0600 file (NEVER log it)."""
    import secrets as _secrets

    token = os.environ.get("GMAIL_MCP_ADMIN_TOKEN")
    if token:
        return token, False
    generated = _secrets.token_urlsafe(32)
    os.environ["GMAIL_MCP_ADMIN_TOKEN"] = generated
    return generated, True


def _write_auto_token_file(token: str) -> str | None:
    """Persist an auto-generated admin token to a 0600 file so the
    orchestrator/tests can pick it up without us logging the secret.
    Returns the file path on success, None on failure (we keep going —
    the caller still has it in `GMAIL_MCP_ADMIN_TOKEN`)."""
    try:
        path = os.path.abspath(_AUTO_TOKEN_FILE)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # The mode arg to os.open() is only applied at creation time; if
        # the file already exists with broader perms (e.g. 0644 from a
        # prior run), the rewrite leaves them. Always re-assert 0600 via
        # fchmod after open. O_NOFOLLOW prevents symlink-tricks.
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW
        fd = os.open(path, flags, 0o600)
        try:
            os.fchmod(fd, 0o600)
            os.write(fd, token.encode("utf-8"))
        finally:
            os.close(fd)
        return path
    except OSError as exc:
        logger.warning("could not persist auto-generated admin token: %s", exc)
        return None


def _assert_safe_bind_combination(host: str, *, token_was_generated: bool) -> None:
    """Refuse to start with the dangerous combination: public bind
    (`0.0.0.0`) AND an auto-generated token (no operator-supplied
    `GMAIL_MCP_ADMIN_TOKEN`). That combo means we'd be exposing
    privileged endpoints to anything that can route to the host with
    a token only the local process knows — fine if the operator chose
    it, but never silently. `GMAIL_MCP_ALLOW_INSECURE=1` is the
    documented opt-out for niche dev setups."""
    if host != "0.0.0.0":
        return
    if not token_was_generated:
        return
    if os.environ.get("GMAIL_MCP_ALLOW_INSECURE") == "1":
        logger.warning(
            "GMAIL_MCP_ALLOW_INSECURE=1: starting with public bind 0.0.0.0 "
            "and an auto-generated admin token — caller accepted the risk."
        )
        return
    raise RuntimeError(
        "refusing to start MCP server: bind host is 0.0.0.0 and no "
        "GMAIL_MCP_ADMIN_TOKEN was supplied (auto-generated tokens "
        "must not be paired with a public bind). Either set "
        "GMAIL_MCP_ADMIN_TOKEN explicitly, switch the bind host to "
        "127.0.0.1 (default), or set GMAIL_MCP_ALLOW_INSECURE=1."
    )


_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "::ffff:127.0.0.1"})


def _is_loopback_client(request) -> bool:
    """True iff the request's client address is loopback. All LEGIT admin
    callers are host-local (the in-process orchestrator and the cc-web
    backend both call localhost:7878). When `request.client` is None we
    can't prove loopback — treat as non-loopback (reject)."""
    client = getattr(request, "client", None)
    if client is None:
        return False
    host = getattr(client, "host", None)
    return host in _LOOPBACK_HOSTS


def _admin_guard(request):
    """Gate every /admin/* route. Two layers, checked client-FIRST so a
    remote caller never even learns whether its token is valid:

    1. The client address must be loopback (127.0.0.1 / ::1) unless the
       operator sets `GMAIL_MCP_ADMIN_ALLOW_REMOTE=1`. Once the firewall
       opens :7878 to the bhatti VM subnet, VMs must not be able to reach
       /admin/* — only host-local callers may. A non-loopback caller is
       rejected with 403 BEFORE any token check, so it never learns
       whether its (possibly stolen) admin token is valid.
    2. The constant-time admin bearer-token check → 401 on failure.

    Returns a JSONResponse to short-circuit the route on denial, or None
    when access is granted."""
    from starlette.responses import JSONResponse

    if os.environ.get("GMAIL_MCP_ADMIN_ALLOW_REMOTE") != "1" and not _is_loopback_client(request):
        return JSONResponse({"error": "forbidden"}, status_code=403)
    if not _check_admin_token(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return None


def _check_admin_token(request) -> bool:
    """Constant-time bearer-token check. Missing / wrong header
    yields 401. Does NOT enforce the loopback gate — use
    `_check_admin_access` on routes; this remains the token-only check."""
    import secrets as _secrets

    expected = os.environ.get("GMAIL_MCP_ADMIN_TOKEN") or ""
    if not expected:
        return False
    header = request.headers.get("authorization", "")
    if not header.startswith("Bearer "):
        return False
    return _secrets.compare_digest(header[len("Bearer ") :], expected)


def _register_blob_route(app) -> None:
    """Public, signature-gated attachment download. The MCP host is the only
    publicly-reachable surface, so a model-fetchable attachment URL must live
    here. Auth is the JWT capability token minted by serve's /api/attachment/{id}/raw
    (claims: aid, uid, exp ~15 min, signed with the shared GMS_SESSION_SECRET) — NOT
    a session. We verify the signature + expiry, then proxy to serve scoped to the
    token's uid (so it can't cross tenants), streaming the bytes back."""
    import httpx
    import jwt
    from starlette.requests import Request
    from starlette.responses import JSONResponse, StreamingResponse

    _BLOB_MAX_BYTES = 30 * 1024 * 1024  # bound the proxy: reject oversized, stream the rest

    @app.custom_route("/attachment", methods=["GET"])
    async def serve_attachment_blob(request: Request):
        token = request.query_params.get("t")
        if not token:
            return JSONResponse({"error": "missing token"}, status_code=400)
        secret = os.environ.get("GMS_SESSION_SECRET")
        if not secret or len(secret) < 32:
            return JSONResponse({"error": "attachment links not configured"}, status_code=503)
        try:
            claims = jwt.decode(token, secret, algorithms=["HS256"], options={"require": ["exp"]})
        except Exception:
            return JSONResponse({"error": "invalid or expired link"}, status_code=403)
        aid, uid = claims.get("aid"), claims.get("uid")
        if aid is None or not uid:
            return JSONResponse({"error": "malformed token"}, status_code=403)
        # Proxy to serve's own attachment endpoint, scoped to the token's uid.
        base = os.environ.get("GMAIL_SEARCH_API_URL", "http://127.0.0.1:8090").rstrip("/")
        headers = {"X-User-Id": str(uid)}
        admin = os.environ.get("GMAIL_MCP_ADMIN_TOKEN")
        if admin:
            headers["Authorization"] = f"Bearer {admin}"
        # Stream (don't buffer) + reject oversized, so a valid link can't pin
        # memory proportional to file size x concurrency.
        client = httpx.AsyncClient(timeout=120)
        try:
            upstream = await client.send(
                client.build_request("GET", f"{base}/api/attachment/{aid}", headers=headers), stream=True
            )
        except Exception:
            await client.aclose()
            return JSONResponse({"error": "upstream unavailable"}, status_code=502)
        if upstream.status_code != 200:
            await upstream.aclose()
            await client.aclose()
            return JSONResponse({"error": "attachment unavailable"}, status_code=upstream.status_code)
        clen = upstream.headers.get("content-length")
        if clen and clen.isdigit() and int(clen) > _BLOB_MAX_BYTES:
            await upstream.aclose()
            await client.aclose()
            return JSONResponse({"error": "attachment too large for fetch"}, status_code=413)
        out_headers = {}
        cd = upstream.headers.get("content-disposition")
        if cd:
            out_headers["content-disposition"] = cd

        async def _body():
            try:
                async for chunk in upstream.aiter_bytes():
                    yield chunk
            finally:
                await upstream.aclose()
                await client.aclose()

        return StreamingResponse(
            _body(), media_type=upstream.headers.get("content-type", "application/octet-stream"), headers=out_headers
        )


def _register_admin_routes(app) -> None:
    """Attach the four admin endpoints. Kept on the same FastMCP app
    via `custom_route` so a single uvicorn process serves both the
    MCP transport and the side-channel."""
    from starlette.requests import Request
    from starlette.responses import JSONResponse

    @app.custom_route("/admin/calls/{session_id}", methods=["GET"])
    async def get_calls(request: Request) -> JSONResponse:
        denied = _admin_guard(request)
        if denied is not None:
            return denied
        session_id = request.path_params["session_id"]
        return JSONResponse({"calls": get_session_calls(session_id)})

    @app.custom_route("/admin/sessions", methods=["POST"])
    async def post_session(request: Request) -> JSONResponse:
        denied = _admin_guard(request)
        if denied is not None:
            return denied
        body = await request.json()
        session_id = body.get("session_id")
        if not session_id:
            return JSONResponse({"error": "session_id required"}, status_code=400)
        # SECURITY: never trust a caller-supplied db_dsn. If the admin
        # token leaks, accepting an arbitrary DSN here would let an
        # attacker point our DB connections at any host they control
        # (SSRF / outbound-DB / credential exfil). The DSN is resolved
        # exclusively from server-side config.
        if body.get("db_dsn"):
            logger.warning(
                "ignoring caller-supplied db_dsn on POST /admin/sessions for "
                "session %s — DSN is resolved server-side",
                session_id,
            )
        register_session(
            session_id,
            evidence_records=body.get("evidence_records"),
            db_dsn=_resolve_server_db_dsn(),
            conversation_id=body.get("conversation_id"),
            workspace=body.get("workspace"),
            user_id=body.get("user_id"),
        )
        return JSONResponse({"ok": True, "session_id": session_id})

    @app.custom_route("/admin/sessions/{session_id}", methods=["DELETE"])
    async def delete_session(request: Request) -> JSONResponse:
        denied = _admin_guard(request)
        if denied is not None:
            return denied
        session_id = request.path_params["session_id"]
        unregister_session(session_id)
        return JSONResponse({"ok": True})

    @app.custom_route("/admin/transport-tokens", methods=["POST"])
    async def mint_transport(request: Request) -> JSONResponse:
        """Mint a per-owner /mcp transport token. Admin-gated. The
        owner's internal `user_id` is resolved SERVER-SIDE from the
        email — the caller cannot supply it (a caller-supplied user_id
        in the body is ignored), so the admin token can never be used
        to mint a token for an arbitrary tenant id."""
        denied = _admin_guard(request)
        if denied is not None:
            return denied
        if _transport_secret() is None:
            return JSONResponse(
                {"error": "transport signing secret unavailable"},
                status_code=503,
            )
        body = await request.json()
        email = body.get("email")
        if not email or not isinstance(email, str):
            return JSONResponse({"error": "email required"}, status_code=400)
        ttl_seconds = body.get("ttl_seconds")
        if ttl_seconds is not None and not isinstance(ttl_seconds, int):
            return JSONResponse({"error": "ttl_seconds must be an integer"}, status_code=400)

        user_id = _resolve_user_id_by_email(email)
        if not user_id:
            return JSONResponse({"error": "unknown email"}, status_code=404)

        try:
            token, exp = mint_transport_token(user_id=user_id, email=email, ttl_seconds=ttl_seconds)
        except RuntimeError:
            return JSONResponse(
                {"error": "transport signing secret unavailable"},
                status_code=503,
            )
        return JSONResponse({"token": token, "expires_at": exp, "user_id": user_id})

    @app.custom_route("/admin/service-tokens", methods=["POST"])
    async def mint_service(request: Request) -> JSONResponse:
        """Mint a SERVICE token for a trusted server-side MCP client
        (claudebox). Admin-gated. Carries NO tenant — the token only
        satisfies /mcp enforcement; per-run scoping stays on the
        registered session. Body may carry an optional integer
        `ttl_seconds` (defaults to the long service TTL)."""
        denied = _admin_guard(request)
        if denied is not None:
            return denied
        if _transport_secret() is None:
            return JSONResponse(
                {"error": "transport signing secret unavailable"},
                status_code=503,
            )
        body = await request.json()
        ttl_seconds = body.get("ttl_seconds")
        if ttl_seconds is not None and not isinstance(ttl_seconds, int):
            return JSONResponse({"error": "ttl_seconds must be an integer"}, status_code=400)

        try:
            token = mint_service_token(ttl_seconds=ttl_seconds)
        except RuntimeError:
            return JSONResponse(
                {"error": "transport signing secret unavailable"},
                status_code=503,
            )
        claims = verify_token(token) or {}
        return JSONResponse({"token": token, "expires_at": claims.get("exp")})


def _require_transport_auth() -> bool:
    """Whether a /mcp request MUST carry a valid transport token. When
    off (default), unauthenticated /mcp requests pass through with no
    transport identity — preserves today's orchestrator behavior during
    rollout. Flipped on before the bhatti firewall port is opened."""
    return os.environ.get("GMAIL_MCP_REQUIRE_TRANSPORT_AUTH") == "1"


# Set by _make_fastmcp_app when GMAIL_MCP_OAUTH_ENABLED: the live OAuth
# provider, so the transport middleware can ALSO accept OAuth-issued access
# tokens (Claude's custom-connector path) and scope them to the owner.
_active_oauth_provider = None
_oauth_owner_uid_cache: str | None = None


def _oauth_owner_uid() -> str:
    """Resolve (and cache) the owner's internal uid for OAuth-authenticated
    requests. OAuth tokens are only ever issued to the single owner (the
    broker Google-login gate enforces it), so every OAuth token scopes to
    this one uid.

    Returns "" on failure and does NOT cache the failure — a transient DB
    hiccup must not stick an empty uid until restart. Callers MUST treat ""
    as a hard auth failure (fail closed), never as a default tenant."""
    global _oauth_owner_uid_cache
    if _oauth_owner_uid_cache:  # only a non-empty uid is ever cached
        return _oauth_owner_uid_cache
    try:
        from gmail_search.agents.mcp_oauth import owner_email
        from gmail_search.auth.write_user import get_user_id_by_email
        from gmail_search.store.db import get_connection

        conn = get_connection(None)
        try:
            uid = get_user_id_by_email(conn, owner_email()) or ""
        finally:
            conn.close()
    except Exception:  # noqa: BLE001 — never let scoping resolution crash auth
        uid = ""
    if uid:
        _oauth_owner_uid_cache = uid
    return uid


class _TransportAuthMiddleware:
    """Pure-ASGI middleware that authenticates `/mcp` requests with a
    per-owner transport token and scopes the request to the token's
    owner.

    Only the `/mcp` path is subject to this middleware's 401 — `/admin/*`
    keeps its own admin-token check and is passed through untouched (the
    host mints tokens via /admin/transport-tokens before transport auth
    would otherwise gate it).

    Behavior on a /mcp request, branching on the token audience:
      - `mcp-transport` (tenant-bound) with a non-empty `uid`: set
        `_transport_user_id` to the token's `uid` for the duration of the
        request; reset after. This is the untrusted bhatti VM path.
      - `mcp-service` (tenantless, trusted server-side client like
        claudebox): authenticated, but DO NOT set `_transport_user_id` —
        scoping falls to the registered session via `_get_session`.
      - Either valid aud → request passes.
      - Missing/invalid/expired/wrong-aud (incl. a transport token with
        no usable `uid`): 401 JSON iff `GMAIL_MCP_REQUIRE_TRANSPORT_AUTH=1`;
        otherwise pass through with no transport identity (legacy
        behavior)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if not path.startswith("/mcp"):
            await self.app(scope, receive, send)
            return

        token = self._bearer_token(scope)
        # Accepts BOTH audiences; returns the claims (incl. `aud`) or None.
        claims = verify_token(token) if token else None
        aud = claims.get("aud") if claims else None

        # A validly-signed transport token with no `uid` carries no usable
        # identity — treat it as an auth failure rather than scoping to an
        # empty user_id (defence-in-depth against a malformed mint).
        uid = str(claims.get("uid") or "") if aud == _TRANSPORT_AUD else ""

        # Authenticated iff a service token, or a transport token that
        # actually carries a uid. Everything else (no token, bad sig,
        # unknown/missing aud, expired, transport-without-uid) is a miss.
        authenticated = aud == _SERVICE_AUD or bool(uid)

        # OAuth path: when an OAuth provider is active, also accept its
        # access tokens (Claude's custom-connector path). They are only ever
        # issued to the owner, so scope the request to the owner's uid. Use
        # load_access_token (OAuth tokens only) — NOT verify_token, which
        # would recurse back into this transport-token check.
        # Only genuine OAuth access tokens (minted as "at_<hex>") take the
        # owner-scoped OAuth path. Transport/service JWTs (the "eyJ..." form)
        # are handled by verify_token above, so a malformed uid-less transport
        # token can't fall through here and get silently upgraded to owner.
        if not authenticated and token and token.startswith("at_") and _active_oauth_provider is not None:
            try:
                oauth_tok = await _active_oauth_provider.load_access_token(token)
            except Exception:  # noqa: BLE001
                oauth_tok = None
            if oauth_tok is not None:
                owner_uid = _oauth_owner_uid()
                if owner_uid:
                    authenticated = True
                    uid = owner_uid
                # else: owner uid unresolved → do NOT authenticate (fail
                # closed). Falls through to the SDK gate, which also can't
                # scope it, rather than scoping to an empty/default tenant.

        # Session-riding hardening: bind a streamable-HTTP mcp-session-id to
        # the FIRST authenticated uid that uses it; a different uid on the
        # same session id is rejected. Server-minted session ids are
        # unguessable UUIDs, but nothing else stops a client that LEARNED
        # one from riding another user's transport. Inert in stateless mode
        # (no session ids exist) and for tenantless service tokens.
        if authenticated and uid:
            sid = self._header(scope, b"mcp-session-id")
            if sid:
                owner = _TRANSPORT_SESSION_OWNERS.get(sid)
                if owner is None:
                    if len(_TRANSPORT_SESSION_OWNERS) >= _MAX_TRANSPORT_SESSION_OWNERS:
                        # Sessions are ephemeral; a full reset just re-binds
                        # each live session on its next request.
                        _TRANSPORT_SESSION_OWNERS.clear()
                    _TRANSPORT_SESSION_OWNERS[sid] = uid
                elif owner != uid:
                    logger.warning(
                        "transport session %s bound to another identity — rejecting request for uid=%s",
                        sid[:12],
                        uid,
                    )
                    await self._forbidden(send)
                    return

        if not authenticated:
            # OAuth on: don't emit our plain 401 — pass through so the SDK's
            # RequireAuthMiddleware produces the spec-compliant 401 with the
            # RFC 9728 WWW-Authenticate resource-metadata pointer Claude needs
            # to discover the OAuth flow. The SDK gate still enforces auth, so
            # this doesn't open /mcp.
            if _active_oauth_provider is not None:
                await self.app(scope, receive, send)
                return
            if _require_transport_auth():
                await self._unauthorized(send)
                return
            # Flag off: pass through with no transport identity.
            await self.app(scope, receive, send)
            return

        if not uid:
            # Service token: authenticated, but tenantless — scoping
            # falls to the registered session. No identity to set.
            await self.app(scope, receive, send)
            return

        reset_token = _transport_user_id.set(uid)
        try:
            await self.app(scope, receive, send)
        finally:
            _transport_user_id.reset(reset_token)

    @staticmethod
    def _bearer_token(scope) -> str | None:
        for key, value in scope.get("headers", []):
            if key == b"authorization":
                raw = value.decode("latin-1")
                if raw.lower().startswith("bearer "):
                    return raw[len("bearer ") :].strip()
                return None
        return None

    @staticmethod
    def _header(scope, name: bytes) -> str | None:
        for key, value in scope.get("headers", []):
            if key == name:
                return value.decode("latin-1") or None
        return None

    @staticmethod
    async def _forbidden(send) -> None:
        body = b'{"error":"transport session bound to another identity"}'
        await send(
            {
                "type": "http.response.start",
                "status": 403,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    @staticmethod
    async def _unauthorized(send) -> None:
        body = b'{"error":"transport token required"}'
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


def build_asgi_app(host: str | None = None, port: int | None = None):
    """Return the servable ASGI app: the FastMCP streamable-HTTP app
    (which also carries the /admin/* custom routes) wrapped in the
    transport-auth middleware. This is what `main()` serves via uvicorn
    and what middleware tests exercise."""
    app = build_app(host=host, port=port)
    return _TransportAuthMiddleware(app.streamable_http_app())


def _make_fastmcp_app(host: str, port: int):
    """Construct the FastMCP instance and register the five tools.
    Split out so tests can build an app without binding a port.

    When GMAIL_MCP_OAUTH_ENABLED is set, the app additionally runs as an
    OAuth 2.1 authorization server (DCR + PKCE + RFC 8414/9728 metadata,
    Google-gated via the broker) so Claude can add it as a custom
    connector. Flag off → byte-for-byte the prior behavior. See mcp_oauth.
    """
    from mcp.server.fastmcp import FastMCP

    from gmail_search.agents import mcp_oauth

    # STATELESS streamable-HTTP (2026-07-03 cross-wiring fix): claude.ai
    # reuses JSON-RPC id 1 for every request and pools one transport
    # session across a user's chats, so concurrent duplicate ids collide
    # in the SDK's per-session routing table (_request_streams keyed by
    # id alone) — one request receives another's response, the other
    # hangs. Stateless mode creates a fresh transport per POST, making
    # the collision impossible by construction. We use no session-only
    # features (no server push, subscriptions, sampling, or event-store
    # resumability; per-call context rides the session_id TOOL ARG).
    # GMAIL_MCP_STATELESS=0 is the operational rollback lever.
    stateless = os.environ.get("GMAIL_MCP_STATELESS", "1").strip().lower() not in ("0", "false", "no", "off")

    global _active_oauth_provider
    oauth_provider = None
    if mcp_oauth.is_oauth_enabled():
        auth_settings, oauth_provider = mcp_oauth.build_auth_settings()
        app = FastMCP(
            name="gmail-search-tools",
            host=host,
            port=port,
            streamable_http_path=DEFAULT_PATH,
            stateless_http=stateless,
            auth_server_provider=oauth_provider,
            auth=auth_settings,
        )
    else:
        app = FastMCP(
            name="gmail-search-tools",
            host=host,
            port=port,
            streamable_http_path=DEFAULT_PATH,
            stateless_http=stateless,
        )

    app.tool(name="search_emails_batch", description=SEARCH_BATCH_DESC)(_tool_search_emails_batch)
    app.tool(name="query_emails_batch", description=QUERY_BATCH_DESC)(_tool_query_emails_batch)
    app.tool(name="get_thread_batch", description=THREAD_BATCH_DESC)(_tool_get_thread_batch)
    app.tool(name="sql_query_batch", description=SQL_QUERY_BATCH_DESC)(_tool_sql_query_batch)
    app.tool(name="find_facts", description=FIND_FACTS_DESC)(_tool_find_facts)
    app.tool(name="describe_schema", description=DESCRIBE_SCHEMA_DESC)(_tool_describe_schema)
    app.tool(name="get_attachment_batch", description=ATTACHMENT_BATCH_DESC)(_tool_get_attachment_batch)
    app.tool(name="publish_artifact_batch", description=PUBLISH_ARTIFACT_BATCH_DESC)(_tool_publish_artifact_batch)

    _register_admin_routes(app)
    _register_blob_route(app)

    if oauth_provider is not None:
        mcp_oauth.register_oauth_callback_route(app, oauth_provider)
        # Resolve the owner uid eagerly and FAIL STARTUP if it can't be
        # resolved — an OAuth server that can't map its tokens to the owner
        # would fail open (empty-tenant scoping). Better a loud refusal than
        # a silent default-tenant exposure. Success populates the cache, so
        # a later transient DB hiccup at request time still has the uid.
        if not _oauth_owner_uid():
            raise RuntimeError(
                "GMAIL_MCP_OAUTH_ENABLED=1 but the owner uid for "
                f"{mcp_oauth.owner_email()!r} could not be resolved — refusing to "
                "start the OAuth server (would scope tokens to an empty tenant)."
            )
    # Publish (or clear) the active provider so _TransportAuthMiddleware can
    # accept OAuth access tokens and scope them to the owner.
    _active_oauth_provider = oauth_provider

    return app


def build_app(host: str | None = None, port: int | None = None):
    """Return a configured FastMCP instance. Used both by the
    `__main__` entrypoint and by tests that introspect the app /
    call its tools directly."""
    resolved_host = host or os.environ.get("GMAIL_MCP_TOOLS_HOST", DEFAULT_HOST)
    resolved_port = port or int(os.environ.get("GMAIL_MCP_TOOLS_PORT", str(DEFAULT_PORT)))
    return _make_fastmcp_app(resolved_host, resolved_port)


def main() -> None:
    """Entrypoint for `python -m gmail_search.agents.mcp_tools_server`.

    We build the FastMCP streamable-HTTP Starlette app, wrap it in the
    transport-auth middleware, and serve it directly with uvicorn.
    (FastMCP's `app.run("streamable-http")` runs uvicorn internally with
    no hook to inject middleware, so we drive uvicorn ourselves.) The
    inner Starlette app's lifespan — which starts FastMCP's session
    manager — still runs because the pure-ASGI middleware forwards
    lifespan events to it."""
    import uvicorn

    from gmail_search.log_config import setup_logging

    # Shared logging: human-readable by default, JSON when GMS_LOG_JSON=1, every
    # line stamped with trace_id (so MCP CALL lines join to serve + agent_events).
    setup_logging()
    token, was_generated = _resolve_admin_token()
    fastmcp = build_app()
    host = fastmcp.settings.host
    port = fastmcp.settings.port
    _assert_safe_bind_combination(host, token_was_generated=was_generated)
    if was_generated:
        token_path = _write_auto_token_file(token)
        if token_path:
            logger.info("auto-generated admin token written to %s (mode 0600)", token_path)
    logger.info(
        "starting gmail-search MCP tools server on http://%s:%s%s",
        host,
        port,
        DEFAULT_PATH,
    )
    # SECURITY: never log the token itself. Operators read it from
    # GMAIL_MCP_ADMIN_TOKEN (operator-supplied) or scripts/mcp_admin_token
    # (auto-generated). See `_resolve_admin_token` + `_write_auto_token_file`.
    logger.info("admin endpoints at /admin/* (token loaded from GMAIL_MCP_ADMIN_TOKEN " "env or auto-generated)")
    if _require_transport_auth():
        logger.info("transport auth ENFORCED: /mcp requires a valid Bearer transport token")
    else:
        logger.info("transport auth NOT enforced (GMAIL_MCP_REQUIRE_TRANSPORT_AUTH unset): /mcp open")
    asgi = _TransportAuthMiddleware(fastmcp.streamable_http_app())
    # SECURITY: proxy_headers=False. This server is NEVER legitimately
    # behind a trusted proxy. uvicorn's default (proxy_headers=True,
    # forwarded_allow_ips=127.0.0.1) would let any host-local proxy hop
    # rewrite request.client.host from an attacker-supplied
    # X-Forwarded-For — a VM sending `X-Forwarded-For: 127.0.0.1` could
    # then masquerade as loopback and defeat the _admin_guard loopback
    # gate. We bind the real peer address only, so the admin gate can't
    # be spoofed via forwarded headers.
    from gmail_search.log_config import uvicorn_log_config

    uvicorn.run(asgi, host=host, port=port, proxy_headers=False, log_config=uvicorn_log_config())


if __name__ == "__main__":
    main()
