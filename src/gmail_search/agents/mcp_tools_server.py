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

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gmail_search.agents.sandbox import SandboxRequest, execute_in_sandbox
from gmail_search.agents.session import save_artifact
from gmail_search.agents.tools import get_thread as _get_thread_impl
from gmail_search.agents.tools import query_emails as _query_emails_impl
from gmail_search.agents.tools import search_emails as _search_emails_impl
from gmail_search.agents.tools import sql_query as _sql_query_impl

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

# Mirrors `analyst._truncate` caps so the MCP shape matches the
# existing ADK shape exactly. Walkers in
# `orchestration._artifact_ids_from_tool_calls` and
# `_cite_refs_from_tool_calls` already handle this exact dict.
RUN_CODE_STDOUT_CAP = 8000
RUN_CODE_STDERR_CAP = 4000
RUN_CODE_TIMEOUT_SECONDS = 60

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


def _record_call(session_id: str, name: str, args: dict, response: dict) -> None:
    """Append one tool call to the session's call log. Stored as the
    full structured response — no stringify, no truncation. Drops
    silently (after one warning per session) once the per-session cap
    is reached so a runaway tool loop can't exhaust memory."""
    import time as _time

    bucket = _SESSION_CALLS.setdefault(session_id, [])
    if len(bucket) >= _MAX_CALLS_PER_SESSION:
        if session_id not in _SESSION_CAP_WARNED:
            _SESSION_CAP_WARNED.add(session_id)
            logger.warning(
                "session %s reached _MAX_CALLS_PER_SESSION=%d; further calls run "
                "but will not be recorded in the side-channel log",
                session_id,
                _MAX_CALLS_PER_SESSION,
            )
        return
    entry = {"name": name, "args": dict(args), "response": response, "ts": _time.time()}
    bucket.append(entry)


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


def register_session(
    session_id: str,
    *,
    evidence_records: list[dict] | dict | None,
    db_dsn: str | None,
    conversation_id: str | None = None,
    workspace: str | None = None,
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
    Claude-supplied file paths."""
    _assert_session_id(session_id)
    if session_id in _SESSIONS:
        _SESSIONS[session_id].close()
    _SESSIONS[session_id] = SessionContext(
        evidence_records=evidence_records,
        db_dsn=db_dsn,
        conversation_id=conversation_id,
        workspace=workspace,
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


# ── Tool implementations (re-routes) ───────────────────────────────


async def _tool_search_emails(
    session_id: str,
    query: str,
    date_from: str = "",
    date_to: str = "",
    top_k: int = 10,
) -> dict:
    """Re-route to the existing async search_emails. `session_id`
    threading is documented in the public docstring registered with
    FastMCP; the impl's only use of session_id is to assert the
    caller registered before calling — search itself is global."""
    _get_session(session_id)
    args = {"query": query, "date_from": date_from, "date_to": date_to, "top_k": top_k}
    response = await _search_emails_impl(**args)
    _record_call(session_id, "search_emails", args, response)
    return response


async def _tool_query_emails(
    session_id: str,
    sender: str = "",
    subject_contains: str = "",
    date_from: str = "",
    date_to: str = "",
    label: str = "",
    has_attachment: bool | None = None,
    order_by: str = "date_desc",
    limit: int = 20,
) -> dict:
    _get_session(session_id)
    args = {
        "sender": sender,
        "subject_contains": subject_contains,
        "date_from": date_from,
        "date_to": date_to,
        "label": label,
        "has_attachment": has_attachment,
        "order_by": order_by,
        "limit": limit,
    }
    response = await _query_emails_impl(**args)
    _record_call(session_id, "query_emails", args, response)
    return response


async def _tool_get_thread(session_id: str, thread_id: str) -> dict:
    _get_session(session_id)
    args = {"thread_id": thread_id}
    response = await _get_thread_impl(**args)
    _record_call(session_id, "get_thread", args, response)
    return response


async def _tool_sql_query(session_id: str, query: str) -> dict:
    _get_session(session_id)
    args = {"query": query}
    response = await _sql_query_impl(**args)
    _record_call(session_id, "sql_query", args, response)
    return response


async def _tool_get_attachment(session_id: str, attachment_id: int, mode: str = "text") -> dict:
    _get_session(session_id)
    args = {"attachment_id": attachment_id, "mode": mode}
    response = await _get_attachment_impl(**args)
    _record_call(session_id, "get_attachment", args, response)
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


async def _tool_publish_artifact(
    session_id: str,
    path: str,
    name: str = "",
    mime_type: str = "",
) -> dict:
    """Publish a file as a user-visible artifact. Read the file from
    the workspace (or the persistent /work scratch dir), upload to
    `agent_artifacts`, return the id the model can cite as
    `[art:<id>]`. The model uses this AFTER it has produced a
    chart/CSV via Bash + Python (or any other means) and wants the
    user to see it in the answer."""
    ctx = _get_session(session_id)
    args = {"path": path, "name": name, "mime_type": mime_type}
    try:
        file_path = _resolve_publish_source(
            path,
            workspace=ctx.workspace,
            conversation_id=ctx.conversation_id,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        response = {"error": str(exc)}
        _record_call(session_id, "publish_artifact", args, response)
        return response

    size = file_path.stat().st_size
    if size > MAX_PUBLISH_BYTES:
        response = {
            "error": f"file too large: {size} bytes; max {MAX_PUBLISH_BYTES}. Save a smaller version (downsample, lower DPI, or summarize)."
        }
        _record_call(session_id, "publish_artifact", args, response)
        return response

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
    response = {"id": art_id, "name": final_name, "mime_type": final_mime, "size": size}
    _record_call(session_id, "publish_artifact", args, response)
    return response


def _truncate(s: str, cap: int) -> str:
    """Match the analyst module's truncation marker exactly so the
    MCP run_code result reads identically to the ADK version."""
    if s is None:
        return ""
    if len(s) <= cap:
        return s
    return s[: cap - 16] + f"\n... (truncated, original {len(s)} chars)"


def _persist_sandbox_artifacts(
    artifacts,
    *,
    session_id: str,
    db_conn,
) -> list[dict[str, Any]]:
    """Upload each sandbox-produced artifact to `agent_artifacts` and
    return the list of `{id, name, mime_type}` rows the model can
    cite. Failures are logged but don't fail the whole tool call —
    matches the existing ADK behaviour."""
    persisted: list[dict[str, Any]] = []
    for art in artifacts:
        try:
            art_id = save_artifact(
                db_conn,
                session_id=session_id,
                name=art.name,
                mime_type=art.mime_type,
                data=art.data,
            )
            persisted.append({"id": art_id, "name": art.name, "mime_type": art.mime_type})
        except Exception as e:  # noqa: BLE001
            logger.warning(f"save_artifact failed for {art.name}: {e}")
    return persisted


def _tool_run_code(session_id: str, code: str) -> dict:
    """Sync — execute_in_sandbox is sync, so wrapping in async would
    just block the event loop on the Docker call without any
    benefit. FastMCP supports both."""
    ctx = _get_session(session_id)
    req = SandboxRequest(
        code=code,
        evidence=ctx.evidence_records,
        db_dsn=ctx.db_dsn,
        timeout_seconds=RUN_CODE_TIMEOUT_SECONDS,
        conversation_id=ctx.conversation_id,
    )
    result = execute_in_sandbox(req)
    # Only open the DB connection when we actually have something to
    # persist. Sessions that never produce an artifact (search-only
    # turns, or sandboxes that just print) shouldn't require DB_DSN to
    # be configured on the MCP server.
    persisted = (
        _persist_sandbox_artifacts(
            result.artifacts,
            session_id=session_id,
            db_conn=ctx.get_db_conn(),
        )
        if result.artifacts
        else []
    )
    response = {
        "exit_code": result.exit_code,
        "stdout": _truncate(result.stdout, RUN_CODE_STDOUT_CAP),
        "stderr": _truncate(result.stderr, RUN_CODE_STDERR_CAP),
        "wall_ms": result.wall_ms,
        "timed_out": result.timed_out,
        "oom_killed": result.oom_killed,
        "artifacts": persisted,
    }
    _record_call(session_id, "run_code", {"code": code}, response)
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

SEARCH_DESC = (
    "Search messages by relevance (semantic + BM25 blend). Use for "
    '"what did we decide about X" / "find messages mentioning Y". '
    "Empty date_from/date_to skip date filter; top_k caps thread "
    f"count (default 10). {SESSION_PARAM_NOTE}"
)

QUERY_DESC = (
    "Filter messages by structured metadata. Use when you know WHAT "
    "fields to filter on — no relevance ranking. Returns threads in "
    f"the requested order (date_desc default). {SESSION_PARAM_NOTE}"
)

THREAD_DESC = (
    "Fetch every message in a thread with body text + attachment "
    "manifest. Bodies clipped to 20k chars. Use AFTER search/query "
    f"when you need the actual words. {SESSION_PARAM_NOTE}"
)

SQL_DESC = (
    "Run a read-only SELECT against the messages DB (gated by the "
    "server's safety check: SELECT/WITH only, no DDL/DML, 500-row + "
    f"10s timeout). Cells > 8000 chars are clipped. {SESSION_PARAM_NOTE}"
)

RUN_CODE_DESC = (
    "Execute a Python snippet in the analysis sandbox. Has access "
    "to `evidence` (pandas DataFrame), `db` (read-only psycopg "
    "connection), `pd`, `np`, `plt`, `sns`, `sklearn`, and "
    "`save_artifact(name, obj)`. Returns exit_code, stdout, stderr, "
    "wall_ms, timed_out, oom_killed, artifacts (list of "
    f"{{id, name, mime_type}}). {SESSION_PARAM_NOTE}"
)

ATTACHMENT_DESC = (
    "Fetch an email attachment. `attachment_id` comes from "
    "get_thread's attachments[].id. `mode` is one of: "
    "'meta' (filename/mime/size only — cheap), "
    "'text' (extracted text from PDFs/docx/OCR — the usual choice), "
    "'rendered_pages' (PDF pages as base64 PNGs — heavy, only when "
    f"text is empty/unhelpful and you need to read images). {SESSION_PARAM_NOTE}"
)

PUBLISH_ARTIFACT_DESC = (
    "Register a file as part of the answer the user sees. Anything you "
    "produce that should appear in the user's answer must be published "
    "first — files you write to disk are invisible by default. Call "
    "this regardless of how you produced the file (Bash, external "
    "command, download, anything). `path` may be relative ('plot.png') "
    "or absolute ('/workspaces/<your-workspace>/plot.png' or "
    "'/work/plot.png'). Optional: `name` for the citation chip, "
    "`mime_type` if auto-detect is wrong. Returns "
    "`{id, name, mime_type, size}`; cite the `id` as `[art:<id>]` in "
    f"your answer. Files >10MB are rejected. {SESSION_PARAM_NOTE}"
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


def _check_admin_token(request) -> bool:
    """Constant-time bearer-token check. Missing / wrong header
    yields 401."""
    import secrets as _secrets

    expected = os.environ.get("GMAIL_MCP_ADMIN_TOKEN") or ""
    if not expected:
        return False
    header = request.headers.get("authorization", "")
    if not header.startswith("Bearer "):
        return False
    return _secrets.compare_digest(header[len("Bearer ") :], expected)


def _register_admin_routes(app) -> None:
    """Attach the four admin endpoints. Kept on the same FastMCP app
    via `custom_route` so a single uvicorn process serves both the
    MCP transport and the side-channel."""
    from starlette.requests import Request
    from starlette.responses import JSONResponse

    @app.custom_route("/admin/calls/{session_id}", methods=["GET"])
    async def get_calls(request: Request) -> JSONResponse:
        if not _check_admin_token(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        session_id = request.path_params["session_id"]
        return JSONResponse({"calls": get_session_calls(session_id)})

    @app.custom_route("/admin/sessions", methods=["POST"])
    async def post_session(request: Request) -> JSONResponse:
        if not _check_admin_token(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
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
        )
        return JSONResponse({"ok": True, "session_id": session_id})

    @app.custom_route("/admin/sessions/{session_id}", methods=["DELETE"])
    async def delete_session(request: Request) -> JSONResponse:
        if not _check_admin_token(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        session_id = request.path_params["session_id"]
        unregister_session(session_id)
        return JSONResponse({"ok": True})


def _make_fastmcp_app(host: str, port: int):
    """Construct the FastMCP instance and register the five tools.
    Split out so tests can build an app without binding a port."""
    from mcp.server.fastmcp import FastMCP

    app = FastMCP(
        name="gmail-search-tools",
        host=host,
        port=port,
        streamable_http_path=DEFAULT_PATH,
    )

    app.tool(name="search_emails", description=SEARCH_DESC)(_tool_search_emails)
    app.tool(name="query_emails", description=QUERY_DESC)(_tool_query_emails)
    app.tool(name="get_thread", description=THREAD_DESC)(_tool_get_thread)
    app.tool(name="sql_query", description=SQL_DESC)(_tool_sql_query)
    app.tool(name="run_code", description=RUN_CODE_DESC)(_tool_run_code)
    app.tool(name="get_attachment", description=ATTACHMENT_DESC)(_tool_get_attachment)
    app.tool(name="publish_artifact", description=PUBLISH_ARTIFACT_DESC)(_tool_publish_artifact)

    _register_admin_routes(app)

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
    FastMCP's `run("streamable-http")` spins up uvicorn internally
    on the host/port we configured at construction."""
    logging.basicConfig(level=logging.INFO)
    token, was_generated = _resolve_admin_token()
    app = build_app()
    _assert_safe_bind_combination(app.settings.host, token_was_generated=was_generated)
    if was_generated:
        token_path = _write_auto_token_file(token)
        if token_path:
            logger.info("auto-generated admin token written to %s (mode 0600)", token_path)
    logger.info(
        "starting gmail-search MCP tools server on http://%s:%s%s",
        app.settings.host,
        app.settings.port,
        DEFAULT_PATH,
    )
    # SECURITY: never log the token itself. Operators read it from
    # GMAIL_MCP_ADMIN_TOKEN (operator-supplied) or scripts/mcp_admin_token
    # (auto-generated). See `_resolve_admin_token` + `_write_auto_token_file`.
    logger.info("admin endpoints at /admin/* (token loaded from GMAIL_MCP_ADMIN_TOKEN " "env or auto-generated)")
    app.run("streamable-http")


if __name__ == "__main__":
    main()
