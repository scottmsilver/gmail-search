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

from gmail_search.agents.session import save_artifact
from gmail_search.agents.tools import describe_schema as _describe_schema_impl
from gmail_search.agents.tools import get_attachment_batch as _get_attachment_batch_impl
from gmail_search.agents.tools import get_thread_batch as _get_thread_batch_impl
from gmail_search.agents.tools import query_emails_batch as _query_emails_batch_impl
from gmail_search.agents.tools import search_emails_batch as _search_emails_batch_impl
from gmail_search.agents.tools import sql_query_batch as _sql_query_batch_impl

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


async def _tool_search_emails_batch(session_id: str, searches: list[dict]) -> dict:
    """Re-route to `search_emails_batch`. `session_id` threading is
    documented in the public docstring registered with FastMCP; the
    impl's only use of session_id is to assert the caller registered
    before calling — search itself is global."""
    _get_session(session_id)
    args = {"searches": searches}
    response = await _search_emails_batch_impl(searches)
    _record_call(session_id, "search_emails_batch", args, response)
    return response


async def _tool_query_emails_batch(session_id: str, filters: list[dict]) -> dict:
    _get_session(session_id)
    args = {"filters": filters}
    response = await _query_emails_batch_impl(filters)
    _record_call(session_id, "query_emails_batch", args, response)
    return response


async def _tool_get_thread_batch(session_id: str, thread_ids: list[str]) -> dict:
    _get_session(session_id)
    args = {"thread_ids": thread_ids}
    response = await _get_thread_batch_impl(thread_ids)
    _record_call(session_id, "get_thread_batch", args, response)
    return response


async def _tool_sql_query_batch(session_id: str, queries: list[str]) -> dict:
    _get_session(session_id)
    args = {"queries": queries}
    response = await _sql_query_batch_impl(queries)
    _record_call(session_id, "sql_query_batch", args, response)
    return response


async def _tool_describe_schema(session_id: str) -> dict:
    _get_session(session_id)
    response = await _describe_schema_impl()
    _record_call(session_id, "describe_schema", {}, response)
    return response


async def _tool_get_attachment_batch(session_id: str, items: list[dict]) -> dict:
    _get_session(session_id)
    args = {"items": items}
    response = await _get_attachment_batch_impl(items)
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
    ctx = _get_session(session_id)
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
    "is a list (1-100) of `{query, date_from?, date_to?, top_k?}` "
    'dicts; `top_k` defaults to 10. Use for relevance lookups ("what '
    'did we decide about X", "find messages mentioning Y"). Returns '
    "`{results: [{input, result}, ...]}` aligned with input. "
    "Per-search errors land in that entry's `result` as `{error: ...}`.\n\n"
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

DESCRIBE_SCHEMA_DESC = (
    "Return markdown documentation for every table the `sql_query` "
    "tool can read. Call this before writing a non-trivial sql_query "
    "if you're unsure about column names. Cheap, no parameters beyond "
    f"session_id. {SESSION_PARAM_NOTE}"
)

ATTACHMENT_BATCH_DESC = (
    "Fetch many email attachments concurrently in ONE call. `items` "
    "is a list (1-100) of `{attachment_id, mode?}` dicts. `mode` is "
    "one of: 'meta' (filename/mime/size only — cheap), 'text' "
    "(extracted text from PDFs/docx/OCR — the usual choice, default), "
    "'rendered_pages' (PDF pages as base64 PNGs — heavy, only when "
    "text is empty/unhelpful). `attachment_id` comes from a thread's "
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

    app.tool(name="search_emails_batch", description=SEARCH_BATCH_DESC)(_tool_search_emails_batch)
    app.tool(name="query_emails_batch", description=QUERY_BATCH_DESC)(_tool_query_emails_batch)
    app.tool(name="get_thread_batch", description=THREAD_BATCH_DESC)(_tool_get_thread_batch)
    app.tool(name="sql_query_batch", description=SQL_QUERY_BATCH_DESC)(_tool_sql_query_batch)
    app.tool(name="describe_schema", description=DESCRIBE_SCHEMA_DESC)(_tool_describe_schema)
    app.tool(name="get_attachment_batch", description=ATTACHMENT_BATCH_DESC)(_tool_get_attachment_batch)
    app.tool(name="publish_artifact_batch", description=PUBLISH_ARTIFACT_BATCH_DESC)(_tool_publish_artifact_batch)

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
