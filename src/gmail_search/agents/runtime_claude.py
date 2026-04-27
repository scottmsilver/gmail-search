"""Claudebox runtime adapter: drop-in replacement for `adk_invoke` that
calls a `psyb0t/docker-claudebox` HTTP endpoint instead of Google ADK.

`claudebox_invoke(agent, prompt, *, workspace, cost_sink=None)` returns
the same `StageResult` the orchestrator already consumes — text plus a
list of tool_call dicts in the canonical `{"name", "args"}` /
`{"name", "response"}` shapes.

The extra `workspace` kwarg is required: each turn maps to one
claudebox workspace, and the orchestrator-side wiring binds it via
`functools.partial` before handing the function off as the
`InvokeFn`.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx

# Test seams: tests patch these to a fake clock + no-op sleep so the
# polling loop can be advanced by hand without real wall time.
_now: Callable[[], float] = time.monotonic
_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep

from gmail_search.agents.jsonl_tail import (  # noqa: F401
    encode_workspace_path,
    map_jsonl_event_to_tool_calls,
    tail_session_events,
)
from gmail_search.agents.orchestration import AgentLike, StageResult  # noqa: F401

logger = logging.getLogger(__name__)


CostSink = Callable[..., None]

_DEFAULT_CLAUDEBOX_URL = "http://localhost:8765"
_DEFAULT_MCP_ADMIN_URL = "http://localhost:7878"
_DEFAULT_MODEL = "opus"
# Native mode + parallel sub-agents + chart-generating run_code routinely
# pushes a single /run past 5 minutes. claudebox kills the subprocess if
# the HTTP client (us) disconnects (returning HTTP 499 to anyone polling),
# so we hold the connection longer than any realistic stage takes. Set
# generously rather than tightly because the model's wall time has wide
# variance and a false timeout discards the entire turn's work.
_REQUEST_TIMEOUT_SECONDS = 900.0
_BUSY_RETRY_ATTEMPTS = 3
_BUSY_RETRY_BASE_DELAY = 1.0


class ClaudeboxError(RuntimeError):
    """Raised when the claudebox `/run` endpoint returns a non-409
    error or a malformed body. The orchestrator already wraps each
    stage in a try/except and emits an `error` event, so we just need
    a clean exception type to surface."""


def _claudebox_url() -> str:
    return os.environ.get("GMAIL_CLAUDEBOX_URL", _DEFAULT_CLAUDEBOX_URL).rstrip("/")


def _claudebox_token() -> str | None:
    return os.environ.get("GMAIL_CLAUDEBOX_TOKEN") or None


def _mcp_admin_url() -> str:
    return os.environ.get("GMAIL_MCP_ADMIN_URL", _DEFAULT_MCP_ADMIN_URL).rstrip("/")


def _mcp_admin_token() -> str | None:
    return os.environ.get("GMAIL_MCP_ADMIN_TOKEN") or None


def _build_mcp_admin_headers() -> dict[str, str]:
    token = _mcp_admin_token()
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


async def register_session_via_admin(
    session_id: str,
    *,
    evidence_records: list[dict] | None,
    db_dsn: str | None = None,
    conversation_id: str | None = None,
    workspace: str | None = None,
) -> None:
    """Register a turn's session_id with the out-of-process MCP server.

    The orchestrator runs in a different process than the MCP server
    (the MCP server is a long-lived daemon that claudebox reaches over
    HTTP), so an in-process `register_session` call would write to a
    dict the daemon can't see. This hits the daemon's admin HTTP
    endpoint instead.

    SECURITY: `db_dsn` is intentionally not transmitted. The MCP
    server resolves the DSN from its own server-side config so a
    leaked admin token can't be used to point our DB connections at
    an attacker-controlled host. The kwarg is kept for backwards
    compatibility but ignored — see
    `mcp_tools_server.post_session` + `_resolve_server_db_dsn`.

    `conversation_id`, when supplied, opts the session into the
    per-conversation persistent sandbox /work mount on the server."""
    if db_dsn:
        logger.debug("register_session_via_admin: ignoring db_dsn (resolved server-side)")
    url = f"{_mcp_admin_url()}/admin/sessions"
    body: dict[str, Any] = {"session_id": session_id, "evidence_records": evidence_records}
    if conversation_id is not None:
        body["conversation_id"] = conversation_id
    if workspace is not None:
        body["workspace"] = workspace
    headers = {"Content-Type": "application/json", **_build_mcp_admin_headers()}
    async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
        response = await client.post(url, json=body, headers=headers)
    if response.status_code != 200:
        raise ClaudeboxError(f"MCP admin register_session failed: HTTP {response.status_code}: {response.text[:300]}")


async def unregister_session_via_admin(session_id: str) -> None:
    """Mirror of `register_session_via_admin`. Idempotent on the
    server side — a missing session is treated as a no-op so callers
    can DELETE in `finally:` blocks without guarding."""
    url = f"{_mcp_admin_url()}/admin/sessions/{session_id}"
    headers = _build_mcp_admin_headers()
    async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
        response = await client.delete(url, headers=headers)
    if response.status_code not in (200, 204, 404):
        logger.warning(
            "MCP admin unregister returned %s for session %s",
            response.status_code,
            session_id,
        )


async def _fetch_structured_tool_calls(session_id: str) -> list[dict]:
    """Pull the side-channel call log for `session_id` from the MCP
    admin endpoint. The MCP server has the FULL structured response
    for every tool call this turn made — claudebox's stringified +
    truncated tool_result content is bypassed entirely.

    Failures (network blip, 401, missing endpoint) return an empty
    list and log a warning — the caller will fall back to the
    truncated message-parsed view rather than crashing the stage."""
    url = f"{_mcp_admin_url()}/admin/calls/{session_id}"
    headers = _build_mcp_admin_headers()
    try:
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.get(url, headers=headers)
        if response.status_code != 200:
            logger.warning(
                "MCP admin /admin/calls returned %s for session %s",
                response.status_code,
                session_id,
            )
            return []
        body = response.json()
        calls = body.get("calls")
        return list(calls) if isinstance(calls, list) else []
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"MCP admin fetch failed for session {session_id}: {exc}")
        return []


def _tool_calls_from_side_channel(records: list[dict]) -> list[dict[str, Any]]:
    """Expand each side-channel record into the orchestrator's
    canonical two-event sequence: `{name, args}` then `{name,
    response}`. Matches what `_extract_tool_calls_from_messages`
    produces from the message stream — but with the full structured
    response instead of the truncated string."""
    tool_calls: list[dict[str, Any]] = []
    for rec in records:
        name = str(rec.get("name") or "")
        args = rec.get("args") or {}
        if not isinstance(args, dict):
            args = {"value": args}
        response = rec.get("response") or {}
        if not isinstance(response, dict):
            response = {"value": response}
        tool_calls.append({"name": name, "args": dict(args)})
        tool_calls.append({"name": name, "response": response})
    return tool_calls


def _resolve_model(agent) -> str:
    explicit = getattr(agent, "model", None)
    if explicit:
        return str(explicit)
    return os.environ.get("GMAIL_CLAUDEBOX_MODEL") or _DEFAULT_MODEL


_SESSION_ID_PROMPT_BLOCK = (
    "\n\n## MCP tool calling\n\n"
    "When invoking any `gmail-tools` MCP tool (search_emails, query_emails, "
    "get_thread, sql_query, run_code), you MUST pass the literal value "
    "`session_id={session_id!r}` as a parameter. The server uses it to bind "
    "per-turn state. Do not invent a different session_id and do not omit it."
)


def _instruction_with_session_binding(agent, session_id: str | None) -> str:
    base = getattr(agent, "instruction", "") or ""
    if not session_id:
        return base
    return base + _SESSION_ID_PROMPT_BLOCK.format(session_id=session_id)


def _build_request_body(
    agent,
    prompt: str,
    *,
    workspace: str,
    session_id: str | None = None,
    resume: str | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "prompt": prompt,
        "workspace": workspace,
        "model": _resolve_model(agent),
        "systemPrompt": _instruction_with_session_binding(agent, session_id),
        "outputFormat": "json-verbose",
    }
    # Pinning the Claude session UUID (`--resume <uuid>`) makes Claude
    # Code append to the same JSONL transcript across turns and gets a
    # real prompt-cache hit on resume. See PER_CONVERSATION_SESSIONS.md
    # for the establishment flow (first turn omits resume; subsequent
    # turns pass the UUID captured from the first response's `sessionId`).
    if resume:
        body["resume"] = resume
    return body


def _extract_session_uuid(response: dict[str, Any]) -> str | None:
    """Pull the Claude session UUID out of a claudebox /run response.
    Field name is `sessionId` (camelCase). Returns None if absent so
    callers handling non-claudebox responses don't need to special-case."""
    sid = response.get("sessionId")
    return str(sid) if sid else None


def _build_auth_headers() -> dict[str, str]:
    token = _claudebox_token()
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


async def _post_claudebox_run(body: dict[str, Any]) -> dict[str, Any]:
    """One HTTP round-trip to claudebox `/run`, with 409-busy retries.

    409 means the workspace is currently running another claude
    invocation; we exponential-back-off and retry up to 3 times.
    Anything else (4xx, 5xx, network error) is surfaced as a
    ClaudeboxError so the caller can stop the stage cleanly."""
    url = f"{_claudebox_url()}/run"
    headers = {"Content-Type": "application/json", **_build_auth_headers()}
    async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
        for attempt in range(_BUSY_RETRY_ATTEMPTS):
            response = await client.post(url, json=body, headers=headers)
            if response.status_code == 409:
                await _sleep_for_busy_retry(attempt)
                continue
            return _parse_claudebox_response(response)
    raise ClaudeboxError(f"claudebox workspace busy after {_BUSY_RETRY_ATTEMPTS} retries")


async def _sleep_for_busy_retry(attempt: int) -> None:
    delay = _BUSY_RETRY_BASE_DELAY * (2**attempt)
    await asyncio.sleep(delay)


def _parse_claudebox_response(response: httpx.Response) -> dict[str, Any]:
    if response.status_code >= 400:
        raise ClaudeboxError(f"claudebox /run failed: HTTP {response.status_code}: {response.text[:500]}")
    try:
        return response.json()
    except ValueError as exc:
        raise ClaudeboxError(f"claudebox /run returned non-JSON body: {exc}") from exc


def _turns_from_response(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Json-verbose assembles events into a top-level `turns` array
    (see `_assemble` in docker-claudebox/jsonpipe.py). Each turn has
    a `role` and a list of `content` blocks. Older builds may use
    `messages`; fall back so a server upgrade doesn't break us."""
    turns = response.get("turns")
    if isinstance(turns, list):
        return turns
    messages = response.get("messages")
    if isinstance(messages, list):
        return messages
    return []


def _extract_tool_calls_from_messages(messages: list[dict]) -> list[dict[str, Any]]:
    """Walk the assembled turns and produce the canonical orchestrator
    tool_calls list. `tool_use` blocks (in assistant turns) become
    `{"name", "args"}`; `tool_result` blocks (in tool_result turns)
    become `{"name", "response"}`. Both shapes coexist in the same
    list, matching what `_extract_text_and_tool_calls` produces in
    the ADK runtime."""
    tool_calls: list[dict[str, Any]] = []
    tool_use_id_to_name: dict[str, str] = {}
    for turn in messages:
        content = turn.get("content") or []
        for block in content:
            block_type = block.get("type")
            if block_type == "tool_use":
                _append_tool_use(block, tool_calls, tool_use_id_to_name)
            elif block_type == "tool_result":
                _append_tool_result(block, tool_calls, tool_use_id_to_name)
    return tool_calls


def _append_tool_use(
    block: dict[str, Any],
    tool_calls: list[dict[str, Any]],
    tool_use_id_to_name: dict[str, str],
) -> None:
    name = str(block.get("name") or "")
    args = block.get("input") or {}
    if not isinstance(args, dict):
        args = {"value": args}
    tool_use_id = block.get("id") or block.get("toolUseId")
    if isinstance(tool_use_id, str) and name:
        tool_use_id_to_name[tool_use_id] = name
    tool_calls.append({"name": name, "args": dict(args)})


def _append_tool_result(
    block: dict[str, Any],
    tool_calls: list[dict[str, Any]],
    tool_use_id_to_name: dict[str, str],
) -> None:
    tool_use_id = block.get("tool_use_id") or block.get("toolUseId") or ""
    name = tool_use_id_to_name.get(tool_use_id, "")
    response = _coerce_tool_result_response(block)
    tool_calls.append({"name": name, "response": response})


def _coerce_tool_result_response(block: dict[str, Any]) -> dict[str, Any]:
    """Tool-result blocks store their payload under `content` (a string,
    truncated by jsonpipe). The orchestrator's downstream inspectors
    (`_artifact_ids_from_tool_calls`, `_cite_refs_from_tool_calls`)
    expect a dict — so wrap the raw content under a `content` key and
    propagate the truncation flags so a debugging operator can tell
    when payloads were clipped."""
    response: dict[str, Any] = {}
    if "content" in block:
        response["content"] = block.get("content")
    if block.get("truncated"):
        response["truncated"] = True
        if "totalLength" in block:
            response["totalLength"] = block["totalLength"]
        elif "total_length" in block:
            response["totalLength"] = block["total_length"]
    is_error = block.get("isError", block.get("is_error"))
    if is_error:
        response["isError"] = True
    return response


def _extract_usage_from_response(response: dict[str, Any]) -> tuple[int, int]:
    """Pull (input_tokens, output_tokens) out of the claudebox result
    blob. Json-verbose normalises keys to camelCase — but older
    claudebox builds left them snake_case, so we probe both."""
    usage = response.get("usage") or {}
    input_tokens = usage.get("inputTokens", usage.get("input_tokens", 0)) or 0
    output_tokens = usage.get("outputTokens", usage.get("output_tokens", 0)) or 0
    return int(input_tokens), int(output_tokens)


def _extract_result_text(response: dict[str, Any]) -> str:
    result = response.get("result")
    if isinstance(result, str):
        return result.strip()
    return ""


def _report_cost(
    agent,
    response: dict[str, Any],
    cost_sink: CostSink | None,
) -> None:
    if cost_sink is None:
        return
    input_tokens, output_tokens = _extract_usage_from_response(response)
    try:
        cost_sink(
            agent_name=getattr(agent, "name", "agent"),
            model=_resolve_model(agent),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"cost_sink failed (non-fatal): {exc}")


EventSink = Callable[[str, dict], Awaitable[None]]

# Host filesystem mirror of `/home/claude/.claude/projects/<encoded>/`
# inside the claudebox container. The docker-compose mount is
# `./claude-config:/home/claude/.claude`, so the host-side path is
# `deploy/claudebox/claude-config/projects/<encoded>/`. We tail the
# newest .jsonl file there to surface tool_use events mid-flight.
_CLAUDEBOX_HOST_PROJECTS_ROOT = Path("deploy/claudebox/claude-config/projects")
# Container-side workspace root the claudebox server passes through.
# Combined with the run's workspace name it yields the absolute path
# the JSONL filename is encoded from.
_CLAUDEBOX_CONTAINER_WORKSPACES = "/workspaces"
# Grace period after the /run POST returns before we abandon the
# tailer task. Drop it if the file system is sluggish — the post-hoc
# parser still gives correct tool_calls.
_TAILER_SHUTDOWN_TIMEOUT_SECONDS = 2.0


def _host_jsonl_dir_for(workspace: str) -> Path:
    """Resolve the host-side directory holding the session's JSONL
    transcript. `workspace="deep-XYZ"` -> mounts under
    `deploy/claudebox/claude-config/projects/-workspaces-deep-XYZ/`."""
    container_path = f"{_CLAUDEBOX_CONTAINER_WORKSPACES}/{workspace}"
    return _CLAUDEBOX_HOST_PROJECTS_ROOT / encode_workspace_path(container_path)


def _make_jsonl_event_handler(
    event_sink: EventSink,
) -> Callable[[dict], Awaitable[None]]:
    """Build the per-line callback the tailer hands each parsed JSONL
    event. We map each line to zero-or-more tool_call dicts and emit
    one `tool_call` event per dict via `event_sink`."""

    async def _handle_jsonl_line(parsed: dict) -> None:
        for tool_call in map_jsonl_event_to_tool_calls(parsed):
            try:
                await event_sink("tool_call", tool_call)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"event_sink raised on streamed tool_call (non-fatal): {exc}")

    return _handle_jsonl_line


async def _shutdown_tailer(task: asyncio.Task, stop_event: asyncio.Event) -> None:
    """Signal the tailer to stop and await its termination with a
    bounded grace period. The tailer is best-effort — if it doesn't
    shut down promptly we drop it and let the post-hoc parser carry
    the load."""
    stop_event.set()
    try:
        await asyncio.wait_for(task, timeout=_TAILER_SHUTDOWN_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        logger.warning("jsonl tailer did not shut down within %ss; dropping", _TAILER_SHUTDOWN_TIMEOUT_SECONDS)
        task.cancel()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"jsonl tailer task raised (non-fatal): {exc}")


# ── Async polling mode ───────────────────────────────────────────
#
# Long claudebox runs (multi-minute deep stages) over a single sync
# POST hit HTTP 499 when intermediate proxies / our own client time
# out. claudebox now exposes `{async: True}` on `/run` which returns
# a runId immediately; we poll `GET /run/result?runId=<id>` for the
# assembled blob. Verified contract from
# `/tmp/docker-claudebox/api_server.py:598-635`:
#   * 404 → run id unknown / already purged
#   * 200 with `{"status": "running"}` while the subprocess is alive
#   * 200 with the full json-verbose blob (same shape sync returns)
#     once status == completed; the result is purged on first read
#   * 200 with `{"status": "failed", "error": ...}` on subprocess
#     failure (also purged); 200 `{"status": "cancelled"}` if
#     externally cancelled.
# We treat anything other than `running` / `completed` as an error.
_POLL_INTERVAL_SECONDS = 2.0
# How long we'll wait without any new JSONL line before treating the
# subprocess as stuck. The model can legitimately produce nothing for
# minutes — a long final answer or a heavy run_code only emits one
# JSONL line at completion. 5 min default; tune up if you have very
# long-form turns. Env override: `GMAIL_CLAUDEBOX_IDLE_TIMEOUT_SECONDS`.
_DEFAULT_IDLE_TIMEOUT_SECONDS = 300.0

# When the parent has any open Task tool_use (sub-agent in flight),
# the parent JSONL goes silent for the duration of the sub-agent's
# work; sub-agent JSONLs are also tailed (jsonl_tail.py) but the
# watchdog gets an extra grace multiplier as belt-and-suspenders.
# Observed: 8-way parallel Task fan-out can keep the parent silent
# for 5+ minutes if sub-agent JSONL discovery lags.
_FANOUT_IDLE_MULTIPLIER = 4.0
_HARD_TIMEOUT_SECONDS = 60 * 60  # 1h hard cap on a single async run.


def _idle_timeout_seconds() -> float:
    raw = os.environ.get("GMAIL_CLAUDEBOX_IDLE_TIMEOUT_SECONDS")
    if not raw:
        return _DEFAULT_IDLE_TIMEOUT_SECONDS
    try:
        v = float(raw)
        if v <= 0:
            return _DEFAULT_IDLE_TIMEOUT_SECONDS
        return v
    except ValueError:
        return _DEFAULT_IDLE_TIMEOUT_SECONDS


_POLL_HTTP_RETRY_ATTEMPTS = 3
_POLL_HTTP_RETRY_DELAY = 0.5
_ASYNC_MODE_ENV_VAR = "GMAIL_CLAUDEBOX_USE_ASYNC"


def _use_async_mode() -> bool:
    """Read the mode-switch env var. Default: async ON. Set
    `GMAIL_CLAUDEBOX_USE_ASYNC=0` to fall back to the sync `/run`
    path (kept verbatim as `_claudebox_invoke_sync` so a deploy can
    revert without a code change)."""
    raw = os.environ.get(_ASYNC_MODE_ENV_VAR, "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


async def claudebox_invoke(
    agent,
    prompt: str,
    *,
    workspace: str,
    session_id: str | None = None,
    cost_sink: CostSink | None = None,
    event_sink: EventSink | None = None,
    resume: str | None = None,
) -> StageResult:
    """Drop-in replacement for `adk_invoke` that talks to a
    docker-claudebox HTTP server.

    Behaviour is gated by `GMAIL_CLAUDEBOX_USE_ASYNC` (default `1`):
      * async: POST `/run` with `{async: True}` → poll
        `GET /run/result?runId=<id>` until done. The JSONL tailer
        doubles as an idle-progress watchdog; if no tool_use events
        appear for `_IDLE_TIMEOUT_SECONDS` we abort client-side.
      * sync (`GMAIL_CLAUDEBOX_USE_ASYNC=0`): the legacy single-POST
        path, kept verbatim as a deploy-time fallback.

    Returns `StageResult(text=..., tool_calls=[...])` matching the
    ADK adapter, so the orchestrator's stage helpers don't need to
    know which backend is wired in.

    `workspace` selects a subdirectory under the claudebox server's
    `/workspaces` root — each deep-mode turn typically maps to its
    own workspace so concurrent stages don't collide on Claude's
    file-mutation tools.

    `session_id`, when provided, switches `tool_calls` to the
    side-channel built by the MCP server (`/admin/calls/<id>`). That
    bypasses claudebox's 2000-char truncation of `tool_result`
    content so the orchestrator's downstream walkers
    (`_cite_refs_from_tool_calls`, `_artifact_ids_from_tool_calls`)
    can read the FULL structured response. Without `session_id` we
    fall back to parsing tool calls out of the (possibly truncated)
    message stream, matching pre-side-channel behaviour.

    `event_sink`, when provided, enables mid-flight tool-call
    streaming: while the `/run` is in flight we tail the per-session
    JSONL transcript on the host filesystem and call
    `event_sink("tool_call", {"name", "args"})` for every tool_use
    block as the model emits it. CALLERS THAT SUPPLY `event_sink`
    MUST NOT re-emit `tool_call` events from `result.tool_calls`
    afterwards — they were already streamed."""
    if _use_async_mode():
        return await _claudebox_invoke_async(
            agent,
            prompt,
            workspace=workspace,
            session_id=session_id,
            cost_sink=cost_sink,
            event_sink=event_sink,
            resume=resume,
        )
    return await _claudebox_invoke_sync(
        agent,
        prompt,
        workspace=workspace,
        session_id=session_id,
        cost_sink=cost_sink,
        event_sink=event_sink,
        resume=resume,
    )


async def _claudebox_invoke_sync(
    agent,
    prompt: str,
    *,
    workspace: str,
    session_id: str | None,
    cost_sink: CostSink | None,
    event_sink: EventSink | None,
    resume: str | None = None,
) -> StageResult:
    """Legacy sync `/run` path. Held in place verbatim as a fallback
    so a misbehaving async deploy can be reverted by flipping
    `GMAIL_CLAUDEBOX_USE_ASYNC=0`."""
    body = _build_request_body(agent, prompt, workspace=workspace, session_id=session_id, resume=resume)
    tailer_task, stop_event = _maybe_start_jsonl_tailer(workspace, event_sink)
    try:
        response = await _post_claudebox_run(body)
    finally:
        if tailer_task is not None and stop_event is not None:
            await _shutdown_tailer(tailer_task, stop_event)
    text = _extract_result_text(response)
    tool_calls = await _resolve_tool_calls(response, session_id=session_id)
    _report_cost(agent, response, cost_sink)
    return StageResult(
        text=text,
        tool_calls=tool_calls,
        claude_session_uuid=_extract_session_uuid(response),
    )


async def _claudebox_invoke_async(
    agent,
    prompt: str,
    *,
    workspace: str,
    session_id: str | None,
    cost_sink: CostSink | None,
    event_sink: EventSink | None,
    resume: str | None = None,
) -> StageResult:
    """Async-mode entry point: kick off `/run` with `async=True`,
    poll for completion, and run the JSONL tailer alongside the poll
    loop as both an event source and an idle-progress watchdog."""
    body = _build_request_body(agent, prompt, workspace=workspace, session_id=session_id, resume=resume)
    body["async"] = True
    initial = await _post_claudebox_run_async(body)
    run_id = _extract_run_id(initial)

    last_event_at = {"t": _now()}
    open_task_ids: set[str] = set()
    stop_event = asyncio.Event()
    tailer_task = _start_progress_tailer(workspace, event_sink, last_event_at, stop_event, open_task_ids=open_task_ids)

    try:
        result_blob = await _poll_until_done(run_id, last_event_at, open_task_ids=open_task_ids)
    finally:
        if tailer_task is not None:
            await _shutdown_tailer(tailer_task, stop_event)

    text = _extract_result_text(result_blob)
    tool_calls = await _resolve_tool_calls(result_blob, session_id=session_id)
    _report_cost(agent, result_blob, cost_sink)
    return StageResult(
        text=text,
        tool_calls=tool_calls,
        claude_session_uuid=_extract_session_uuid(result_blob),
    )


def _extract_run_id(initial: dict[str, Any]) -> str:
    """Pull the runId off the initial async POST response. Missing
    runId means the server isn't honouring async mode — fail loud
    rather than fall through to a polling loop with `None`."""
    run_id = initial.get("runId") or initial.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ClaudeboxError(f"claudebox /run async returned no runId; body={str(initial)[:300]}")
    return run_id


async def _post_claudebox_run_async(body: dict[str, Any]) -> dict[str, Any]:
    """POST `/run` in async mode. Reuses the 409-busy retry behaviour
    of the sync helper so workspace contention is handled identically;
    only the body shape (`{async: True}`) and the response shape
    (`{runId, status: "running"}`) differ."""
    return await _post_claudebox_run(body)


def _start_progress_tailer(
    workspace: str,
    event_sink: EventSink | None,
    last_event_at: dict[str, float],
    stop_event: asyncio.Event,
    open_task_ids: set[str] | None = None,
) -> asyncio.Task | None:
    """Spawn a JSONL tailer that updates the idle-watchdog timestamp
    on every parsed event and (if a sink is wired) forwards the
    canonical `("tool_call", {...})` events to the caller. The tailer
    runs whenever async mode is on — even without an event_sink — so
    the watchdog has a progress signal."""
    host_dir = _host_jsonl_dir_for(workspace)
    handler = _make_progress_handler(event_sink, last_event_at, open_task_ids=open_task_ids)
    return asyncio.create_task(
        tail_session_events(host_dir, handler, stop_event=stop_event),
    )


def _make_progress_handler(
    event_sink: EventSink | None,
    last_event_at: dict[str, float],
    open_task_ids: set[str] | None = None,
) -> Callable[[dict], Awaitable[None]]:
    """Build the per-line callback the tailer uses. Every line bumps
    the watchdog timestamp; lines mappable to tool_calls also fan out
    to the caller's event_sink when one is wired.

    If `open_task_ids` is supplied, the handler tracks open Task-tool
    invocations so the idle-timeout check can apply a longer grace
    period while sub-agents are running. We add the tool_use_id when
    we see a `Task` `tool_use` block, and remove it when the matching
    `tool_result` arrives."""

    async def _handle(parsed: dict) -> None:
        last_event_at["t"] = _now()
        if open_task_ids is not None:
            _update_open_task_ids(parsed, open_task_ids)
        if event_sink is None:
            return
        for tool_call in map_jsonl_event_to_tool_calls(parsed):
            try:
                await event_sink("tool_call", tool_call)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"event_sink raised on streamed tool_call (non-fatal): {exc}")

    return _handle


def _update_open_task_ids(parsed: dict, open_task_ids: set[str]) -> None:
    """Mutate `open_task_ids` based on one parsed JSONL line.

    Adds tool_use_id when the line is an assistant Task tool_use;
    removes it when a matching user tool_result lands. Robust to the
    many shapes claudebox JSONL lines can take — silently skips
    anything we don't recognize. The watchdog only needs a rough
    approximation: if Task is in flight, give the run more breathing
    room; if not, fall back to the standard threshold."""
    msg_type = parsed.get("type")
    message = parsed.get("message")
    if not isinstance(message, dict):
        return
    content = message.get("content")
    if not isinstance(content, list):
        return
    if msg_type == "assistant":
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use" and str(block.get("name") or "") == "Task":
                tool_use_id = block.get("id")
                if isinstance(tool_use_id, str) and tool_use_id:
                    open_task_ids.add(tool_use_id)
    elif msg_type == "user":
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_result":
                tool_use_id = block.get("tool_use_id")
                if isinstance(tool_use_id, str):
                    open_task_ids.discard(tool_use_id)


async def _poll_until_done(
    run_id: str,
    last_event_at: dict[str, float],
    *,
    open_task_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Poll `GET /run/result` until the run completes, fails, or one
    of two watchdogs trips:
      * idle: no JSONL progress for `_idle_timeout_seconds()` →
        claudebox subprocess looks wedged; abort client-side. While
        any Task tool_use is open (sub-agent in flight), the threshold
        is multiplied by `_FANOUT_IDLE_MULTIPLIER`.
      * hard: total wall time exceeds `_HARD_TIMEOUT_SECONDS`."""
    started = _now()
    while True:
        _check_hard_timeout(run_id, started)
        await _sleep(_POLL_INTERVAL_SECONDS)
        status, body = await _poll_run_result(run_id)
        if status == "done":
            return body  # type: ignore[return-value]
        _check_idle_timeout(run_id, last_event_at, open_task_ids=open_task_ids)


def _check_hard_timeout(run_id: str, started: float) -> None:
    if _now() - started > _HARD_TIMEOUT_SECONDS:
        raise ClaudeboxError(f"hard timeout {_HARD_TIMEOUT_SECONDS}s exceeded for run {run_id}")


def _check_idle_timeout(
    run_id: str,
    last_event_at: dict[str, float],
    *,
    open_task_ids: set[str] | None = None,
) -> None:
    idle = _now() - last_event_at["t"]
    threshold = _idle_timeout_seconds()
    if open_task_ids:
        threshold *= _FANOUT_IDLE_MULTIPLIER
    if idle > threshold:
        fanout_note = (
            f" (Task fan-out active with {len(open_task_ids)} open sub-agent(s); " f"threshold was {threshold:.0f}s)"
            if open_task_ids
            else ""
        )
        raise ClaudeboxError(
            f"no JSONL progress for {idle:.0f}s on run {run_id}; "
            f"claudebox subprocess appears stuck (no tool_use events emitted "
            f"since the last log line){fanout_note}. Aborting client-side; "
            f"the server-side run may still complete and is purged after 6h."
        )


async def _poll_run_result(run_id: str) -> tuple[str, dict[str, Any] | None]:
    """One poll of `GET /run/result?runId=<id>`. Returns
    `("running", None)` while the run is alive, `("done", blob)` on
    completion. Raises ClaudeboxError on any other status (failed,
    cancelled, 4xx, 5xx)."""
    url = f"{_claudebox_url()}/run/result"
    headers = _build_auth_headers()
    response = await _get_with_transient_retries(url, params={"runId": run_id}, headers=headers)
    return _interpret_run_result_response(response, run_id)


async def _get_with_transient_retries(
    url: str,
    *,
    params: dict[str, str],
    headers: dict[str, str],
) -> httpx.Response:
    """GET with a small bounded retry on connect-level failures.
    HTTP-level errors (4xx/5xx) come back as a Response and are
    interpreted by the caller — only network blips are retried here."""
    last_exc: Exception | None = None
    for attempt in range(_POLL_HTTP_RETRY_ATTEMPTS):
        try:
            async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
                return await client.get(url, params=params, headers=headers)
        except httpx.HTTPError as exc:
            last_exc = exc
            if attempt + 1 < _POLL_HTTP_RETRY_ATTEMPTS:
                await _sleep(_POLL_HTTP_RETRY_DELAY)
    raise ClaudeboxError(f"claudebox /run/result transient error: {last_exc}")


def _interpret_run_result_response(
    response: httpx.Response,
    run_id: str,
) -> tuple[str, dict[str, Any] | None]:
    """Map a `/run/result` HTTP response onto the polling state
    machine. Mirrors the contract documented above the polling
    constants; any deviation is surfaced as a ClaudeboxError."""
    if response.status_code == 404:
        raise ClaudeboxError(f"claudebox /run/result: unknown runId {run_id} (404)")
    if response.status_code >= 400:
        raise ClaudeboxError(f"claudebox /run/result HTTP {response.status_code}: {response.text[:500]}")
    body = _parse_run_result_json(response)
    status = body.get("status")
    if status == "running":
        return "running", None
    if status in ("failed", "cancelled"):
        error = body.get("error") or status
        raise ClaudeboxError(f"claudebox run {run_id} {status}: {error}")
    # No `status` field → completed run; the server returns the
    # full json-verbose blob (which has its own top-level `type`,
    # `result`, `usage`, `turns` etc., NOT a `status` field).
    return "done", body


def _parse_run_result_json(response: httpx.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError as exc:
        raise ClaudeboxError(f"claudebox /run/result non-JSON body: {exc}") from exc
    if not isinstance(body, dict):
        raise ClaudeboxError(f"claudebox /run/result body not an object: {str(body)[:200]}")
    return body


def _maybe_start_jsonl_tailer(
    workspace: str,
    event_sink: EventSink | None,
) -> tuple[asyncio.Task | None, asyncio.Event | None]:
    """Spawn the JSONL tailer iff a sink is wired. Returns
    (task, stop_event) or (None, None) so the caller can short-circuit
    cleanup when streaming is off."""
    if event_sink is None:
        return None, None
    host_dir = _host_jsonl_dir_for(workspace)
    stop_event = asyncio.Event()
    handler = _make_jsonl_event_handler(event_sink)
    task = asyncio.create_task(
        tail_session_events(host_dir, handler, stop_event=stop_event),
    )
    return task, stop_event


async def _resolve_tool_calls(
    response: dict[str, Any],
    *,
    session_id: str | None,
) -> list[dict[str, Any]]:
    """Pick the source of truth for `tool_calls`. With a session_id
    we trust the MCP side-channel exclusively — that's what survives
    truncation. Without one we walk the message stream like before."""
    if session_id is None:
        return _extract_tool_calls_from_messages(_turns_from_response(response))
    records = await _fetch_structured_tool_calls(session_id)
    return _tool_calls_from_side_channel(records)
