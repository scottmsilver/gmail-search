"""Single-agent deep-analysis turn through the Pi agent harness.

One `pi --mode rpc` process per turn, inside the `pi-sandbox`
container, driven over stdin/stdout. Tool calls stream to
`agent_events` as they happen; the MCP side channel supplies the full
structured responses afterwards, exactly as `claude_native` does.
Public entry point: `pi_run()`.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

from gmail_search.agents import pi_protocol as pp
from gmail_search.agents.deep_events import (
    emit_analyst_events,
    emit_error,
    emit_plan_event,
    emit_retriever_events,
    emit_writer_and_final,
    sweep_and_extend_final_text,
)
from gmail_search.agents.pi_rpc import PiRpcClient, PiRpcError
from gmail_search.agents.session import append_event, finalize_session
from gmail_search.store.db import get_connection

logger = logging.getLogger(__name__)

AGENT_NAME = "pi"
_DEFAULT_MODEL = "google/gemini-3.7-flash"
_DEFAULT_THINKING = "medium"
_DEFAULT_CONTAINER = "pi-sandbox"
_DEFAULT_EXTENSION_PATH = "/opt/gmail-tools"
_DEFAULT_HARD_TIMEOUT = 900.0
_DEFAULT_IDLE_TIMEOUT = 300.0
_ABORT_GRACE = 5.0
_STATS_TIMEOUT = 15.0
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_BUILTIN_TOOLS_OFF = {"0", "false", "no", "off"}

PI_INSTRUCTION = """You are a deep-analysis agent over the user's personal Gmail archive. Your
job is to answer one question with grounded, cited reasoning.

# Tools

Every tool below takes a LIST as its main argument — even when you have
just one item, you pass a one-item list. There are no single-call
versions; that's deliberate, to keep you in batch-mode by default.

- `search_emails_batch(session_id, searches=[{query, date_from?,
  date_to?, top_k?, detail?, max_matches?}, ...])` — semantic search,
  fan out across phrasings/date-windows in one call. Each result
  thread has a `cite_ref` field. `detail="refs"` returns ONE compact
  line per thread — use it for fan-out inventory sweeps (e.g. one
  search per entity in a list) so the batch payload stays small.
- `query_emails_batch(session_id, filters=[{sender?,
  subject_contains?, date_from?, date_to?, label?, has_attachment?,
  order_by?, limit?}, ...])` — structured-metadata filter; multiple
  filter combos in one call.
- `get_thread_batch(session_id, thread_ids=[...])` — full message
  bodies for many threads. Per-thread payload includes `attachments`
  array with `{id, filename, mime_type}`.
- `get_attachment_batch(session_id, items=[{attachment_id, mode?}, ...])`
  — `mode="text"` (default) returns extracted PDF/docx/OCR text;
  `mode="meta"` returns just filename/mime/size; avoid
  `mode="rendered_pages"` (heavy base64 PNGs) unless text extraction
  is empty and you need the visual layout.
- `sql_query_batch(session_id, queries=[...])` — read-only SQL,
  many queries concurrently. ParadeDB BM25 is enforced server-side
  (LIKE/ILIKE on indexed columns is rejected). Call `describe_schema`
  first if unsure about column names.
- `find_facts(session_id, query, exhaustive?, k?)` — ENUMERATE every
  instance of an entity/attribute across the whole mailbox in ONE call
  (e.g. "all my license plates", "all my account numbers"). Use this
  for exhaustive "list ALL my X" questions instead of many
  `search_emails_batch` reformulations. Each returned fact carries a
  `message_id` back-pointer to cite/verify via `get_thread_batch`.
- `describe_schema(session_id)` — markdown docs for every queryable
  table. Cheap; call before writing a non-trivial sql_query.
- `publish_artifact_batch(session_id, items=[{path, name?,
  mime_type?}, ...])` — register files as part of the answer. Returns
  ids you cite as `[art:<id>]`. Files >10MB are rejected per item.
- `bash` — run shell/python inside your workspace to compute, chart
  (matplotlib is installed) and write files. Publish any file the user
  should see.

  **Rule: anything you produce that should appear in the user's
  answer must be published.** Files you write to disk are invisible
  to the user by default. Whether you produced the file via Bash, an
  external command, a download, or anything else, you must include
  it in a `publish_artifact_batch` call before citing `[art:<id>]`.

# Workflow

1. Briefly think about what evidence you need. Don't write a long plan
   upfront — just decide on the first move and go.
2. Retrieve. Use search / query / sql to find threads or aggregate counts.
3. Re-search if your first pass missed something. You can iterate freely.
4. Write the final answer in markdown.

# Parallelism — built into the tools

Each retrieval tool takes a list and runs every item concurrently in
ONE call. Wall clock for `sql_query_batch(queries=[q])` ≈
`sql_query_batch(queries=[q1, ..., q20])`. The way you parallelize is
by packing more items into each batch call — NOT by issuing many
single tool_use blocks per turn (the tools don't accept singles).

**Rule: before every assistant turn, ask "what are ALL the things I
need next?" — then pack them into a single batch call per tool.**

Concrete patterns:

- **Hypothesis fan-out.** Investigating "what happened with my Delta
  refund" → one `search_emails_batch` with 5 different queries
  (sender phrasing, subject keyword, body keyword, etc.).
- **Multiple SQL angles.** "Compare X across years" → one
  `sql_query_batch` with one query per year-bucket.
- **Thread fetches.** When `search_emails_batch` returns 6 candidate
  threads, fetch them all in one `get_thread_batch` call.
- **Mixed tools in parallel.** Bash, `sql_query_batch`, and
  `get_thread_batch` calls don't share state — emit them as
  multiple `tool_use` blocks in the SAME assistant turn when each
  answers a different piece of the question.

There is no sub-agent tool. Heavy fan-out goes through larger batch calls.

**Don't be conservative about volume.** Retrieval is cheap. A batch
of 10 searches that covers every angle beats 1 careful search that
misses something and forces a re-investigation.

The only reason to serialize across turns is when query N+1
*literally cannot be written* without query N's results.

# Citations — IMPORTANT

- Cite threads as `[ref:<cite_ref>]`, using the `cite_ref` field
  returned by `search_emails_batch` / `query_emails_batch`. Use the
  value EXACTLY as returned — do not shorten or truncate it.
- Cite artifacts as `[art:<id>]`, using the `id` returned by
  `publish_artifact_batch` for files you registered.
- Do NOT invent citation refs. Only use values that actually appeared in
  your tool results.
- If you couldn't find evidence, say so plainly. Don't guess.

# Output

Plain markdown. No JSON wrapper, no "Here's my analysis:" preamble. Just
the answer.

# Before you finish

Before you write your final answer, walk through this checklist:

1. List every file you produced during this turn (via bash, python,
   anything). Read your own tool history to count
   them.
2. For each file, decide: should the user see it?
   - Yes → confirm you called `publish_artifact` for it and got back an
     `id` you've cited as `[art:<id>]` in your answer.
   - No → it stays in the workspace as scratch. Fine.
3. If you find a file that should be user-visible but isn't published
   yet, publish it now BEFORE writing your final answer.

There IS a server-side safety net that auto-publishes any unpublished
file you wrote — but auto-published files lack the human-readable name
you'd give them. ALWAYS prefer publishing explicitly with a meaningful
name.
"""


# ── Settings (env, matching the GMAIL_CLAUDEBOX_* pattern) ────────────


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    try:
        v = float(raw) if raw else default
    except ValueError:
        return default
    return v if v > 0 else default


def pi_model() -> str:
    return os.environ.get("GMAIL_PI_MODEL") or _DEFAULT_MODEL


def pi_thinking() -> str | None:
    raw = os.environ.get("GMAIL_PI_THINKING", _DEFAULT_THINKING).strip().lower()
    return None if raw in ("", "off", "none") else raw


def pi_container() -> str:
    return os.environ.get("GMAIL_PI_CONTAINER") or _DEFAULT_CONTAINER


def pi_extension_path() -> str:
    return os.environ.get("GMAIL_PI_EXTENSION_PATH") or _DEFAULT_EXTENSION_PATH


def pi_builtin_tools() -> bool:
    """Whether pi's built-in tools (bash, file read/write, ...) are
    enabled for the turn. Default on; set `GMAIL_PI_BUILTIN_TOOLS` to
    one of 0/false/no/off (any case) to disable."""
    raw = os.environ.get("GMAIL_PI_BUILTIN_TOOLS", "").strip().lower()
    return raw not in _BUILTIN_TOOLS_OFF


def hard_timeout_seconds() -> float:
    return _env_float("GMAIL_PI_HARD_TIMEOUT", _DEFAULT_HARD_TIMEOUT)


def idle_timeout_seconds() -> float:
    return _env_float("GMAIL_PI_IDLE_TIMEOUT", _DEFAULT_IDLE_TIMEOUT)


def session_path_for(conversation_id: str | None) -> str | None:
    """Deterministic in-container session file per conversation. No
    conversation (one-off probes) → ephemeral run."""
    if not conversation_id or not _SESSION_ID_RE.match(conversation_id):
        return None
    return f"/sessions/{conversation_id}.jsonl"


async def _spawn_client(argv: list[str]) -> PiRpcClient:
    return await PiRpcClient.spawn(argv)


# ── Per-conversation serialization ─────────────────────────────────
#
# Two pi processes writing the same `--session` file concurrently
# corrupt the transcript (spike finding). One lock per conversation_id
# ensures turns for the same conversation run strictly one at a time.

_conversation_locks: dict[str, asyncio.Lock] = {}


def _lock_for(conversation_id: str | None) -> asyncio.Lock:
    if conversation_id is None:
        return asyncio.Lock()
    if conversation_id not in _conversation_locks:
        _conversation_locks[conversation_id] = asyncio.Lock()
    return _conversation_locks[conversation_id]


# ── Turn loop ───────────────────────────────────────────────────────


@dataclass
class TurnOutcome:
    final_text: str
    local_tool_calls: list[dict] = field(default_factory=list)
    usage: pp.UsageStats | None = None


class _TurnState:
    def __init__(self) -> None:
        self.final_text = ""
        self.local_tool_calls: list[dict] = []
        self.open_bash: dict[str, dict] = {}
        self.stop_reason: str | None = None
        self.error_message: str | None = None


ToolEventSink = Callable[[str, dict], Awaitable[None]]


def _raise_if_no_answer(state: _TurnState) -> None:
    """Raise PiRpcError if pi stopped with error or has no final answer."""
    if state.stop_reason in ("error", "aborted"):
        msg = f"pi stopped with {state.stop_reason}: {state.error_message or 'no error message'}"
        raise PiRpcError(msg)
    if not state.final_text.strip():
        msg = f"pi finished without an assistant answer (stop reason: {state.stop_reason})"
        raise PiRpcError(msg)


async def drive_turn(
    client,
    question: str,
    *,
    on_tool_event: ToolEventSink,
    hard_timeout: float,
    idle_timeout: float,
) -> TurnOutcome:
    """Send the prompt, consume events until `agent_end`, then fetch
    usage. Raises PiRpcError on EOF, idle timeout or hard timeout; the
    caller aborts the client."""
    started = time.monotonic()
    state = _TurnState()
    await client.send({"type": "prompt", "message": question})
    while True:
        remaining = hard_timeout - (time.monotonic() - started)
        if remaining <= 0:
            raise PiRpcError(f"hard timeout after {hard_timeout:.0f}s")
        try:
            rec = await client.read_record(min(idle_timeout, remaining))
        except asyncio.TimeoutError as exc:
            raise PiRpcError(f"idle timeout: no event for {idle_timeout:.0f}s") from exc
        if rec is None:
            raise PiRpcError("pi exited before agent_end")
        if rec.get("type") == "agent_end":
            break
        await _handle_record(rec, state, on_tool_event)
    _raise_if_no_answer(state)
    usage = await _fetch_usage(client)
    return TurnOutcome(final_text=state.final_text, local_tool_calls=state.local_tool_calls, usage=usage)


async def _handle_record(rec: dict, state: _TurnState, on_tool_event: ToolEventSink) -> None:
    kind = rec.get("type")
    if kind == "tool_execution_start":
        await on_tool_event("tool_call", pp.tool_call_args_entry(rec))
        if rec.get("toolName") == "bash":
            state.open_bash[str(rec.get("toolCallId"))] = rec
    elif kind == "tool_execution_end":
        await on_tool_event("tool_call", pp.tool_call_response_entry(rec))
        start = state.open_bash.pop(str(rec.get("toolCallId")), None)
        if start is not None:
            state.local_tool_calls.extend(pp.bash_as_run_code(start, rec))
    elif kind == "message_end":
        text = pp.assistant_text(rec)
        if text:
            state.final_text = text
        stop_reason, error_message = pp.assistant_stop(rec)
        if stop_reason is not None:
            state.stop_reason = stop_reason
        if error_message is not None:
            state.error_message = error_message
    elif kind == "extension_error":
        logger.error("pi extension error: %s", rec)


async def _fetch_usage(client) -> pp.UsageStats | None:
    try:
        resp = await client.request({"type": "get_session_stats"}, timeout=_STATS_TIMEOUT)
    except (PiRpcError, asyncio.TimeoutError) as exc:
        logger.warning("get_session_stats failed (cost not recorded): %s", exc)
        return None
    return pp.usage_from_stats_response(resp)


# ── Glue: DB events, side channel, cost ─────────────────────────────


def _make_tool_event_sink(conn, session_id: str) -> ToolEventSink:
    async def _sink(kind: str, payload: dict) -> None:
        try:
            append_event(conn, session_id=session_id, agent_name=AGENT_NAME, kind=kind, payload=payload)
        except Exception:
            logger.exception("streaming append_event failed for session %s", session_id)

    return _sink


def _report_cost(cost_sink, model: str, usage: pp.UsageStats | None) -> None:
    if cost_sink is None or usage is None:
        return
    try:
        cost_sink(
            agent_name=AGENT_NAME,
            model=model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            usd_override=usage.cost_usd,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("cost_sink failed (non-fatal): %s", exc)


async def _side_channel_tool_calls(session_id: str) -> list[dict]:
    from gmail_search.agents import runtime_claude as rc

    records = await rc._fetch_structured_tool_calls(session_id)
    return rc._tool_calls_from_side_channel(records)


def _build_argv(session_id: str, workspace: str, conversation_id: str | None, model: str) -> list[str]:
    return pp.build_pi_argv(
        container=pi_container(),
        session_id=session_id,
        workspace=workspace,
        session_path=session_path_for(conversation_id),
        extension_path=pi_extension_path(),
        model=model,
        thinking=pi_thinking(),
        system_prompt=PI_INSTRUCTION,
        builtin_tools=pi_builtin_tools(),
    )


async def _kill_stray_pi(session_path: str | None) -> None:
    """Killing the host-side `docker exec` client does not always kill
    the pi process inside the container. Best effort: pkill by the
    session path, which is unique per conversation. Ephemeral runs
    (no session path) are left to exit on their own."""
    if not session_path:
        return
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            pi_container(),
            "pkill",
            "-f",
            session_path,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.wait(), 10.0)
    except Exception as exc:  # noqa: BLE001
        logger.warning("pkill of stray pi for %s failed: %s", session_path, exc)


async def _run_turn(
    conn, *, session_id: str, workspace: str, conversation_id: str | None, question: str, model: str
) -> TurnOutcome:
    client = await _spawn_client(_build_argv(session_id, workspace, conversation_id, model))
    try:
        outcome = await drive_turn(
            client,
            question,
            on_tool_event=_make_tool_event_sink(conn, session_id),
            hard_timeout=hard_timeout_seconds(),
            idle_timeout=idle_timeout_seconds(),
        )
    except BaseException:
        await client.abort_and_close(grace=_ABORT_GRACE)
        await _kill_stray_pi(session_path_for(conversation_id))
        raise
    await client.close()
    return outcome


def _finish_ok(
    conn, *, session_id, workspace, conversation_id, turn_started_at, outcome: TurnOutcome, side_calls: list[dict]
) -> None:
    all_calls = side_calls + outcome.local_tool_calls
    emit_retriever_events(conn, session_id, all_calls, skip_per_tool_emission=True)
    emit_analyst_events(conn, session_id, all_calls, skip_per_tool_emission=True)
    final_text = sweep_and_extend_final_text(
        conn,
        session_id=session_id,
        workspace=workspace,
        conversation_id=conversation_id,
        turn_started_at=turn_started_at,
        base_text=outcome.final_text,
    )
    emit_writer_and_final(conn, session_id, final_text)
    finalize_session(conn, session_id, status="done", final_answer=final_text)


def _finish_error(conn, session_id: str, exc: BaseException) -> None:
    logger.exception("pi_run failed for session %s: %s", session_id, exc)
    try:
        emit_error(conn, session_id, exc, agent_name=AGENT_NAME)
    except Exception:
        logger.exception("failed to emit error event for %s", session_id)
    try:
        finalize_session(conn, session_id, status="error")
    except Exception:
        logger.exception("failed to finalize session %s on error path", session_id)


async def pi_run(
    *,
    db_path: Path,
    session_id: str,
    workspace: str,
    conversation_id: str | None,
    question: str,
    model: str | None,
    cost_sink: Callable[..., None] | None,
    user_id: str | None = None,
) -> None:
    """Run one deep-mode turn through pi. Same contract as
    `runtime_claude_native.native_run` minus resume plumbing: the
    session file path is derived from `conversation_id`.

    Holds a per-conversation lock for the whole turn: two pi processes
    on one `--session` file corrupt the transcript (spike finding)."""
    from gmail_search.agents import runtime_claude as rc

    async with _lock_for(conversation_id):
        turn_started_at = time.time()
        resolved_model = model or pi_model()
        conn = get_connection(db_path)
        registered = False
        try:
            await rc.register_session_via_admin(
                session_id, evidence_records=None, conversation_id=conversation_id, workspace=workspace, user_id=user_id
            )
            registered = True
            emit_plan_event(conn, session_id, agent_name=AGENT_NAME, approach="single pi agent loop with all tools")
            outcome = await _run_turn(
                conn,
                session_id=session_id,
                workspace=workspace,
                conversation_id=conversation_id,
                question=question,
                model=resolved_model,
            )
            _report_cost(cost_sink, resolved_model, outcome.usage)
            side_calls = await _side_channel_tool_calls(session_id)
            _finish_ok(
                conn,
                session_id=session_id,
                workspace=workspace,
                conversation_id=conversation_id,
                turn_started_at=turn_started_at,
                outcome=outcome,
                side_calls=side_calls,
            )
        except Exception as exc:  # noqa: BLE001
            _finish_error(conn, session_id, exc)
        finally:
            if registered:
                try:
                    await rc.unregister_session_via_admin(session_id)
                except Exception:
                    logger.exception("unregister_session failed for %s", session_id)
            try:
                conn.close()
            except Exception:
                logger.exception("closing conn for session %s failed", session_id)
