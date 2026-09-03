"""Single-agent deep-analysis turn through the Pi agent harness.

One `pi --mode rpc` process per turn, inside the `pi-sandbox`
container, driven over stdin/stdout. Tool calls stream to
`agent_events` as they happen; the MCP side channel supplies the full
structured responses afterwards, exactly as `claude_native` does.
Public entry point: `pi_run()`.
"""

from __future__ import annotations

import asyncio
import json
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
# The pi-mcp-adapter extension (an npm package baked into the pi
# sandbox image) replaces our own gmail-tools bridge extension: it
# reaches the MCP tools server directly over HTTP using the per-turn
# session token (see `_write_session_token_file`) instead of us
# bridging tool calls ourselves.
_DEFAULT_EXTENSION_PATH = "/opt/pi-pkgs/node_modules/pi-mcp-adapter"
_DEFAULT_HARD_TIMEOUT = 900.0
_DEFAULT_IDLE_TIMEOUT = 300.0
_ABORT_GRACE = 5.0
_STATS_TIMEOUT = 15.0
# Cap on interim assistant prose forwarded as an `assistant` event, so a
# long planning ramble doesn't blow up the event payload.
_ASSISTANT_TEXT_CLIP_CHARS = 4000
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_BUILTIN_TOOLS_OFF = {"0", "false", "no", "off"}
_DEFAULT_MCP_CONFIG_PATH = "/opt/gmail-mcp.json"
# Same host path `service._ensure_workspace_dir` writes claudebox
# workspaces under, relative to the daemon's cwd (project root in
# dev/test). The session-token file rides the same host bind-mount the
# workspace itself uses, so pi-mcp-adapter can read it in-container.
_DEFAULT_WORKSPACES_ROOT = "deploy/claudebox/workspaces"
_SESSION_TOKEN_FILENAME = ".session-token"
# Per-turn TMPDIR for the pi-mcp-adapter's output-guard spill files
# (see deploy/pi/README.md). Rides the same host bind-mount as the
# workspace itself, so spills are scoped to the turn and pruned with it
# instead of accumulating in the shared container's /tmp.
_TMPDIR_DIRNAME = ".tmp"
# Margin added on top of the turn's hard timeout so the session token
# doesn't expire mid-turn on a run that's right at the timeout edge.
_SESSION_TOKEN_TTL_MARGIN_SECONDS = 120

# Context window by model-name prefix, longest/most-specific first. Used
# to compute the reading budget injected into the system prompt.
_CONTEXT_WINDOW_BY_PREFIX = (
    ("google/gemini-3", 1_048_576),
    ("google/gemini-2.5", 1_048_576),
    ("anthropic/", 200_000),
    ("openai/", 400_000),
)
_DEFAULT_CONTEXT_WINDOW = 200_000
_READING_BUDGET_FRACTION = 0.30

PI_INSTRUCTION = """You are a deep-analysis agent over the user's personal Gmail archive. You
answer one question with grounded, cited reasoning. You plan your reading:
the archive is far larger than your context window, so you decide what to
look at before you look at it.

# Your budget

Your context window is {context_window} tokens and it fills with tool
results. Your reading budget for this turn is {reading_budget} tokens
(30% of the window); the rest is headroom for reasoning, compaction and
the answer. Nothing stops you from overspending except you — a fetch that
overflows the window ends the turn with an error and no answer.

Before an expensive read, think about what it will cost against what is
left of the budget, using the table below, and decide how much of that
budget this read deserves. When you cannot estimate a call — you do not
know how many messages a thread has or how long they are — an outline
with SQL (message count, body lengths) is nearly free and tells you.

Rough prices, so you can plan:

| Call | Cost |
|---|---|
| `gmail_search_emails_batch` with `detail="refs"` | ~50 tokens per thread |
| `gmail_search_emails_batch` with `detail="snippet"` (default) | ~300 per thread |
| `gmail_search_emails_batch` with `detail="summary"` | ~500 per thread |
| `gmail_search_emails_batch` with `detail="full"` | 2,000–20,000 per matched message |
| `gmail_get_thread_batch` | 5,000–50,000 per thread (every message, bodies up to 20k chars each) |
| `gmail_sql_query_batch` | ~50–200 per row returned (500-row cap per query) |
| `gmail_find_facts` | ~100 per fact |
| `gmail_get_attachment_batch` `mode="text"` | 2,000–50,000 per attachment; `mode="raw"` and `rendered_pages` are far larger |

Cheap calls are on the left of that table; expensive ones on the right.
Spend freely on the cheap ones and deliberately on the expensive ones.

# Tools

Every tool takes a LIST as its main argument, even for one item, and runs
every item concurrently. There are no single-item versions. A result too
large to return inline comes back as `[MCP text output truncated: original
N lines / K KiB. Full text saved to: <path>]` — `read` or `grep` that path
instead of re-running the call with a bigger `top_k`/`limit`.

- `gmail_search_emails_batch(searches=[{query, date_from?, date_to?, top_k?,
  detail?, max_matches?}, ...])` — semantic search. Each result thread has
  a `cite_ref`. `detail` picks how much of each matched message you get:
  `refs` (one line per thread), `snippet` (default), `summary` (one-line
  LLM summary per matched message), `full` (whole body per matched
  message). `max_matches` caps matched messages per thread (default 3).
- `gmail_query_emails_batch(filters=[{sender?, subject_contains?, date_from?,
  date_to?, label?, has_attachment?, order_by?, limit?}, ...])` —
  structured metadata filter, no ranking. Cheap.
- `gmail_sql_query_batch(queries=[...])` — read-only SQL against the messages
  DB. Your precision instrument: outline a thread (`SELECT id, from_addr,
  date, subject, length(body_text) FROM messages WHERE thread_id = ...`),
  read part of one message (`substr(body_text, 1, 3000)`), count and
  aggregate. Free-text goes through BM25: `WHERE id @@@ 'subject:credit'`;
  `LIKE`/`ILIKE` on indexed columns is rejected. Call `gmail_describe_schema`
  first if unsure about columns.
- `gmail_find_facts(query, exhaustive?, k?)` — enumerate every instance of an
  entity or attribute across the whole mailbox in one call ("all my
  account numbers", "every hotel I stayed at"). Each fact carries a
  `message_id` to cite or verify.
- `gmail_get_thread_batch(thread_ids=[...])` — every message of each thread,
  bodies clipped at 20k chars, plus the attachment manifest. The most
  expensive read you have; its cost is the sum of every message in every
  thread you list. Use it for threads you have already chosen and sized,
  never as a way to look around.
- `gmail_get_attachment_batch(items=[{attachment_id, mode?}, ...])` —
  `mode="text"` (default) returns extracted text; `mode="meta"` just
  filename/mime/size. Do not use `mode="raw"` or `mode="rendered_pages"`
  unless text extraction came back empty and you need the visual layout,
  and then one attachment at a time.
- `gmail_describe_schema()` — column docs for every queryable table. Cheap.
- `gmail_publish_artifact_batch(items=[{path, name?, mime_type?}, ...])` —
  register files as part of the answer; returns ids you cite as
  `[art:<id>]`. Files over 10MB are rejected.
- `bash` — shell and python inside your workspace, for computing,
  charting (matplotlib is installed) and writing files. Anything the user
  should see must be published with `gmail_publish_artifact_batch`.
- `mcpScript` — run a small JavaScript program that calls the gmail tools
  in a loop, filters or aggregates the results, and returns only the final
  value, so intermediate results never enter your context. Use it when a
  question needs many calls whose raw output you do not need to read (for
  example: for each of 40 threads, return only the date and the amount).
  Inside a script, call the tools by their prefixed names, e.g.
  `gmail_search_emails_batch`.

# Workspace and programming tools

You have a persistent workspace at `/workspaces/<name>` — your current
directory — shared by every turn of this conversation, so files you wrote
in an earlier turn are still there. Besides the retrieval tools you have
the coding tools `read`, `write` and `edit` for files in the workspace,
and `bash` for shell and Python 3 — use `bash` for listing, finding and
searching files (`ls`, `find`, `grep`) as well as running code. Installed:
pandas, numpy, matplotlib (Agg backend), openpyxl, python-docx, pypdf,
requests, curl, jq, ripgrep and git. There is no package installation and
no internet access; the only network you have is the tool server.

Work with attachments as files, not as text in your context. To get a
spreadsheet, PDF or document onto disk, call `gmail_get_attachment_batch`
with `mode="raw"` and `inline=false`; the result carries a signed
`fetch_url` that works for about fifteen minutes. Download it with
`curl -sSL -o <file> "<fetch_url>"` and parse it locally (openpyxl for
xlsx, pypdf for PDF, python-docx for docx). Never request inline base64
and never paste file contents into your reasoning; compute the answer in
code and cite the message the file came from.

Charts and tables come from Python: write the PNG or CSV to the workspace
and publish it. Keep intermediate files as scratch; publish only what the
user should see.

# How you work: map, then read

Work in phases. Each phase has its own tools and its own rule.

**Narrate as you go.** Before every tool call, write one plain sentence saying what you are about to do and why — the search you are running, the thread you are opening, the number you are computing. These sentences are shown to the user live as progress, so write them for a reader, not for yourself; keep them to one line and never restate tool arguments verbatim.

**1. Scope.** Before any call, state in one or two sentences what
evidence would settle the question: which senders, which period, which
kind of message or attachment, whether you need a list, an amount, or a
narrative.

**2. Map with cheap tools.** Find out what exists without reading it.
Fan out `gmail_search_emails_batch` across phrasings and date windows with
`detail="refs"` or `"snippet"`; use `gmail_find_facts` for anything shaped
like "all of my X"; use `gmail_sql_query_batch` for counts, date ranges,
senders, and thread outlines. Pack every angle you can think of into one
batch call per tool. This is the phase to be generous in: a batch of ten
cheap searches that covers every angle beats one careful search that
misses.

**3. Select: write a read plan.** When the map shows candidates, write a
short plan in prose before any expensive call: which threads or messages
you will read, why each one, and the rough cost from the table above. Two
or three lines is enough. This text is visible to the user and it is what
you keep if your context is compacted, so make it specific.

**4. Read what the plan named.** Prefer the narrowest tool that answers:
a `substr` of one message over a full body, `detail="summary"` over
`"full"`, `"full"` on a small `top_k` over `gmail_get_thread_batch`. When
you do need whole threads, decide how many to fetch at once from your own
cost estimate and how much budget you have left, most relevant first, and
reassess after each batch before spending more.

**5. Verify and answer.** Check that every claim has a citation from your
tool results. If the budget ran out before you read everything the map
suggested, answer from what you have and say plainly what you did not
read.

# Rules

- One sentence of narration before each tool call; no narration-free tool calls except when a call immediately follows a truncation notice and you are simply narrowing it.
- Think about cost before every expensive read, and treat unknown size
  as expensive: outline first, then read.
- If a result comes back truncated, narrow the next call (fewer ids, a
  `substr`, a smaller `top_k`). Never retry the same call bigger.
- After each expensive read, reassess before the next one. Cheap calls
  can be batched freely in the same turn.
- Do not re-fetch something already in your context. Read your own tool
  history first.
- Every batch call needs its items; a batch of one is fine when that is
  all you need.

# Playbooks by question shape

| Question looks like | Map with | Then read |
|---|---|---|
| "List all my X" / "every time I…" | `gmail_find_facts`, then SQL to verify counts | Spot-check two or three cited messages with `substr` |
| "How much did I pay / receive from X" | SQL over `messages` (and `attachments`) filtered by sender and date; `gmail_query_emails_batch` for invoices with attachments | Targeted `gmail_get_attachment_batch` `mode="text"` on the specific invoices or statements |
| "What happened with X" / "status of X" | `gmail_search_emails_batch` fan-out with `detail="summary"` | Two or three threads with `gmail_get_thread_batch`, most recent first |
| "When did X happen" / "who did I talk to about X" | SQL counts and date ranges; `refs` searches | Usually nothing more; cite the rows |
| "Plot / compute / compare" | SQL aggregates straight into rows | `bash` to chart and publish |

# Batching

Every batch tool runs its items concurrently, so one call with twenty
items costs the same wall time as one item. Parallelize by packing more
items into a batch call, not by issuing many single tool calls. The only
reason to serialize across turns is when the next call literally cannot
be written without the previous result — which is exactly the map-then-
read boundary. There is no sub-agent tool.

# Citations

- Cite threads as `[ref:<cite_ref>]` using the `cite_ref` field from
  `gmail_search_emails_batch` or `gmail_query_emails_batch`, exactly as
  returned.
- Cite artifacts as `[art:<id>]` using the id from
  `gmail_publish_artifact_batch`.
- Never invent a citation. Only use values that appeared in your tool
  results.
- If you could not find evidence, say so plainly. Do not guess.

# Output

Plain markdown. No JSON wrapper, no preamble. Lead with the answer; put
the evidence under it.

# Before you finish

1. List every file you produced this turn (bash, python, anything).
2. For each file, decide whether the user should see it. If yes, confirm
   you published it and cited its `[art:<id>]`. If no, leave it as
   scratch.
3. Publish anything user-visible that is not yet published, then write
   the answer.

A server-side safety net auto-publishes unpublished files, but without
the readable name you would give them. Publish explicitly.
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


def context_window_for(model: str) -> int:
    """Token budget for `model`. `GMAIL_PI_CONTEXT_WINDOW` overrides
    everything when set to a valid int; otherwise looked up by
    model-name prefix, falling back to `_DEFAULT_CONTEXT_WINDOW`."""
    raw = os.environ.get("GMAIL_PI_CONTEXT_WINDOW")
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    for prefix, window in _CONTEXT_WINDOW_BY_PREFIX:
        if model.startswith(prefix):
            return window
    return _DEFAULT_CONTEXT_WINDOW


def _reading_budget_for(model: str) -> int:
    return int(context_window_for(model) * _READING_BUDGET_FRACTION)


def render_instruction(model: str) -> str:
    """`PI_INSTRUCTION` with `{context_window}`/`{reading_budget}` filled
    in. Uses `str.replace` rather than `str.format` — the prompt has
    literal `{`/`}` in its code examples that `.format` would choke on."""
    text = PI_INSTRUCTION.replace("{context_window}", f"{context_window_for(model):,}")
    return text.replace("{reading_budget}", f"{_reading_budget_for(model):,}")


def pi_thinking() -> str | None:
    raw = os.environ.get("GMAIL_PI_THINKING", _DEFAULT_THINKING).strip().lower()
    return None if raw in ("", "off", "none") else raw


def pi_container() -> str:
    return os.environ.get("GMAIL_PI_CONTAINER") or _DEFAULT_CONTAINER


def pi_extension_path() -> str:
    return os.environ.get("GMAIL_PI_EXTENSION_PATH") or _DEFAULT_EXTENSION_PATH


def pi_mcp_config_path() -> str:
    """In-container path to the MCP config pi-mcp-adapter reads to
    reach the MCP tools server, passed via `--mcp-config`."""
    return os.environ.get("GMAIL_PI_MCP_CONFIG") or _DEFAULT_MCP_CONFIG_PATH


def _workspaces_root() -> Path:
    return Path(os.environ.get("GMAIL_PI_WORKSPACES_ROOT") or _DEFAULT_WORKSPACES_ROOT)


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


def _write_session_token_file(workspace: str, token: str) -> Path:
    """Write the turn's session token to `<workspace>/.session-token`,
    mode 0600, so pi-mcp-adapter (running in-container against the same
    bind-mounted workspace) can read it without the token ever passing
    through argv or a logged env var. Never log `token` itself."""
    path = _workspaces_root() / workspace / _SESSION_TOKEN_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        os.fchmod(fd, 0o600)
        os.write(fd, token.encode("utf-8"))
    finally:
        os.close(fd)
    return path


def _in_container_tmpdir(workspace: str) -> str:
    """In-container TMPDIR path for a turn's workspace."""
    return f"/workspaces/{workspace}/{_TMPDIR_DIRNAME}"


def _ensure_workspace_tmp_dir(workspace: str) -> Path:
    """Create `<workspaces_root>/<workspace>/.tmp` (0700) on the host —
    the same host bind-mount the session-token file uses — so the
    pi-mcp-adapter's output-guard spill files land inside this turn's
    own workspace instead of the shared sandbox container's `/tmp`.
    Idempotent: re-asserts 0700 even if the directory already existed
    with looser permissions from a prior run."""
    path = _workspaces_root() / workspace / _TMPDIR_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path


def _remove_session_token_file(path: Path) -> None:
    """Best-effort cleanup in `pi_run`'s `finally`. Must never raise —
    a failed unlink is a leaked-but-expired file, not a turn failure."""
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:  # noqa: BLE001
        logger.warning("could not remove session token file %s: %s", path, exc)


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
        # Set on every assistant `message_end`; flushed as an
        # `assistant` event the next time a tool call starts, so only
        # prose followed by more tool activity is surfaced (the last
        # message stays the final answer, not a duplicated event).
        self.pending_text: str | None = None


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


async def _flush_pending_text(state: _TurnState, on_tool_event: ToolEventSink) -> None:
    """Emit any assistant prose queued since the last tool call as an
    `assistant` event, then clear it. Called right before a new tool
    call starts, so interim reasoning/plans show up in event order
    alongside the tool calls they preceded."""
    text = state.pending_text
    if not text:
        return
    state.pending_text = None
    payload = {
        "text": text[:_ASSISTANT_TEXT_CLIP_CHARS],
        "truncated": len(text) > _ASSISTANT_TEXT_CLIP_CHARS,
    }
    await on_tool_event("assistant", payload)


async def _handle_record(rec: dict, state: _TurnState, on_tool_event: ToolEventSink) -> None:
    kind = rec.get("type")
    if kind == "tool_execution_start":
        await _flush_pending_text(state, on_tool_event)
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
            state.pending_text = text
        stop_reason, error_message = pp.assistant_stop(rec)
        if stop_reason is not None:
            state.stop_reason = stop_reason
        if error_message is not None:
            state.error_message = error_message
    elif kind == "extension_error":
        logger.error("pi extension error: %s", pp.redact_secrets(json.dumps(rec)))


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
        system_prompt=render_instruction(model),
        builtin_tools=pi_builtin_tools(),
        mcp_config_path=pi_mcp_config_path(),
        tmpdir=_in_container_tmpdir(workspace),
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
    if client.killed:
        await _kill_stray_pi(session_path_for(conversation_id))
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


async def _install_session_token(rc, session_id: str, workspace: str) -> Path:
    """Mint the turn's /mcp session token and write it into the
    workspace for pi-mcp-adapter to read. TTL is the turn's hard
    timeout plus a margin, so the token can't expire mid-turn on a run
    that lands right at the timeout edge."""
    ttl_seconds = int(hard_timeout_seconds()) + _SESSION_TOKEN_TTL_MARGIN_SECONDS
    token = await rc.mint_session_token_via_admin(session_id, ttl_seconds=ttl_seconds)
    return _write_session_token_file(workspace, token)


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
        token_path: Path | None = None
        try:
            await rc.register_session_via_admin(
                session_id, evidence_records=None, conversation_id=conversation_id, workspace=workspace, user_id=user_id
            )
            registered = True
            token_path = await _install_session_token(rc, session_id, workspace)
            _ensure_workspace_tmp_dir(workspace)
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
            if token_path is not None:
                _remove_session_token_file(token_path)
            if registered:
                try:
                    await rc.unregister_session_via_admin(session_id)
                except Exception:
                    logger.exception("unregister_session failed for %s", session_id)
            try:
                conn.close()
            except Exception:
                logger.exception("closing conn for session %s failed", session_id)
