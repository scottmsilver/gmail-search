"""Single-agent Claude Code loop for the deep-analysis pipeline.

Unlike the orchestrator backend (`adk` / `claude_code`) which fans the
turn into a Planner → Retriever → Analyst → Writer → Critic chain, this
backend runs ONE Claude Code invocation with all five MCP tools
available and synthesizes the same `agent_events` shape the UI already
renders. Net effect: identical UX, fewer round trips, no critic loop.

Public entrypoint is `native_run()` — service.py invokes it inside the
existing _real_run branching.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

from gmail_search.agents.deep_events import RETRIEVAL_TOOL_NAMES as _RETRIEVAL_TOOL_NAMES  # noqa: F401
from gmail_search.agents.deep_events import emit_analyst_events as _emit_analyst_events
from gmail_search.agents.deep_events import emit_error as _emit_error
from gmail_search.agents.deep_events import emit_retriever_events as _emit_retriever_events
from gmail_search.agents.deep_events import emit_writer_and_final as _emit_writer_and_final
from gmail_search.agents.deep_events import sweep_and_extend_final_text as _sweep_and_extend_final_text
from gmail_search.agents.session import append_event, finalize_session
from gmail_search.store.db import get_connection

logger = logging.getLogger(__name__)


CostSink = Callable[..., None]


NATIVE_INSTRUCTION = """You are a deep-analysis agent over the user's personal Gmail archive. Your
job is to answer one question with grounded, cited reasoning.

# Tools

Every tool below takes a LIST as its main argument — even when you have
just one item, you pass a one-item list. There are no single-call
versions; that's deliberate, to keep you in batch-mode by default.
Always pass the `session_id` provided in this prompt as the first arg.

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

Sub-agents via the `Task` tool are for HEAVIER independent
investigations (e.g., "audit every Delta thread for refund status"
runs as one Task, "audit every United thread" runs as a parallel
Task). Use `Task` when each branch needs its own context window.

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

1. List every file you produced during this turn (via Bash, run_code,
   external commands, anything). Read your own tool history to count
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


def _build_native_agent(model: str | None) -> SimpleNamespace:
    """Build the lightweight agent object `claudebox_invoke` consumes.
    Matches the shape `_resolve_model` + `_instruction_with_session_binding`
    expect: `name`, `model`, `instruction`."""
    return SimpleNamespace(
        name="claude_native",
        instruction=NATIVE_INSTRUCTION,
        model=model or "sonnet",
    )


def _emit_plan_event(conn, session_id: str) -> None:
    from gmail_search.agents.deep_events import emit_plan_event

    emit_plan_event(conn, session_id, approach="single-agent claude code loop with all tools")


async def native_run(
    *,
    db_path: Path,
    session_id: str,
    workspace: str,
    conversation_id: str | None,
    question: str,
    model: str | None,
    cost_sink: CostSink | None,
    resume: str | None = None,
    on_session_uuid: Callable[[str], None] | None = None,
    user_id: str | None = None,
) -> None:
    """Run one deep-mode turn through a single Claude Code invocation.

    Registers the MCP session, fires ONE `claudebox_invoke` with all
    tools available, then synthesizes the orchestrator's event vocabulary
    (`plan` / `tool_call` / `evidence` / `code_run` / `analysis` /
    `draft` / `final`) so the existing SSE consumer in the UI keeps
    working unchanged. Always unregisters in finally.

    Imports `runtime_claude` lazily (function-local) so unit tests can
    monkeypatch `runtime_claude.{claudebox_invoke,register_session_via_admin,
    unregister_session_via_admin}` and have those substitutions visible
    here at call time."""
    import time as _time

    from gmail_search.agents import runtime_claude as rc

    # Capture wall time at the very top so the auto-publish sweep can
    # discriminate "files written this turn" from pre-existing scratch.
    turn_started_at = _time.time()

    conn = get_connection(db_path)
    registered = False
    try:
        await rc.register_session_via_admin(
            session_id,
            evidence_records=None,
            conversation_id=conversation_id,
            workspace=workspace,
            user_id=user_id,
        )
        registered = True
        _emit_plan_event(conn, session_id)
        agent = _build_native_agent(model)

        # Stream tool_call events to agent_events as the JSONL transcript
        # surfaces them. service.py's poller turns those rows into SSE
        # frames so the user sees tool calls in real time. We tag them
        # under "claude_native" — the UI's tool-call lane is keyed by
        # `kind=="tool_call"`, not by agent_name, so the existing
        # consumer renders them without a UI change.
        async def _stream_tool_call(kind: str, payload: dict) -> None:
            try:
                append_event(
                    conn,
                    session_id=session_id,
                    agent_name="claude_native",
                    kind=kind,
                    payload=payload,
                )
            except Exception:
                logger.exception(f"streaming append_event failed for session {session_id}")

        result = await rc.claudebox_invoke(
            agent,
            question,
            workspace=workspace,
            session_id=session_id,
            cost_sink=cost_sink,
            event_sink=_stream_tool_call,
            resume=resume,
        )
        # First-turn UUID capture: when this turn ran without `resume`
        # (i.e. the conversation hadn't yet pinned a Claude session
        # UUID), surface the UUID claudebox returned so the caller can
        # persist it for subsequent turns to `--resume` against.
        if resume is None and result.claude_session_uuid and on_session_uuid is not None:
            try:
                on_session_uuid(result.claude_session_uuid)
            except Exception:
                logger.exception(f"on_session_uuid callback raised for session {session_id}")
        # Skip per-tool emission downstream: the JSONL tailer already
        # streamed every tool_use as a `tool_call` event. The aggregate
        # `evidence` / `analysis` events still fire so the UI's
        # retriever/analyst panels populate.
        _emit_retriever_events(conn, session_id, result.tool_calls, skip_per_tool_emission=True)
        _emit_analyst_events(conn, session_id, result.tool_calls, skip_per_tool_emission=True)
        # End-of-turn auto-publish sweep: catch any file Claude wrote
        # without explicitly publishing it, append chips to the final
        # text so the UI's `[art:<id>]` renderer surfaces them inline.
        # Both `draft` and `final` events carry the extended text — the
        # UI may consume either.
        final_text = _sweep_and_extend_final_text(
            conn,
            session_id=session_id,
            workspace=workspace,
            conversation_id=conversation_id,
            turn_started_at=turn_started_at,
            base_text=result.text,
        )
        _emit_writer_and_final(conn, session_id, final_text)
        finalize_session(conn, session_id, status="done", final_answer=final_text)
    except Exception as exc:  # noqa: BLE001
        logger.exception(f"native_run failed for session {session_id}: {exc}")
        try:
            _emit_error(conn, session_id, exc)
        except Exception:
            logger.exception(f"failed to emit error event for {session_id}")
        try:
            finalize_session(conn, session_id, status="error")
        except Exception:
            logger.exception(f"failed to finalize session {session_id} on error path")
    finally:
        if registered:
            try:
                await rc.unregister_session_via_admin(session_id)
            except Exception:
                logger.exception(f"unregister_session failed for {session_id}")
        try:
            conn.close()
        except Exception:
            logger.exception(f"closing conn for session {session_id} failed")
