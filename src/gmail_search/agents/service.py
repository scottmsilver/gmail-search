"""HTTP surface for the deep-analysis agent.

Lives inside the main gmail-search FastAPI app (wired in by
`create_app` when ADK / agent deps are available). Two endpoints:

  POST /api/agent/analyze  — kicks off a deep-mode turn. Returns an
                             SSE stream of `agent_events` as the
                             sub-agents run. Phase 1 implementation
                             is a STUB: it fires a handful of fake
                             events to exercise the persistence +
                             streaming pipe. Real ADK orchestration
                             lands in Phase 4.
  GET  /api/artifact/<id>  — returns the artifact bytes (plot PNG,
                             CSV, etc.) cited by [art:<id>] in the
                             final answer.

Events are persisted to `agent_events` as they're produced; the SSE
handler polls the DB by seq so a disconnected client can resume at
`?after=<last_seq>`.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from gmail_search.agents.session import (
    append_event,
    create_session,
    fetch_events_after,
    finalize_session,
    get_artifact,
    new_session_id,
)
from gmail_search.store.db import get_connection


def _use_real_pipeline() -> bool:
    """`GMAIL_DEEP_REAL=1` flips from the Phase-1 stub to the live
    ADK pipeline. Off by default so a fresh install doesn't make a
    live Gemini call the moment you POST to /api/agent/analyze."""
    import os

    return os.environ.get("GMAIL_DEEP_REAL", "").lower() in ("1", "true", "yes")


logger = logging.getLogger(__name__)


class AnalyzeRequest(BaseModel):
    """Input shape for POST /api/agent/analyze.

    Intentionally narrow for v1: one question per turn, no prior
    conversation history (the planner decides what retrieval it
    needs). Multi-turn deep sessions are a v2 concern.
    """

    conversation_id: str | None = None
    question: str
    # When None, each sub-agent falls back to its env var
    # (GMAIL_PLANNER_MODEL etc.) or its hardcoded default. Set this
    # to override every stage with the same model — the path the
    # web picker takes when the user picks a non-default model for
    # deep mode.
    model: str | None = None
    # Which runtime adapter to use: `adk` (Gemini via google-adk) or
    # `claude_code` (claudebox HTTP + MCP side-channel). When None,
    # falls back to the `GMAIL_DEEP_BACKEND` env var (default `adk`).
    backend: str | None = None


def _sse(event_type: str, data: dict) -> str:
    """Format one SSE message frame. The Next.js proxy forwards these
    verbatim; the client reads `event:` to route and `data:` for the
    payload JSON."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def _emit_auto_published_event(
    conn,
    *,
    session_id: str,
    workspace: str | None,
    conversation_id: str | None,
    turn_started_at: float,
) -> list[int]:
    """Run the auto-publish sweep AFTER the orchestrator's final event
    has fired and emit a follow-up `auto_published` event with the
    artifact ids. The orchestrator path is single-shot from the
    Writer's perspective — its `final` text is already in the SSE
    stream — so we can't mutate it; the UI consumes the new event
    type to surface "also produced" files.

    Returns the list of newly-published artifact ids (empty on no-op
    or failure). Never raises — the sweep is a safety net."""
    from gmail_search.agents.auto_publish import auto_publish_unpublished_files
    from gmail_search.agents.session import append_event

    try:
        published = auto_publish_unpublished_files(
            conn,
            session_id=session_id,
            workspace=workspace,
            conversation_id=conversation_id,
            turn_started_at=turn_started_at,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("auto_publish sweep raised for session %s: %s", session_id, exc)
        return []
    if not published:
        return []
    artifact_ids = [int(row["id"]) for row in published]
    note = (
        "Files produced during this turn but not explicitly published "
        "by the agent. They're available via /api/artifact/<id>."
    )
    try:
        append_event(
            conn,
            session_id=session_id,
            agent_name="root",
            kind="auto_published",
            payload={
                "artifact_ids": artifact_ids,
                "artifacts": published,
                "note": note,
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "auto_publish: append_event failed for session %s: %s",
            session_id,
            exc,
        )
    return artifact_ids


async def _stub_run(db_path: Path, session_id: str, question: str) -> AsyncIterator[str]:
    """Phase-1 placeholder: emit a few fake events so the UI + SSE +
    persistence pipe can be developed and tested end-to-end BEFORE the
    real ADK agents land. Each event is both persisted (so a reload
    replays it) and streamed (so the live UI updates).

    Replace this function with the real root-agent `.run()` in Phase 4.
    """
    conn = get_connection(db_path)
    try:
        # Planner fake: restates the question as a trivial plan so the
        # panel has something to render.
        seq = append_event(
            conn,
            session_id=session_id,
            agent_name="planner",
            kind="plan",
            payload={
                "steps": [
                    {"agent": "retriever", "note": "(stub) pull top-N matches"},
                    {"agent": "analyst", "note": "(stub) no-op"},
                    {"agent": "writer", "note": "(stub) echo"},
                ],
                "question": question,
            },
        )
        yield _sse("plan", {"seq": seq, "question": question})
        await asyncio.sleep(0.05)

        # Retriever fake
        seq = append_event(
            conn,
            session_id=session_id,
            agent_name="retriever",
            kind="tool_result",
            payload={"note": "stub", "threads_found": 0},
        )
        yield _sse("retriever", {"seq": seq, "threads_found": 0})
        await asyncio.sleep(0.05)

        # Writer fake
        draft = f"(stub) I received: {question!r}. Phase 1 scaffolding only."
        seq = append_event(
            conn,
            session_id=session_id,
            agent_name="writer",
            kind="draft",
            payload={"text": draft},
        )
        yield _sse("draft", {"seq": seq, "text": draft})
        await asyncio.sleep(0.05)

        # Critic fake: always passes in the stub
        seq = append_event(
            conn,
            session_id=session_id,
            agent_name="critic",
            kind="critique",
            payload={"accepted": True, "notes": []},
        )
        yield _sse("critique", {"seq": seq, "accepted": True})

        # Final
        seq = append_event(
            conn,
            session_id=session_id,
            agent_name="root",
            kind="final",
            payload={"text": draft},
        )
        yield _sse("final", {"seq": seq, "text": draft})
        finalize_session(conn, session_id, status="done", final_answer=draft)
    except Exception as e:
        logger.exception(f"stub run failed for session {session_id}: {e}")
        finalize_session(conn, session_id, status="error")
        yield _sse("error", {"message": str(e)})
    finally:
        conn.close()


_VALID_BACKENDS = ("adk", "claude_code", "claude_native")


def _deep_backend(override: str | None = None) -> str:
    """Pick the runtime adapter for the deep pipeline.

    Order of precedence:
      1. `override` (typically `AnalyzeRequest.backend` from the UI),
      2. `GMAIL_DEEP_BACKEND` env (operator-level default),
      3. `adk` (project default — preserves pre-claude_code behaviour).

    Unrecognised values fall back to `adk` rather than raising — the
    UI is free to send fresh values; we don't want one malformed
    request to crash the deep route. The frontend's request schema
    keeps the field as a free string for the same reason."""
    import os as _os

    candidates = (
        (override or "").strip().lower(),
        _os.environ.get("GMAIL_DEEP_BACKEND", "").strip().lower(),
    )
    for c in candidates:
        if c in _VALID_BACKENDS:
            return c
    return "adk"


def _claudebox_workspace_for(session_id: str) -> str:
    """One claudebox workspace per deep-mode turn so concurrent
    stages can't collide on file mutations."""
    return f"deep-{session_id}"


def _ensure_workspace_dir(workspace: str) -> None:
    """Claudebox refuses to run in a workspace it can't see on disk.
    Create it on the host filesystem before the first invoke."""
    base = Path("deploy/claudebox/workspaces") / workspace
    base.mkdir(parents=True, exist_ok=True)


_HISTORY_MAX_TURNS = 4
_HISTORY_MAX_CHARS = 8000


def _extract_text_from_parts_json(parts_raw: str) -> str:
    """Pull plain text out of the JSON-encoded parts column. Messages
    can have many block types (text, data-deep-stage, data-debug-id,
    etc.); we only surface text blocks for the history preamble — the
    rest is UI metadata the model doesn't need."""
    try:
        parts = json.loads(parts_raw)
    except (json.JSONDecodeError, TypeError):
        return ""
    if not isinstance(parts, list):
        return ""
    pieces: list[str] = []
    for p in parts:
        if isinstance(p, dict) and p.get("type") == "text":
            text = p.get("text")
            if isinstance(text, str) and text.strip():
                pieces.append(text.strip())
    return "\n\n".join(pieces)


def _build_conversation_history_preamble(
    conn,
    conversation_id: str | None,
    *,
    max_turns: int = _HISTORY_MAX_TURNS,
    max_chars: int = _HISTORY_MAX_CHARS,
) -> str:
    """Pull recent (user, assistant) message pairs for `conversation_id`
    and format them as a prompt preamble. Returns empty string when
    there's no conversation_id, no prior turns, or only the in-progress
    user message.

    The most recent message is dropped if it's role='user' — that's the
    in-progress turn we're about to answer, already passed in as
    `question`. Truncates to the last `max_turns` pairs and to
    `max_chars` total, dropping oldest first."""
    if not conversation_id:
        return ""
    try:
        rows = conn.execute(
            "SELECT role, parts FROM conversation_messages WHERE conversation_id = %s ORDER BY seq ASC",
            (conversation_id,),
        ).fetchall()
    except Exception as exc:  # noqa: BLE001
        # Test stubs and degraded DB states shouldn't crash a turn.
        # Logged but tolerated — the turn just runs without prior-context
        # awareness, same as a fresh conversation.
        logger.warning(f"history preamble fetch failed for conversation {conversation_id}: {exc}")
        return ""
    if not rows:
        return ""
    if rows[-1]["role"] == "user":
        rows = rows[:-1]
    if not rows:
        return ""
    rows = rows[-max_turns * 2 :]
    blocks: list[str] = []
    for r in rows:
        text = _extract_text_from_parts_json(r["parts"])
        if text:
            blocks.append(f"**{r['role'].title()}:** {text}")
    body = "\n\n".join(blocks)
    while len(body) > max_chars and len(blocks) >= 2:
        blocks = blocks[2:]
        body = "\n\n".join(blocks)
    if not body:
        return ""
    return (
        "# Prior conversation in this chat\n\n"
        "These are the previous user questions and your answers in the same\n"
        "conversation. Use them for context — but only answer the LATEST\n"
        "user question (below the divider).\n\n"
        f"{body}\n\n---\n\n# Latest user question (answer this)\n\n"
    )


async def _real_run(
    db_path: Path,
    session_id: str,
    question: str,
    default_model: str | None = None,
    backend: str | None = None,
    conversation_id: str | None = None,
) -> AsyncIterator[str]:
    """Live pipeline: fire the Orchestrator as a background task while
    this generator polls the session's event log and streams every
    new row to the client as an SSE frame.

    The orchestrator writes events to `agent_events` directly (that's
    its durable log); we just watch. This keeps the orchestrator free
    of streaming concerns — it's a pure async function that blocks
    until the turn is done.

    `default_model`, if provided, applies to every sub-agent (Planner,
    Retriever, Analyst, Writer, Critic). Per-stage GMAIL_*_MODEL env
    overrides still win — they're the power-user lever for running
    Writer on Pro and the rest on Flash. When `default_model` is None,
    each stage falls back to its env var or its hardcoded default.

    `GMAIL_DEEP_BACKEND=claude_code` swaps in the claudebox adapter
    and registers a side-channel MCP session for the turn.
    """
    import asyncio
    import time as _time

    from gmail_search.agents.analyst import ANALYST_INSTRUCTION, build_analyst_agent, instruction_with_skills
    from gmail_search.agents.cost import record_agent_cost
    from gmail_search.agents.critic import build_critic_agent
    from gmail_search.agents.orchestration import Orchestrator
    from gmail_search.agents.planner import build_planner_agent
    from gmail_search.agents.retriever import build_retriever_agent
    from gmail_search.agents.runtime import adk_invoke
    from gmail_search.agents.session import append_event
    from gmail_search.agents.writer import build_writer_agent

    # Wall-clock at the very top so the post-orchestrator auto-publish
    # sweep can pick out files written during this turn vs. pre-existing
    # scratch. native_run captures its own equivalent — these don't share
    # because the two backends have separate lifetimes.
    turn_started_at = _time.time()

    # Two distinct DB connections: one for the orchestrator's writes
    # (events, costs, finalize) and one for the poller's reads
    # (`fetch_events_after`). psycopg does NOT allow concurrent
    # `execute` on the same connection — sharing one would race the
    # moment a stage finishes while the poller is mid-fetch and
    # produce `OperationalError: another command is already in
    # progress`. Two connections cost almost nothing here (one deep
    # turn = one pair); the safety win is worth it.
    conn = get_connection(db_path)
    poll_conn = get_connection(db_path)

    # Multi-turn memory: prepend the prior (user, assistant) pairs from
    # this conversation so the model sees its own earlier answers and
    # the user's earlier questions. Without this every turn is amnesic
    # — Claude can't resolve "build a spreadsheet for THAT" because it
    # doesn't know what THAT refers to.
    history_preamble = _build_conversation_history_preamble(conn, conversation_id)
    if history_preamble:
        question = history_preamble + question

    # Cost sink: every ADK call lands one row in `costs` with
    # operation='deep_<agent_name>' so the existing spend breakdown
    # automatically segments deep-mode per stage. We ALSO emit a
    # `cost` event on the session transcript so the UI can surface
    # per-turn spend inline.
    turn_cost_usd = 0.0

    def _record_cost(*, agent_name: str, model: str, input_tokens: int, output_tokens: int) -> None:
        nonlocal turn_cost_usd
        usd = record_agent_cost(
            conn,
            session_id=session_id,
            agent_name=agent_name,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        turn_cost_usd += usd
        append_event(
            conn,
            session_id=session_id,
            agent_name=agent_name,
            kind="cost",
            payload={
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "usd": round(usd, 5),
                "turn_total_usd": round(turn_cost_usd, 5),
            },
        )

    # Backend selection: ADK (default), claude_code (claudebox + MCP +
    # full orchestrator), or claude_native (single-agent claudebox loop
    # with all MCP tools, no orchestrator). The first two preserve the
    # orchestrator's InvokeFn contract; claude_native is a separate
    # path that owns its own event emission + finalization.
    backend = _deep_backend(backend)
    workspace = _claudebox_workspace_for(session_id)
    if backend == "claude_native":
        from gmail_search.agents.runtime_claude_native import native_run

        _ensure_workspace_dir(workspace)
        native_task = asyncio.create_task(
            native_run(
                db_path=db_path,
                session_id=session_id,
                workspace=workspace,
                conversation_id=conversation_id,
                question=question,
                model=default_model,
                cost_sink=_record_cost,
            )
        )
        last_seq = 0
        try:
            while True:
                new_events = list(fetch_events_after(poll_conn, session_id, after_seq=last_seq))
                for ev in new_events:
                    last_seq = max(last_seq, ev.seq)
                    yield _sse(
                        ev.kind,
                        {"seq": ev.seq, "agent": ev.agent_name, "payload": ev.payload},
                    )
                if native_task.done():
                    break
                await asyncio.sleep(0.1)
            for ev in fetch_events_after(poll_conn, session_id, after_seq=last_seq):
                yield _sse(
                    ev.kind,
                    {"seq": ev.seq, "agent": ev.agent_name, "payload": ev.payload},
                )
            exc = native_task.exception()
            if exc is not None:
                logger.exception(f"native_run raised in session {session_id}: {exc}")
        finally:
            try:
                poll_conn.close()
            except Exception:
                logger.exception(f"closing poll_conn for session {session_id} failed")
            conn.close()
        return

    claude_session_active = False
    if backend == "claude_code":
        from gmail_search.agents.runtime_claude import (
            claudebox_invoke,
            register_session_via_admin,
            unregister_session_via_admin,
        )

        _ensure_workspace_dir(workspace)
        await register_session_via_admin(
            session_id,
            evidence_records=None,
            db_dsn=None,
            conversation_id=conversation_id,
            workspace=workspace,
        )
        claude_session_active = True

        # Stream tool_call events to the session's agent_events table
        # as they arrive in the per-session JSONL transcript. The
        # poller above turns those rows into SSE frames, so the user
        # sees tool calls in real time instead of after the stage
        # finishes. The runtime adapter assigns an agent name; we
        # tag streamed events as "claude_code" so the UI can route
        # them, and the orchestrator runs with skip_per_tool_emission
        # so we don't double-emit when the stage finally returns.
        async def _stream_tool_call_to_events(kind: str, payload: dict) -> None:
            try:
                append_event(
                    conn,
                    session_id=session_id,
                    agent_name="claude_code",
                    kind=kind,
                    payload=payload,
                )
            except Exception:
                logger.exception(f"streaming append_event failed for session {session_id}")

        async def _invoke(agent, prompt):
            return await claudebox_invoke(
                agent,
                prompt,
                workspace=workspace,
                session_id=session_id,
                cost_sink=_record_cost,
                event_sink=_stream_tool_call_to_events,
            )

    else:

        async def _invoke(agent, prompt):
            return await adk_invoke(agent, prompt, cost_sink=_record_cost)

    # Analyst factory: closure-bound so run_code persists to THIS
    # session's artifacts. Instruction gets skill-matched text
    # appended if a SKILL.md matches the question.
    def _analyst_factory(evidence_records):
        instr = instruction_with_skills(ANALYST_INSTRUCTION, question=question)
        return build_analyst_agent(
            evidence_records=evidence_records,
            db_dsn=None,
            session_id=session_id,
            db_conn=conn,
            instruction=instr,
            model=default_model,
            conversation_id=conversation_id,
        )

    orch = Orchestrator(
        session_id=session_id,
        conn=conn,
        planner=build_planner_agent(model=default_model),
        retriever=build_retriever_agent(model=default_model),
        writer=build_writer_agent(model=default_model),
        critic=build_critic_agent(model=default_model),
        analyst_factory=_analyst_factory,
        invoke=_invoke,
        # claude_code already streams tool_call events mid-flight via
        # the runtime adapter's event_sink — turn off the orchestrator's
        # per-tool emission to avoid duplicates. ADK has no streaming
        # path so it keeps the post-hoc emission.
        skip_per_tool_emission=(backend == "claude_code"),
    )

    # Kick off the orchestration; a parallel poller drains events as
    # they land and yields SSE frames. The poller uses its OWN
    # connection (`poll_conn`) so it can read while the orchestrator
    # is mid-write on `conn` — see comment above.
    orch_task = asyncio.create_task(orch.run(question))
    last_seq = 0
    try:
        while True:
            # Any new events since last tick.
            new_events = list(fetch_events_after(poll_conn, session_id, after_seq=last_seq))
            for ev in new_events:
                last_seq = max(last_seq, ev.seq)
                yield _sse(
                    ev.kind,
                    {"seq": ev.seq, "agent": ev.agent_name, "payload": ev.payload},
                )
            if orch_task.done():
                break
            await asyncio.sleep(0.1)
        # Drain any stragglers written after the done flag flipped.
        for ev in fetch_events_after(poll_conn, session_id, after_seq=last_seq):
            last_seq = max(last_seq, ev.seq)
            yield _sse(
                ev.kind,
                {"seq": ev.seq, "agent": ev.agent_name, "payload": ev.payload},
            )
        # Post-orchestrator auto-publish sweep: catch any file the
        # orchestrator's stages wrote without explicitly publishing.
        # The orchestrator already emitted its `final` event with text
        # we can't cleanly mutate mid-stream, so we emit a separate
        # `auto_published` event the UI can render as follow-up chips.
        # Skipped on the orchestrator-raised path — finalization is
        # already broken and we don't want to add to the noise.
        if orch_task.exception() is None:
            _emit_auto_published_event(
                conn,
                session_id=session_id,
                workspace=workspace,
                conversation_id=conversation_id,
                turn_started_at=turn_started_at,
            )
            for ev in fetch_events_after(poll_conn, session_id, after_seq=last_seq):
                yield _sse(
                    ev.kind,
                    {"seq": ev.seq, "agent": ev.agent_name, "payload": ev.payload},
                )
        # Surface any orchestrator exception that didn't make it to
        # the event log (orchestrator logs `error` itself on raise,
        # but re-raising lets the SSE client see a failed stream).
        exc = orch_task.exception()
        if exc is not None:
            logger.exception(f"orchestrator raised in session {session_id}: {exc}")
    finally:
        # Drop the side-channel session BEFORE closing DB conns —
        # unregister_session is itself idempotent + non-DB-touching,
        # so order vs. close() is just hygiene.
        if claude_session_active:
            try:
                from gmail_search.agents.runtime_claude import unregister_session_via_admin

                await unregister_session_via_admin(session_id)
            except Exception:
                logger.exception(f"unregister_session failed for {session_id}")
        # Close the poller first (it's the read-only one) then the
        # orchestrator's writer. Both must close even if one raises
        # during close — the suppression is intentional.
        try:
            poll_conn.close()
        except Exception:
            logger.exception(f"closing poll_conn for session {session_id} failed")
        conn.close()


async def _probe_claudebox_streaming() -> None:
    """Smoke-test that the JSONL tailer sees mid-flight events.

    Fires a trivial claudebox `/run` ("say PONG") with `event_sink=`
    capturing emitted events. If the tailer caught at least one, log
    success — otherwise log an ERROR pointing at the JSONL path so an
    operator can debug (wrong mount, image upgrade changed format,
    permissions, etc.).

    Opt-in via `GMAIL_DEEP_PROBE_STREAMING=1` so a fresh boot doesn't
    burn an Anthropic call. Skipped automatically under pytest
    (`PYTEST_CURRENT_TEST` set)."""
    import os as _os

    if _os.environ.get("GMAIL_DEEP_PROBE_STREAMING", "").lower() not in ("1", "true", "yes"):
        return
    if _os.environ.get("PYTEST_CURRENT_TEST"):
        return
    from types import SimpleNamespace

    from gmail_search.agents.runtime_claude import _host_jsonl_dir_for, claudebox_invoke

    workspace = "deep-probe-streaming"
    captured: list[tuple[str, dict]] = []

    async def _sink(kind: str, payload: dict) -> None:
        captured.append((kind, payload))

    _ensure_workspace_dir(workspace)
    agent = SimpleNamespace(name="probe", model="sonnet", instruction="Reply with PONG.")
    try:
        await claudebox_invoke(
            agent,
            "say PONG",
            workspace=workspace,
            event_sink=_sink,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "claudebox streaming probe FAILED to invoke claudebox: %s. "
            "Mid-flight tool_call streaming will not work for deep mode. "
            "JSONL path expected at: %s",
            exc,
            _host_jsonl_dir_for(workspace),
        )
        return
    if not captured:
        logger.error(
            "claudebox streaming probe captured ZERO mid-flight events. "
            "Tool_call streaming is broken — check the JSONL transcript "
            "directory: %s. Likely causes: image upgrade changed the "
            "transcript format, the host mount is missing, or the workspace "
            "encoding scheme changed.",
            _host_jsonl_dir_for(workspace),
        )
    else:
        logger.info(
            "claudebox streaming probe captured %d mid-flight event(s); streaming OK",
            len(captured),
        )


def _probe_adk_imports() -> None:
    """Boot-time check: try to import every ADK module the deep
    pipeline reaches for inside `_real_run`. Surfaces broken installs
    at server startup instead of at the first /api/agent/analyze
    request — operators see the problem when they boot, not when a
    user clicks "Deep mode" hours later. Chat mode is unaffected, so
    we log a warning and return rather than crashing the server."""
    try:
        # Touch every import the live pipeline performs. If any of
        # these is broken the deep path is broken.
        from gmail_search.agents import (  # noqa: F401
            analyst,
            critic,
            orchestration,
            planner,
            retriever,
            runtime,
            writer,
        )
    except Exception as e:
        logger.warning(
            "ADK imports failed at startup — deep mode (/api/agent/analyze) "
            "will fail at request time. Chat mode is unaffected. Underlying "
            f"error: {type(e).__name__}: {e}"
        )


def register_agent_routes(app: FastAPI, db_path: Path) -> None:
    """Attach the deep-agent endpoints to an existing FastAPI app.
    Called from `create_app` so the agent surface is an opt-in add-on
    that doesn't affect chat-mode code paths."""
    # Probe ADK imports at registration time so a broken install is
    # visible in the server logs at boot rather than at the first
    # deep-mode request hours later.
    _probe_adk_imports()

    # Optional: smoke-test the JSONL streaming pipeline so a broken
    # mount / image upgrade is loud at boot. Off by default — opt in
    # via `GMAIL_DEEP_PROBE_STREAMING=1`.
    @app.on_event("startup")
    async def _streaming_probe_on_startup() -> None:
        try:
            await _probe_claudebox_streaming()
        except Exception:
            logger.exception("claudebox streaming probe raised unexpectedly")

    @app.post("/api/agent/analyze")
    async def analyze(req: AnalyzeRequest) -> Response:
        """Start a deep-mode turn. Creates the session row, then
        streams SSE as each sub-agent emits events. `question` is
        required; conversation_id is optional (links the session back
        to the chat thread in the UI)."""
        session_id = new_session_id()
        conn = get_connection(db_path)
        try:
            create_session(
                conn,
                session_id=session_id,
                conversation_id=req.conversation_id,
                mode="deep",
                question=req.question,
            )
        finally:
            conn.close()

        async def stream() -> AsyncIterator[str]:
            # First frame announces the session id so the UI can
            # (a) show it, (b) resume via ?after=<seq> after drops.
            yield _sse("session", {"session_id": session_id})
            if _use_real_pipeline():
                # Real path: run the orchestrator, mirror events into
                # the SSE stream by polling agent_events (the
                # orchestrator itself only writes to the DB).
                async for frame in _real_run(
                    db_path,
                    session_id,
                    req.question,
                    default_model=req.model,
                    backend=req.backend,
                    conversation_id=req.conversation_id,
                ):
                    yield frame
            else:
                async for frame in _stub_run(db_path, session_id, req.question):
                    yield frame

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/api/agent/analyze/{session_id}/events")
    async def events(session_id: str, after: int = 0, request: Request = None) -> Response:
        """Replay events after `seq=<after>`. Used by the UI when
        reconnecting to an in-flight session (back-button, tab swap,
        network blip). Keeps streaming as new events arrive by polling
        the DB at a 250 ms tick — not elegant but fine for human-rate
        turns. If the client disconnects we stop polling."""

        async def replay() -> AsyncIterator[str]:
            last_seen = after
            while True:
                if request is not None and await request.is_disconnected():
                    return
                conn = get_connection(db_path)
                try:
                    new_events = list(fetch_events_after(conn, session_id, after_seq=last_seen))
                    # Also check whether the session has terminated so
                    # we can close the stream cleanly instead of
                    # polling forever.
                    srow = conn.execute(
                        "SELECT status FROM agent_sessions WHERE id = %s",
                        (session_id,),
                    ).fetchone()
                finally:
                    conn.close()
                for ev in new_events:
                    last_seen = max(last_seen, ev.seq)
                    yield _sse(
                        ev.kind,
                        {
                            "seq": ev.seq,
                            "agent": ev.agent_name,
                            "payload": ev.payload,
                        },
                    )
                if srow and srow["status"] in {"done", "error"}:
                    return
                await asyncio.sleep(0.25)

        return StreamingResponse(replay(), media_type="text/event-stream")

    @app.get("/api/artifact/{artifact_id}")
    async def artifact(artifact_id: int) -> Response:
        """Serve a single artifact (plot PNG, CSV, etc.) by id.
        Mime type comes from the stored row. No listing endpoint —
        artifacts are always fetched by a specific id cited in an
        agent response."""
        conn = get_connection(db_path)
        try:
            row = get_artifact(conn, artifact_id)
        finally:
            conn.close()
        if row is None:
            return JSONResponse({"error": "Artifact not found"}, status_code=404)
        name, mime_type, data = row
        return Response(
            content=data,
            media_type=mime_type,
            headers={"Content-Disposition": f'inline; filename="{name}"'},
        )
