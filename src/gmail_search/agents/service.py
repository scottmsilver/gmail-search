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


def _claudebox_workspace_for(conversation_id: str | None, session_id: str) -> str:
    """One claudebox workspace per CONVERSATION (stable across turns)
    so the per-conversation `--resume` thread appends to the same
    JSONL transcript and the workspace dir accumulates files across
    turns. Falls back to per-session naming when there's no
    conversation_id (one-off probes / tests / non-chat flows).

    The conversation_id is already validated against
    `^[A-Za-z0-9_-]{1,64}$` at the API boundary
    (server.py:_validate_conversation_id), so we can interpolate
    directly into the dir name without sanitization here."""
    if conversation_id:
        return f"deep-conv-{conversation_id}"
    return f"deep-{session_id}"


def _ensure_workspace_dir(workspace: str) -> None:
    """Claudebox refuses to run in a workspace it can't see on disk.
    Create it on the host filesystem before the first invoke."""
    base = Path("deploy/claudebox/workspaces") / workspace
    base.mkdir(parents=True, exist_ok=True)


def _ensure_workspace_dir_exists(workspace: str) -> bool:
    """Return True if the workspace dir exists on disk. Used by the
    resume-recovery path to detect a stale mapping that points at a
    workspace prune_conversation_workspaces deleted."""
    return (Path("deploy/claudebox/workspaces") / workspace).is_dir()


def _final_text_from_session(conn, session_id: str) -> str:
    """Pull the writer's final text out of `agent_events` for one
    session. Both the orchestrated path (writer/critic emits `final`)
    and the native path (root emits `final`) write a `kind='final'`
    event with `payload.text`. Returns "" if absent — the persist
    path then writes an empty-text bubble rather than crashing."""
    row = conn.execute(
        """SELECT payload FROM agent_events
           WHERE session_id = %s AND kind = 'final'
           ORDER BY seq DESC LIMIT 1""",
        (session_id,),
    ).fetchone()
    if not row:
        return ""
    payload = row["payload"] or {}
    if isinstance(payload, dict):
        text = payload.get("text")
        if isinstance(text, str):
            return text
    return ""


def _build_assistant_parts_from_events(
    events: list,
    *,
    session_id: str,
    final_text: str,
) -> list[dict]:
    """Reconstruct a Vercel AI SDK UIMessage `parts` array from a
    deep-mode session's `agent_events` log.

    Output schema mirrors what the chat-thread Thread renderer
    consumes (sampled from real Gemini chat-mode rows): a leading
    `data-debug-id` block, then per-tool-call `tool-<name>` blocks
    with both input + output, then the final text. The renderer
    falls back gracefully on unknown `tool-<name>` types — the
    payload is still browsable via the JSON inspector.

    Tool calls come from `mcp_tool_call_full` events when they're
    present (Phase 2 — the durable record carries the structured
    response too) and fall back to the streamed `tool_call` events
    (input only, no response captured) when the mcp write didn't
    land. We dedupe by (name, json(args)) so each tool call shows
    up exactly once."""
    import json as _json

    parts: list[dict] = []
    parts.append(
        {
            "type": "data-debug-id",
            "id": f"dbg-agent-session-{session_id}",
            "data": {"id": session_id},
        }
    )
    seen: set[tuple] = set()
    # First pass: rich versions from Phase 2's `mcp_tool_call_full`.
    for ev in events:
        if ev.kind != "mcp_tool_call_full":
            continue
        payload = ev.payload or {}
        name = str(payload.get("name") or "?")
        args = payload.get("args") or {}
        response = payload.get("response") or {}
        try:
            sig = (name, _json.dumps(args, sort_keys=True))
        except TypeError:
            sig = (name, repr(args))
        seen.add(sig)
        parts.append(
            {
                "type": f"tool-{name}",
                "toolCallId": f"tc-{ev.seq}",
                "state": "output-available",
                "input": args,
                "output": response,
            }
        )
    # Second pass: streamed tool_use blocks the JSONL tailer
    # forwarded. Skip ones already covered by mcp_tool_call_full;
    # what remains are tool calls whose response wasn't durably
    # captured (claudebox-internal tools like Bash, Task, etc.).
    for ev in events:
        if ev.kind != "tool_call":
            continue
        payload = ev.payload or {}
        name = str(payload.get("name") or "?")
        args = payload.get("args") or {}
        try:
            sig = (name, _json.dumps(args, sort_keys=True))
        except TypeError:
            sig = (name, repr(args))
        if sig in seen:
            continue
        seen.add(sig)
        parts.append(
            {
                "type": f"tool-{name}",
                "toolCallId": f"tc-stream-{ev.seq}",
                "state": "input-available",
                "input": args,
            }
        )
    parts.append({"type": "text", "text": final_text or ""})
    return parts


def _persist_rich_assistant_message(
    conn,
    *,
    conversation_id: str | None,
    session_id: str,
    final_text: str | None = None,
) -> bool:
    """At the end of a deep-mode turn, INSERT a rich assistant message
    into `conversation_messages` reconstructed from `agent_events`.
    The chat-thread UI reads from `conversation_messages.parts`
    (NOT `agent_events`), and the front-end's deep-mode persist
    path is text-only — without this call, the UI loses every tool
    call on reload.

    Best-effort: a failure logs and continues. The deep turn already
    succeeded — chat-thread display is the only thing affected.

    Pairs with the front-end change in `web/app/api/chat/route.ts`
    that disables `persistAssistantText` for deep mode (otherwise
    the front-end's PUT would race + clobber this row with a
    text-only version)."""
    import json as _json

    if not conversation_id:
        return False
    try:
        if final_text is None:
            final_text = _final_text_from_session(conn, session_id)
        events = list(fetch_events_after(conn, session_id, after_seq=0))
        parts = _build_assistant_parts_from_events(events, session_id=session_id, final_text=final_text)
        next_seq_row = conn.execute(
            "SELECT COALESCE(MAX(seq), -1) + 1 AS next_seq " "FROM conversation_messages WHERE conversation_id = %s",
            (conversation_id,),
        ).fetchone()
        next_seq = int(next_seq_row["next_seq"])
        conn.execute(
            """INSERT INTO conversation_messages
               (conversation_id, seq, role, parts) VALUES (%s, %s, 'assistant', %s)""",
            (conversation_id, next_seq, _json.dumps(parts)),
        )
        # Bump conversations.updated_at so the sidebar's
        # most-recent-first ordering picks up this turn immediately.
        conn.execute(
            "UPDATE conversations SET updated_at = NOW() WHERE id = %s",
            (conversation_id,),
        )
        conn.commit()
        logger.info(
            "persisted rich assistant msg for conv=%s session=%s parts=%d",
            conversation_id,
            session_id,
            len(parts),
        )
        return True
    except Exception:
        logger.exception(
            f"persist rich assistant message failed for conv={conversation_id} "
            f"session={session_id} (chat thread may show only the front-end's "
            "text-only fallback on reload)"
        )
        return False


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
    # Per-turn credential refresh: catches the case where the host
    # rotated `~/.claude/.credentials.json` between server startup and
    # this turn (Claude Code refreshes via OAuth roughly every 24h, so
    # a long-running web server WILL drift if we only sync at boot).
    # No-op when mtimes already match.
    if backend in ("claude_native", "claude_code"):
        try:
            from gmail_search.claudebox_creds import sync_credentials_if_stale

            sync_credentials_if_stale()
        except Exception:
            logger.exception("per-turn claudebox credential sync failed (continuing)")
    workspace = _claudebox_workspace_for(conversation_id, session_id)
    if backend == "claude_native":
        from gmail_search.agents.runtime_claude_native import native_run
        from gmail_search.agents.session import (
            claim_first_turn_lock,
            lookup_claude_session_uuid,
            record_claude_session_uuid,
        )

        _ensure_workspace_dir(workspace)
        # Resolve the Claude session UUID this conversation is pinned to
        # (or None if this is its first deep turn). When None, native_run
        # will run without `--resume` and call `_persist_first_uuid` once
        # claudebox returns the UUID it generated. Subsequent turns of
        # the same conversation hit the fast-path lookup with no lock.
        # Wrap defensively: a DB failure here should NOT crash the turn
        # — we just fall through to "no resume, treat as first turn".
        resume_uuid = None
        if conversation_id:
            try:
                resume_uuid = lookup_claude_session_uuid(conn, conversation_id)
            except Exception:
                logger.warning(
                    f"resume-uuid lookup failed for conversation {conversation_id}; "
                    "starting fresh Claude session for this turn",
                    exc_info=True,
                )
            else:
                # Recovery: a mapping that points at a workspace whose
                # JSONL was deleted (e.g. by prune_conversation_workspaces
                # while the conversation was idle) would crash claudebox
                # with "No conversation found with session ID". Detect
                # the gap up-front by checking the workspace dir; if
                # missing, drop the stale mapping and start a fresh
                # Claude session. The user's chat continuity (Postgres
                # conversation_messages) is unaffected.
                if resume_uuid and not _ensure_workspace_dir_exists(workspace):
                    logger.warning(
                        f"workspace {workspace!r} missing on disk for conversation "
                        f"{conversation_id} (mapping pointed at claude_session_uuid "
                        f"{resume_uuid!r}). Dropping stale mapping and starting fresh."
                    )
                    try:
                        conn.execute(
                            "DELETE FROM conversation_claude_session WHERE conversation_id = %s",
                            (conversation_id,),
                        )
                        conn.commit()
                    except Exception:
                        logger.exception(
                            f"failed to drop stale mapping for {conversation_id}; "
                            "next turn may still attempt to resume against missing state"
                        )
                    resume_uuid = None

        def _persist_first_uuid(claude_uuid: str) -> None:
            """First-turn establisher: take the advisory lock, re-check
            for a row another concurrent caller may have just persisted,
            INSERT if we're still the first writer. Same-conversation
            concurrent first-turns are also serialized at the workspace
            level by claudebox's busy_workspaces lock, but the DB lock
            here is the source-of-truth across orchestrator processes."""
            if not conversation_id:
                return
            claim_conn = get_connection(db_path)
            try:
                existing = claim_first_turn_lock(claim_conn, conversation_id)
                if not existing:
                    record_claude_session_uuid(claim_conn, conversation_id, claude_uuid)
                claim_conn.commit()
            except Exception:
                claim_conn.rollback()
                logger.exception(f"persist claude_session_uuid failed for conversation {conversation_id}")
            finally:
                claim_conn.close()

        native_task = asyncio.create_task(
            native_run(
                db_path=db_path,
                session_id=session_id,
                workspace=workspace,
                conversation_id=conversation_id,
                question=question,
                model=default_model,
                cost_sink=_record_cost,
                resume=resume_uuid,
                on_session_uuid=_persist_first_uuid,
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
            else:
                # Persist a rich assistant message into conversation_messages
                # so the chat-thread UI shows tool calls on reload (without
                # this, the front-end's persistAssistantText writes a text-
                # only version and we lose all debug detail). Emit a
                # `persist_ok` SSE frame ONLY on commit success — the
                # front-end's serverPersistedRichAssistant flag gates on
                # this frame, so a silent persist failure causes the
                # front-end to fall back to its text-only PUT (loss of
                # rich detail, but at least an assistant bubble exists
                # on reload). Without the explicit `persist_ok` gate, a
                # persist failure would leave NO assistant row at all.
                persisted = _persist_rich_assistant_message(
                    conn,
                    conversation_id=conversation_id,
                    session_id=session_id,
                )
                if persisted:
                    yield _sse("persist_ok", {"session_id": session_id})
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
        from gmail_search.agents.session import (
            claim_first_turn_lock,
            lookup_claude_session_uuid,
            record_claude_session_uuid,
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

        # Mirror the native-branch claim flow: look up any existing
        # Claude session UUID for this conversation; first sub-agent
        # invoke without it captures one and we cache it for the
        # remaining sub-agents in this turn so all stages share the
        # same Claude conversation. See PER_CONVERSATION_SESSIONS.md
        # for the establishment protocol.
        claude_uuid_for_turn: dict[str, str | None] = {"v": None}
        if conversation_id:
            try:
                claude_uuid_for_turn["v"] = lookup_claude_session_uuid(conn, conversation_id)
            except Exception:
                logger.warning(
                    f"resume-uuid lookup failed for conversation {conversation_id}; "
                    "starting fresh Claude session for this turn",
                    exc_info=True,
                )
            else:
                if claude_uuid_for_turn["v"] and not _ensure_workspace_dir_exists(workspace):
                    logger.warning(
                        f"workspace {workspace!r} missing on disk for conversation "
                        f"{conversation_id} (mapping pointed at claude_session_uuid "
                        f"{claude_uuid_for_turn['v']!r}). Dropping stale mapping."
                    )
                    try:
                        conn.execute(
                            "DELETE FROM conversation_claude_session WHERE conversation_id = %s",
                            (conversation_id,),
                        )
                        conn.commit()
                    except Exception:
                        logger.exception(f"failed to drop stale mapping for {conversation_id}")
                    claude_uuid_for_turn["v"] = None

        def _persist_first_uuid_code(claude_uuid: str) -> None:
            """First-turn establisher for the orchestrated claude_code
            backend. Same advisory-lock + idempotent-INSERT shape as
            the claude_native variant — kept inline rather than shared
            so each branch can evolve independently without coupling."""
            if not conversation_id:
                return
            claim_conn = get_connection(db_path)
            try:
                existing = claim_first_turn_lock(claim_conn, conversation_id)
                if not existing:
                    record_claude_session_uuid(claim_conn, conversation_id, claude_uuid)
                claim_conn.commit()
            except Exception:
                claim_conn.rollback()
                logger.exception(f"persist claude_session_uuid failed for conversation {conversation_id}")
            finally:
                claim_conn.close()

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
            resume = claude_uuid_for_turn["v"]
            result = await claudebox_invoke(
                agent,
                prompt,
                workspace=workspace,
                session_id=session_id,
                cost_sink=_record_cost,
                event_sink=_stream_tool_call_to_events,
                resume=resume,
            )
            # First sub-agent in this turn that ran without resume —
            # capture the Claude session UUID claudebox returned and
            # persist it so subsequent sub-agents (this turn) and
            # subsequent turns of this conversation can `--resume`.
            if resume is None and result.claude_session_uuid:
                claude_uuid_for_turn["v"] = result.claude_session_uuid
                _persist_first_uuid_code(result.claude_session_uuid)
            return result

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
            # Persist a rich assistant message into conversation_messages
            # so the chat-thread UI shows tool calls on reload (without
            # this, the front-end's persistAssistantText writes a text-
            # only version and we lose all debug detail). Emit
            # `persist_ok` ONLY on commit success — see twin block in
            # the claude_native branch above for why this gate matters.
            persisted = _persist_rich_assistant_message(
                conn,
                conversation_id=conversation_id,
                session_id=session_id,
            )
            if persisted:
                yield _sse("persist_ok", {"session_id": session_id})
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
