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
    model: str = "gemini-2.5-pro"


def _sse(event_type: str, data: dict) -> str:
    """Format one SSE message frame. The Next.js proxy forwards these
    verbatim; the client reads `event:` to route and `data:` for the
    payload JSON."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


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


async def _real_run(db_path: Path, session_id: str, question: str) -> AsyncIterator[str]:
    """Live pipeline: fire the Orchestrator as a background task while
    this generator polls the session's event log and streams every
    new row to the client as an SSE frame.

    The orchestrator writes events to `agent_events` directly (that's
    its durable log); we just watch. This keeps the orchestrator free
    of streaming concerns — it's a pure async function that blocks
    until the turn is done.
    """
    import asyncio

    from gmail_search.agents.analyst import ANALYST_INSTRUCTION, build_analyst_agent, instruction_with_skills
    from gmail_search.agents.cost import record_agent_cost
    from gmail_search.agents.critic import build_critic_agent
    from gmail_search.agents.orchestration import Orchestrator
    from gmail_search.agents.planner import build_planner_agent
    from gmail_search.agents.retriever import build_retriever_agent
    from gmail_search.agents.runtime import adk_invoke
    from gmail_search.agents.session import append_event
    from gmail_search.agents.writer import build_writer_agent

    # Shared DB connection for the whole turn. The orchestrator does
    # one event append per stage, not a hot loop, so serialising on
    # one connection is fine.
    conn = get_connection(db_path)

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

    # Wrap adk_invoke so the orchestrator's InvokeFn contract
    # (two-arg) is preserved while the cost sink gets threaded in
    # via closure — no signature change to the Orchestrator class.
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
        )

    orch = Orchestrator(
        session_id=session_id,
        conn=conn,
        planner=build_planner_agent(),
        retriever=build_retriever_agent(),
        writer=build_writer_agent(),
        critic=build_critic_agent(),
        analyst_factory=_analyst_factory,
        invoke=_invoke,
    )

    # Kick off the orchestration; a parallel poller drains events as
    # they land and yields SSE frames.
    orch_task = asyncio.create_task(orch.run(question))
    last_seq = 0
    try:
        while True:
            # Any new events since last tick.
            new_events = list(fetch_events_after(conn, session_id, after_seq=last_seq))
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
        for ev in fetch_events_after(conn, session_id, after_seq=last_seq):
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
        conn.close()


def register_agent_routes(app: FastAPI, db_path: Path) -> None:
    """Attach the deep-agent endpoints to an existing FastAPI app.
    Called from `create_app` so the agent surface is an opt-in add-on
    that doesn't affect chat-mode code paths."""

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
                async for frame in _real_run(db_path, session_id, req.question):
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
