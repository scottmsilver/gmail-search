"""Tests for the deep-analysis orchestration state machine.

We CI the invariant shape of a deep-mode turn — order of stages,
when to skip the Analyst, critic revision loop behavior, event
emission — using fake agents that return canned strings. No live
Gemini calls are made here; those are reserved for the Phase-6
integration suite.

The Orchestrator takes an `invoke` callable for its per-stage
execution. By substituting a test double, we control exactly what
each sub-agent "returns" and can assert the resulting event
transcript.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

import pytest

from gmail_search.agents.orchestration import MAX_CRITIC_ROUNDS, Orchestrator, StageResult
from gmail_search.agents.session import create_session, fetch_events_after, new_session_id
from gmail_search.store.db import get_connection, init_db

# ── Test doubles ─────────────────────────────────────────────────


@dataclass
class _FakeAgent:
    """Satisfies the orchestration's AgentLike protocol. The
    orchestrator only reads `.name` for logging / routing; the
    canned output is decided by the `invoke` shim."""

    name: str


def _invoke_with_scripted_outputs(scripts: dict[str, list[str]]):
    """Build an `invoke` callable that returns scripted outputs per
    agent name. `scripts["planner"]` is a list of strings — the
    orchestrator consumes them in order for each planner invocation.
    """

    def make() -> Callable:
        counters: dict[str, int] = {n: 0 for n in scripts}

        async def invoke(agent: _FakeAgent, prompt: str) -> StageResult:  # noqa: ARG001
            name = agent.name
            idx = counters.get(name, 0)
            outputs = scripts.get(name, [])
            assert idx < len(outputs), f"agent {name!r} called more times than scripted"
            counters[name] = idx + 1
            return StageResult(text=outputs[idx])

        return invoke

    return make()


def _make_orchestrator(conn, *, sid: str, invoke, analyst_factory=None):
    """Helper so each test assembles a minimal Orchestrator."""
    return Orchestrator(
        session_id=sid,
        conn=conn,
        planner=_FakeAgent("planner"),
        retriever=_FakeAgent("retriever"),
        writer=_FakeAgent("writer"),
        critic=_FakeAgent("critic"),
        analyst_factory=analyst_factory or (lambda _evidence: _FakeAgent("analyst")),
        invoke=invoke,
    )


def _all_events(conn, sid: str) -> list[tuple[str, str]]:
    """Fetch (agent_name, kind) for every event on a session, in
    seq order — makes order assertions readable."""
    return [(e.agent_name, e.kind) for e in fetch_events_after(conn, sid)]


# ── Tests ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_happy_path_critic_accepts_first_round(db_backend):
    """Canonical linear flow: Planner → Retriever → Analyst → Writer
    → Critic (accepts). No revision. The event log must contain
    exactly the expected (agent, kind) tuples in order, ending with
    a root/final."""
    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)
    sid = new_session_id()
    create_session(conn, session_id=sid, conversation_id=None, mode="deep", question="q")

    invoke = _invoke_with_scripted_outputs(
        {
            "planner": [json.dumps({"question_type": "analytical", "analysis": [{"step": "plot"}]})],
            "retriever": ["found: 3 threads\n- [ref:abcdef01] foo"],
            "analyst": ["ran analysis, produced art:7"],
            "writer": ["Final draft with [ref:abcdef01] and [art:7]."],
            "critic": [json.dumps({"accepted": True, "violations": []})],
        }
    )
    orch = _make_orchestrator(conn, sid=sid, invoke=invoke)
    final = await orch.run("analyze my spending")

    assert "[ref:abcdef01]" in final
    events = _all_events(conn, sid)
    # Stage order locked in. No revision, no analyst-skipped.
    assert events == [
        ("planner", "plan"),
        ("retriever", "evidence"),
        ("analyst", "analysis"),
        ("writer", "draft"),
        ("critic", "critique"),
        ("root", "final"),
    ]
    conn.close()


@pytest.mark.asyncio
async def test_analyst_skipped_when_plan_has_no_analysis_steps(db_backend):
    """Factual questions where the Planner emits `analysis: []` must
    skip the Analyst entirely. We emit a `skipped` event for
    visibility in the UI transcript."""
    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)
    sid = new_session_id()
    create_session(conn, session_id=sid, conversation_id=None, mode="deep", question="q")

    invoke = _invoke_with_scripted_outputs(
        {
            "planner": [json.dumps({"question_type": "factual", "analysis": []})],
            "retriever": ["found: 1 thread\n- [ref:deadbeef] bar"],
            "writer": ["Short factual answer with [ref:deadbeef]."],
            "critic": [json.dumps({"accepted": True, "violations": []})],
        }
    )
    orch = _make_orchestrator(conn, sid=sid, invoke=invoke)
    await orch.run("when was the receipt signed?")

    events = _all_events(conn, sid)
    assert ("analyst", "skipped") in events
    assert ("analyst", "analysis") not in events
    conn.close()


@pytest.mark.asyncio
async def test_critic_rejection_triggers_writer_revision(db_backend):
    """Critic rejects round 1 → Writer revises → Critic accepts
    round 2. The event log captures the revision as writer/revision."""
    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)
    sid = new_session_id()
    create_session(conn, session_id=sid, conversation_id=None, mode="deep", question="q")

    invoke = _invoke_with_scripted_outputs(
        {
            "planner": [json.dumps({"question_type": "synthesis", "analysis": []})],
            "retriever": ["found: 2 threads\n- [ref:aaaaaaaa] one"],
            "writer": [
                "First draft (ungrounded claim).",
                "Revised draft with [ref:aaaaaaaa] citation.",
            ],
            "critic": [
                json.dumps(
                    {
                        "accepted": False,
                        "violations": [{"kind": "ungrounded", "quote": "ungrounded claim", "note": "cite it"}],
                    }
                ),
                json.dumps({"accepted": True, "violations": []}),
            ],
        }
    )
    orch = _make_orchestrator(conn, sid=sid, invoke=invoke)
    final = await orch.run("summarize")

    assert "Revised draft" in final
    events = _all_events(conn, sid)
    # Two critic events, one writer revision.
    assert events.count(("critic", "critique")) == 2
    assert ("writer", "revision") in events
    assert ("root", "final") in events
    conn.close()


@pytest.mark.asyncio
async def test_critic_loop_respects_max_rounds_cap(db_backend):
    """Even if the critic never accepts, we ship SOMETHING after
    MAX_CRITIC_ROUNDS. The loop's job is to bound tail latency / cost,
    not to block on an unresolvable disagreement."""
    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)
    sid = new_session_id()
    create_session(conn, session_id=sid, conversation_id=None, mode="deep", question="q")

    # Planner / retriever / writer all fine; critic always rejects.
    rejections = [
        json.dumps({"accepted": False, "violations": [{"kind": "ungrounded", "quote": "x", "note": "n"}]})
        for _ in range(MAX_CRITIC_ROUNDS + 2)
    ]
    invoke = _invoke_with_scripted_outputs(
        {
            "planner": [json.dumps({"analysis": []})],
            "retriever": ["found: 0 threads"],
            "writer": ["draft-1", "draft-2", "draft-3"],
            "critic": rejections,
        }
    )
    orch = _make_orchestrator(conn, sid=sid, invoke=invoke)
    final = await orch.run("impossible")

    # Final is whatever the last draft was (orchestrator ships on
    # cap — the UI surfaces the remaining critic notes for the user).
    assert final in {"draft-1", "draft-2", "draft-3"}
    events = _all_events(conn, sid)
    # Exactly MAX_CRITIC_ROUNDS critic events; one fewer writer
    # revisions (writer doesn't re-run after the cap trips).
    assert events.count(("critic", "critique")) == MAX_CRITIC_ROUNDS
    assert events.count(("writer", "revision")) == MAX_CRITIC_ROUNDS - 1
    conn.close()


@pytest.mark.asyncio
async def test_planner_output_is_parsed_and_persisted(db_backend):
    """Plan JSON should land in the event payload as a dict (not raw
    string) so the UI doesn't have to double-parse."""
    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)
    sid = new_session_id()
    create_session(conn, session_id=sid, conversation_id=None, mode="deep", question="q")

    plan = {"question_type": "exploratory", "retrieval": [], "analysis": []}
    invoke = _invoke_with_scripted_outputs(
        {
            "planner": [json.dumps(plan)],
            "retriever": ["found: 0"],
            "writer": ["done"],
            "critic": [json.dumps({"accepted": True, "violations": []})],
        }
    )
    orch = _make_orchestrator(conn, sid=sid, invoke=invoke)
    await orch.run("explore")

    events = list(fetch_events_after(conn, sid))
    plan_event = next(e for e in events if e.kind == "plan")
    assert plan_event.payload["plan"] == plan
    conn.close()


@pytest.mark.asyncio
async def test_handles_malformed_planner_json_gracefully(db_backend):
    """Planner occasionally wraps in a code fence or adds prose.
    Orchestrator should parse what it can, skip Analyst (empty
    plan), and finish cleanly."""
    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)
    sid = new_session_id()
    create_session(conn, session_id=sid, conversation_id=None, mode="deep", question="q")

    invoke = _invoke_with_scripted_outputs(
        {
            # Unparseable plan (the orchestrator falls back to {}).
            "planner": ["I think the plan is: analyze spending quarterly."],
            "retriever": ["found: 0"],
            "writer": ["done"],
            "critic": [json.dumps({"accepted": True, "violations": []})],
        }
    )
    orch = _make_orchestrator(conn, sid=sid, invoke=invoke)
    final = await orch.run("q")
    assert final == "done"
    events = _all_events(conn, sid)
    assert ("analyst", "skipped") in events  # empty plan → skip
    conn.close()
