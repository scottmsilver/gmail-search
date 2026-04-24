"""Orchestration for a deep-analysis turn.

One `Orchestrator.run(question)` coroutine drives the five sub-agents
in the canonical order — planner → retriever → analyst → writer →
critic → (maybe writer revision) — and emits a structured event
stream (`agent_events` rows) as each stage completes.

The flow is deliberately NOT expressed as an ADK `SequentialAgent`
because the critic's feedback loop is non-linear: a rejected draft
goes BACK to the writer with the critic's notes. That's a cycle, so
we orchestrate in Python and use ADK's `Runner` as the per-sub-agent
execution primitive.

Testability is the other reason: the orchestration state machine
(order of stages, when to skip the Analyst, critic revision cap) is
invariant we want to CI — and we can CI it with a fake
`invoke_agent` function that returns canned strings, without ever
touching Gemini. Integration tests with the live LLM live in Phase 6.

Every sub-agent invocation goes through `_invoke` so the
mock-friendly indirection happens at one place.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol

from gmail_search.agents.session import append_event, finalize_session

logger = logging.getLogger(__name__)


# Hard caps so one runaway stage can't bleed the whole turn.
MAX_CRITIC_ROUNDS = 2
MAX_RETRIEVER_ROUNDS = 1  # retriever is already tool-bounded; 1 outer pass is plenty


class AgentLike(Protocol):
    """Minimal shape we rely on from both real ADK agents and test
    fakes. Real `google.adk.Agent` instances satisfy this; test code
    supplies `_FakeAgent` with the same attribute names."""

    name: str


@dataclass
class StageResult:
    """What one sub-agent returns. `text` is the model's primary
    output (JSON for Planner/Critic, markdown for Writer, summary for
    Retriever, stdout+summary for Analyst). `tool_calls` is a log of
    every tool the agent invoked this stage — the service layer
    emits `tool_call` events from it."""

    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


InvokeFn = Callable[[AgentLike, str], Awaitable[StageResult]]
"""Signature of the per-stage invocation shim. Real implementation
wraps ADK Runner.run_async; tests supply a mock."""


@dataclass
class Orchestrator:
    """Owns one deep-analysis turn.

    Construction is decoupled from execution so tests can build an
    orchestrator with mock agents and exercise `.run()` without any
    network or model dependency.

    `conn` is a live DB connection used to append events + persist
    artifacts. Each sub-agent gets the SAME connection — one
    transaction-per-event keeps partial failures recoverable.
    """

    session_id: str
    conn: Any  # psycopg connection
    planner: AgentLike
    retriever: AgentLike
    writer: AgentLike
    critic: AgentLike
    # Analyst is built PER SESSION (its run_code tool closure is
    # bound to this session's evidence + artifact sink), so we hand
    # in a factory instead of a pre-built agent. Called with the
    # retrieval evidence when analysis is needed.
    analyst_factory: Callable[[list[dict] | dict | None], AgentLike]
    invoke: InvokeFn

    async def run(self, question: str) -> str:
        """Drive all five stages for `question` and return the final
        accepted markdown. Emits events throughout so an SSE reader
        sees progress. Any unhandled exception lands in an `error`
        event before we re-raise."""
        try:
            plan = await self._run_planner(question)
            evidence = await self._run_retriever(question, plan)
            analysis = await self._run_analyst_if_needed(question, plan, evidence)
            draft = await self._run_writer(question, plan, evidence, analysis)
            draft = await self._run_critic_loop(draft, evidence, analysis)
            self._emit("root", "final", {"text": draft})
            finalize_session(self.conn, self.session_id, status="done", final_answer=draft)
            return draft
        except Exception as e:
            logger.exception(f"deep-run {self.session_id} failed: {e}")
            self._emit("root", "error", {"message": str(e), "type": type(e).__name__})
            finalize_session(self.conn, self.session_id, status="error")
            raise

    # ── Per-stage helpers ──────────────────────────────────────

    async def _run_planner(self, question: str) -> dict:
        """Planner emits a JSON plan. We parse it but don't insist —
        a malformed plan still lets the rest of the pipeline run,
        just with an empty scaffold."""
        result = await self.invoke(self.planner, question)
        plan = _parse_json_or_empty(result.text)
        self._emit("planner", "plan", {"plan": plan, "raw": result.text})
        return plan

    async def _run_retriever(self, question: str, plan: dict) -> str:
        """Retriever returns a short summary of what it found +
        cite_refs. We pass it downstream as free text, not
        structured, because the Writer+Critic prompts were written
        to consume that shape."""
        prompt = _retriever_input(question, plan)
        result = await self.invoke(self.retriever, prompt)
        for tc in result.tool_calls:
            self._emit("retriever", "tool_call", tc)
        self._emit("retriever", "evidence", {"summary": result.text})
        return result.text

    async def _run_analyst_if_needed(self, question: str, plan: dict, evidence: str) -> str | None:
        """Skip the Analyst entirely when the plan has zero analysis
        steps — factual questions don't need code execution, and the
        skip makes the turn faster + cheaper + less error-prone."""
        steps = plan.get("analysis") or []
        if not steps:
            self._emit("analyst", "skipped", {"reason": "no analysis steps in plan"})
            return None
        evidence_records = _evidence_to_records(evidence)
        analyst = self.analyst_factory(evidence_records)
        prompt = _analyst_input(question, plan, evidence)
        result = await self.invoke(analyst, prompt)
        for tc in result.tool_calls:
            self._emit("analyst", "code_run", tc)
        self._emit("analyst", "analysis", {"summary": result.text})
        return result.text

    async def _run_writer(self, question: str, plan: dict, evidence: str, analysis: str | None) -> str:
        """Writer composes the draft. Feed it question + all upstream
        outputs — it's the agent that has to ground citations, so it
        needs everything."""
        prompt = _writer_input(question, plan, evidence, analysis)
        result = await self.invoke(self.writer, prompt)
        self._emit("writer", "draft", {"text": result.text})
        return result.text

    async def _run_critic_loop(self, draft: str, evidence: str, analysis: str | None) -> str:
        """Critic → Writer revision loop, capped at MAX_CRITIC_ROUNDS.
        Each pass: critic emits a JSON verdict; if rejected, writer
        produces one revision; next round critic reviews again. We
        stop on accept OR when the cap trips — whichever comes
        first."""
        for attempt in range(MAX_CRITIC_ROUNDS):
            prompt = _critic_input(draft, evidence, analysis)
            verdict_result = await self.invoke(self.critic, prompt)
            verdict = _parse_json_or_empty(verdict_result.text)
            accepted = bool(verdict.get("accepted"))
            self._emit(
                "critic",
                "critique",
                {
                    "round": attempt + 1,
                    "accepted": accepted,
                    "violations": verdict.get("violations") or [],
                    "raw": verdict_result.text,
                },
            )
            if accepted:
                return draft
            # Hard stop: if this was the last round, we return the
            # draft anyway with the critic's notes attached to the
            # event — the UI can surface "Critic not fully satisfied".
            if attempt == MAX_CRITIC_ROUNDS - 1:
                break
            revision_prompt = _writer_revision_input(draft, verdict)
            revision = await self.invoke(self.writer, revision_prompt)
            draft = revision.text
            self._emit("writer", "revision", {"text": draft, "round": attempt + 2})
        return draft

    # ── Event helper ──────────────────────────────────────────

    def _emit(self, agent_name: str, kind: str, payload: dict[str, Any]) -> int:
        """Append one row to the session's event log and return its
        seq. The service layer's SSE reader polls by seq to stream
        these frames to the UI."""
        return append_event(
            self.conn,
            session_id=self.session_id,
            agent_name=agent_name,
            kind=kind,
            payload=payload,
        )


# ── Prompt composition (plain strings — intentionally simple) ──


def _retriever_input(question: str, plan: dict) -> str:
    return f"Question: {question}\n\nPlan: {json.dumps(plan, indent=2)}\n\nRetrieve the evidence."


def _analyst_input(question: str, plan: dict, evidence: str) -> str:
    return (
        f"Question: {question}\n\nPlan: {json.dumps(plan, indent=2)}\n\n"
        f"Retrieved evidence:\n{evidence}\n\nRun analysis per the plan."
    )


def _writer_input(question: str, plan: dict, evidence: str, analysis: str | None) -> str:
    analysis_section = f"\n\nAnalyst output:\n{analysis}" if analysis else ""
    return (
        f"Question: {question}\n\nEvidence:\n{evidence}{analysis_section}\n\n"
        "Compose the final markdown answer. Cite every factual claim."
    )


def _writer_revision_input(draft: str, verdict: dict) -> str:
    notes = verdict.get("violations") or []
    notes_txt = "\n".join(f"- {v.get('kind','?')}: {v.get('note','')}  (quote: {v.get('quote','')})" for v in notes)
    return (
        f"Previous draft:\n{draft}\n\nCritic found these violations:\n{notes_txt}\n\n"
        "Revise the draft. Fix every violation. Keep the rest."
    )


def _critic_input(draft: str, evidence: str, analysis: str | None) -> str:
    analysis_section = f"\n\nAnalyst output:\n{analysis}" if analysis else ""
    return (
        f"Draft answer:\n{draft}\n\nRetrieved evidence:\n{evidence}{analysis_section}\n\n"
        "Review and emit your JSON verdict."
    )


# ── Helpers ─────────────────────────────────────────────────────


def _parse_json_or_empty(text: str) -> dict:
    """Agents are instructed to emit pure JSON, but LLMs occasionally
    wrap in code fences or prose. Try the cleanest-looking slice
    first, fall back to {}."""
    if not text:
        return {}
    s = text.strip()
    if s.startswith("```"):
        # Strip code fence.
        s = s.split("\n", 1)[-1]
        if s.endswith("```"):
            s = s.rsplit("```", 1)[0]
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # Last resort: find the first { and last }.
        start = s.find("{")
        end = s.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(s[start : end + 1])
            except json.JSONDecodeError:
                pass
        return {}


def _evidence_to_records(evidence_text: str) -> list[dict] | None:
    """The Retriever returns free-form text; the Analyst's sandbox
    wants a structured records list (pandas DataFrame seed). Phase 5
    leaves this as None — the Analyst still gets `evidence` (empty
    DataFrame) and can query the DB via `db` if it needs rows. Phase 6
    will plumb structured records through once the Retriever's output
    stabilises."""
    return None
