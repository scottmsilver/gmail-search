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

# Input-prompt budget per downstream stage. Writer + Critic both
# receive evidence + analysis + draft concatenated; without clipping,
# a chatty Retriever or a noisy Analyst stdout can push the combined
# prompt past Gemini's 1M-token input cap and the whole turn fails
# with 400 INVALID_ARGUMENT. 80k chars ≈ ~20k tokens per field, well
# under the cap even if all fields are full. The clip keeps the HEAD
# (where the model usually put the important summary) plus a short
# tail hint so truncation is obvious.
STAGE_FIELD_CHAR_CAP = 80_000


def _clip_for_prompt(text: str | None, *, cap: int = STAGE_FIELD_CHAR_CAP) -> str:
    """Truncate a stage-to-stage field to `cap` chars with an explicit
    "[truncated: N chars dropped]" marker. None is passed through as
    empty string so callers don't have to guard.

    NOTE on the head-bias (cap chars from the FRONT, tail dropped):
    this is only safe because Writer + Critic both consume citations
    from the explicit "Allowed citations" block we render in
    `_format_allowed_citations` — that allow-list is built from
    `_cite_refs_from_tool_calls` / `_artifact_ids_from_tool_calls`,
    NOT by scraping `[ref:...]` / `[art:N]` tokens out of the
    evidence/analysis text we clip here. So a `[ref:...]` that
    happens to live in the truncated tail of evidence is still in the
    allow-list and the Writer can still cite it.
    If anyone ever rips out the allow-list mechanism and goes back to
    "Writer scrapes citations from the evidence text directly", this
    head-only truncation will silently drop tail citations and the
    Writer will produce uncited claims (or worse, invent ids). Keep
    those two pieces in lockstep — either both or neither.
    """
    if not text:
        return ""
    if len(text) <= cap:
        return text
    dropped = len(text) - cap
    return text[:cap] + f"\n\n[truncated: {dropped:,} chars dropped]"


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
            evidence, cite_refs = await self._run_retriever(question, plan)
            analysis = await self._run_analyst_if_needed(question, plan, evidence)
            artifact_ids = list(analysis["artifact_ids"]) if analysis else []
            draft = await self._run_writer(question, plan, evidence, analysis, cite_refs, artifact_ids)
            draft = await self._run_critic_loop(draft, evidence, analysis, cite_refs, artifact_ids)
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

    async def _run_retriever(self, question: str, plan: dict) -> tuple[str, list[str]]:
        """Retriever returns a short summary of what it found +
        cite_refs. We pass it downstream as free text (the
        summary) AND as a structured list of cite_refs extracted
        from the retrieval tool results — the Writer and Critic both
        need the explicit list so they can ground citations instead
        of inventing them.
        """
        prompt = _retriever_input(question, plan)
        result = await self.invoke(self.retriever, prompt)
        for tc in result.tool_calls:
            self._emit("retriever", "tool_call", tc)
        cite_refs = _cite_refs_from_tool_calls(result.tool_calls)
        self._emit(
            "retriever",
            "evidence",
            {"summary": result.text, "cite_refs": cite_refs},
        )
        return result.text, cite_refs

    async def _run_analyst_if_needed(self, question: str, plan: dict, evidence: str) -> dict | None:
        """Run the Analyst when the plan has analysis steps. Skip
        entirely on empty plans (factual questions don't need code).

        Returns a dict carrying the analyst's summary text AND the
        list of `artifact_ids` the run_code tool actually persisted
        (empty if the analyst decided code wasn't needed). The
        Writer + Critic both consume artifact_ids from this dict so
        they can't invent `[art:N]` citations.
        """
        steps = plan.get("analysis") or []
        if not steps:
            self._emit("analyst", "skipped", {"reason": "no analysis steps in plan"})
            return None
        evidence_records = _evidence_to_records(evidence)
        analyst = self.analyst_factory(evidence_records)
        prompt = _analyst_input(question, plan, evidence)
        result = await self.invoke(analyst, prompt)
        artifact_ids = _artifact_ids_from_tool_calls(result.tool_calls)
        for tc in result.tool_calls:
            self._emit("analyst", "code_run", tc)
        self._emit(
            "analyst",
            "analysis",
            {
                "summary": result.text,
                "artifact_ids": artifact_ids,
                "called_run_code": bool(result.tool_calls),
            },
        )
        return {"summary": result.text, "artifact_ids": artifact_ids}

    async def _run_writer(
        self,
        question: str,
        plan: dict,
        evidence: str,
        analysis: dict | None,
        cite_refs: list[str],
        artifact_ids: list[int],
    ) -> str:
        """Writer composes the draft. Feed it question + all upstream
        outputs AND the explicit allowed-citation lists so it can't
        invent [ref:X] or [art:N] tokens that don't exist.
        """
        analysis_text = analysis["summary"] if analysis else None
        prompt = _writer_input(question, plan, evidence, analysis_text, cite_refs, artifact_ids)
        result = await self.invoke(self.writer, prompt)
        self._emit("writer", "draft", {"text": result.text})
        return result.text

    async def _run_critic_loop(
        self,
        draft: str,
        evidence: str,
        analysis: dict | None,
        cite_refs: list[str],
        artifact_ids: list[int],
    ) -> str:
        """Critic → Writer revision loop, capped at MAX_CRITIC_ROUNDS.
        Each pass: critic emits a JSON verdict; if rejected, writer
        produces one revision; next round critic reviews again. We
        stop on accept OR when the cap trips — whichever comes
        first.
        """
        analysis_text = analysis["summary"] if analysis else None
        for attempt in range(MAX_CRITIC_ROUNDS):
            prompt = _critic_input(draft, evidence, analysis_text, cite_refs, artifact_ids)
            verdict_result = await self.invoke(self.critic, prompt)
            verdict, parse_failed = _parse_json_or_empty_with_status(verdict_result.text)
            # Surface parse failures explicitly so a consistently-broken
            # Critic doesn't silently burn the full revision budget on
            # garbage. The orchestrator still treats the round as
            # rejected (accepted=False), but the UI now sees WHY.
            if parse_failed:
                self._emit(
                    "critic",
                    "critique_parse_error",
                    {
                        "round": attempt + 1,
                        "raw": verdict_result.text,
                    },
                )
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
        f"Retrieved evidence:\n{_clip_for_prompt(evidence)}\n\nRun analysis per the plan."
    )


def _format_allowed_citations(cite_refs: list[str], artifact_ids: list[int]) -> str:
    """Render the grounded-citation allow-list both Writer + Critic
    consume. If a list is empty we still emit the header so the
    agent knows that class of citation is off-limits rather than
    missing."""
    ref_line = ", ".join(f"[ref:{r}]" for r in cite_refs) if cite_refs else "(none)"
    art_line = ", ".join(f"[art:{a}]" for a in artifact_ids) if artifact_ids else "(none)"
    return (
        "Allowed citations (these are the ONLY values you may use):\n"
        f"  threads: {ref_line}\n"
        f"  artifacts: {art_line}"
    )


def _writer_input(
    question: str,
    plan: dict,
    evidence: str,
    analysis: str | None,
    cite_refs: list[str],
    artifact_ids: list[int],
) -> str:
    # Both evidence + analysis can be arbitrarily long (Retriever
    # narrative, Analyst stdout). Clip each field independently so
    # one huge field doesn't starve the other.
    evidence_clipped = _clip_for_prompt(evidence)
    analysis_clipped = _clip_for_prompt(analysis) if analysis else ""
    analysis_section = f"\n\nAnalyst output:\n{analysis_clipped}" if analysis else ""
    allowed = _format_allowed_citations(cite_refs, artifact_ids)
    return (
        f"Question: {question}\n\nEvidence:\n{evidence_clipped}{analysis_section}\n\n"
        f"{allowed}\n\n"
        "Compose the final markdown answer. Cite every factual claim you make, "
        "using ONLY the citations from the allowed list above. If a claim has no "
        "matching citation available, state the claim without one rather than "
        "inventing a ref/artifact id."
    )


def _writer_revision_input(draft: str, verdict: dict) -> str:
    notes = verdict.get("violations") or []
    notes_txt = "\n".join(f"- {v.get('kind','?')}: {v.get('note','')}  (quote: {v.get('quote','')})" for v in notes)
    return (
        f"Previous draft:\n{draft}\n\nCritic found these violations:\n{notes_txt}\n\n"
        "Revise the draft. Fix every violation. Keep the rest."
    )


def _critic_input(
    draft: str,
    evidence: str,
    analysis: str | None,
    cite_refs: list[str],
    artifact_ids: list[int],
) -> str:
    evidence_clipped = _clip_for_prompt(evidence)
    analysis_clipped = _clip_for_prompt(analysis) if analysis else ""
    analysis_section = f"\n\nAnalyst output:\n{analysis_clipped}" if analysis else ""
    allowed = _format_allowed_citations(cite_refs, artifact_ids)
    return (
        f"Draft answer:\n{draft}\n\nRetrieved evidence:\n{evidence_clipped}{analysis_section}\n\n"
        f"{allowed}\n\n"
        "Review and emit your JSON verdict. ANY [ref:*] or [art:*] in the draft "
        "that is NOT in the allowed list above is an `invented_citation` violation. "
        "Cross-check every citation token in the draft against the allowed list."
    )


# ── Tool-call inspectors ───────────────────────────────────────────


def _has_run_code_invocation(tool_calls: list[dict]) -> bool:
    """True if the Analyst's tool_calls contain at least one
    invocation of `run_code`. Used for telemetry only — we do NOT
    force the Analyst to call the tool; prose-only is a legitimate
    outcome when the evidence already answers the question."""
    return any(tc.get("name") == "run_code" for tc in tool_calls)


def _artifact_ids_from_tool_calls(tool_calls: list[dict]) -> list[int]:
    """Walk Analyst tool_results (both function_call and
    function_response shapes ADK produces) and collect the artifact
    ids that `run_code` persisted. These are the ONLY ids the Writer
    may cite as `[art:N]`."""
    ids: list[int] = []
    for tc in tool_calls:
        resp = tc.get("response")
        if not isinstance(resp, dict):
            continue
        for art in resp.get("artifacts") or []:
            if isinstance(art, dict) and isinstance(art.get("id"), int):
                ids.append(int(art["id"]))
    return ids


def _cite_refs_from_tool_calls(tool_calls: list[dict]) -> list[str]:
    """Walk Retriever tool_results and collect the `cite_ref` tokens
    returned by search_emails / query_emails. These are the ONLY
    tokens the Writer may cite as `[ref:CITE_REF]`."""
    refs: list[str] = []
    seen: set[str] = set()
    for tc in tool_calls:
        resp = tc.get("response")
        if not isinstance(resp, dict):
            continue
        # search_emails + query_emails both return {results: [...]}
        for row in resp.get("results") or []:
            cr = row.get("cite_ref") if isinstance(row, dict) else None
            if isinstance(cr, str) and cr not in seen:
                seen.add(cr)
                refs.append(cr)
    return refs


# ── Helpers ─────────────────────────────────────────────────────


def _parse_json_or_empty(text: str) -> dict:
    """Agents are instructed to emit pure JSON, but LLMs occasionally
    wrap in code fences or prose. Try the cleanest-looking slice
    first, fall back to {}."""
    parsed, _failed = _parse_json_or_empty_with_status(text)
    return parsed


def _parse_json_or_empty_with_status(text: str) -> tuple[dict, bool]:
    """Same as `_parse_json_or_empty` but ALSO returns whether parsing
    failed. Empty input is treated as "nothing to parse" (failed=False)
    so we don't spam parse-error events when an agent legitimately
    returned no text. Non-empty input that we couldn't coerce into a
    dict comes back as ({}, True) — caller surfaces a parse-error
    event."""
    if not text:
        return {}, False
    s = text.strip()
    if s.startswith("```"):
        # Strip code fence.
        s = s.split("\n", 1)[-1]
        if s.endswith("```"):
            s = s.rsplit("```", 1)[0]
    try:
        result = json.loads(s)
        return (result, False) if isinstance(result, dict) else ({}, True)
    except json.JSONDecodeError:
        # Last resort: find the first { and last }.
        start = s.find("{")
        end = s.rfind("}")
        if start >= 0 and end > start:
            try:
                result = json.loads(s[start : end + 1])
                return (result, False) if isinstance(result, dict) else ({}, True)
            except json.JSONDecodeError:
                pass
        return {}, True


def _evidence_to_records(evidence_text: str) -> list[dict] | None:
    """The Retriever returns free-form text; the Analyst's sandbox
    wants a structured records list (pandas DataFrame seed). Phase 5
    leaves this as None — the Analyst still gets `evidence` (empty
    DataFrame) and can query the DB via `db` if it needs rows. Phase 6
    will plumb structured records through once the Retriever's output
    stabilises."""
    return None
