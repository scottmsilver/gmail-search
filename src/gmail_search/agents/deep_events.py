"""Event emitters shared by the single-agent deep backends
(`claude_native`, `pi`).

Both backends run ONE agent loop with every tool available and then
synthesize the orchestrator's event vocabulary (`plan` / `tool_call` /
`evidence` / `code_run` / `analysis` / `draft` / `final`) so the UI's
deep-mode panels render without change. This module owns that
synthesis; the runtimes own only how they drive their agent.
"""

from __future__ import annotations

import logging

from gmail_search.agents.orchestration import _artifact_ids_from_tool_calls, _cite_refs_from_tool_calls
from gmail_search.agents.session import append_event

logger = logging.getLogger(__name__)

# Names of the retrieval tools the UI surfaces under the "retriever"
# agent. `run_code` is treated separately as the Analyst's tool.
RETRIEVAL_TOOL_NAMES = frozenset({"search_emails", "query_emails", "get_thread", "sql_query"})


def _is_args_entry(tc: dict) -> bool:
    """Side-channel records are split into one `{name, args}` and one
    `{name, response}` entry per tool call (see
    `_tool_calls_from_side_channel`). For tool_call telemetry we only
    want the args-shape — emitting both would double-count."""
    return "args" in tc and "response" not in tc


def _is_response_entry(tc: dict) -> bool:
    return "response" in tc


def _retrieval_args_entries(tool_calls: list[dict]) -> list[dict]:
    """Args-shape entries for the retrieval tools (search/query/get/sql).
    These are the ones the UI's retriever panel renders."""
    return [tc for tc in tool_calls if _is_args_entry(tc) and tc.get("name") in RETRIEVAL_TOOL_NAMES]


def _run_code_response_entries(tool_calls: list[dict]) -> list[dict]:
    """Response-shape entries for `run_code`. These carry the
    artifact ids the UI's analyst panel needs to render code-run cards."""
    return [tc for tc in tool_calls if _is_response_entry(tc) and tc.get("name") == "run_code"]


def _has_run_code(tool_calls: list[dict]) -> bool:
    return any(tc.get("name") == "run_code" for tc in tool_calls)


def _retriever_summary(retrieval_calls: list[dict]) -> str:
    """One-liner summarizing what the retrieval pass did. Empty when
    no retrieval calls fired so the UI can hide the panel."""
    if not retrieval_calls:
        return "No retrieval tools invoked."
    by_name: dict[str, int] = {}
    for tc in retrieval_calls:
        n = str(tc.get("name") or "?")
        by_name[n] = by_name.get(n, 0) + 1
    parts = [f"{count}× {name}" for name, count in sorted(by_name.items())]
    return f"Retrieval calls: {', '.join(parts)}."


def _analyst_summary(run_code_calls: list[dict], artifact_ids: list[int]) -> str:
    """One-liner for the analyst panel describing the run_code pass."""
    if not run_code_calls:
        return "No code execution."
    art_blurb = f", produced {len(artifact_ids)} artifact(s)" if artifact_ids else ""
    return f"Ran {len(run_code_calls)} code block(s){art_blurb}."


def emit_plan_event(conn, session_id: str, *, agent_name: str = "native", approach: str) -> None:
    append_event(
        conn,
        session_id=session_id,
        agent_name=agent_name,
        kind="plan",
        payload={
            "native_mode": True,
            "approach": approach,
        },
    )


def emit_retriever_events(
    conn,
    session_id: str,
    tool_calls: list[dict],
    *,
    skip_per_tool_emission: bool = False,
) -> None:
    """Mirror Orchestrator._run_retriever's emission shape: one
    `tool_call` per retrieval invocation, then one rolled-up `evidence`
    event with the cite_refs the Writer would have been allowed to use.

    `skip_per_tool_emission` short-circuits the per-call loop when
    those events were already streamed mid-flight via the runtime's
    `event_sink`. The aggregate `evidence` event still fires because
    downstream UI panels depend on it."""
    retrieval_calls = _retrieval_args_entries(tool_calls)
    if not skip_per_tool_emission:
        for tc in retrieval_calls:
            append_event(
                conn,
                session_id=session_id,
                agent_name="retriever",
                kind="tool_call",
                payload=tc,
            )
    cite_refs = _cite_refs_from_tool_calls(tool_calls)
    append_event(
        conn,
        session_id=session_id,
        agent_name="retriever",
        kind="evidence",
        payload={
            "summary": _retriever_summary(retrieval_calls),
            "cite_refs": cite_refs,
        },
    )


def emit_analyst_events(
    conn,
    session_id: str,
    tool_calls: list[dict],
    *,
    skip_per_tool_emission: bool = False,
) -> None:
    """Mirror Orchestrator._run_analyst_if_needed: one `code_run` per
    run_code response + one rolled-up `analysis` event. Only fires the
    `analysis` event if the agent actually ran code — matches the
    orchestrator's "skipped" branch by simply staying silent.

    `skip_per_tool_emission` skips the per-call code_run emission
    when those events were already streamed via `event_sink`. The
    aggregate `analysis` event still fires."""
    run_code_calls = _run_code_response_entries(tool_calls)
    if not run_code_calls:
        return
    if not skip_per_tool_emission:
        for tc in run_code_calls:
            append_event(
                conn,
                session_id=session_id,
                agent_name="analyst",
                kind="code_run",
                payload=tc,
            )
    artifact_ids = _artifact_ids_from_tool_calls(tool_calls)
    append_event(
        conn,
        session_id=session_id,
        agent_name="analyst",
        kind="analysis",
        payload={
            "summary": _analyst_summary(run_code_calls, artifact_ids),
            "artifact_ids": artifact_ids,
            "called_run_code": _has_run_code(tool_calls),
        },
    )


def emit_writer_and_final(conn, session_id: str, text: str) -> None:
    """Writer panel surfaces the final markdown via `draft`; the root
    event signals turn-complete to the SSE proxy. Both carry the same
    text so a UI reading either gets the right answer."""
    append_event(
        conn,
        session_id=session_id,
        agent_name="writer",
        kind="draft",
        payload={"text": text},
    )
    append_event(
        conn,
        session_id=session_id,
        agent_name="root",
        kind="final",
        payload={"text": text},
    )


def emit_error(conn, session_id: str, exc: BaseException, *, agent_name: str = "native") -> None:
    append_event(
        conn,
        session_id=session_id,
        agent_name=agent_name,
        kind="error",
        payload={"message": str(exc)},
    )


def sweep_and_extend_final_text(
    conn,
    *,
    session_id: str,
    workspace: str | None,
    conversation_id: str | None,
    turn_started_at: float,
    base_text: str,
) -> str:
    """End-of-turn auto-publish sweep + footer assembly. Catches any
    file Claude wrote but didn't explicitly publish, inserts each
    into agent_artifacts, and appends `[art:<id>] <name>` chips to
    the final answer text.

    Failures are logged but never raised — the sweep is a safety net,
    not a hard dependency."""
    # Inline imports because the formatter strips module-level imports
    # that aren't referenced at module top-level.
    from gmail_search.agents.auto_publish import auto_publish_unpublished_files, build_auto_publish_footer

    try:
        published = auto_publish_unpublished_files(
            conn,
            session_id=session_id,
            workspace=workspace,
            conversation_id=conversation_id,
            turn_started_at=turn_started_at,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "auto_publish sweep raised for session %s: %s",
            session_id,
            exc,
        )
        return base_text
    if not published:
        return base_text
    return base_text + build_auto_publish_footer(published)
