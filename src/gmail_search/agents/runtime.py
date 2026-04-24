"""ADK runtime adapter: turns an `LlmAgent` + prompt into a
`StageResult` the orchestrator consumes.

The orchestrator (Phase 5) takes an abstract `invoke: InvokeFn`
callable so tests can substitute a mock. This module provides the
REAL implementation — one that spins up an ADK `Runner` with an
in-memory session service, feeds the prompt as the first user
message, and drains the event stream until the agent yields a final
text.

Kept intentionally thin. Everything ADK-idiomatic (session state
shared across stages, explicit plugin wiring, artifact service
integration) is deferred to a follow-on because we haven't needed
any of it yet — each sub-agent gets its own one-shot session, and
the orchestrator already owns evidence handoff through prompts
rather than through ADK's session state.
"""

from __future__ import annotations

import logging
from typing import Any

from gmail_search.agents.orchestration import StageResult

logger = logging.getLogger(__name__)


# Identifier for the ADK session service + runner. One name for every
# sub-agent invocation is fine because each call creates its own
# session-id and the runner only uses app_name for logging.
_APP_NAME = "gmail_search_deep"
_USER_ID = "local"


def _extract_text_and_tool_calls(events: list) -> StageResult:
    """Walk ADK events produced by one agent run and collapse them
    into a single StageResult.

    Text: we concatenate `content.parts[*].text` from every event
    that has content and flag the LAST event as the final response
    (ADK marks it via `is_final_response()` on EventActions).

    Tool calls: every event that references a `function_call` or
    `function_response` gets recorded. The orchestrator emits these
    as `tool_call` events on its own transcript so the UI can show
    "Retriever called search_emails(q=...)" live.
    """
    final_text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for ev in events:
        # Extract tool calls (emitted as function_call parts).
        content = getattr(ev, "content", None)
        parts = getattr(content, "parts", None) if content is not None else None
        if parts:
            for p in parts:
                fc = getattr(p, "function_call", None)
                if fc is not None:
                    tool_calls.append(
                        {
                            "name": getattr(fc, "name", ""),
                            "args": dict(getattr(fc, "args", {}) or {}),
                        }
                    )
                fr = getattr(p, "function_response", None)
                if fr is not None:
                    tool_calls.append(
                        {
                            "name": getattr(fr, "name", ""),
                            "response": dict(getattr(fr, "response", {}) or {}),
                        }
                    )
                # Text parts of the final response.
                text = getattr(p, "text", None)
                if text and _is_final_response(ev):
                    final_text_parts.append(text)
    return StageResult(text="".join(final_text_parts).strip(), tool_calls=tool_calls)


def _is_final_response(event) -> bool:
    """ADK tags the terminal event with `is_final_response() == True`
    on its EventActions. Not every ADK version exposes that as a
    method, so we probe defensively."""
    check = getattr(event, "is_final_response", None)
    if callable(check):
        try:
            return bool(check())
        except Exception:  # noqa: BLE001
            return False
    return False


async def adk_invoke(agent, prompt: str) -> StageResult:
    """Real `invoke` for the orchestrator. Creates a fresh ADK
    session per call, feeds the prompt as the user message, drains
    the event stream, and returns the collapsed StageResult.

    One session per invocation is a deliberate choice: the
    orchestrator already curates context explicitly via per-stage
    prompts, so we don't need ADK's session memory to carry state
    across stages. Each sub-agent's "world" is whatever the
    orchestrator handed it.
    """
    from google.adk import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types as _types

    session_service = InMemorySessionService()
    # create_session may be sync or async depending on ADK version;
    # await when it returns a coroutine, otherwise treat as value.
    session_obj = session_service.create_session(app_name=_APP_NAME, user_id=_USER_ID)
    if hasattr(session_obj, "__await__"):
        session_obj = await session_obj
    session_id = getattr(session_obj, "id", None) or getattr(session_obj, "session_id", None)

    runner = Runner(app_name=_APP_NAME, agent=agent, session_service=session_service)

    user_message = _types.Content(role="user", parts=[_types.Part(text=prompt)])

    events = []
    async for ev in runner.run_async(
        user_id=_USER_ID,
        session_id=session_id,
        new_message=user_message,
    ):
        events.append(ev)

    return _extract_text_and_tool_calls(events)
