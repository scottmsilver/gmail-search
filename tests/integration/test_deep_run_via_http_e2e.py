"""End-to-end test of the full deep-mode pipeline via HTTP.

POSTs to the live FastAPI `/api/agent/analyze` endpoint with the
`claude_code` backend, consumes the SSE stream, and asserts a
`final` event lands. Triggers all five stages (planner, retriever,
analyst-if-needed, writer, critic) end-to-end. Cost: ~$0.005 per
run for a trivial sql-only question.
"""

from __future__ import annotations

import asyncio
import json
import re

import httpx
import pytest

from . import _stack_probe

pytestmark = pytest.mark.integration


_ANALYZE_URL = f"{_stack_probe.GMAIL_FASTAPI_URL}/api/agent/analyze"


async def _stream_analyze(body: dict, timeout: float = 180.0) -> list[tuple[str, dict]]:
    """POST to /api/agent/analyze and collect every SSE event as
    `(event_type, data_dict)` pairs. Times out generously — the deep
    pipeline can take ~30s on a cold cache."""
    events: list[tuple[str, dict]] = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", _ANALYZE_URL, json=body) as response:
            response.raise_for_status()
            current_event: str | None = None
            async for raw_line in response.aiter_lines():
                line = raw_line.rstrip("\r")
                if not line:
                    current_event = None
                    continue
                if line.startswith("event: "):
                    current_event = line[len("event: ") :].strip()
                    continue
                if line.startswith("data: "):
                    payload_text = line[len("data: ") :]
                    try:
                        payload = json.loads(payload_text)
                    except ValueError:
                        payload = {"_raw": payload_text}
                    events.append((current_event or "message", payload))
    return events


def _extract_final_text(events: list[tuple[str, dict]]) -> str | None:
    """Pull the writer's final answer text out of the event log.
    The orchestrator emits `event: final` with `payload.text`. The
    older shape (`{seq, text}`) is also accepted for resilience to
    downstream payload renames."""
    for kind, payload in events:
        if kind != "final":
            continue
        text = payload.get("text")
        if isinstance(text, str) and text.strip():
            return text
        nested = payload.get("payload") or {}
        nested_text = nested.get("text") if isinstance(nested, dict) else None
        if isinstance(nested_text, str) and nested_text.strip():
            return nested_text
    return None


def _session_id_from_events(events: list[tuple[str, dict]]) -> str | None:
    """First frame is `event: session` carrying the new session_id."""
    for kind, payload in events:
        if kind == "session":
            sid = payload.get("session_id")
            if isinstance(sid, str) and sid:
                return sid
    return None


def test_analyze_endpoint_streams_final_event_for_claude_code(
    live_stack,
    integration_env,
    fresh_conversation_id,
    monkeypatch,
):
    """The full pipeline must produce a `final` event with non-empty
    text containing a digit (the answer to "how many messages")."""
    # `_real_run` only fires when GMAIL_DEEP_REAL=1; otherwise the
    # service falls back to the stub pipeline that emits canned
    # events. We want the live path here.
    monkeypatch.setenv("GMAIL_DEEP_REAL", "1")

    body = {
        "conversation_id": fresh_conversation_id,
        "question": "How many emails are in the messages table? Reply with just the number.",
        "model": "sonnet",
        "backend": "claude_code",
    }

    events = asyncio.run(_stream_analyze(body))

    assert events, "no SSE events received from /api/agent/analyze"
    final_text = _extract_final_text(events)
    assert final_text, f"no `final` event with text in stream; " f"got events: {[k for k, _ in events][:20]}"
    assert re.search(r"\d", final_text), f"final answer should contain a digit (count); got {final_text!r}"


def test_backend_field_overrides_env(
    live_stack,
    integration_env,
    mcp_admin_token,
    fresh_conversation_id,
    monkeypatch,
):
    """`backend: "claude_code"` in the request body must beat the
    env-default `adk`. We verify the run actually went through
    claudebox by checking the side-channel admin endpoint records
    tool calls for the session_id the service announced."""
    monkeypatch.setenv("GMAIL_DEEP_REAL", "1")
    monkeypatch.setenv("GMAIL_DEEP_BACKEND", "adk")  # env default that should be overridden

    body = {
        "conversation_id": fresh_conversation_id,
        "question": "How many emails are in the messages table? Reply with just the number.",
        "model": "sonnet",
        "backend": "claude_code",
    }

    events = asyncio.run(_stream_analyze(body))

    session_id = _session_id_from_events(events)
    assert session_id, f"no `session` event in stream; got {[k for k, _ in events][:10]}"

    # If we went through the claude_code backend, the orchestrator
    # would have registered the session_id with the MCP server.
    # Calls might already be cleared (unregister fires in finally),
    # so we instead look for tool_call events emitted to the SSE
    # stream — those are populated from the side-channel only when
    # the claude_code path runs.
    tool_call_events = [k for k, _ in events if k == "tool_call"]
    assert tool_call_events, (
        "no tool_call events in stream — claude_code backend should have "
        f"emitted at least one. Event kinds: {sorted(set(k for k, _ in events))}"
    )
