"""End-to-end test of the claude_native deep-mode pipeline via HTTP.

POSTs to the live FastAPI `/api/agent/analyze` with `backend:
"claude_native"`, consumes the SSE stream, and asserts a `final` event
lands. Single-agent path — one claudebox round-trip, no orchestrator.
Cost: ~$0.005 per run for a trivial count question.
"""

from __future__ import annotations

import asyncio
import re

import pytest

from . import _stack_probe
from .test_deep_run_via_http_e2e import _extract_final_text, _stream_analyze

pytestmark = pytest.mark.integration


_ANALYZE_URL = f"{_stack_probe.GMAIL_FASTAPI_URL}/api/agent/analyze"


def test_analyze_endpoint_streams_final_event_for_claude_native(
    live_stack,
    integration_env,
    fresh_conversation_id,
    monkeypatch,
):
    """The single-agent native pipeline must produce a `final` event
    with non-empty text containing a digit. Trivial question so the
    run is cheap (~$0.005)."""
    monkeypatch.setenv("GMAIL_DEEP_REAL", "1")

    body = {
        "conversation_id": fresh_conversation_id,
        "question": "How many emails from joy@gmail.com? Reply with just the number.",
        "model": "sonnet",
        "backend": "claude_native",
    }

    events = asyncio.run(_stream_analyze(body))

    assert events, "no SSE events received from /api/agent/analyze"
    final_text = _extract_final_text(events)
    assert final_text, f"no `final` event with text; got {[k for k, _ in events][:20]}"
    assert re.search(r"\d", final_text), f"final answer should contain a digit; got {final_text!r}"
    # Plan event uses agent_name="native" — confirms we took the
    # native_run path rather than the orchestrator.
    plan_payloads = [p for k, p in events if k == "plan"]
    assert plan_payloads, "expected a `plan` event in the stream"
    assert any(
        (p.get("payload") or {}).get("native_mode") is True or p.get("native_mode") is True for p in plan_payloads
    ), f"plan event should carry native_mode=True; got {plan_payloads[:2]}"
