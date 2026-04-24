"""Tests for the ADK runtime adapter and the real-pipeline flag.

Only exercises shape + flag parsing here. Full end-to-end with live
Gemini is Phase 7 (integration suite, gated on GEMINI_API_KEY).
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

try:
    import google.adk  # noqa: F401

    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False


def test_use_real_pipeline_flag_parsing(monkeypatch):
    """`GMAIL_DEEP_REAL=1` (or true/yes) flips the pipeline from the
    stub to the live orchestrator. Anything else keeps the stub."""
    from gmail_search.agents.service import _use_real_pipeline

    monkeypatch.delenv("GMAIL_DEEP_REAL", raising=False)
    assert _use_real_pipeline() is False

    for truthy in ("1", "true", "True", "YES", "yes"):
        monkeypatch.setenv("GMAIL_DEEP_REAL", truthy)
        assert _use_real_pipeline() is True, f"{truthy!r} should be truthy"

    for falsy in ("0", "false", "no", "", "anything_else"):
        monkeypatch.setenv("GMAIL_DEEP_REAL", falsy)
        assert _use_real_pipeline() is False, f"{falsy!r} should be falsy"


@pytest.mark.skipif(not ADK_AVAILABLE, reason="google-adk not installed")
def test_adk_invoke_is_an_async_callable_matching_invoke_fn_shape():
    """The orchestrator's `invoke: InvokeFn` contract expects
    `async (agent, prompt: str) -> StageResult`. Lock the shape
    here so a signature drift breaks CI, not a production request."""
    from gmail_search.agents.runtime import adk_invoke

    sig = inspect.signature(adk_invoke)
    assert list(sig.parameters.keys()) == ["agent", "prompt"]
    assert asyncio.iscoroutinefunction(adk_invoke)


def test_extract_text_and_tool_calls_handles_empty_events():
    """Degenerate path: no events → empty StageResult. An agent run
    that yielded nothing (connection dropped, safety refusal) must
    not crash the orchestrator; it gets an empty draft to pass
    downstream."""
    from gmail_search.agents.runtime import _extract_text_and_tool_calls

    result = _extract_text_and_tool_calls([])
    assert result.text == ""
    assert result.tool_calls == []
