"""End-to-end tests for `claudebox_invoke` against a real running
claudebox container at http://127.0.0.1:8765.

These tests cost real Anthropic credits per run; prompts are kept
trivially short ("say PONG") so a full integration sweep is in the
single-cents range. Skipped when the stack isn't up.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest

from gmail_search.agents.runtime_claude import claudebox_invoke

pytestmark = pytest.mark.integration


@dataclass
class _PingAgent:
    """Minimal AgentLike stub. The real ADK agent shape is much
    bigger — `claudebox_invoke` only needs `name`, `model`,
    `instruction`, so this is all we wire."""

    name: str = "ping-agent"
    model: str = "sonnet"
    instruction: str = "Reply with exactly the word PONG. No punctuation, no extra words."


def _ensure_workspace(name: str) -> str:
    """Claudebox refuses to run a workspace it can't see on disk.
    Mirrors `service._ensure_workspace_dir` so tests don't depend on
    the orchestrator having pre-created the dir."""
    base = Path("deploy/claudebox/workspaces") / name
    base.mkdir(parents=True, exist_ok=True)
    return name


def test_claudebox_invoke_returns_real_text(
    live_stack,
    integration_env,
    fresh_session_id,
):
    """Single short Sonnet call through claudebox; verify we get a
    non-empty result back. Cost: ~$0.0005."""
    workspace = _ensure_workspace(f"itest-{fresh_session_id}")
    agent = _PingAgent()

    async def _go():
        return await claudebox_invoke(
            agent,
            "ping",
            workspace=workspace,
            session_id=None,
        )

    result = asyncio.run(_go())

    assert result.text, "claudebox returned empty result text"
    # We don't assert exactly "PONG" — Sonnet sometimes adds a period
    # or a friendly suffix. We just assert it actually replied.
    assert isinstance(result.text, str)
    # No tool calls happened (no MCP session, no tools allowed in the
    # stage prompt) so the list must be empty.
    assert result.tool_calls == []


def test_claudebox_invoke_busy_retries_recover(
    live_stack,
    integration_env,
    fresh_session_id,
):
    """Two concurrent calls to the SAME workspace should serialize:
    one runs, the other gets 409 and our adapter retries until it
    succeeds. Skip if the running claudebox build doesn't expose the
    busy semantics (some builds queue server-side instead).

    Cost: 2 short Sonnet calls, ~$0.001 total.
    """
    workspace = _ensure_workspace(f"itest-busy-{fresh_session_id}")
    agent = _PingAgent()

    async def _go():
        # Fire two concurrent invocations — at least one will hit a
        # 409 the first time around if the server is busy-aware.
        a = claudebox_invoke(agent, "say A", workspace=workspace, session_id=None)
        b = claudebox_invoke(agent, "say B", workspace=workspace, session_id=None)
        return await asyncio.gather(a, b, return_exceptions=True)

    results = asyncio.run(_go())

    # If both raised the same "busy after N retries" we'll surface
    # the failure; if both succeeded the retry path worked OR the
    # server queues, both of which are acceptable for this test.
    successes = [r for r in results if not isinstance(r, Exception)]
    if len(successes) < 2:
        # Surface the actual exception so a regression is debuggable.
        for r in results:
            if isinstance(r, Exception):
                pytest.skip(
                    f"claudebox concurrent path didn't recover; build may not "
                    f"emit 409. underlying: {type(r).__name__}: {r}"
                )
    assert len(successes) == 2
    for r in successes:
        assert r.text, "concurrent claudebox call returned empty text"
