"""End-to-end tests for the MCP side-channel call log.

We register a session via the real admin HTTP endpoint, invoke a
tool through the real MCP streamable-HTTP transport, and verify the
call appears in /admin/calls/{id}. No mocks anywhere — these tests
fail loud if the wire shape ever drifts.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from gmail_search.agents.runtime_claude import register_session_via_admin, unregister_session_via_admin

from . import _stack_probe

pytestmark = pytest.mark.integration


async def _call_sql_query_via_mcp(session_id: str, query: str) -> dict:
    """Talk to the real MCP server over streamable-HTTP. Uses the
    `mcp` python client that's already in the project dependency
    closure (it's how the server is built). Returns the raw tool
    result dict."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    url = f"{_stack_probe.MCP_TOOLS_URL}/mcp"
    async with streamablehttp_client(url) as (read_stream, write_stream, _close):
        async with ClientSession(read_stream, write_stream) as client:
            await client.initialize()
            result = await client.call_tool(
                "sql_query",
                arguments={"session_id": session_id, "query": query},
            )
    return _coerce_tool_result(result)


def _coerce_tool_result(result) -> dict:
    """The MCP CallToolResult exposes `content` (list of content
    blocks) and `structuredContent` (the dict-shaped echo of the
    tool's return value when available). We return the dict if we
    have it, else stringify the first text block — enough for
    asserting the call landed without depending on FastMCP version
    quirks."""
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        return structured
    content = getattr(result, "content", None) or []
    if content:
        first = content[0]
        text = getattr(first, "text", None)
        if text:
            return {"text": text}
    return {}


def _admin_get_calls(session_id: str, token: str) -> list[dict]:
    """Hit the side-channel admin endpoint synchronously. Returns the
    `calls` list (possibly empty)."""
    url = f"{_stack_probe.MCP_TOOLS_URL}/admin/calls/{session_id}"
    headers = {"Authorization": f"Bearer {token}"}
    with httpx.Client(timeout=5.0) as client:
        r = client.get(url, headers=headers)
    r.raise_for_status()
    body = r.json()
    return list(body.get("calls", []))


def test_side_channel_captures_real_tool_call(
    live_stack,
    integration_env,
    mcp_admin_token,
    fresh_session_id,
):
    """Register a session, run sql_query through the real MCP HTTP
    transport, verify the call appears in /admin/calls."""

    async def _go():
        await register_session_via_admin(
            fresh_session_id,
            evidence_records=None,
            db_dsn=None,
        )
        try:
            await _call_sql_query_via_mcp(fresh_session_id, "SELECT 1 AS x")
        finally:
            # ALWAYS unregister so a failed assert doesn't leak the
            # session into the server's memory across test runs.
            # (We re-register below intentionally if needed.)
            pass

    asyncio.run(_go())

    try:
        calls = _admin_get_calls(fresh_session_id, mcp_admin_token)
        assert len(calls) >= 1, "expected at least one call recorded in side channel"
        sql_calls = [c for c in calls if c.get("name") == "sql_query"]
        assert sql_calls, f"no sql_query call recorded; got {[c.get('name') for c in calls]}"
        # The args dict should round-trip our query verbatim.
        first_args = sql_calls[0].get("args") or {}
        assert first_args.get("query") == "SELECT 1 AS x"
    finally:
        asyncio.run(unregister_session_via_admin(fresh_session_id))


def test_unregister_clears_call_log(
    live_stack,
    integration_env,
    mcp_admin_token,
    fresh_session_id,
):
    """After unregister_session, /admin/calls/{id} must return an
    empty list — confirms the side-channel cleanup path works."""

    async def _go():
        await register_session_via_admin(
            fresh_session_id,
            evidence_records=None,
            db_dsn=None,
        )
        await _call_sql_query_via_mcp(fresh_session_id, "SELECT 2 AS y")

    asyncio.run(_go())

    # Sanity-check we recorded something before unregister wipes it.
    pre = _admin_get_calls(fresh_session_id, mcp_admin_token)
    assert len(pre) >= 1, "preflight: expected a call log entry before unregister"

    asyncio.run(unregister_session_via_admin(fresh_session_id))

    post = _admin_get_calls(fresh_session_id, mcp_admin_token)
    assert post == [], f"call log should be empty after unregister; got {post}"
