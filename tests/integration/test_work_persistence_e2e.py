"""End-to-end tests for per-conversation /work persistence.

Goes through the real MCP server -> real `execute_in_sandbox` ->
real Docker container, with a host bind-mount at
`data/agent_scratch/<conversation_id>/`. Verifies a file written in
one run_code call survives across a SECOND run_code call in the
same conversation, and confirms backward-compat (no conversation
id => fresh tmpdir per call).
"""

from __future__ import annotations

import asyncio

import pytest

from gmail_search.agents.runtime_claude import register_session_via_admin, unregister_session_via_admin

from . import _stack_probe

pytestmark = pytest.mark.integration


async def _call_run_code_via_mcp(session_id: str, code: str) -> dict:
    """Run `code` in the sandbox via the real MCP transport. Returns
    the structured tool result with stdout/stderr/exit_code."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    url = f"{_stack_probe.MCP_TOOLS_URL}/mcp"
    async with streamablehttp_client(url) as (read_stream, write_stream, _close):
        async with ClientSession(read_stream, write_stream) as client:
            await client.initialize()
            result = await client.call_tool(
                "run_code",
                arguments={"session_id": session_id, "code": code},
            )
    return _coerce_run_code_result(result)


def _maybe_skip_on_server_db_config(result: dict) -> None:
    """The MCP server's run_code path opens a DB connection
    unconditionally to persist artifacts. If the running daemon was
    started without DB_DSN, every run_code call returns
    `session has no db_dsn — cannot persist artifacts` and we can't
    exercise /work persistence at all. That's a server-config issue,
    not a test issue — skip cleanly so an operator with a properly
    configured stack still gets coverage."""
    blob = (result.get("stdout") or "") + (result.get("stderr") or "")
    if "session has no db_dsn" in blob:
        pytest.skip(
            "MCP server has no DB_DSN configured; run_code path can't open "
            "its artifact-persist connection. Restart the MCP server with "
            "DB_DSN set to exercise this test."
        )


def _coerce_run_code_result(result) -> dict:
    """run_code returns a dict via FastMCP's structured-output path.
    Both `structuredContent` and the raw text block are accepted —
    the FastMCP version landed in this project sometimes serialises
    one way and sometimes the other."""
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        return structured
    content = getattr(result, "content", None) or []
    if content:
        first = content[0]
        text = getattr(first, "text", None)
        if text:
            import json

            try:
                return json.loads(text)
            except ValueError:
                return {"stdout": text}
    return {}


def test_work_dir_persists_across_two_sandbox_calls_via_mcp(
    live_stack,
    integration_env,
    fresh_session_id,
    fresh_conversation_id,
    scratch_cleanup,
):
    """Two run_code calls with the same conversation_id should share
    /work — file written in call 1 is readable in call 2. Goes
    through the full real stack: MCP transport -> session registry
    -> execute_in_sandbox -> docker bind-mount."""
    scratch_cleanup(fresh_conversation_id)

    async def _go():
        # Register the session WITH a conversation_id so the server
        # threads it into SandboxRequest.
        await register_session_via_admin(
            fresh_session_id,
            evidence_records=None,
            db_dsn=None,
            conversation_id=fresh_conversation_id,
        )
        try:
            first = await _call_run_code_via_mcp(
                fresh_session_id,
                "open('/work/probe.txt', 'w').write('first')",
            )
            second = await _call_run_code_via_mcp(
                fresh_session_id,
                "print(open('/work/probe.txt').read())",
            )
            return first, second
        finally:
            await unregister_session_via_admin(fresh_session_id)

    first, second = asyncio.run(_go())

    _maybe_skip_on_server_db_config(first)
    _maybe_skip_on_server_db_config(second)
    assert first.get("exit_code") == 0, f"first call failed: {first}"
    assert second.get("exit_code") == 0, f"second call failed: {second}"
    assert "first" in (second.get("stdout") or ""), (
        f"expected 'first' to persist across calls; got stdout={second.get('stdout')!r}, "
        f"stderr={second.get('stderr')!r}"
    )


def test_no_conversation_id_means_ephemeral(
    live_stack,
    integration_env,
    fresh_session_id,
):
    """Without a conversation_id /work is per-call ephemeral. Two
    calls in the same session must NOT see each other's files —
    confirms persistence is opt-in."""

    async def _go():
        await register_session_via_admin(
            fresh_session_id,
            evidence_records=None,
            db_dsn=None,
            # NB: no conversation_id => ephemeral
        )
        try:
            first = await _call_run_code_via_mcp(
                fresh_session_id,
                "open('/work/leak.txt', 'w').write('LEAK')",
            )
            second = await _call_run_code_via_mcp(
                fresh_session_id,
                # Print "MISSING" if the file is gone, else its content.
                # We don't want a non-zero exit on the missing-file path
                # because exit-code asserts above would fire.
                "import os\n" "p = '/work/leak.txt'\n" "print(open(p).read() if os.path.exists(p) else 'MISSING')",
            )
            return first, second
        finally:
            await unregister_session_via_admin(fresh_session_id)

    first, second = asyncio.run(_go())

    _maybe_skip_on_server_db_config(first)
    _maybe_skip_on_server_db_config(second)
    assert first.get("exit_code") == 0, f"first call failed: {first}"
    assert second.get("exit_code") == 0, f"second call failed: {second}"
    second_stdout = second.get("stdout") or ""
    assert "MISSING" in second_stdout, (
        "ephemeral /work should not have leaked the file across calls; " f"got stdout={second_stdout!r}"
    )
    assert "LEAK" not in second_stdout, "leaked content from prior ephemeral call"
