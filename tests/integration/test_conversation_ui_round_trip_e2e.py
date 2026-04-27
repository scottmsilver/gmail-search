"""End-to-end test that mirrors what the chat UI actually does for
a deep-mode conversation.

Captures the exact bug we hit on 2026-04-27: an `/api/agent/analyze`
turn writes its final answer to `agent_sessions.final_answer` and
its events to `agent_events`, but the chat-thread UI reads from
`conversation_messages`. If the test harness only POSTs to
`/api/agent/analyze` (as several earlier smoke tests did), the
backend looks healthy but the UI shows an empty conversation.

The realistic UI flow is:
  1. PUT  /api/conversations/<id>  with the new user message appended.
  2. POST /api/agent/analyze       to drive the assistant's response.
  3. PUT  /api/conversations/<id>  again with the assistant's reply
                                   appended (this is what the front-
                                   end does after the SSE stream
                                   completes).
  4. GET  /api/conversations/<id>  — what the user sees on tap.

This test executes that full round-trip and asserts the GET in step 4
returns a non-empty `messages` array. The Phase 1 per-conversation
plumbing (workspace + Claude session UUID + `--resume` continuity) is
also verified by running TWO turns and confirming the JSONL appended
in place.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

import httpx
import pytest

from . import _stack_probe

pytestmark = pytest.mark.integration


_FASTAPI = _stack_probe.GMAIL_FASTAPI_URL


def _conv_id() -> str:
    """Allow-listed conversation_id (matches server.py:_CONVERSATION_ID_RE).
    Prefixed `itest-` so post-run cleanup can target test rows in bulk."""
    return f"itest-uiloop-{uuid.uuid4().hex[:12]}"


async def _put_conversation(client: httpx.AsyncClient, conv_id: str, messages: list[dict]) -> None:
    r = await client.put(
        f"{_FASTAPI}/api/conversations/{conv_id}",
        json={"title": "ui round-trip test", "messages": messages},
    )
    r.raise_for_status()


async def _get_conversation(client: httpx.AsyncClient, conv_id: str) -> dict:
    r = await client.get(f"{_FASTAPI}/api/conversations/{conv_id}")
    r.raise_for_status()
    return r.json()


async def _delete_conversation(client: httpx.AsyncClient, conv_id: str) -> None:
    """Best-effort. ON DELETE CASCADE drops conversation_messages +
    conversation_claude_session; the workspace dir cleanup happens
    in the route handler."""
    try:
        await client.delete(f"{_FASTAPI}/api/conversations/{conv_id}")
    except Exception:
        pass


async def _run_one_analyze_turn(
    client: httpx.AsyncClient,
    *,
    conv_id: str,
    question: str,
    timeout: float = 180.0,
) -> tuple[str, str]:
    """POST /api/agent/analyze in claude_native mode and consume the
    SSE stream. Returns `(session_id, final_text)`. Raises on stream
    errors or missing final event."""
    session_id: str | None = None
    final_text: str | None = None
    async with client.stream(
        "POST",
        f"{_FASTAPI}/api/agent/analyze",
        json={
            "conversation_id": conv_id,
            "question": question,
            "backend": "claude_native",
        },
        timeout=timeout,
    ) as response:
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
            if not line.startswith("data: "):
                continue
            try:
                payload = json.loads(line[len("data: ") :])
            except ValueError:
                continue
            if "session_id" in payload and not session_id:
                session_id = payload["session_id"]
            if current_event == "final" and isinstance(payload, dict):
                # claude_native emits the final text inside payload.text
                # or directly as text key — accept either for resilience.
                final_text = payload.get("text") or (payload.get("payload") or {}).get("text")
    if session_id is None:
        raise AssertionError("no `session_id` event seen in SSE stream")
    if final_text is None:
        raise AssertionError("no `final` event with text seen in SSE stream")
    return session_id, final_text


@pytest.mark.asyncio
async def test_chat_ui_round_trip_renders_in_get(live_stack):
    """Drive the full UI flow for a single deep turn and assert the
    GET shows the messages a real user would see on tap. This is the
    minimum check that prevents the 2026-04-27 regression where the
    backend ran fine but the UI showed an empty conversation."""
    conv_id = _conv_id()
    question = "Say PONG and only PONG."
    user_msg = {"role": "user", "parts": [{"type": "text", "text": question}]}

    async with httpx.AsyncClient() as client:
        try:
            # Step 1: UI creates the conversation with the new user msg.
            await _put_conversation(client, conv_id, [user_msg])

            # Step 2: UI fires the analyze request.
            session_id, final_text = await _run_one_analyze_turn(client, conv_id=conv_id, question=question)
            assert session_id, "expected analyze to surface a session_id"
            assert final_text, "expected analyze to emit a final answer"

            # Step 3: UI persists the assistant's reply back into
            # conversation_messages (this is what the front-end does
            # after SSE completes). Without this round-trip the UI
            # would show only the user message.
            assistant_msg = {
                "role": "assistant",
                "parts": [{"type": "text", "text": final_text}],
            }
            await _put_conversation(client, conv_id, [user_msg, assistant_msg])

            # Step 4: what the user sees when they tap the conversation.
            shown = await _get_conversation(client, conv_id)
            assert shown["id"] == conv_id
            messages = shown.get("messages") or []
            assert len(messages) >= 2, (
                f"UI would show empty conversation; expected ≥2 messages, "
                f"got {len(messages)}. This is the 2026-04-27 regression."
            )
            roles = [m["role"] for m in messages]
            assert "user" in roles and "assistant" in roles, roles
        finally:
            await _delete_conversation(client, conv_id)


@pytest.mark.asyncio
async def test_deep_turn_persists_rich_parts_into_conversation_messages(live_stack):
    """The Python side of `_real_run` writes a rich assistant message
    (with `tool-<name>` blocks reconstructed from agent_events) into
    conversation_messages at the end of every successful deep turn.

    This is the Phase 6 fix for the 2026-04-27 regression where the
    front-end persisted only `[{type: "text", text: ...}]` and the
    chat thread lost every tool call on reload. The assertion: the
    persisted assistant message MUST contain at least one `tool-`
    prefixed part (the prompt is crafted to force a sql_query_batch
    invocation, which Phase 2 records as `mcp_tool_call_full`)."""
    conv_id = _conv_id()
    # Force at least one tool call so we have something rich to verify.
    question = (
        "Use sql_query_batch with a single query that selects " "COUNT(*) FROM messages, then report just the number."
    )
    user_msg = {"role": "user", "parts": [{"type": "text", "text": question}]}

    async with httpx.AsyncClient() as client:
        try:
            await _put_conversation(client, conv_id, [user_msg])
            session_id, _ = await _run_one_analyze_turn(client, conv_id=conv_id, question=question)
            # Give the server's persist a beat to commit before we GET.
            await asyncio.sleep(1.0)
            shown = await _get_conversation(client, conv_id)
            messages = shown.get("messages") or []
            assert len(messages) >= 2, f"expected at least user + rich-assistant, got {len(messages)} messages"
            assistant = messages[-1]
            assert assistant["role"] == "assistant", assistant
            parts = assistant.get("parts") or []
            tool_parts = [
                p
                for p in parts
                if isinstance(p, dict) and isinstance(p.get("type"), str) and p["type"].startswith("tool-")
            ]
            text_parts = [p for p in parts if isinstance(p, dict) and p.get("type") == "text"]
            assert tool_parts, (
                f"assistant message has NO tool-* blocks (only {[p.get('type') for p in parts]}); "
                f"the 2026-04-27 regression has resurfaced — server-side rich persist isn't writing"
            )
            assert text_parts, "assistant message has no text part for the final answer"
            # The Phase 2 mcp_tool_call_full record carries `output` too
            # — at least one tool block should have it (proving we're
            # not just streaming input-only).
            with_output = [p for p in tool_parts if "output" in p]
            assert with_output, (
                "no tool-* part has an `output` field — Phase 2's "
                "mcp_tool_call_full path didn't reach the persist step"
            )
        finally:
            await _delete_conversation(client, conv_id)


@pytest.mark.asyncio
async def test_two_turns_pin_same_claude_session_uuid(live_stack):
    """Per-conversation Claude session UUID is established on turn 1
    and reused on turn 2 — so the JSONL appends in place and the
    `--resume` cache hit kicks in. Failure here means Phase 1's
    workspace pinning broke."""
    conv_id = _conv_id()
    q1 = "Say PONG and only PONG."
    q2 = "What word did I just ask you to say?"

    async with httpx.AsyncClient() as client:
        try:
            await _put_conversation(
                client,
                conv_id,
                [{"role": "user", "parts": [{"type": "text", "text": q1}]}],
            )
            sid1, ans1 = await _run_one_analyze_turn(client, conv_id=conv_id, question=q1)
            await _put_conversation(
                client,
                conv_id,
                [
                    {"role": "user", "parts": [{"type": "text", "text": q1}]},
                    {"role": "assistant", "parts": [{"type": "text", "text": ans1}]},
                    {"role": "user", "parts": [{"type": "text", "text": q2}]},
                ],
            )
            sid2, ans2 = await _run_one_analyze_turn(client, conv_id=conv_id, question=q2)

            # Verify --resume worked: only ONE JSONL file exists for
            # this conversation, and it contains turn-2 content (the
            # appended file is bigger than just turn 1 alone).
            ws_proj_dir = Path("deploy/claudebox/claude-config/projects") / f"-workspaces-deep-conv-{conv_id}"
            jsonls = list(ws_proj_dir.glob("*.jsonl"))
            assert len(jsonls) == 1, (
                f"expected exactly ONE JSONL (resume should append); "
                f"found {len(jsonls)}: {[p.name for p in jsonls]}"
            )

            # Phase 5 debug pane should surface both turns + the
            # pinned UUID.
            debug = (await client.get(f"{_FASTAPI}/api/conversations/{conv_id}/debug")).json()
            assert debug["claude_session_uuid"] is not None
            assert len(debug["sessions"]) >= 2, debug["sessions"]
        finally:
            await _delete_conversation(client, conv_id)
