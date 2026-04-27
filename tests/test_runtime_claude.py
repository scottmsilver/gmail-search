"""Tests for the claudebox runtime adapter.

Mocks httpx so no real claudebox server is needed. Canned response
blobs mirror the json-verbose shape that
`docker-claudebox/jsonpipe.py:_assemble` produces — `turns` array with
assistant + tool_result entries, camelCased keys, truncation metadata
on long tool outputs.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from gmail_search.agents.runtime_claude import (
    ClaudeboxError,
    _extract_tool_calls_from_messages,
    _extract_usage_from_response,
    claudebox_invoke,
)


@dataclass
class _FakeAgent:
    name: str = "writer"
    model: str = "claude-opus-4-7"
    instruction: str = "You are the writer."


def _canned_response(*, result: str = "final answer", turns: list | None = None) -> dict[str, Any]:
    """Mirrors a real json-verbose claudebox response: top-level result
    + usage + cost, plus the assembled `turns` array (post-camelCase)."""
    return {
        "type": "result",
        "subtype": "success",
        "result": result,
        "usage": {"inputTokens": 1234, "outputTokens": 567},
        "costUsd": 0.0123,
        "sessionId": "sess_abc",
        "runId": "run_xyz",
        "workspace": "/workspaces/test",
        "turns": turns or [],
        "system": {"sessionId": "sess_abc", "model": "claude-opus-4-7", "cwd": "/workspaces/test", "tools": []},
    }


def _assistant_with_tool_use(tool_id: str, name: str, input_dict: dict) -> dict:
    return {
        "role": "assistant",
        "content": [
            {"type": "text", "text": f"calling {name}"},
            {"type": "tool_use", "id": tool_id, "name": name, "input": input_dict},
        ],
    }


def _tool_result_turn(tool_use_id: str, content_text: str, *, truncated: bool = False) -> dict:
    block: dict = {
        "type": "tool_result",
        "toolUseId": tool_use_id,
        "isError": False,
        "content": content_text,
    }
    if truncated:
        block["truncated"] = True
        block["totalLength"] = len(content_text) * 10
        block["sha256"] = "deadbeef"
    return {"role": "tool_result", "content": [block]}


class _FakeHttpResponse:
    """Quacks like httpx.Response for the small surface our parser
    touches (status_code, json(), text)."""

    def __init__(self, status_code: int, body: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._body = body
        self.text = text or (str(body) if body else "")

    def json(self):
        if self._body is None:
            raise ValueError("no body")
        return self._body


def _patched_post(responses: list[_FakeHttpResponse]):
    """Build an AsyncMock-friendly side_effect that yields each
    canned response in order. Lets the busy-retry test assert the
    second call returns the success body."""
    calls = iter(responses)

    async def _post(*args, **kwargs):
        return next(calls)

    return _post


def _patch_async_client(post_side_effect):
    """Patch httpx.AsyncClient so the body of `_post_claudebox_run`
    receives our fake response. Using `with` on the AsyncClient is
    handled by the MagicMock auto-async-context-manager pattern."""
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=post_side_effect)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    return patch("gmail_search.agents.runtime_claude.httpx.AsyncClient", return_value=mock_client)


def _force_sync_mode(monkeypatch) -> None:
    """Pin the runtime to the legacy sync `/run` path so a test can
    reuse the pre-existing single-POST mock pattern. New async tests
    leave the default in place and exercise the polling flow
    directly."""
    monkeypatch.setenv("GMAIL_CLAUDEBOX_USE_ASYNC", "0")


def test_claudebox_invoke_extracts_text_and_tool_calls(monkeypatch):
    """End-to-end happy path: response with one tool_use + one
    tool_result lands as two entries in `tool_calls`, with `args` and
    `response` shapes matching the ADK runtime contract."""
    _force_sync_mode(monkeypatch)
    monkeypatch.delenv("GMAIL_CLAUDEBOX_TOKEN", raising=False)
    response_body = _canned_response(
        result="here is the answer",
        turns=[
            _assistant_with_tool_use("tu_1", "search_emails", {"q": "from:bob"}),
            _tool_result_turn("tu_1", '{"results": [{"cite_ref": "r1"}]}'),
        ],
    )
    fake_responses = [_FakeHttpResponse(200, response_body)]

    with _patch_async_client(_patched_post(fake_responses)):
        result = asyncio.run(
            claudebox_invoke(_FakeAgent(), "do the thing", workspace="ws_1"),
        )

    assert result.text == "here is the answer"
    assert len(result.tool_calls) == 2
    assert result.tool_calls[0] == {"name": "search_emails", "args": {"q": "from:bob"}}
    assert result.tool_calls[1]["name"] == "search_emails"
    assert result.tool_calls[1]["response"]["content"] == '{"results": [{"cite_ref": "r1"}]}'


def test_claudebox_invoke_calls_cost_sink_with_token_counts(monkeypatch):
    """`cost_sink` must receive the same kwargs as the ADK path:
    agent_name, model, input_tokens, output_tokens."""
    _force_sync_mode(monkeypatch)
    captured: dict[str, Any] = {}

    def sink(*, agent_name: str, model: str, input_tokens: int, output_tokens: int) -> None:
        captured["agent_name"] = agent_name
        captured["model"] = model
        captured["input_tokens"] = input_tokens
        captured["output_tokens"] = output_tokens

    body = _canned_response(turns=[])
    fake_responses = [_FakeHttpResponse(200, body)]

    with _patch_async_client(_patched_post(fake_responses)):
        asyncio.run(
            claudebox_invoke(
                _FakeAgent(name="critic", model="claude-opus-4-7"),
                "review",
                workspace="ws_1",
                cost_sink=sink,
            ),
        )

    assert captured == {
        "agent_name": "critic",
        "model": "claude-opus-4-7",
        "input_tokens": 1234,
        "output_tokens": 567,
    }


def test_claudebox_invoke_retries_on_409(monkeypatch):
    """Workspace-busy 409 → retry up to 3 times; second attempt
    succeeds. The exponential backoff sleep is patched to a no-op so
    the test is fast."""
    _force_sync_mode(monkeypatch)
    monkeypatch.setattr(
        "gmail_search.agents.runtime_claude._sleep_for_busy_retry",
        AsyncMock(return_value=None),
    )

    busy = _FakeHttpResponse(409, text="busy")
    success_body = _canned_response(result="ok after retry", turns=[])
    success = _FakeHttpResponse(200, success_body)

    with _patch_async_client(_patched_post([busy, success])):
        result = asyncio.run(
            claudebox_invoke(_FakeAgent(), "go", workspace="ws_busy"),
        )

    assert result.text == "ok after retry"


def test_claudebox_invoke_gives_up_after_three_409s(monkeypatch):
    """Three consecutive 409s → ClaudeboxError, no infinite loop."""
    _force_sync_mode(monkeypatch)
    monkeypatch.setattr(
        "gmail_search.agents.runtime_claude._sleep_for_busy_retry",
        AsyncMock(return_value=None),
    )

    busy_responses = [_FakeHttpResponse(409, text="busy") for _ in range(3)]

    with _patch_async_client(_patched_post(busy_responses)):
        with pytest.raises(ClaudeboxError, match="busy"):
            asyncio.run(
                claudebox_invoke(_FakeAgent(), "go", workspace="ws_busy"),
            )


def test_claudebox_invoke_surfaces_4xx_as_clean_exception(monkeypatch):
    """A 400 (bad workspace) becomes a ClaudeboxError with the body
    text included so the orchestrator's `error` event has something
    actionable."""
    _force_sync_mode(monkeypatch)
    bad = _FakeHttpResponse(400, text="workspace not found: /workspaces/nope")

    with _patch_async_client(_patched_post([bad])):
        with pytest.raises(ClaudeboxError, match="workspace not found"):
            asyncio.run(
                claudebox_invoke(_FakeAgent(), "go", workspace="nope"),
            )


def test_extract_tool_calls_from_messages_threads_name_through_results():
    """tool_result blocks don't carry the tool name — only the
    tool_use_id. The walker must remember the name from the
    earlier tool_use block so the orchestrator's `tc["name"]` lookups
    keep working."""
    turns = [
        _assistant_with_tool_use("tu_42", "run_code", {"code": "print(1)"}),
        _tool_result_turn("tu_42", '{"artifacts": [{"id": 7}]}', truncated=True),
    ]

    tool_calls = _extract_tool_calls_from_messages(turns)

    assert tool_calls[0] == {"name": "run_code", "args": {"code": "print(1)"}}
    assert tool_calls[1]["name"] == "run_code"  # threaded from the tool_use
    assert tool_calls[1]["response"]["content"] == '{"artifacts": [{"id": 7}]}'
    assert tool_calls[1]["response"]["truncated"] is True


def test_extract_usage_handles_both_camel_and_snake_keys():
    """json-verbose normalises to camelCase, but older builds may
    leave snake_case. Both must yield the same (in, out) tuple."""
    camel = {"usage": {"inputTokens": 10, "outputTokens": 20}}
    snake = {"usage": {"input_tokens": 10, "output_tokens": 20}}
    missing = {}
    assert _extract_usage_from_response(camel) == (10, 20)
    assert _extract_usage_from_response(snake) == (10, 20)
    assert _extract_usage_from_response(missing) == (0, 0)


# ── Side-channel merge: tool_calls come from the MCP admin endpoint
#    when session_id is provided, replacing the truncated message-stream
#    view ────────────────────────────────────────────────────────────


def test_claudebox_invoke_uses_side_channel_when_session_id_given(monkeypatch):
    """When `session_id` is provided, `claudebox_invoke` must fetch
    structured tool calls from `/admin/calls/<id>` and replace the
    (truncated) message-parsed view with them. This is the load-bearing
    test for the whole side-channel design.

    Setup: claudebox returns a tool_result whose `content` field has been
    stringified + truncated to 2000 chars (claudebox's behaviour). The
    mocked MCP admin endpoint returns the FULL structured response. The
    final StageResult.tool_calls must reflect the side channel, not the
    truncated string."""
    from gmail_search.agents import runtime_claude as rc

    truncated_string = '{"results": [{"cite_ref": "abc12345", "prev' + ("x" * 1500)
    full_structured = {
        "results": [
            {"thread_id": "t1", "cite_ref": "abc12345", "preview": "x" * 5000},
            {"thread_id": "t2", "cite_ref": "deadbeef", "preview": "y" * 5000},
        ],
        "totalLength": 12345,
    }

    response_body = _canned_response(
        result="answer",
        turns=[
            _assistant_with_tool_use("tu_1", "search_emails", {"q": "from:x"}),
            _tool_result_turn("tu_1", truncated_string, truncated=True),
        ],
    )

    side_channel_records = [
        {
            "name": "search_emails",
            "args": {"q": "from:x"},
            "response": full_structured,
            "ts": 1.0,
        }
    ]

    async def fake_fetch(session_id):
        assert session_id == "s-merge"
        return list(side_channel_records)

    _force_sync_mode(monkeypatch)
    monkeypatch.setattr(rc, "_fetch_structured_tool_calls", fake_fetch)
    monkeypatch.delenv("GMAIL_CLAUDEBOX_TOKEN", raising=False)

    with _patch_async_client(_patched_post([_FakeHttpResponse(200, response_body)])):
        result = asyncio.run(
            rc.claudebox_invoke(
                _FakeAgent(),
                "go",
                workspace="ws",
                session_id="s-merge",
            ),
        )

    # Two entries (args + response) per side-channel record.
    assert len(result.tool_calls) == 2
    assert result.tool_calls[0] == {"name": "search_emails", "args": {"q": "from:x"}}
    # The response must be the FULL structured dict, not the truncated string.
    assert result.tool_calls[1] == {"name": "search_emails", "response": full_structured}
    assert result.tool_calls[1]["response"]["totalLength"] == 12345
    assert len(result.tool_calls[1]["response"]["results"]) == 2


def test_claudebox_invoke_falls_back_to_messages_when_no_session_id(monkeypatch):
    """Backwards compat: callers that don't pass session_id (e.g. unit
    tests, or a path that doesn't have an MCP server) keep the old
    message-stream parser behaviour."""
    _force_sync_mode(monkeypatch)
    response_body = _canned_response(
        result="answer",
        turns=[
            _assistant_with_tool_use("tu_1", "search_emails", {"q": "x"}),
            _tool_result_turn("tu_1", '{"results": []}'),
        ],
    )

    with _patch_async_client(_patched_post([_FakeHttpResponse(200, response_body)])):
        result = asyncio.run(claudebox_invoke(_FakeAgent(), "go", workspace="ws"))

    assert len(result.tool_calls) == 2
    assert result.tool_calls[1]["response"]["content"] == '{"results": []}'


def test_register_session_via_admin_sends_conversation_id(monkeypatch):
    """When `conversation_id` is supplied, the JSON body sent to the
    MCP admin endpoint must include it so the server can opt the
    session into the per-conversation persistent /work mount."""
    from gmail_search.agents import runtime_claude as rc

    monkeypatch.setenv("GMAIL_MCP_ADMIN_URL", "http://mcp.test:7878")
    monkeypatch.setenv("GMAIL_MCP_ADMIN_TOKEN", "tok-r")

    captured: dict = {}

    async def fake_post(self, url, json=None, headers=None):
        captured["url"] = url
        captured["json"] = json
        return _FakeHttpResponse(200, {"ok": True})

    mock_client = AsyncMock()
    mock_client.post = fake_post.__get__(mock_client)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("gmail_search.agents.runtime_claude.httpx.AsyncClient", return_value=mock_client):
        asyncio.run(
            rc.register_session_via_admin(
                "sess-Q",
                evidence_records=None,
                conversation_id="conv-foo",
            ),
        )

    assert captured["url"].endswith("/admin/sessions")
    assert captured["json"]["session_id"] == "sess-Q"
    assert captured["json"]["conversation_id"] == "conv-foo"


def test_register_session_via_admin_omits_conversation_id_when_absent(monkeypatch):
    """Backwards-compat: when no conversation_id is passed, the body
    must not carry one (so the server keeps the legacy ephemeral path)."""
    from gmail_search.agents import runtime_claude as rc

    monkeypatch.setenv("GMAIL_MCP_ADMIN_URL", "http://mcp.test:7878")
    monkeypatch.setenv("GMAIL_MCP_ADMIN_TOKEN", "tok-r")

    captured: dict = {}

    async def fake_post(self, url, json=None, headers=None):
        captured["json"] = json
        return _FakeHttpResponse(200, {"ok": True})

    mock_client = AsyncMock()
    mock_client.post = fake_post.__get__(mock_client)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("gmail_search.agents.runtime_claude.httpx.AsyncClient", return_value=mock_client):
        asyncio.run(rc.register_session_via_admin("sess-Q2", evidence_records=None))

    assert "conversation_id" not in captured["json"]


def test_claudebox_invoke_streams_tool_calls_via_event_sink(monkeypatch, tmp_path):
    """When `event_sink` is supplied, the JSONL tailer must surface
    each tool_use as a streamed `("tool_call", {name, args})` event
    BEFORE the /run POST resolves. We pre-create the JSONL file in
    the host-mirrored projects dir, then patch httpx so the POST
    sleeps long enough for the tailer to drain the file."""
    import json as _json

    from gmail_search.agents import runtime_claude as rc
    from gmail_search.agents.jsonl_tail import encode_workspace_path

    _force_sync_mode(monkeypatch)
    # Redirect the host projects root to tmp_path so we don't need to
    # touch the real deploy/ tree.
    monkeypatch.setattr(rc, "_CLAUDEBOX_HOST_PROJECTS_ROOT", tmp_path)

    workspace = "deep-test"
    encoded = encode_workspace_path(f"/workspaces/{workspace}")
    session_dir = tmp_path / encoded
    session_dir.mkdir(parents=True)
    jsonl = session_dir / "session1.jsonl"
    lines = [
        _json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "tool_use", "id": "tu_1", "name": "search_emails", "input": {"q": "x"}}]
                },
            }
        ),
        _json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "tool_use", "id": "tu_2", "name": "run_code", "input": {"code": "print(1)"}}]
                },
            }
        ),
    ]
    jsonl.write_text("\n".join(lines) + "\n")

    captured: list[tuple[str, dict]] = []

    async def sink(kind: str, payload: dict) -> None:
        captured.append((kind, payload))

    body = _canned_response(result="done", turns=[])

    async def slow_post(*args, **kwargs):
        # Give the tailer a couple of poll cycles to see the file.
        await asyncio.sleep(0.15)
        return _FakeHttpResponse(200, body)

    with _patch_async_client(slow_post):
        result = asyncio.run(
            rc.claudebox_invoke(
                _FakeAgent(),
                "go",
                workspace=workspace,
                event_sink=sink,
            ),
        )

    assert result.text == "done"
    kinds = [k for k, _ in captured]
    names = [p.get("name") for _, p in captured]
    assert kinds == ["tool_call", "tool_call"]
    assert "search_emails" in names
    assert "run_code" in names


def test_claudebox_invoke_without_event_sink_does_not_tail(monkeypatch, tmp_path):
    """Backwards compat: when `event_sink` is None, no tailer is
    spawned. The POST returns immediately and tool_calls come from
    the message-stream parser as before."""
    from gmail_search.agents import runtime_claude as rc

    _force_sync_mode(monkeypatch)
    monkeypatch.setattr(rc, "_CLAUDEBOX_HOST_PROJECTS_ROOT", tmp_path)
    body = _canned_response(
        result="ok",
        turns=[_assistant_with_tool_use("tu_1", "search_emails", {"q": "x"})],
    )
    with _patch_async_client(_patched_post([_FakeHttpResponse(200, body)])):
        result = asyncio.run(rc.claudebox_invoke(_FakeAgent(), "go", workspace="ws"))

    # No event_sink => no streaming; tool_calls still extracted post-hoc.
    assert any(tc.get("name") == "search_emails" for tc in result.tool_calls)


def test_fetch_structured_tool_calls_hits_admin_endpoint(monkeypatch):
    """`_fetch_structured_tool_calls` must GET /admin/calls/<id> with
    the bearer token and return the calls list."""
    from gmail_search.agents import runtime_claude as rc

    monkeypatch.setenv("GMAIL_MCP_ADMIN_URL", "http://mcp.test:7878")
    monkeypatch.setenv("GMAIL_MCP_ADMIN_TOKEN", "tok-1")

    captured: dict = {}

    async def fake_get(self, url, headers=None):
        captured["url"] = url
        captured["headers"] = headers
        return _FakeHttpResponse(200, {"calls": [{"name": "x", "args": {}, "response": {}}]})

    mock_client = AsyncMock()
    mock_client.get = fake_get.__get__(mock_client)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("gmail_search.agents.runtime_claude.httpx.AsyncClient", return_value=mock_client):
        out = asyncio.run(rc._fetch_structured_tool_calls("sess-A"))

    assert captured["url"] == "http://mcp.test:7878/admin/calls/sess-A"
    assert captured["headers"]["Authorization"] == "Bearer tok-1"
    assert out == [{"name": "x", "args": {}, "response": {}}]


# ── Async polling mode ────────────────────────────────────────────
#
# The async path POSTs `/run` with `{async: True}`, gets a runId
# back, then polls `GET /run/result?runId=...` until done. A JSONL
# tailer fires events into `event_sink` AND bumps an idle-progress
# watchdog; tests below patch httpx + the test seams (`_now`,
# `_sleep`) so the polling state machine can be advanced without
# real wall time.


class _FakeClock:
    """Mutable monotonic clock the runtime's `_now` test seam reads.
    `advance(s)` jumps wall time forward; tests use it to walk the
    polling loop into the idle-watchdog or hard-timeout branch on
    demand."""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def _build_async_post_get_client(post_responses: list, get_responses: list):
    """AsyncMock httpx client with separate POST + GET queues. The
    POST queue feeds the initial `/run` call; the GET queue feeds
    the polling loop. Used by the async-mode tests."""
    post_iter = iter(post_responses)
    get_iter = iter(get_responses)

    async def _post(*args, **kwargs):
        return next(post_iter)

    async def _get(*args, **kwargs):
        return next(get_iter)

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=_post)
    mock_client.get = AsyncMock(side_effect=_get)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    return mock_client


def _install_async_mode_seams(monkeypatch, clock: _FakeClock) -> None:
    """Replace the runtime's `_now` + `_sleep` seams with the fake
    clock + a no-op sleep that auto-advances time so the polling
    loop converges instantly without real time passing."""
    from gmail_search.agents import runtime_claude as rc

    async def _fake_sleep(seconds: float) -> None:
        clock.advance(seconds)
        # Yield once so a concurrent JSONL tailer task can interleave
        # between polls. Without this, AsyncMock-only HTTP keeps the
        # poll loop greedy and the tailer never gets scheduled.
        await asyncio.sleep(0)

    monkeypatch.setattr(rc, "_now", clock.now)
    monkeypatch.setattr(rc, "_sleep", _fake_sleep)
    # Ensure async mode is on (default) — undo any earlier override.
    monkeypatch.setenv("GMAIL_CLAUDEBOX_USE_ASYNC", "1")

    # Do not start any real JSONL tailer in tests that don't care
    # about the file system; replace it with a no-op task.
    async def _noop_tail(*args, **kwargs):
        return None

    monkeypatch.setattr(rc, "tail_session_events", _noop_tail)


def test_async_happy_path_polls_and_returns_stage_result(monkeypatch):
    """async POST returns a runId; two GETs say running; the third
    GET returns the assembled blob. Resulting StageResult must mirror
    what the sync path produces."""
    from gmail_search.agents import runtime_claude as rc

    clock = _FakeClock()
    _install_async_mode_seams(monkeypatch, clock)

    initial = _FakeHttpResponse(200, {"runId": "r1", "workspace": "ws", "status": "running"})
    final_blob = _canned_response(
        result="async answer",
        turns=[
            _assistant_with_tool_use("tu_1", "search_emails", {"q": "from:bob"}),
            _tool_result_turn("tu_1", '{"results": [{"cite_ref": "r1"}]}'),
        ],
    )
    polls = [
        _FakeHttpResponse(200, {"runId": "r1", "workspace": "ws", "status": "running"}),
        _FakeHttpResponse(200, {"runId": "r1", "workspace": "ws", "status": "running"}),
        _FakeHttpResponse(200, final_blob),
    ]
    client = _build_async_post_get_client([initial], polls)

    with patch("gmail_search.agents.runtime_claude.httpx.AsyncClient", return_value=client):
        result = asyncio.run(rc.claudebox_invoke(_FakeAgent(), "go", workspace="ws"))

    assert result.text == "async answer"
    assert result.tool_calls[0] == {"name": "search_emails", "args": {"q": "from:bob"}}
    assert result.tool_calls[1]["response"]["content"] == '{"results": [{"cite_ref": "r1"}]}'


def test_async_event_sink_fires_for_streamed_jsonl_lines(monkeypatch):
    """Mid-flight tailer events must surface to the caller's
    event_sink as `("tool_call", {name, args})` tuples — once per
    JSONL line the tailer parses."""
    from gmail_search.agents import runtime_claude as rc

    clock = _FakeClock()
    _install_async_mode_seams(monkeypatch, clock)

    # Replace the tailer with one that synthesises two parsed JSONL
    # lines through the supplied handler before exiting.
    streamed_lines = [
        {
            "type": "assistant",
            "message": {"content": [{"type": "tool_use", "id": "tu_a", "name": "search_emails", "input": {"q": "x"}}]},
        },
        {
            "type": "assistant",
            "message": {"content": [{"type": "tool_use", "id": "tu_b", "name": "run_code", "input": {"code": "p"}}]},
        },
    ]

    async def fake_tail(workspace_dir, on_event, *, stop_event, **_kwargs):
        for line in streamed_lines:
            await on_event(line)
        return None

    monkeypatch.setattr(rc, "tail_session_events", fake_tail)

    initial = _FakeHttpResponse(200, {"runId": "r1", "status": "running"})
    final_blob = _canned_response(result="done", turns=[])
    polls = [_FakeHttpResponse(200, final_blob)]
    client = _build_async_post_get_client([initial], polls)

    captured: list[tuple[str, dict]] = []

    async def sink(kind: str, payload: dict) -> None:
        captured.append((kind, payload))

    with patch("gmail_search.agents.runtime_claude.httpx.AsyncClient", return_value=client):
        result = asyncio.run(
            rc.claudebox_invoke(_FakeAgent(), "go", workspace="ws", event_sink=sink),
        )

    assert result.text == "done"
    assert [k for k, _ in captured] == ["tool_call", "tool_call"]
    names = [p["name"] for _, p in captured]
    assert names == ["search_emails", "run_code"]


def test_async_idle_timeout_raises_when_no_jsonl_progress(monkeypatch):
    """If `/run/result` keeps reporting `running` AND the JSONL
    tailer never fires, the idle-progress watchdog must trip after
    `_IDLE_TIMEOUT_SECONDS`."""
    from gmail_search.agents import runtime_claude as rc

    clock = _FakeClock()
    _install_async_mode_seams(monkeypatch, clock)

    initial = _FakeHttpResponse(200, {"runId": "r1", "status": "running"})

    # Infinite supply of "running" polls.
    def running_forever():
        while True:
            yield _FakeHttpResponse(200, {"runId": "r1", "status": "running"})

    polls_iter = running_forever()

    async def _post(*args, **kwargs):
        return initial

    async def _get(*args, **kwargs):
        return next(polls_iter)

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=_post)
    mock_client.get = AsyncMock(side_effect=_get)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("gmail_search.agents.runtime_claude.httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(ClaudeboxError, match="no JSONL progress"):
            asyncio.run(rc.claudebox_invoke(_FakeAgent(), "go", workspace="ws"))


def test_async_jsonl_progress_keeps_idle_watchdog_alive(monkeypatch):
    """The idle watchdog must NOT trip when JSONL progress is
    arriving — even if the run takes many polls. A tailer that
    bumps `last_event_at` between polls keeps the function happy
    until the GET finally returns the assembled blob."""
    from gmail_search.agents import runtime_claude as rc

    clock = _FakeClock()
    _install_async_mode_seams(monkeypatch, clock)

    # Tailer that fires one JSONL event per call to bump the
    # watchdog; the test relies on _make_progress_handler updating
    # last_event_at via _now() (which is the fake clock).
    progress_calls: list[int] = []

    async def fake_tail(workspace_dir, on_event, *, stop_event, **_kwargs):
        # Drip-feed an event every poll cycle while the run is alive.
        for _ in range(6):
            if stop_event.is_set():
                return
            await on_event(
                {
                    "type": "assistant",
                    "message": {"content": [{"type": "tool_use", "id": "tu", "name": "search_emails", "input": {}}]},
                }
            )
            progress_calls.append(1)
            await asyncio.sleep(0)  # let the poller run
        return None

    monkeypatch.setattr(rc, "tail_session_events", fake_tail)

    initial = _FakeHttpResponse(200, {"runId": "r1", "status": "running"})
    final_blob = _canned_response(result="done after progress", turns=[])
    polls = [
        _FakeHttpResponse(200, {"runId": "r1", "status": "running"}),
        _FakeHttpResponse(200, {"runId": "r1", "status": "running"}),
        _FakeHttpResponse(200, {"runId": "r1", "status": "running"}),
        _FakeHttpResponse(200, {"runId": "r1", "status": "running"}),
        _FakeHttpResponse(200, {"runId": "r1", "status": "running"}),
        _FakeHttpResponse(200, final_blob),
    ]
    client = _build_async_post_get_client([initial], polls)

    with patch("gmail_search.agents.runtime_claude.httpx.AsyncClient", return_value=client):
        result = asyncio.run(rc.claudebox_invoke(_FakeAgent(), "go", workspace="ws"))

    assert result.text == "done after progress"
    assert len(progress_calls) >= 1


def test_async_initial_post_409_retries(monkeypatch):
    """A 409 on the initial async POST must still go through the
    busy-retry helper; the second attempt's runId is then polled."""
    from gmail_search.agents import runtime_claude as rc

    clock = _FakeClock()
    _install_async_mode_seams(monkeypatch, clock)
    monkeypatch.setattr(
        "gmail_search.agents.runtime_claude._sleep_for_busy_retry",
        AsyncMock(return_value=None),
    )

    busy = _FakeHttpResponse(409, text="busy")
    success_initial = _FakeHttpResponse(200, {"runId": "r2", "status": "running"})
    final_blob = _canned_response(result="ok after busy", turns=[])
    poll = _FakeHttpResponse(200, final_blob)
    client = _build_async_post_get_client([busy, success_initial], [poll])

    with patch("gmail_search.agents.runtime_claude.httpx.AsyncClient", return_value=client):
        result = asyncio.run(rc.claudebox_invoke(_FakeAgent(), "go", workspace="ws"))

    assert result.text == "ok after busy"


def test_sync_fallback_path_runs_when_env_disables_async(monkeypatch):
    """With `GMAIL_CLAUDEBOX_USE_ASYNC=0`, `claudebox_invoke` must
    take the legacy sync path: one POST, no polling, returned blob
    parsed identically. Confirms the deploy-time revert lever works."""
    monkeypatch.setenv("GMAIL_CLAUDEBOX_USE_ASYNC", "0")
    body = _canned_response(result="sync path ok", turns=[])
    with _patch_async_client(_patched_post([_FakeHttpResponse(200, body)])):
        result = asyncio.run(claudebox_invoke(_FakeAgent(), "go", workspace="ws"))
    assert result.text == "sync path ok"


def test_async_hard_timeout_raises_on_endless_run(monkeypatch):
    """If a run keeps making JSONL progress but never completes
    within `_HARD_TIMEOUT_SECONDS`, the hard cap must trip."""
    from gmail_search.agents import runtime_claude as rc

    clock = _FakeClock()
    _install_async_mode_seams(monkeypatch, clock)

    # Tailer that keeps bumping the watchdog so idle timeout never trips.
    async def busy_tail(workspace_dir, on_event, *, stop_event, **_kwargs):
        while not stop_event.is_set():
            await on_event(
                {
                    "type": "assistant",
                    "message": {"content": [{"type": "tool_use", "id": "x", "name": "n", "input": {}}]},
                }
            )
            await asyncio.sleep(0)
        return None

    monkeypatch.setattr(rc, "tail_session_events", busy_tail)

    initial = _FakeHttpResponse(200, {"runId": "rH", "status": "running"})

    # A fake sleep that jumps wall time well past the 1h cap each
    # tick AND yields so the busy_tail interleaves and bumps the
    # idle watchdog. Without that yield, idle timeout would race
    # the hard-timeout check.
    async def big_sleep(seconds: float) -> None:
        clock.advance(rc._HARD_TIMEOUT_SECONDS + 10)
        await asyncio.sleep(0)

    monkeypatch.setattr(rc, "_sleep", big_sleep)

    def running_forever():
        while True:
            yield _FakeHttpResponse(200, {"runId": "rH", "status": "running"})

    polls_iter = running_forever()

    async def _post(*args, **kwargs):
        return initial

    async def _get(*args, **kwargs):
        return next(polls_iter)

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=_post)
    mock_client.get = AsyncMock(side_effect=_get)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("gmail_search.agents.runtime_claude.httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(ClaudeboxError, match="hard timeout"):
            asyncio.run(rc.claudebox_invoke(_FakeAgent(), "go", workspace="ws"))


def test_async_failed_run_status_surfaces_as_claudebox_error(monkeypatch):
    """When the server returns `{status: "failed", error: "..."}`,
    the polling loop must raise ClaudeboxError with the error text."""
    from gmail_search.agents import runtime_claude as rc

    clock = _FakeClock()
    _install_async_mode_seams(monkeypatch, clock)

    initial = _FakeHttpResponse(200, {"runId": "rF", "status": "running"})
    failed = _FakeHttpResponse(200, {"runId": "rF", "status": "failed", "error": "claude crashed"})
    client = _build_async_post_get_client([initial], [failed])

    with patch("gmail_search.agents.runtime_claude.httpx.AsyncClient", return_value=client):
        with pytest.raises(ClaudeboxError, match="claude crashed"):
            asyncio.run(rc.claudebox_invoke(_FakeAgent(), "go", workspace="ws"))
