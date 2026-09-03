"""Pure parsers for pi RPC records. No I/O."""

from __future__ import annotations

from gmail_search.agents import pi_protocol as pp


def test_tool_call_args_entry_copies_name_and_args():
    ev = {
        "type": "tool_execution_start",
        "toolCallId": "c1",
        "toolName": "search_emails_batch",
        "args": {"searches": [{"query": "x"}]},
    }
    assert pp.tool_call_args_entry(ev) == {"name": "search_emails_batch", "args": {"searches": [{"query": "x"}]}}


def test_tool_call_args_entry_wraps_non_dict_args():
    ev = {"type": "tool_execution_start", "toolName": "bash", "args": "ls"}
    assert pp.tool_call_args_entry(ev) == {"name": "bash", "args": {"value": "ls"}}


def test_tool_call_response_entry_joins_text_and_clips():
    ev = {
        "type": "tool_execution_end",
        "toolName": "bash",
        "isError": False,
        "result": {"content": [{"type": "text", "text": "a" * 10}, {"type": "text", "text": "b"}]},
    }
    entry = pp.tool_call_response_entry(ev, clip=6)
    assert entry["name"] == "bash"
    assert entry["response"]["text"] == "aaaaaa"
    assert entry["response"]["is_error"] is False


def test_tool_call_response_entry_flags_error():
    ev = {"type": "tool_execution_end", "toolName": "x", "isError": True, "result": {"content": []}}
    assert pp.tool_call_response_entry(ev)["response"]["is_error"] is True


def test_assistant_text_joins_text_blocks_only():
    ev = {
        "type": "message_end",
        "message": {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "hmm"},
                {"type": "text", "text": "Hi"},
                {"type": "text", "text": " there"},
            ],
        },
    }
    assert pp.assistant_text(ev) == "Hi there"


def test_assistant_text_none_for_user_or_empty():
    assert (
        pp.assistant_text(
            {"type": "message_end", "message": {"role": "user", "content": [{"type": "text", "text": "q"}]}}
        )
        is None
    )
    assert (
        pp.assistant_text({"type": "message_end", "message": {"role": "assistant", "content": [{"type": "toolCall"}]}})
        is None
    )
    assert pp.assistant_text({"type": "turn_end"}) is None


def test_assistant_stop_extracts_stop_reason_and_error():
    ev = {
        "type": "message_end",
        "message": {"role": "assistant", "content": [], "stopReason": "error", "errorMessage": "boom"},
    }
    assert pp.assistant_stop(ev) == ("error", "boom")


def test_assistant_stop_none_for_non_assistant():
    assert pp.assistant_stop({"type": "message_end", "message": {"role": "user", "content": []}}) == (None, None)
    assert pp.assistant_stop({"type": "turn_end"}) == (None, None)


def test_bash_as_run_code_produces_args_then_response():
    start = {"type": "tool_execution_start", "toolName": "bash", "args": {"command": "python plot.py"}}
    end = {
        "type": "tool_execution_end",
        "toolName": "bash",
        "isError": False,
        "result": {"content": [{"type": "text", "text": "saved chart.png"}]},
    }
    assert pp.bash_as_run_code(start, end) == [
        {"name": "run_code", "args": {"code": "python plot.py"}},
        {"name": "run_code", "response": {"stdout": "saved chart.png", "artifacts": []}},
    ]


def test_usage_from_stats_response():
    resp = {
        "type": "response",
        "command": "get_session_stats",
        "success": True,
        "data": {"tokens": {"input": 10, "output": 4, "cacheRead": 3, "cacheWrite": 1}, "cost": 0.25},
    }
    u = pp.usage_from_stats_response(resp)
    assert (u.input_tokens, u.output_tokens, u.cache_read_tokens, u.cache_write_tokens, u.cost_usd) == (
        10,
        4,
        3,
        1,
        0.25,
    )


def test_usage_from_stats_response_tolerates_missing_fields():
    u = pp.usage_from_stats_response({"type": "response", "success": True, "data": {}})
    assert (u.input_tokens, u.output_tokens, u.cost_usd) == (0, 0, None)


def test_build_pi_argv_shape():
    argv = pp.build_pi_argv(
        container="pi-sandbox",
        session_id="s1",
        workspace="deep-conv-c1",
        session_path="/sessions/c1.jsonl",
        extension_path="/opt/gmail-tools",
        model="anthropic/claude-x",
        thinking="medium",
        system_prompt="SYS",
    )
    assert argv[:3] == ["docker", "exec", "-i"]
    assert "-e" in argv and "GMS_SESSION_ID=s1" in argv
    assert argv[argv.index("-w") + 1] == "/workspaces/deep-conv-c1"
    assert "pi-sandbox" in argv
    tail = argv[argv.index("pi") :]
    assert tail[:3] == ["pi", "--mode", "rpc"]
    for flag in ("--no-extensions", "--no-skills", "--no-context-files", "--no-prompt-templates"):
        assert flag in tail
    assert tail[tail.index("-e") + 1] == "/opt/gmail-tools"
    assert tail[tail.index("--session") + 1] == "/sessions/c1.jsonl"
    assert tail[tail.index("--model") + 1] == "anthropic/claude-x"
    assert tail[tail.index("--thinking") + 1] == "medium"
    assert tail[tail.index("--system-prompt") + 1] == "SYS"


def test_build_pi_argv_without_session_uses_no_session():
    argv = pp.build_pi_argv(
        container="c",
        session_id="s",
        workspace="w",
        session_path=None,
        extension_path="/x",
        model="m",
        thinking=None,
        system_prompt="p",
    )
    assert "--no-session" in argv and "--session" not in argv and "--thinking" not in argv


def test_build_pi_argv_can_disable_builtin_tools():
    argv = pp.build_pi_argv(
        container="c",
        session_id="s",
        workspace="w",
        session_path=None,
        extension_path="/x",
        model="m",
        thinking=None,
        system_prompt="p",
        builtin_tools=False,
    )
    assert "--no-builtin-tools" in argv
    default = pp.build_pi_argv(
        container="c",
        session_id="s",
        workspace="w",
        session_path=None,
        extension_path="/x",
        model="m",
        thinking=None,
        system_prompt="p",
    )
    assert "--no-builtin-tools" not in default
