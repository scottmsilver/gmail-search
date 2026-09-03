"""Pure parsers for pi RPC records. No I/O."""

from __future__ import annotations

import pytest

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


def test_build_pi_argv_rejects_traversal_session_id():
    with pytest.raises(ValueError):
        pp.build_pi_argv(
            container="c",
            session_id="../x",
            workspace="w",
            session_path=None,
            extension_path="/x",
            model="m",
            thinking=None,
            system_prompt="p",
        )


def test_build_pi_argv_rejects_slash_in_workspace():
    with pytest.raises(ValueError):
        pp.build_pi_argv(
            container="c",
            session_id="s",
            workspace="a/b",
            session_path=None,
            extension_path="/x",
            model="m",
            thinking=None,
            system_prompt="p",
        )


def test_build_pi_argv_accepts_74_char_workspace():
    workspace = "deep-conv-" + "a" * 64
    argv = pp.build_pi_argv(
        container="c",
        session_id="s",
        workspace=workspace,
        session_path=None,
        extension_path="/x",
        model="m",
        thinking=None,
        system_prompt="p",
    )
    assert argv[argv.index("-w") + 1] == f"/workspaces/{workspace}"


def test_build_pi_argv_rejects_81_char_workspace():
    workspace = "a" * 81
    with pytest.raises(ValueError):
        pp.build_pi_argv(
            container="c",
            session_id="s",
            workspace=workspace,
            session_path=None,
            extension_path="/x",
            model="m",
            thinking=None,
            system_prompt="p",
        )


def test_build_pi_argv_appends_mcp_config_flag_after_extension():
    argv = pp.build_pi_argv(
        container="c",
        session_id="s",
        workspace="w",
        session_path=None,
        extension_path="/opt/pi-pkgs/node_modules/pi-mcp-adapter",
        model="m",
        thinking=None,
        system_prompt="p",
        mcp_config_path="/opt/gmail-mcp.json",
    )
    tail = argv[argv.index("pi") :]
    ext_idx = tail.index("-e")
    assert tail[ext_idx + 1] == "/opt/pi-pkgs/node_modules/pi-mcp-adapter"
    assert tail[ext_idx + 2] == "--mcp-config"
    assert tail[ext_idx + 3] == "/opt/gmail-mcp.json"


def test_build_pi_argv_includes_tmpdir_flag_when_given():
    argv = pp.build_pi_argv(
        container="c",
        session_id="s",
        workspace="deep-conv-c1",
        session_path=None,
        extension_path="/x",
        model="m",
        thinking=None,
        system_prompt="p",
        tmpdir="/workspaces/deep-conv-c1/.tmp",
    )
    assert "-e" in argv
    assert "TMPDIR=/workspaces/deep-conv-c1/.tmp" in argv
    # Sits next to GMS_SESSION_ID, before the `-w` workdir flag.
    tmpdir_idx = argv.index("TMPDIR=/workspaces/deep-conv-c1/.tmp")
    assert argv[tmpdir_idx - 1] == "-e"
    assert argv.index("GMS_SESSION_ID=s") < tmpdir_idx


def test_build_pi_argv_omits_tmpdir_flag_by_default():
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
    assert not any(a.startswith("TMPDIR=") for a in argv)


def test_build_pi_argv_omits_mcp_config_flag_by_default():
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
    assert "--mcp-config" not in argv


def test_redact_secrets_leaves_ordinary_text_unchanged():
    text = "The search found 3 threads about hotel refunds, sorted by date."
    assert pp.redact_secrets(text) == text


def test_redact_secrets_google_api_key():
    text = "key is AIzaSyD-abcdefghijklmnopqrstuvwxyz1234"
    redacted = pp.redact_secrets(text)
    assert "AIzaSyD" not in redacted
    assert "[REDACTED]" in redacted


def test_redact_secrets_bearer_token():
    text = "Authorization: Bearer abcDEF123456.token-value_here"
    redacted = pp.redact_secrets(text)
    assert "abcDEF123456" not in redacted
    assert redacted.startswith("Authorization: Bearer [REDACTED]")


def test_redact_secrets_sk_style_key():
    text = "found sk-abcdefghijklmnopqrstuvwxyz in output"
    redacted = pp.redact_secrets(text)
    assert "sk-abcdefghijklmnopqrstuvwxyz" not in redacted
    assert "[REDACTED]" in redacted


def test_redact_secrets_env_style_assignment():
    assert pp.redact_secrets("GMAIL_MCP_SERVICE_TOKEN=abc.def") == "GMAIL_MCP_SERVICE_TOKEN=[REDACTED]"
    assert pp.redact_secrets("GEMINI_API_KEY=xyz123") == "GEMINI_API_KEY=[REDACTED]"
    assert pp.redact_secrets("MY_SECRET=shh") == "MY_SECRET=[REDACTED]"
    assert pp.redact_secrets("DB_PASSWORD=hunter2") == "DB_PASSWORD=[REDACTED]"


def test_redact_secrets_is_case_insensitive():
    assert pp.redact_secrets("gemini_api_key=xyz123") == "gemini_api_key=[REDACTED]"


def test_tool_call_response_entry_redacts_secrets_in_bash_output():
    ev = {
        "type": "tool_execution_end",
        "toolName": "bash",
        "isError": False,
        "result": {"content": [{"type": "text", "text": "GMAIL_MCP_SERVICE_TOKEN=abc.def\nOK"}]},
    }
    entry = pp.tool_call_response_entry(ev)
    assert "abc.def" not in entry["response"]["text"]
    assert "[REDACTED]" in entry["response"]["text"]
