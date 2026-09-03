"""pi_run: one deep turn through a fake PiRpcClient."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from gmail_search.agents import runtime_claude as rc
from gmail_search.agents import runtime_pi


class _FakeConn:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.finalized: list[dict[str, Any]] = []
        self.closed = False
        self._seq = 0

    def execute(self, sql: str, params: tuple):
        low = sql.lower()
        if "insert into agent_events" in low:
            self._seq += 1
            session_id, _a, agent_name, kind, payload_json = params
            self.events.append({"agent_name": agent_name, "kind": kind, "payload": json.loads(payload_json)})
            return _Row({"seq": self._seq})
        if "update agent_sessions" in low:
            status, final_answer, session_id = params
            self.finalized.append({"status": status, "final_answer": final_answer})
        return _Row(None)

    def commit(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class _Row:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class _FakeClient:
    """Replays scripted records; records commands sent."""

    def __init__(self, records: list[dict], stats: dict | None = None, *, hang_after: int | None = None) -> None:
        self._records = list(records)
        self._stats = stats or {}
        self._hang_after = hang_after
        self.sent: list[dict] = []
        self.aborted = False
        self.closed = False
        self.killed = False
        self.stray: list[dict] = []
        self._served = 0

    async def send(self, command: dict) -> None:
        self.sent.append(command)

    async def read_record(self, timeout: float) -> dict | None:
        if self._hang_after is not None and self._served >= self._hang_after:
            await asyncio.sleep(timeout)
            raise asyncio.TimeoutError()
        if not self._records:
            return None
        self._served += 1
        return self._records.pop(0)

    async def request(self, command: dict, *, timeout: float) -> dict:
        self.sent.append(command)
        return {"type": "response", "command": command["type"], "success": True, "data": self._stats}

    async def abort_and_close(self, *, grace: float = 5.0) -> None:
        self.aborted = True
        self.closed = True

    async def close(self) -> None:
        self.closed = True


def _happy_records() -> list[dict]:
    return [
        {"type": "response", "command": "prompt", "success": True},
        {"type": "agent_start"},
        {
            "type": "tool_execution_start",
            "toolCallId": "c1",
            "toolName": "search_emails_batch",
            "args": {"searches": [{"query": "hotel"}]},
        },
        {
            "type": "tool_execution_end",
            "toolCallId": "c1",
            "toolName": "search_emails_batch",
            "isError": False,
            "result": {"content": [{"type": "text", "text": '{"results": []}'}]},
        },
        {"type": "tool_execution_start", "toolCallId": "c2", "toolName": "bash", "args": {"command": "python plot.py"}},
        {
            "type": "tool_execution_end",
            "toolCallId": "c2",
            "toolName": "bash",
            "isError": False,
            "result": {"content": [{"type": "text", "text": "wrote chart.png"}]},
        },
        {
            "type": "message_end",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "Final answer"}]},
        },
        {"type": "agent_end", "messages": []},
    ]


def _install_common(monkeypatch, *, side_channel: list[dict] | None = None):
    """Everything `_install` sets up except the spawn client itself, so
    tests that need custom spawn behaviour (e.g. concurrency) can reuse
    this and set `runtime_pi._spawn_client` themselves."""
    conn = _FakeConn()
    monkeypatch.setattr(runtime_pi, "get_connection", lambda _p: conn)
    calls: dict[str, list] = {"register": [], "unregister": []}

    async def fake_register(session_id, **kw):
        calls["register"].append({"session_id": session_id, **kw})

    async def fake_unregister(session_id):
        calls["unregister"].append(session_id)

    async def fake_mint(session_id, **kw):
        calls.setdefault("mint", []).append({"session_id": session_id, **kw})
        return "tok-123"

    async def fake_fetch(session_id):
        return side_channel or []

    monkeypatch.setattr(rc, "register_session_via_admin", fake_register)
    monkeypatch.setattr(rc, "unregister_session_via_admin", fake_unregister)
    monkeypatch.setattr(rc, "mint_session_token_via_admin", fake_mint)
    monkeypatch.setattr(rc, "_fetch_structured_tool_calls", fake_fetch)
    monkeypatch.setattr(runtime_pi, "sweep_and_extend_final_text", lambda conn, **kw: kw["base_text"])

    async def fake_kill(session_path):
        calls.setdefault("kill", []).append(session_path)

    monkeypatch.setattr(runtime_pi, "_kill_stray_pi", fake_kill)
    return conn, calls


def _install(monkeypatch, client: _FakeClient, *, side_channel: list[dict] | None = None):
    conn, calls = _install_common(monkeypatch, side_channel=side_channel)
    monkeypatch.setattr(runtime_pi, "_spawn_client", _make_spawn(client))
    return conn, calls


def _make_spawn(client):
    async def _spawn(argv):
        _spawn.argv = argv
        return client

    return _spawn


def _run(**overrides):
    kwargs = dict(
        db_path=Path("/tmp/x.db"),
        session_id="s1",
        workspace="deep-conv-c1",
        conversation_id="c1",
        question="hotels?",
        model="anthropic/claude-test",
        cost_sink=None,
        user_id="u1",
    )
    kwargs.update(overrides)
    asyncio.run(runtime_pi.pi_run(**kwargs))


def test_happy_path_emits_full_event_sequence(monkeypatch):
    client = _FakeClient(
        _happy_records(), stats={"tokens": {"input": 100, "output": 20, "cacheRead": 5, "cacheWrite": 1}, "cost": 0.03}
    )
    side = [{"name": "search_emails_batch", "args": {"searches": []}, "response": {"results": [{"cite_ref": "r1"}]}}]
    conn, calls = _install(monkeypatch, client, side_channel=side)
    costs: list[dict] = []
    _run(cost_sink=lambda **kw: costs.append(kw))

    kinds = [e["kind"] for e in conn.events]
    assert kinds[0] == "plan"
    assert kinds.count("tool_call") == 4  # start+end for search, start+end for bash
    assert "evidence" in kinds and "analysis" in kinds
    assert kinds[-2:] == ["draft", "final"]
    assert conn.events[-1]["payload"]["text"] == "Final answer"
    assert conn.finalized == [{"status": "done", "final_answer": "Final answer"}]
    assert costs == [
        {
            "agent_name": "pi",
            "model": "anthropic/claude-test",
            "input_tokens": 100,
            "output_tokens": 20,
            "usd_override": 0.03,
            "cache_read_tokens": 5,
            "cache_write_tokens": 1,
        }
    ]
    assert calls["register"][0]["workspace"] == "deep-conv-c1" and calls["register"][0]["user_id"] == "u1"
    assert calls["unregister"] == ["s1"]
    assert client.sent[0]["type"] == "prompt" and client.sent[0]["message"] == "hotels?"
    assert client.closed and conn.closed


def test_argv_uses_conversation_session_path(monkeypatch):
    client = _FakeClient(_happy_records())
    _install(monkeypatch, client)
    monkeypatch.setenv("GMAIL_PI_CONTAINER", "pi-test")
    _run()
    argv = runtime_pi._spawn_client.argv
    assert argv[argv.index("--session") + 1] == "/sessions/c1.jsonl"
    assert "pi-test" in argv and "GMS_SESSION_ID=s1" in argv


def test_no_conversation_runs_without_session(monkeypatch):
    client = _FakeClient(_happy_records())
    _install(monkeypatch, client)
    _run(conversation_id=None)
    assert "--no-session" in runtime_pi._spawn_client.argv


def test_idle_timeout_aborts_and_emits_error(monkeypatch):
    client = _FakeClient(_happy_records()[:2], hang_after=2)
    conn, calls = _install(monkeypatch, client)
    monkeypatch.setenv("GMAIL_PI_IDLE_TIMEOUT", "0.2")
    _run()
    assert client.aborted
    assert calls["kill"] == ["/sessions/c1.jsonl"]
    assert conn.events[-1]["kind"] == "error" and "idle" in conn.events[-1]["payload"]["message"]
    assert conn.finalized == [{"status": "error", "final_answer": None}]


def test_eof_before_agent_end_is_an_error(monkeypatch):
    client = _FakeClient(_happy_records()[:3])
    conn, calls = _install(monkeypatch, client)
    _run()
    assert conn.events[-1]["kind"] == "error"
    assert calls["unregister"] == ["s1"]


def test_extension_error_is_logged_not_fatal(monkeypatch, caplog):
    records = _happy_records()
    records.insert(
        2, {"type": "extension_error", "extensionPath": "/opt/gmail-tools", "event": "tool_call", "error": "kaboom"}
    )
    client = _FakeClient(records)
    conn, _ = _install(monkeypatch, client)
    _run()
    assert conn.finalized[0]["status"] == "done"
    assert "kaboom" in caplog.text


def test_session_path_for_rejects_bad_ids():
    assert runtime_pi.session_path_for("abc-123") == "/sessions/abc-123.jsonl"
    assert runtime_pi.session_path_for("../etc") is None
    assert runtime_pi.session_path_for(None) is None


def test_builtin_tools_disabled_via_env(monkeypatch):
    client = _FakeClient(_happy_records())
    _install(monkeypatch, client)
    monkeypatch.setenv("GMAIL_PI_BUILTIN_TOOLS", "0")
    _run()
    assert "--no-builtin-tools" in runtime_pi._spawn_client.argv


def test_same_conversation_turns_are_serialized(monkeypatch):
    """Two pi_run calls for the same conversation must not spawn a
    second pi process until the first client is fully closed — two
    processes writing the same `--session` file corrupts the
    transcript (spike finding)."""
    events: list[str] = []

    class _SlowFakeClient(_FakeClient):
        def __init__(self, name: str, records: list[dict]) -> None:
            super().__init__(records)
            self.name = name

        async def read_record(self, timeout: float) -> dict | None:
            await asyncio.sleep(0.05)
            return await super().read_record(timeout)

        async def close(self) -> None:
            events.append(f"close:{self.name}")
            await super().close()

    clients = [_SlowFakeClient("a", _happy_records()), _SlowFakeClient("b", _happy_records())]
    remaining = list(clients)

    async def _spawn(argv):
        client = remaining.pop(0)
        events.append(f"spawn:{client.name}")
        return client

    _install_common(monkeypatch)
    monkeypatch.setattr(runtime_pi, "_spawn_client", _spawn)

    async def run_both():
        await asyncio.gather(
            runtime_pi.pi_run(
                db_path=Path("/tmp/x.db"),
                session_id="s1",
                workspace="w1",
                conversation_id="c1",
                question="q1",
                model="m",
                cost_sink=None,
                user_id="u1",
            ),
            runtime_pi.pi_run(
                db_path=Path("/tmp/x.db"),
                session_id="s2",
                workspace="w2",
                conversation_id="c1",
                question="q2",
                model="m",
                cost_sink=None,
                user_id="u1",
            ),
        )

    asyncio.run(run_both())

    spawn_order = [e for e in events if e.startswith("spawn:")]
    assert spawn_order == ["spawn:a", "spawn:b"]
    assert events.index("close:a") < events.index("spawn:b")


def test_error_stop_reason_surfaces_as_error_event(monkeypatch):
    records = _happy_records()
    records[-2] = {
        "type": "message_end",
        "message": {"role": "assistant", "content": [], "stopReason": "error", "errorMessage": "context too large"},
    }
    client = _FakeClient(records)
    conn, _ = _install(monkeypatch, client)
    _run()
    assert conn.events[-1]["kind"] == "error"
    assert "context too large" in conn.events[-1]["payload"]["message"]
    assert conn.finalized == [{"status": "error", "final_answer": None}]


def test_killed_on_normal_close_triggers_stray_kill(monkeypatch):
    """When `close()` had to kill the process (e.g. pi didn't exit on
    stdin EOF), `_run_turn` must still clean up the in-container
    process — not just on the exception path."""

    class _KilledOnCloseClient(_FakeClient):
        async def close(self) -> None:
            self.killed = True
            await super().close()

    client = _KilledOnCloseClient(_happy_records())
    conn, calls = _install(monkeypatch, client)
    _run()
    assert client.closed
    assert calls["kill"] == ["/sessions/c1.jsonl"]
    assert conn.finalized == [{"status": "done", "final_answer": "Final answer"}]


def test_empty_answer_surfaces_as_error_event(monkeypatch):
    records = _happy_records()
    records[-2] = {
        "type": "message_end",
        "message": {"role": "assistant", "content": [], "stopReason": "stop"},
    }
    client = _FakeClient(records)
    conn, _ = _install(monkeypatch, client)
    _run()
    assert conn.events[-1]["kind"] == "error"
    assert "without an assistant answer" in conn.events[-1]["payload"]["message"]
    assert conn.finalized == [{"status": "error", "final_answer": None}]


def test_render_instruction_injects_gemini3_budget():
    text = runtime_pi.render_instruction("google/gemini-3.7-flash")
    assert "1,048,576" in text
    assert "314,572" in text
    assert "{context_window}" not in text
    assert "{reading_budget}" not in text
    assert "session_id" not in text


def test_context_window_env_override(monkeypatch):
    monkeypatch.setenv("GMAIL_PI_CONTEXT_WINDOW", "500000")
    text = runtime_pi.render_instruction("google/gemini-3.7-flash")
    assert "500,000" in text
    assert "150,000" in text


def test_render_instruction_uses_prefixed_tool_names_and_mcp_script():
    """Tool names in the prompt must match the adapter-registered names
    (gmail_-prefixed), and mcpScript must be documented. No bare
    (unprefixed) call form of a gmail tool should remain — that would
    point the model at a tool name the server doesn't register."""
    text = runtime_pi.render_instruction("anthropic/claude-test")
    assert "gmail_search_emails_batch" in text
    assert "mcpScript" in text
    assert "search_emails_batch(" not in text.replace("gmail_search_emails_batch(", "")


def test_context_window_for_unknown_model_falls_back_to_default():
    assert runtime_pi.context_window_for("some/unknown-model") == 200_000


def test_context_window_for_known_prefixes():
    assert runtime_pi.context_window_for("google/gemini-2.5-pro") == 1_048_576
    assert runtime_pi.context_window_for("anthropic/claude-test") == 200_000


# ── session-token file lifecycle ────────────────────────────────────


def test_argv_includes_default_mcp_config_flag(monkeypatch):
    client = _FakeClient(_happy_records())
    _install(monkeypatch, client)
    _run()
    argv = runtime_pi._spawn_client.argv
    assert argv[argv.index("--mcp-config") + 1] == "/opt/gmail-mcp.json"


def test_argv_mcp_config_flag_honors_env_override(monkeypatch):
    client = _FakeClient(_happy_records())
    _install(monkeypatch, client)
    monkeypatch.setenv("GMAIL_PI_MCP_CONFIG", "/custom/mcp.json")
    _run()
    argv = runtime_pi._spawn_client.argv
    assert argv[argv.index("--mcp-config") + 1] == "/custom/mcp.json"


def test_session_token_file_written_0600_and_removed_on_happy_path(monkeypatch, tmp_path):
    client = _FakeClient(_happy_records())
    conn, calls = _install(monkeypatch, client)
    monkeypatch.setenv("GMAIL_PI_WORKSPACES_ROOT", str(tmp_path))

    written_path = tmp_path / "deep-conv-c1" / ".session-token"
    original_write = runtime_pi._write_session_token_file
    captured: dict = {}

    def spy_write(workspace, token):
        path = original_write(workspace, token)
        captured["path"] = path
        captured["mode"] = path.stat().st_mode & 0o777
        captured["content"] = path.read_text()
        return path

    monkeypatch.setattr(runtime_pi, "_write_session_token_file", spy_write)
    _run()

    assert calls["mint"] == [{"session_id": "s1", "ttl_seconds": int(runtime_pi.hard_timeout_seconds()) + 120}]
    assert captured["path"] == written_path
    assert captured["mode"] == 0o600
    assert captured["content"] == "tok-123"
    # Removed once the turn finishes.
    assert not written_path.exists()


def test_session_token_file_removed_on_error_path(monkeypatch, tmp_path):
    """Cleanup must run even when the turn itself errors out (e.g. an
    idle timeout) — the token file must never outlive the turn."""
    client = _FakeClient(_happy_records()[:2], hang_after=2)
    conn, calls = _install(monkeypatch, client)
    monkeypatch.setenv("GMAIL_PI_WORKSPACES_ROOT", str(tmp_path))
    monkeypatch.setenv("GMAIL_PI_IDLE_TIMEOUT", "0.2")

    _run()

    assert conn.events[-1]["kind"] == "error"
    written_path = tmp_path / "deep-conv-c1" / ".session-token"
    assert not written_path.exists()


def test_argv_scopes_tmpdir_to_workspace(monkeypatch):
    """Output-guard spill files must land under the turn's own
    workspace (TMPDIR), not the shared sandbox container's /tmp."""
    client = _FakeClient(_happy_records())
    _install(monkeypatch, client)
    _run()
    argv = runtime_pi._spawn_client.argv
    assert "TMPDIR=/workspaces/deep-conv-c1/.tmp" in argv


def test_run_creates_workspace_tmp_dir_0700(monkeypatch, tmp_path):
    client = _FakeClient(_happy_records())
    _install(monkeypatch, client)
    monkeypatch.setenv("GMAIL_PI_WORKSPACES_ROOT", str(tmp_path))
    _run()
    tmp_dir = tmp_path / "deep-conv-c1" / ".tmp"
    assert tmp_dir.is_dir()
    assert (tmp_dir.stat().st_mode & 0o777) == 0o700


def test_ensure_workspace_tmp_dir_reasserts_0700_on_existing_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("GMAIL_PI_WORKSPACES_ROOT", str(tmp_path))
    loose = tmp_path / "deep-conv-c1" / ".tmp"
    loose.mkdir(parents=True)
    loose.chmod(0o755)
    path = runtime_pi._ensure_workspace_tmp_dir("deep-conv-c1")
    assert path == loose
    assert (path.stat().st_mode & 0o777) == 0o700


def test_session_token_file_not_written_when_registration_never_happens(monkeypatch, tmp_path):
    """If register_session_via_admin itself fails, no token is minted
    and there is nothing to clean up — the finally block must not
    explode on a token_path that was never set."""
    client = _FakeClient(_happy_records())
    conn, calls = _install(monkeypatch, client)
    monkeypatch.setenv("GMAIL_PI_WORKSPACES_ROOT", str(tmp_path))

    async def failing_register(session_id, **kw):
        raise RuntimeError("admin unreachable")

    monkeypatch.setattr(rc, "register_session_via_admin", failing_register)
    _run()

    assert conn.events[-1]["kind"] == "error"
    assert not (tmp_path / "deep-conv-c1").exists()
    assert runtime_pi.context_window_for("openai/gpt-5") == 400_000
