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

    async def fake_fetch(session_id):
        return side_channel or []

    monkeypatch.setattr(rc, "register_session_via_admin", fake_register)
    monkeypatch.setattr(rc, "unregister_session_via_admin", fake_unregister)
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
