"""Tests for the deep-analysis HTTP surface (`gmail_search.agents.service`).

Covers registration-time concerns (the ADK import probe). The streaming
endpoints themselves are exercised against the orchestrator directly in
`test_agent_orchestration.py`, without spinning up FastAPI.
"""

from __future__ import annotations

import builtins
import logging
import sys
from pathlib import Path

from fastapi import FastAPI

from gmail_search.agents import service


def test_register_agent_routes_calls_adk_probe(monkeypatch):
    """`register_agent_routes` must invoke the ADK probe at boot so a
    broken install surfaces in the server logs immediately, not at the
    first /api/agent/analyze request hours later. We don't care here
    HOW the probe checks; we care that it RAN."""
    called: list[bool] = []

    def _spy() -> None:
        called.append(True)

    monkeypatch.setattr(service, "_probe_adk_imports", _spy, raising=True)
    app = FastAPI()
    service.register_agent_routes(app, Path("/tmp/unused.db"))
    assert called == [True], "register_agent_routes did not invoke the ADK probe"


def test_probe_adk_imports_warns_on_broken_submodule(monkeypatch, caplog):
    """The probe's contract: when one of the ADK-touching submodules
    fails to import, log a WARNING and return cleanly. Chat mode must
    stay healthy even though deep mode is wedged.

    We simulate "ADK importable-but-broken" by intercepting the
    builtin `__import__` for the duration of the probe call so that
    any attempt to re-import a key submodule raises ImportError. This
    matches the shape of a real-world failure (e.g. `google.adk` is
    installed but a transitive dep is the wrong version)."""
    # The probe does `from gmail_search.agents import (analyst, ...)`.
    # Drop the cached submodules so the import statement actually
    # executes the loader path (and our patched __import__ sees it).
    cached: dict[str, object] = {}
    target = "gmail_search.agents.retriever"
    if target in sys.modules:
        cached[target] = sys.modules.pop(target)

    real_import = builtins.__import__

    def _patched_import(name, globals=None, locals=None, fromlist=(), level=0):
        # Trip the import we want to fake-break. Any other import goes
        # through normally so the rest of the test machinery works.
        if name == "gmail_search.agents" and "retriever" in (fromlist or ()):
            raise ImportError("simulated ADK breakage: retriever submodule")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _patched_import)

    try:
        with caplog.at_level(logging.WARNING, logger=service.logger.name):
            # Must NOT raise — the contract is wrap-and-warn.
            service._probe_adk_imports()

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert any(
            "ADK imports failed" in r.getMessage() for r in warnings
        ), f"expected ADK-failure warning, got: {[r.getMessage() for r in warnings]}"
    finally:
        # Restore any submodule we evicted so other tests aren't poisoned.
        for k, v in cached.items():
            sys.modules[k] = v


def test_probe_adk_imports_silent_on_healthy_install(caplog):
    """Sanity check: when imports succeed (the dev/CI machine has a
    working ADK install), the probe is silent. No warning, no error,
    no crash."""
    with caplog.at_level(logging.WARNING, logger=service.logger.name):
        service._probe_adk_imports()
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert not any(
        "ADK imports failed" in r.getMessage() for r in warnings
    ), f"healthy install should not warn; got: {[r.getMessage() for r in warnings]}"


# ── claude_code backend wiring ─────────────────────────────────────


def test_real_run_claude_code_backend_calls_register_invoke_unregister(monkeypatch, tmp_path):
    """`GMAIL_DEEP_BACKEND=claude_code` must:
       1. ensure the workspace dir exists,
       2. register the MCP session BEFORE the orchestrator runs,
       3. route every Orchestrator invoke through `claudebox_invoke`
          with `workspace=` and `session_id=` bound,
       4. unregister the MCP session in the finally cleanup.

    We don't bring up actual containers — `claudebox_invoke`,
    `register_session`, and `unregister_session` are all swapped for
    spies, and the Orchestrator's `run` is stubbed so the harness
    returns instantly."""
    import asyncio

    monkeypatch.setenv("GMAIL_DEEP_BACKEND", "claude_code")

    events: list[str] = []
    invoke_calls: list[dict] = []

    async def fake_claudebox_invoke(
        agent,
        prompt,
        *,
        workspace,
        session_id=None,
        cost_sink=None,
        event_sink=None,
        resume=None,
    ):
        from gmail_search.agents.orchestration import StageResult

        invoke_calls.append(
            {
                "agent": getattr(agent, "name", "?"),
                "workspace": workspace,
                "session_id": session_id,
                "has_cost_sink": cost_sink is not None,
                "has_event_sink": event_sink is not None,
            }
        )
        return StageResult(text="{}", tool_calls=[])

    async def fake_register_session(session_id, *, evidence_records, db_dsn, conversation_id=None, workspace=None):
        events.append(f"register:{session_id}:conv={conversation_id}:ws={workspace}")

    async def fake_unregister_session(session_id):
        events.append(f"unregister:{session_id}")

    workspace_dirs: list[str] = []

    def fake_ensure_workspace(workspace):
        workspace_dirs.append(workspace)

    # Stub orchestrator.run so the test doesn't actually run any
    # planner/retriever/etc — we only need to verify wiring.
    class _FakeOrch:
        def __init__(
            self,
            *,
            session_id,
            conn,
            planner,
            retriever,
            writer,
            critic,
            analyst_factory,
            invoke,
            skip_per_tool_emission=False,
        ):
            self.session_id = session_id
            self.invoke = invoke
            self.skip_per_tool_emission = skip_per_tool_emission

        async def run(self, question):
            class _A:
                name = "planner"

            await self.invoke(_A(), "x")  # fire one invoke to capture wiring
            return None

    # Stub DB + builders so no real ADK / DB activity happens.
    class _FakeConn:
        def close(self):
            pass

    monkeypatch.setattr(service, "get_connection", lambda _path: _FakeConn())
    monkeypatch.setattr(service, "fetch_events_after", lambda *a, **kw: [])
    monkeypatch.setattr(service, "_ensure_workspace_dir", fake_ensure_workspace)

    import gmail_search.agents.runtime_claude as rc

    monkeypatch.setattr(rc, "register_session_via_admin", fake_register_session)
    monkeypatch.setattr(rc, "unregister_session_via_admin", fake_unregister_session)
    monkeypatch.setattr(rc, "claudebox_invoke", fake_claudebox_invoke)

    # Bypass the actual Orchestrator + sub-agent factories.
    import gmail_search.agents.orchestration as orch_mod

    monkeypatch.setattr(orch_mod, "Orchestrator", _FakeOrch)
    for builder in ("build_planner_agent", "build_retriever_agent", "build_writer_agent", "build_critic_agent"):
        for mod_name, attr in [
            ("planner", "build_planner_agent"),
            ("retriever", "build_retriever_agent"),
            ("writer", "build_writer_agent"),
            ("critic", "build_critic_agent"),
        ]:
            mod = __import__(f"gmail_search.agents.{mod_name}", fromlist=[attr])
            monkeypatch.setattr(mod, attr, lambda *a, **kw: object(), raising=True)

    async def consume():
        async for _ in service._real_run(tmp_path / "x.db", "sess-XYZ", "what happened"):
            pass

    asyncio.run(consume())

    # Workspace dir created with the expected naming scheme.
    assert workspace_dirs == ["deep-sess-XYZ"]
    # register fired before any invoke; unregister fired last. The
    # register payload includes the conversation_id (None here — the
    # test calls _real_run without one).
    assert events[0] == "register:sess-XYZ:conv=None:ws=deep-sess-XYZ"
    assert events[-1] == "unregister:sess-XYZ"
    # Every invoke saw the right workspace + session_id + cost_sink.
    assert invoke_calls and all(c["workspace"] == "deep-sess-XYZ" for c in invoke_calls)
    assert all(c["session_id"] == "sess-XYZ" for c in invoke_calls)
    assert all(c["has_cost_sink"] for c in invoke_calls)


def test_real_run_claude_native_routes_to_native_run(monkeypatch, tmp_path):
    """`GMAIL_DEEP_BACKEND=claude_native` must:
       1. ensure the workspace dir exists,
       2. delegate to `native_run` (NOT the orchestrator),
       3. forward the right kwargs (db_path, session_id, workspace,
          conversation_id, question, model, cost_sink),
       4. SKIP every orchestrator/sub-agent builder.

    We swap `native_run` for a spy and assert the call shape. The
    Orchestrator factory is also stubbed to assert it never runs."""
    import asyncio

    monkeypatch.setenv("GMAIL_DEEP_BACKEND", "claude_native")

    native_calls: list[dict] = []

    async def fake_native_run(
        *,
        db_path,
        session_id,
        workspace,
        conversation_id,
        question,
        model,
        cost_sink,
        resume=None,
        on_session_uuid=None,
    ):
        native_calls.append(
            {
                "db_path": db_path,
                "session_id": session_id,
                "workspace": workspace,
                "conversation_id": conversation_id,
                "question": question,
                "model": model,
                "has_cost_sink": cost_sink is not None,
                "resume": resume,
                "has_session_uuid_callback": on_session_uuid is not None,
            }
        )

    import gmail_search.agents.runtime_claude_native as rcn

    monkeypatch.setattr(rcn, "native_run", fake_native_run)

    workspace_dirs: list[str] = []

    def fake_ensure_workspace(workspace):
        workspace_dirs.append(workspace)

    monkeypatch.setattr(service, "_ensure_workspace_dir", fake_ensure_workspace)

    # Asserting orchestrator never runs: blow up if anything tries to
    # construct it.
    class _OrchestratorMustNotRun:
        def __init__(self, *args, **kwargs):
            raise AssertionError("Orchestrator should not be constructed for claude_native")

    import gmail_search.agents.orchestration as orch_mod

    monkeypatch.setattr(orch_mod, "Orchestrator", _OrchestratorMustNotRun)

    # DB stub: support fetch_events_after returning [] so the poller
    # exits as soon as native_task is done.
    class _FakeConn:
        def close(self):
            pass

    monkeypatch.setattr(service, "get_connection", lambda _path: _FakeConn())
    monkeypatch.setattr(service, "fetch_events_after", lambda *a, **kw: [])

    async def consume():
        async for _ in service._real_run(
            tmp_path / "x.db",
            "sess-NAT",
            "what happened",
            default_model="opus",
            conversation_id="conv-7",
        ):
            pass

    asyncio.run(consume())

    # Per-conversation workspace naming (Phase 1): when conversation_id
    # is supplied, the workspace is `deep-conv-<conversation_id>` and
    # stays stable across turns so claudebox can `--resume` into the
    # same JSONL transcript.
    assert workspace_dirs == ["deep-conv-conv-7"]
    assert len(native_calls) == 1
    call = native_calls[0]
    assert call["session_id"] == "sess-NAT"
    assert call["workspace"] == "deep-conv-conv-7"
    assert call["conversation_id"] == "conv-7"
    # Resume + on_session_uuid plumbing must reach native_run; the
    # `_FakeConn` here can't execute SELECTs so resume is None and the
    # callback is wired but never fired in this test.
    assert call["resume"] is None
    assert call["has_session_uuid_callback"] is True
    assert call["question"] == "what happened"
    assert call["model"] == "opus"
    assert call["has_cost_sink"] is True


def test_real_run_adk_backend_does_not_register_mcp_session(monkeypatch, tmp_path):
    """Default backend must NOT touch the MCP session registry —
    that path is entirely claude_code-only."""
    import asyncio

    monkeypatch.delenv("GMAIL_DEEP_BACKEND", raising=False)

    register_calls: list[str] = []
    unregister_calls: list[str] = []

    import gmail_search.agents.runtime_claude as rc

    async def fake_register(sid, **kw):
        register_calls.append(sid)

    async def fake_unregister(sid):
        unregister_calls.append(sid)

    monkeypatch.setattr(rc, "register_session_via_admin", fake_register)
    monkeypatch.setattr(rc, "unregister_session_via_admin", fake_unregister)

    class _FakeOrch:
        def __init__(self, **kw):
            pass

        async def run(self, question):
            return None

    class _FakeConn:
        def close(self):
            pass

    monkeypatch.setattr(service, "get_connection", lambda _path: _FakeConn())
    monkeypatch.setattr(service, "fetch_events_after", lambda *a, **kw: [])
    import gmail_search.agents.orchestration as orch_mod

    monkeypatch.setattr(orch_mod, "Orchestrator", _FakeOrch)
    for mod_name, attr in [
        ("planner", "build_planner_agent"),
        ("retriever", "build_retriever_agent"),
        ("writer", "build_writer_agent"),
        ("critic", "build_critic_agent"),
    ]:
        mod = __import__(f"gmail_search.agents.{mod_name}", fromlist=[attr])
        monkeypatch.setattr(mod, attr, lambda *a, **kw: object(), raising=True)

    async def consume():
        async for _ in service._real_run(tmp_path / "x.db", "adk-sess", "q"):
            pass

    asyncio.run(consume())

    assert register_calls == []
    assert unregister_calls == []


# ── conversation history preamble ──────────────────────────────────


class _FakeRow(dict):
    """psycopg-style row that supports both dict-key and attribute access."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, sql, params):
        return _FakeCursor(self._rows)


def _row(role, text):
    import json as _json

    return _FakeRow(role=role, parts=_json.dumps([{"type": "text", "text": text}]))


def test_history_preamble_empty_when_no_conversation_id():
    assert service._build_conversation_history_preamble(_FakeConn([]), None) == ""


def test_history_preamble_empty_for_first_turn():
    """Only the in-progress user message exists yet — no prior history."""
    conn = _FakeConn([_row("user", "first question")])
    assert service._build_conversation_history_preamble(conn, "conv-1") == ""


def test_history_preamble_includes_prior_turns():
    """Two completed turns + an in-progress user message: preamble has
    the first two pairs, drops the trailing user."""
    conn = _FakeConn(
        [
            _row("user", "first question"),
            _row("assistant", "first answer"),
            _row("user", "second question"),
            _row("assistant", "second answer"),
            _row("user", "third question (in progress)"),
        ]
    )
    out = service._build_conversation_history_preamble(conn, "conv-1")
    assert "first question" in out
    assert "first answer" in out
    assert "second question" in out
    assert "second answer" in out
    assert "third question (in progress)" not in out
    assert out.endswith("# Latest user question (answer this)\n\n")


def test_history_preamble_truncates_to_max_turns():
    """When more than max_turns pairs exist, only the most recent are
    kept."""
    conn = _FakeConn(
        [
            _row("user", "very old question"),
            _row("assistant", "very old answer"),
            _row("user", "old question"),
            _row("assistant", "old answer"),
            _row("user", "recent question"),
            _row("assistant", "recent answer"),
            _row("user", "in progress"),
        ]
    )
    out = service._build_conversation_history_preamble(conn, "conv-1", max_turns=1)
    assert "very old" not in out
    assert "old question" not in out
    assert "recent question" in out
    assert "recent answer" in out


def test_history_preamble_skips_non_text_blocks():
    """Messages with only data-deep-stage / data-debug-id blocks
    contribute no text and don't appear in the preamble."""
    import json as _json

    conn = _FakeConn(
        [
            _row("user", "real question"),
            _FakeRow(role="assistant", parts=_json.dumps([{"type": "data-debug-id", "id": "x"}])),
            _row("user", "in progress"),
        ]
    )
    out = service._build_conversation_history_preamble(conn, "conv-1")
    assert "real question" in out
    # No assistant text appeared — but the preamble still has the user
    # turn, which is enough to establish topic context.
