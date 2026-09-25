"""Tests for the deep-analysis HTTP surface (`gmail_search.agents.service`).

Covers backend selection and the claude_code wiring. The streaming
endpoints themselves are exercised against the orchestrator directly in
`test_agent_orchestration.py`, without spinning up FastAPI.
"""

from __future__ import annotations


import pytest

from gmail_search.agents import service


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

    async def fake_register_session(
        session_id, *, evidence_records, db_dsn, conversation_id=None, workspace=None, user_id=None
    ):
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

    # Stub DB + builders so no real model / DB activity happens.
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
        user_id=None,
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
                "user_id": user_id,
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


@pytest.mark.parametrize("selected_model", [None, "openrouter/meta/muse-spark-1.3"])
def test_real_run_pi_backend_routes_to_pi_run(monkeypatch, tmp_path, selected_model):
    """backend="pi" must ensure the workspace, call pi_run with the
    turn's selected model (or None for the configured default),
    skip the claudebox credential preflight, and never build the
    orchestrator."""
    import asyncio

    pi_calls: list[dict] = []

    async def fake_pi_run(*, db_path, session_id, workspace, conversation_id, question, model, cost_sink, user_id=None):
        pi_calls.append(
            {
                "session_id": session_id,
                "workspace": workspace,
                "conversation_id": conversation_id,
                "question": question,
                "model": model,
                "has_cost_sink": cost_sink is not None,
                "user_id": user_id,
            }
        )

    import gmail_search.agents.runtime_pi as rp

    monkeypatch.setattr(rp, "pi_run", fake_pi_run)

    workspace_dirs: list[str] = []
    monkeypatch.setattr(service, "_ensure_workspace_dir", lambda w: workspace_dirs.append(w))

    def _preflight_must_not_run():
        raise AssertionError("credential preflight must be skipped for pi")

    import gmail_search.claudebox_creds as creds

    monkeypatch.setattr(creds, "credentials_health", _preflight_must_not_run)

    class _OrchestratorMustNotRun:
        def __init__(self, *a, **kw):
            raise AssertionError("Orchestrator should not be constructed for pi")

    import gmail_search.agents.orchestration as orch_mod

    monkeypatch.setattr(orch_mod, "Orchestrator", _OrchestratorMustNotRun)

    class _FakeConn:
        def close(self):
            pass

    monkeypatch.setattr(service, "get_connection", lambda _p: _FakeConn())
    monkeypatch.setattr(service, "fetch_events_after", lambda *a, **kw: [])
    monkeypatch.setattr(service, "_persist_rich_assistant_message", lambda *a, **kw: True)

    frames: list[str] = []

    async def consume():
        async for frame in service._real_run(
            tmp_path / "x.db",
            "sess-PI",
            "what happened",
            default_model=selected_model,
            backend="pi",
            conversation_id="conv-9",
            user_id="u1",
        ):
            frames.append(frame)

    asyncio.run(consume())

    assert workspace_dirs == ["deep-conv-conv-9"]
    assert pi_calls == [
        {
            "session_id": "sess-PI",
            "workspace": "deep-conv-conv-9",
            "conversation_id": "conv-9",
            "question": "what happened",
            "model": selected_model,
            "has_cost_sink": True,
            "user_id": "u1",
        }
    ]
    assert any("persist_ok" in f for f in frames)


def test_deep_backend_accepts_pi():
    assert service._deep_backend("pi") == "pi"


def test_deep_backend_defaults_to_pi_and_no_longer_knows_adk(monkeypatch):
    # ADK was removed; a stale "adk" from an old client gets the default.
    monkeypatch.delenv("GMAIL_DEEP_BACKEND", raising=False)
    assert service._deep_backend(None) == service.DEFAULT_BACKEND == "pi"
    assert service._deep_backend("adk") == "pi"


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


def test_pi_cost_sink_forwards_cache_counts_to_the_ledger(monkeypatch, tmp_path):
    """Regression, 2026-09-04: runtime_pi._report_cost has always sent
    cache_read_tokens / cache_write_tokens, but the service closure
    swallowed them in **extra and called record_agent_cost without them,
    so the `costs` table stored no context-cache data and could not be
    reconciled against the invoice. The closure must forward them."""
    import asyncio

    captured: dict = {}

    def fake_record_agent_cost(conn, **kw):
        captured.update(kw)
        return 0.93146

    monkeypatch.setattr(service, "record_agent_cost", fake_record_agent_cost, raising=False)
    import gmail_search.agents.cost as cost_mod

    monkeypatch.setattr(cost_mod, "record_agent_cost", fake_record_agent_cost)

    async def fake_pi_run(*, db_path, session_id, workspace, conversation_id, question, model, cost_sink, user_id=None):
        # Exactly the kwargs runtime_pi._report_cost emits.
        cost_sink(
            agent_name="pi",
            model="google/gemini-3.7-flash",
            input_tokens=759_360,
            output_tokens=26_845,
            usd_override=0.93146,
            cache_read_tokens=3_483_568,
            cache_write_tokens=17,
        )

    import gmail_search.agents.runtime_pi as rp

    monkeypatch.setattr(rp, "pi_run", fake_pi_run)
    monkeypatch.setattr(service, "_ensure_workspace_dir", lambda w: None)

    class _FakeConn:
        def close(self):
            pass

    monkeypatch.setattr(service, "get_connection", lambda _p: _FakeConn())
    monkeypatch.setattr(service, "fetch_events_after", lambda *a, **kw: [])
    monkeypatch.setattr(service, "_persist_rich_assistant_message", lambda *a, **kw: True)
    monkeypatch.setattr(service, "append_event", lambda *a, **kw: None, raising=False)

    async def consume():
        async for _ in service._real_run(
            tmp_path / "x.db",
            "sess-CACHE",
            "q",
            default_model="opus",
            backend="pi",
            conversation_id="conv-1",
            user_id="u1",
        ):
            pass

    asyncio.run(consume())

    assert captured.get("cache_read_tokens") == 3_483_568
    assert captured.get("cache_write_tokens") == 17
    assert captured.get("input_tokens") == 759_360


def test_history_includes_both_deep_battle_answers():
    import json

    parts = [{"type": "data-battle", "data": {"answer_a": "First finding", "answer_b": "Second finding"}}]
    text = service._extract_text_from_parts_json(json.dumps(parts))
    assert "First finding" in text
    assert "Second finding" in text


@pytest.mark.parametrize("backend", ["adk", "claude_native"])
def test_removed_backends_rejected_at_api_boundary(backend):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        service.AnalyzeRequest(question="q", backend=backend)


def test_use_real_pipeline_flag_parsing(monkeypatch):
    """`GMAIL_DEEP_REAL=1` (or true/yes) flips the pipeline from the
    stub to the live orchestrator. Anything else keeps the stub."""
    from gmail_search.agents.service import _use_real_pipeline

    monkeypatch.delenv("GMAIL_DEEP_REAL", raising=False)
    assert _use_real_pipeline() is False

    for truthy in ("1", "true", "True", "YES", "yes"):
        monkeypatch.setenv("GMAIL_DEEP_REAL", truthy)
        assert _use_real_pipeline() is True, f"{truthy!r} should be truthy"

    for falsy in ("0", "false", "no", "", "anything_else"):
        monkeypatch.setenv("GMAIL_DEEP_REAL", falsy)
        assert _use_real_pipeline() is False, f"{falsy!r} should be falsy"


def test_every_stage_builds_without_a_model_sdk(monkeypatch):
    # claude_code's orchestrator needs only name/model/instruction per stage;
    # the builders once imported google-adk, which was never installed.
    from gmail_search.agents.analyst import build_analyst_agent
    from gmail_search.agents.critic import build_critic_agent
    from gmail_search.agents.orchestration import DEFAULT_STAGE_MODEL, StageAgent
    from gmail_search.agents.planner import build_planner_agent
    from gmail_search.agents.retriever import build_retriever_agent
    from gmail_search.agents.writer import build_writer_agent

    monkeypatch.setenv("GMAIL_WRITER_MODEL", "writer-model")
    monkeypatch.delenv("GMAIL_CRITIC_MODEL", raising=False)
    stages = [build_planner_agent(model="m"), build_retriever_agent(model="m", user_id="u1"),
              build_writer_agent(), build_critic_agent(), build_analyst_agent(model="m", instruction="do it")]
    assert [type(s) for s in stages] == [StageAgent] * 5
    assert [s.name for s in stages] == ["planner", "retriever", "writer", "critic", "analyst"]
    assert stages[2].model == "writer-model" and stages[3].model == DEFAULT_STAGE_MODEL
    assert stages[4].instruction == "do it" and all(s.instruction for s in stages)
