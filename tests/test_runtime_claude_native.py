"""Tests for the single-agent Claude Native runtime adapter.

Verifies that `native_run` synthesizes the full orchestrator-shape
event sequence (`plan` / `tool_call` / `evidence` / `code_run` /
`analysis` / `draft` / `final`) from one `claudebox_invoke` call,
plus the error / cleanup paths.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from gmail_search.agents import runtime_claude as rc
from gmail_search.agents import runtime_claude_native as native
from gmail_search.agents.orchestration import StageResult

# ── Fake DB connection: records every append_event call ────────────


class _FakeConn:
    """Captures `append_event` writes + `finalize_session` calls so
    tests can inspect the synthesized SSE sequence without standing up
    psycopg. `execute()` returns a thin row-stub with `fetchone()`
    so `append_event` and `finalize_session` succeed."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.finalized: list[dict[str, Any]] = []
        self.closed = False
        self._next_seq = 0

    def execute(self, sql: str, params: tuple):
        sql_lower = sql.lower()
        if "insert into agent_events" in sql_lower:
            self._next_seq += 1
            session_id, _sid_again, agent_name, kind, payload_json = params
            import json

            self.events.append(
                {
                    "seq": self._next_seq,
                    "session_id": session_id,
                    "agent_name": agent_name,
                    "kind": kind,
                    "payload": json.loads(payload_json),
                }
            )
            return _ReturningRow({"seq": self._next_seq})
        if "update agent_sessions" in sql_lower:
            status, final_answer, session_id = params
            self.finalized.append({"session_id": session_id, "status": status, "final_answer": final_answer})
            return _ReturningRow(None)
        return _ReturningRow(None)

    def commit(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class _ReturningRow:
    """Mimics psycopg's RETURNING-then-fetchone pattern."""

    def __init__(self, row: dict | None) -> None:
        self._row = row

    def fetchone(self):
        return self._row


# ── Fixtures / helpers ────────────────────────────────────────────


def _install_fakes(
    monkeypatch,
    *,
    invoke_result: StageResult | None = None,
    invoke_raises: BaseException | None = None,
    streamed_tool_calls: list[dict] | None = None,
) -> tuple[_FakeConn, list[dict]]:
    """Wire `claudebox_invoke`, the admin register/unregister, and the
    DB connection factory to in-memory fakes. Returns `(conn, calls)`
    where `calls` records every fake-claudebox invocation so tests can
    assert how it was called.

    `streamed_tool_calls`, when supplied, is a list of `{name, args}`
    dicts the fake invoke will fire through `event_sink` BEFORE
    returning — simulating mid-flight JSONL streaming. Use it to
    verify native_run records the streamed events AND skips the
    post-hoc per-tool emission."""
    conn = _FakeConn()
    monkeypatch.setattr(native, "get_connection", lambda _path: conn)

    register_calls: list[str] = []
    unregister_calls: list[str] = []

    async def fake_register(session_id, *, evidence_records, conversation_id=None, workspace=None):
        register_calls.append(session_id)

    async def fake_unregister(session_id):
        unregister_calls.append(session_id)

    invoke_calls: list[dict] = []

    async def fake_invoke(agent, prompt, *, workspace, session_id=None, cost_sink=None, event_sink=None):
        invoke_calls.append(
            {
                "agent_name": getattr(agent, "name", None),
                "agent_model": getattr(agent, "model", None),
                "prompt": prompt,
                "workspace": workspace,
                "session_id": session_id,
                "cost_sink": cost_sink,
                "event_sink": event_sink,
            }
        )
        # Simulate the JSONL tailer surfacing mid-flight tool_calls
        # via `event_sink` so tests can assert streamed events landed.
        if streamed_tool_calls and event_sink is not None:
            for tc in streamed_tool_calls:
                await event_sink("tool_call", tc)
        if invoke_raises is not None:
            raise invoke_raises
        return invoke_result if invoke_result is not None else StageResult(text="", tool_calls=[])

    monkeypatch.setattr(rc, "register_session_via_admin", fake_register)
    monkeypatch.setattr(rc, "unregister_session_via_admin", fake_unregister)
    monkeypatch.setattr(rc, "claudebox_invoke", fake_invoke)

    # Stash both call lists on conn for convenience.
    conn._register_calls = register_calls  # type: ignore[attr-defined]
    conn._unregister_calls = unregister_calls  # type: ignore[attr-defined]
    return conn, invoke_calls


def _kinds(conn: _FakeConn) -> list[str]:
    return [e["kind"] for e in conn.events]


# ── Happy path: full event sequence ───────────────────────────────


def test_native_run_emits_full_event_sequence_in_order(monkeypatch):
    """A turn with both retrieval AND run_code calls must produce:
    plan → tool_call(s) (streamed mid-flight) → evidence → analysis →
    draft → final, in that order. Mirrors what the existing UI consumer
    expects.

    Note: with mid-flight streaming wired in, `tool_call` events fire
    via `event_sink` (agent_name="claude_native") before invoke
    returns, and the post-hoc per-tool emission is skipped (no more
    duplicate `tool_call`/`code_run` events). The aggregate `evidence`
    + `analysis` events still fire."""
    tool_calls = [
        {"name": "search_emails", "args": {"query": "foo"}},
        {
            "name": "search_emails",
            "response": {"results": [{"cite_ref": "abc12345"}, {"cite_ref": "deadbeef"}]},
        },
        {"name": "run_code", "args": {"code": "print(1)"}},
        {"name": "run_code", "response": {"artifacts": [{"id": 7}, {"id": 9}]}},
    ]
    streamed = [
        {"name": "search_emails", "args": {"query": "foo"}},
        {"name": "run_code", "args": {"code": "print(1)"}},
    ]
    conn, invoke_calls = _install_fakes(
        monkeypatch,
        invoke_result=StageResult(text="here is the answer", tool_calls=tool_calls),
        streamed_tool_calls=streamed,
    )

    asyncio.run(
        native.native_run(
            db_path=Path("/unused.db"),
            session_id="sess-A",
            workspace="deep-sess-A",
            conversation_id="conv-1",
            question="how many?",
            model="opus",
            cost_sink=None,
        ),
    )

    # Plan first.
    assert _kinds(conn)[0] == "plan"
    # Then mid-flight streamed tool_calls (one per JSONL tool_use).
    tool_call_kinds = [e for e in conn.events if e["kind"] == "tool_call"]
    assert len(tool_call_kinds) == 2
    assert tool_call_kinds[0]["agent_name"] == "claude_native"
    # Then evidence with cite_refs extracted from FULL tool_calls.
    evidence = next(e for e in conn.events if e["kind"] == "evidence")
    assert evidence["payload"]["cite_refs"] == ["abc12345", "deadbeef"]
    # Per-tool code_run is skipped now — streaming covered it.
    code_runs = [e for e in conn.events if e["kind"] == "code_run"]
    assert len(code_runs) == 0
    # Aggregate analysis still fires with artifact_ids extracted.
    analysis = next(e for e in conn.events if e["kind"] == "analysis")
    assert analysis["payload"]["artifact_ids"] == [7, 9]
    assert analysis["payload"]["called_run_code"] is True
    # Draft + final carry the answer text.
    draft = next(e for e in conn.events if e["kind"] == "draft")
    final = next(e for e in conn.events if e["kind"] == "final")
    assert draft["payload"]["text"] == "here is the answer"
    assert final["payload"]["text"] == "here is the answer"
    # Order: plan < tool_call(streamed) < evidence < analysis < draft < final.
    seq_by_kind = {e["kind"]: e["seq"] for e in conn.events}
    assert (
        seq_by_kind["plan"]
        < seq_by_kind["tool_call"]
        < seq_by_kind["evidence"]
        < seq_by_kind["analysis"]
        < seq_by_kind["draft"]
        < seq_by_kind["final"]
    )
    # Session finalized as done.
    assert conn.finalized == [
        {"session_id": "sess-A", "status": "done", "final_answer": "here is the answer"},
    ]
    # MCP register/unregister bracketed the invoke.
    assert conn._register_calls == ["sess-A"]
    assert conn._unregister_calls == ["sess-A"]
    # Invoke saw the right wiring.
    assert invoke_calls[0]["workspace"] == "deep-sess-A"
    assert invoke_calls[0]["session_id"] == "sess-A"
    assert invoke_calls[0]["agent_name"] == "claude_native"
    assert invoke_calls[0]["agent_model"] == "opus"


def test_native_run_skips_code_events_when_no_run_code(monkeypatch):
    """A retrieval-only turn (no run_code) must NOT emit code_run or
    analysis events — matches the orchestrator's `_run_analyst_if_needed`
    skip behaviour."""
    tool_calls = [
        {"name": "search_emails", "args": {"query": "foo"}},
        {"name": "search_emails", "response": {"results": [{"cite_ref": "r1"}]}},
    ]
    streamed = [{"name": "search_emails", "args": {"query": "foo"}}]
    conn, _ = _install_fakes(
        monkeypatch,
        invoke_result=StageResult(text="answer", tool_calls=tool_calls),
        streamed_tool_calls=streamed,
    )

    asyncio.run(
        native.native_run(
            db_path=Path("/unused.db"),
            session_id="sess-B",
            workspace="ws",
            conversation_id=None,
            question="q",
            model=None,
            cost_sink=None,
        ),
    )

    kinds = _kinds(conn)
    assert "code_run" not in kinds
    assert "analysis" not in kinds
    assert kinds == ["plan", "tool_call", "evidence", "draft", "final"]


def test_native_run_with_empty_tool_calls_still_emits_draft_and_final(monkeypatch):
    """A turn where Claude answers without invoking any tool still
    needs the full event tail so the UI can render the final markdown."""
    conn, _ = _install_fakes(
        monkeypatch,
        invoke_result=StageResult(text="prose-only answer", tool_calls=[]),
    )

    asyncio.run(
        native.native_run(
            db_path=Path("/unused.db"),
            session_id="sess-C",
            workspace="ws",
            conversation_id=None,
            question="q",
            model=None,
            cost_sink=None,
        ),
    )

    kinds = _kinds(conn)
    # plan + evidence (empty cite_refs) + draft + final. NO tool_call,
    # NO code_run, NO analysis.
    assert "tool_call" not in kinds
    assert "code_run" not in kinds
    assert "analysis" not in kinds
    assert kinds == ["plan", "evidence", "draft", "final"]
    final = next(e for e in conn.events if e["kind"] == "final")
    assert final["payload"]["text"] == "prose-only answer"


def test_native_run_emits_error_event_when_claudebox_invoke_raises(monkeypatch):
    """A claudebox failure must surface as an `error` event AND mark
    the session as `error` so the UI doesn't poll forever."""
    conn, _ = _install_fakes(
        monkeypatch,
        invoke_raises=RuntimeError("simulated claudebox 500"),
    )

    asyncio.run(
        native.native_run(
            db_path=Path("/unused.db"),
            session_id="sess-D",
            workspace="ws",
            conversation_id=None,
            question="q",
            model=None,
            cost_sink=None,
        ),
    )

    error_events = [e for e in conn.events if e["kind"] == "error"]
    assert len(error_events) == 1
    assert "simulated claudebox 500" in error_events[0]["payload"]["message"]
    # Session finalized as error.
    assert conn.finalized == [
        {"session_id": "sess-D", "status": "error", "final_answer": None},
    ]
    # Unregister still ran (registered before the raise).
    assert conn._unregister_calls == ["sess-D"]


def test_native_run_extracts_cite_refs_and_artifact_ids_correctly(monkeypatch):
    """Cross-check the orchestration helpers we delegate to:
    `_cite_refs_from_tool_calls` should pull every unique cite_ref,
    `_artifact_ids_from_tool_calls` every artifact id."""
    tool_calls = [
        {"name": "search_emails", "args": {"query": "x"}},
        {
            "name": "search_emails",
            "response": {
                "results": [
                    {"cite_ref": "aaa11111"},
                    {"cite_ref": "bbb22222"},
                    {"cite_ref": "aaa11111"},  # dupe — must be deduped
                ]
            },
        },
        {"name": "query_emails", "args": {"sender": "y"}},
        {
            "name": "query_emails",
            "response": {"results": [{"cite_ref": "ccc33333"}]},
        },
        {"name": "run_code", "args": {"code": "p"}},
        {
            "name": "run_code",
            "response": {"artifacts": [{"id": 1}, {"id": 2}]},
        },
        {"name": "run_code", "args": {"code": "q"}},
        {
            "name": "run_code",
            "response": {"artifacts": [{"id": 3}]},
        },
    ]
    streamed = [
        {"name": "search_emails", "args": {"query": "x"}},
        {"name": "query_emails", "args": {"sender": "y"}},
        {"name": "run_code", "args": {"code": "p"}},
        {"name": "run_code", "args": {"code": "q"}},
    ]
    conn, _ = _install_fakes(
        monkeypatch,
        invoke_result=StageResult(text="ok", tool_calls=tool_calls),
        streamed_tool_calls=streamed,
    )

    asyncio.run(
        native.native_run(
            db_path=Path("/unused.db"),
            session_id="sess-E",
            workspace="ws",
            conversation_id=None,
            question="q",
            model=None,
            cost_sink=None,
        ),
    )

    evidence = next(e for e in conn.events if e["kind"] == "evidence")
    assert evidence["payload"]["cite_refs"] == ["aaa11111", "bbb22222", "ccc33333"]
    analysis = next(e for e in conn.events if e["kind"] == "analysis")
    assert analysis["payload"]["artifact_ids"] == [1, 2, 3]
    # Streamed tool_calls (one per JSONL tool_use): four entries.
    # No post-hoc code_run emission (skipped because streaming
    # already covered it).
    assert sum(1 for e in conn.events if e["kind"] == "tool_call") == 4
    assert sum(1 for e in conn.events if e["kind"] == "code_run") == 0


def test_native_run_unregisters_session_even_on_error(monkeypatch):
    """The finally cleanup must drop the MCP session regardless of
    whether the invoke succeeded or raised — otherwise leaked sessions
    accumulate on the MCP server."""
    conn, _ = _install_fakes(
        monkeypatch,
        invoke_raises=RuntimeError("boom"),
    )

    asyncio.run(
        native.native_run(
            db_path=Path("/unused.db"),
            session_id="sess-F",
            workspace="ws",
            conversation_id=None,
            question="q",
            model=None,
            cost_sink=None,
        ),
    )

    assert conn._register_calls == ["sess-F"]
    assert conn._unregister_calls == ["sess-F"]
    assert conn.closed is True


def test_native_run_default_model_is_sonnet(monkeypatch):
    """When `model=None`, the agent passed to claudebox_invoke must
    default to `sonnet` (matches the spec's `model or "sonnet"`)."""
    conn, invoke_calls = _install_fakes(
        monkeypatch,
        invoke_result=StageResult(text="x", tool_calls=[]),
    )

    asyncio.run(
        native.native_run(
            db_path=Path("/unused.db"),
            session_id="sess-G",
            workspace="ws",
            conversation_id=None,
            question="q",
            model=None,
            cost_sink=None,
        ),
    )

    assert invoke_calls[0]["agent_model"] == "sonnet"


# Pytest sanity: importing this module must not require live psycopg.
def test_module_imports_cleanly():
    assert callable(native.native_run)
    assert isinstance(native.NATIVE_INSTRUCTION, str)
    assert "session_id" in native.NATIVE_INSTRUCTION
