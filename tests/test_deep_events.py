"""deep_events: emitters shared by the single-agent deep backends."""

from __future__ import annotations

import json
from typing import Any

from gmail_search.agents import deep_events


class _FakeConn:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self._seq = 0

    def execute(self, sql: str, params: tuple):
        if "insert into agent_events" in sql.lower():
            self._seq += 1
            session_id, _again, agent_name, kind, payload_json = params
            self.events.append({"agent_name": agent_name, "kind": kind, "payload": json.loads(payload_json)})
            return _Row({"seq": self._seq})
        return _Row(None)

    def commit(self) -> None:
        pass


class _Row:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


def test_plan_event_uses_supplied_agent_name_and_approach():
    conn = _FakeConn()
    deep_events.emit_plan_event(conn, "s1", agent_name="pi", approach="single pi loop")
    assert conn.events == [
        {"agent_name": "pi", "kind": "plan", "payload": {"native_mode": True, "approach": "single pi loop"}}
    ]


def test_retriever_events_emit_tool_calls_then_evidence():
    conn = _FakeConn()
    calls = [
        {"name": "search_emails", "args": {"q": "delta"}},
        {"name": "search_emails", "response": {"results": [{"cite_ref": "t1"}, {"cite_ref": "t2"}]}},
    ]
    deep_events.emit_retriever_events(conn, "s1", calls)
    kinds = [e["kind"] for e in conn.events]
    assert kinds == ["tool_call", "evidence"]
    assert conn.events[1]["payload"]["cite_refs"] == ["t1", "t2"]
    assert conn.events[1]["payload"]["summary"] == "Retrieval calls: 1× search_emails."


def test_retriever_events_skip_per_tool_when_streamed():
    conn = _FakeConn()
    calls = [{"name": "search_emails", "args": {"q": "x"}}]
    deep_events.emit_retriever_events(conn, "s1", calls, skip_per_tool_emission=True)
    assert [e["kind"] for e in conn.events] == ["evidence"]


def test_analyst_events_silent_without_run_code():
    conn = _FakeConn()
    deep_events.emit_analyst_events(conn, "s1", [{"name": "search_emails", "args": {}}])
    assert conn.events == []


def test_analyst_events_collect_artifact_ids():
    conn = _FakeConn()
    calls = [
        {"name": "run_code", "args": {"code": "print(1)"}},
        {"name": "run_code", "response": {"artifacts": [{"id": 7}]}},
    ]
    deep_events.emit_analyst_events(conn, "s1", calls, skip_per_tool_emission=True)
    assert [e["kind"] for e in conn.events] == ["analysis"]
    assert conn.events[0]["payload"]["artifact_ids"] == [7]
    assert conn.events[0]["payload"]["called_run_code"] is True


def test_writer_and_final_carry_same_text():
    conn = _FakeConn()
    deep_events.emit_writer_and_final(conn, "s1", "answer")
    assert [(e["agent_name"], e["kind"]) for e in conn.events] == [("writer", "draft"), ("root", "final")]
    assert all(e["payload"] == {"text": "answer"} for e in conn.events)


def test_error_event_uses_agent_name():
    conn = _FakeConn()
    deep_events.emit_error(conn, "s1", RuntimeError("boom"), agent_name="pi")
    assert conn.events == [{"agent_name": "pi", "kind": "error", "payload": {"message": "boom"}}]


def test_retriever_events_recognize_batch_tool_names():
    conn = _FakeConn()
    calls = [
        {"name": "search_emails_batch", "args": {"searches": [{"q": "delta"}]}},
        {
            "name": "search_emails_batch",
            "response": {"results": [{"input": {"q": "delta"}, "result": {"results": [{"cite_ref": "t1"}]}}]},
        },
    ]
    deep_events.emit_retriever_events(conn, "s1", calls)
    kinds = [e["kind"] for e in conn.events]
    assert kinds == ["tool_call", "evidence"]
    assert conn.events[1]["payload"]["summary"] == "Retrieval calls: 1× search_emails_batch."
