"""Tests for cross-session retrieval leads extraction, JEV quality gate,
conversation superseding, battle-loser exclusion, passive 👍/👎 feedback,
and system-prompt lead injection."""

from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from gmail_search.agents.leads import (
    apply_battle_outcome_to_sessions,
    backfill_session_leads,
    extract_session_leads,
    find_prior_leads,
    format_prior_leads_guidance,
    get_session_feedback,
    is_correction_followup,
    record_session_leads,
    set_session_feedback,
    verify_answer_with_jev,
)
from gmail_search.agents.runtime_pi import render_instruction
from gmail_search.agents.service import register_agent_routes
from gmail_search.gateway.jev import JevConfig


class _InMemoryLeadsConn:
    """Minimal in-memory DB connection modeling `agent_sessions`,
    `agent_events`, and `model_battles` for unit tests."""

    def __init__(self) -> None:
        self.sessions: dict[str, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []
        self.battles: list[dict[str, Any]] = []
        self.commits = 0

    def add_session(
        self,
        session_id: str,
        *,
        user_id: str = "u1",
        conversation_id: str | None = None,
        question: str = "",
        final_answer: str | None = None,
        status: str = "done",
        plan: dict[str, Any] | None = None,
        started_at: str = "2026-10-22T10:00:00+00:00",
        finished_at: str = "2026-10-22T10:00:15+00:00",
    ) -> None:
        self.sessions[session_id] = {
            "id": session_id,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "question": question,
            "final_answer": final_answer,
            "status": status,
            "plan": plan,
            "started_at": started_at,
            "finished_at": finished_at,
        }

    def execute(self, sql: str, params: tuple = ()):
        norm = " ".join(sql.split()).lower()
        if norm.startswith("select plan, conversation_id from agent_sessions where id = %s"):
            sid = params[0]
            s = self.sessions.get(sid)
            return _Cursor([{"plan": s["plan"], "conversation_id": s["conversation_id"]}] if s else [])
        if norm.startswith("select id, plan from agent_sessions where conversation_id = %s and id != %s"):
            conv_id, excl = params
            rows = [
                {"id": s["id"], "plan": s["plan"]}
                for s in self.sessions.values()
                if s["conversation_id"] == conv_id and s["id"] != excl
            ]
            return _Cursor(rows)
        if norm.startswith("select id, plan from agent_sessions where id = %s and user_id = %s"):
            sid, uid = params
            s = self.sessions.get(sid)
            if s and s["user_id"] == uid:
                return _Cursor([{"id": s["id"], "plan": s["plan"]}])
            return _Cursor([])
        if norm.startswith("select plan from agent_sessions where id = %s and user_id = %s"):
            sid, uid = params
            s = self.sessions.get(sid)
            if s and s["user_id"] == uid:
                return _Cursor([{"plan": s["plan"]}])
            return _Cursor([])
        if norm.startswith("update agent_sessions set plan = %s::jsonb where id = %s"):
            plan_json, sid = params
            if sid in self.sessions:
                self.sessions[sid]["plan"] = json.loads(plan_json)
            return _Cursor([])
        if "from agent_sessions where user_id = %s and status = 'done' and plan is not null" in norm:
            uid = params[0]
            rows = [
                s
                for s in self.sessions.values()
                if s["user_id"] == uid and s["status"] == "done" and s["plan"] is not None
            ]
            rows.sort(key=lambda r: r["started_at"], reverse=True)
            return _Cursor(rows)
        if norm.startswith("select id, conversation_id, question, final_answer, user_id, plan from agent_sessions"):
            uid = params[0] if params else None
            rows = [
                s
                for s in self.sessions.values()
                if s["status"] == "done" and (uid is None or s["user_id"] == uid)
            ]
            rows.sort(key=lambda r: r["started_at"])
            return _Cursor(rows)
        if "from agent_events where session_id = %s" in norm:
            sid = params[0]
            rows = [e for e in self.events if e["session_id"] == sid]
            return _Cursor(rows)
        if "from model_battles" in norm:
            uid = params[0] if params else None
            rows = [b for b in self.battles if uid is None or b["user_id"] == uid]
            return _Cursor(rows)
        raise AssertionError(f"Unhandled SQL in _InMemoryLeadsConn: {sql}")

    def commit(self) -> None:
        self.commits += 1

    def close(self) -> None:
        pass


class _Cursor:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def fetchone(self) -> dict[str, Any] | None:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list[dict[str, Any]]:
        return list(self._rows)


def test_extract_session_leads_links_winning_queries_threads_and_attachments():
    side_calls = [
        {
            "name": "search_emails_batch",
            "args": {
                "searches": [
                    {"query": "91wpc drywall schedule", "top_k": 6},
                    {"query": "unrelated kitchen faucet", "top_k": 6},
                ]
            },
        },
        {
            "name": "search_emails_batch",
            "response": {
                "results": [
                    {
                        "input": {"query": "91wpc drywall schedule", "top_k": 6},
                        "result": {"results": [{"thread_id": "19cfa8fe31026be6", "cite_ref": "19cfa8fe31026be6"}]},
                    },
                    {
                        "input": {"query": "unrelated kitchen faucet", "top_k": 6},
                        "result": {"results": [{"thread_id": "1900000000000001", "cite_ref": "1900000000000001"}]},
                    },
                ]
            },
        },
        {
            "name": "get_thread_batch",
            "args": {"thread_ids": ["19cfa8fe31026be6", "1900000000000001"]},
        },
        {
            "name": "get_thread_batch",
            "response": {
                "results": [
                    {
                        "thread_id": "19cfa8fe31026be6",
                        "result": {
                            "cite_ref": "19cfa8fe31026be6",
                            "messages": [
                                {
                                    "id": "m1",
                                    "attachments": [{"id": 18421, "filename": "schedule.pdf"}],
                                }
                            ],
                        },
                    },
                    {
                        "thread_id": "1900000000000001",
                        "result": {
                            "cite_ref": "1900000000000001",
                            "messages": [
                                {
                                    "id": "m2",
                                    "attachments": [{"id": 99999, "filename": "faucet.pdf"}],
                                }
                            ],
                        },
                    },
                ]
            },
        },
        {
            "name": "get_attachment_batch",
            "args": {"items": [{"attachment_id": 18421, "mode": "text"}, {"attachment_id": 99999, "mode": "text"}]},
        },
        {
            "name": "get_attachment_batch",
            "response": {
                "results": [
                    {"input": {"attachment_id": 18421}, "result": {"attachment_id": 18421, "extracted_text": "Drywall starts May 12"}},
                    {"input": {"attachment_id": 99999}, "result": {"attachment_id": 99999, "extracted_text": "Faucet"}},
                ]
            },
        },
    ]
    leads = extract_session_leads(
        "in the schedule for 91wpc when will they start drywall?",
        "Drywall at 91 WPC is scheduled to start on May 12 [ref:19cfa8fe31026be6].",
        side_calls,
    )
    assert leads["eligible"] is True
    assert leads["cited_thread_ids"] == ["19cfa8fe31026be6"]
    # Only attachment 18421 belongs to the cited thread 19cfa8fe31026be6 (not 99999 from uncited thread).
    assert leads["cited_attachment_ids"] == [18421]
    assert leads["winning_queries"] == [{"tool": "search_emails", "query": "91wpc drywall schedule"}]


def test_extract_session_leads_rejects_uncited_or_hallucinated_refs():
    side_calls = [
        {
            "name": "search_emails_batch",
            "args": {"searches": [{"query": "appraisal"}]},
            "response": {
                "results": [
                    {
                        "input": {"query": "appraisal"},
                        "result": {"results": [{"thread_id": "19aaaa", "cite_ref": "19aaaa"}]},
                    }
                ]
            },
        }
    ]
    # Cites a thread ID that never appeared in tool results:
    leads = extract_session_leads(
        "when is the appraisal?",
        "The appraisal is tomorrow [ref:deadbeef999].",
        side_calls,
    )
    assert leads["cited_thread_ids"] == []
    assert leads["eligible"] is False


def test_verify_answer_with_jev_gates_unanswered_responses(monkeypatch):
    def fake_post(self, url, *, headers=None, json=None):
        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "answers": {
                        "quality": {
                            "probabilities": {"answered": 0.12, "unanswered": 0.88},
                        }
                    }
                }

        return _Resp()

    monkeypatch.setattr("httpx.Client.post", fake_post)
    ok, prob = verify_answer_with_jev(
        "what is the alarm code?",
        "I searched your emails [ref:19aaaa] but could not find any alarm code.",
        JevConfig("https://jev.example/v1/noul", "secret"),
    )
    assert ok is False
    assert prob == 0.12


def test_conversation_superseding_and_correction_followup():
    conn = _InMemoryLeadsConn()
    conn.add_session(
        "s_turn1",
        user_id="u1",
        conversation_id="conv_1",
        question="what did we pay for the 91wpc appraisal?",
        final_answer="You paid $650 [ref:19t1].",
        started_at="2026-10-22T10:00:00+00:00",
    )
    record_session_leads(
        conn,
        session_id="s_turn1",
        conversation_id="conv_1",
        question="what did we pay for the 91wpc appraisal?",
        final_answer="You paid $650 [ref:19t1].",
        side_calls=[
            {
                "name": "search_emails",
                "args": {"query": "91wpc appraisal"},
                "response": {"results": [{"cite_ref": "19t1"}]},
            }
        ],
        check_jev=False,
    )
    assert conn.sessions["s_turn1"]["plan"]["leads"]["eligible"] is True
    assert "superseded_by" not in conn.sessions["s_turn1"]["plan"]

    # Turn 2 in the same conversation corrects Turn 1 even without citing a new thread:
    assert is_correction_followup("No, that's wrong, that was for Costello not 91wpc") is True
    conn.add_session(
        "s_turn2",
        user_id="u1",
        conversation_id="conv_1",
        question="No, that's wrong, that was for Costello not 91wpc",
        final_answer="You are right — I could not find a separate 91wpc appraisal.",
        started_at="2026-10-22T10:02:00+00:00",
    )
    record_session_leads(
        conn,
        session_id="s_turn2",
        conversation_id="conv_1",
        question="No, that's wrong, that was for Costello not 91wpc",
        final_answer="You are right — I could not find a separate 91wpc appraisal.",
        side_calls=[],
        check_jev=False,
    )
    assert conn.sessions["s_turn1"]["plan"]["superseded_by"] == "s_turn2"
    # A subsequent conversation asking about 91wpc appraisal must NOT receive s_turn1's superseded lead:
    leads = find_prior_leads(conn, user_id="u1", question="how much was the 91wpc appraisal?")
    assert leads == []


def test_feedback_and_battle_exclusion_and_prior_lead_formatting():
    conn = _InMemoryLeadsConn()
    for sid, q, tid in (
        ("s_good", "when will 91wpc start drywall schedule?", "19drywall_good"),
        ("s_bad", "when will 91wpc start drywall framing?", "19drywall_bad"),
        ("s_loser", "when will 91wpc start drywall inspection?", "19drywall_loser"),
    ):
        conn.add_session(
            sid,
            user_id="u1",
            question=q,
            final_answer=f"Answer [ref:{tid}]",
        )
        record_session_leads(
            conn,
            session_id=sid,
            question=q,
            final_answer=f"Answer [ref:{tid}]",
            side_calls=[
                {
                    "name": "search_emails",
                    "args": {"query": "91wpc drywall"},
                    "response": {"results": [{"cite_ref": tid}]},
                }
            ],
            check_jev=False,
        )

    # Thumb-up s_good and thumb-down s_bad:
    assert set_session_feedback(conn, session_id="s_good", user_id="u1", feedback="up") is True
    assert set_session_feedback(conn, session_id="s_bad", user_id="u1", feedback="down") is True
    assert get_session_feedback(conn, session_id="s_good", user_id="u1") == "up"
    assert get_session_feedback(conn, session_id="s_bad", user_id="u1") == "down"
    # Cross-user attempt returns False:
    assert set_session_feedback(conn, session_id="s_good", user_id="other_user", feedback="down") is False

    # Mark s_loser as a battle loser:
    apply_battle_outcome_to_sessions(
        conn,
        user_id="u1",
        winner="a",
        session_a="s_good",
        session_b="s_loser",
    )

    matched = find_prior_leads(conn, user_id="u1", question="what is the 91wpc drywall schedule?")
    assert [m["session_id"] for m in matched] == ["s_good"]

    block = format_prior_leads_guidance(matched)
    assert block is not None
    assert "# Prior Investigation Leads" in block
    assert "`19drywall_good`" in block
    assert 'search_emails("91wpc drywall")' in block

    rendered = render_instruction("google/gemini-3.8-flash", prior_leads_guidance=block)
    assert "# Prior Investigation Leads" in rendered
    assert "`19drywall_good`" in rendered


def test_feedback_http_endpoint(monkeypatch, tmp_path):
    conn = _InMemoryLeadsConn()
    conn.add_session("sess123", user_id="user_a", question="q", final_answer="a [ref:t1]")
    record_session_leads(
        conn,
        session_id="sess123",
        question="q",
        final_answer="a [ref:t1]",
        side_calls=[{"name": "search_emails", "args": {"query": "q"}, "response": {"results": [{"cite_ref": "t1"}]}}],
        check_jev=False,
    )

    monkeypatch.setattr("gmail_search.agents.service.get_connection", lambda _p: conn)
    app = FastAPI()
    from gmail_search.auth import require_user_id

    app.dependency_overrides[require_user_id] = lambda: "user_a"
    register_agent_routes(app, tmp_path / "dummy.db")
    client = TestClient(app)

    resp = client.post("/api/agent/sessions/sess123/feedback", json={"feedback": "down"})
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "session_id": "sess123", "feedback": "down"}
    assert conn.sessions["sess123"]["plan"]["feedback"] == "down"
    assert conn.sessions["sess123"]["plan"]["leads"]["eligible"] is False

    # Invalid feedback value -> 400
    resp_bad = client.post("/api/agent/sessions/sess123/feedback", json={"feedback": "meh"})
    assert resp_bad.status_code == 400

    # Other user's session -> 404
    app.dependency_overrides[require_user_id] = lambda: "user_b"
    resp_other = client.post("/api/agent/sessions/sess123/feedback", json={"feedback": "up"})
    assert resp_other.status_code == 404


def test_backfill_session_leads_from_agent_events():
    conn = _InMemoryLeadsConn()
    conn.add_session(
        "s_hist",
        user_id="u1",
        question="# Prior conversation in this chat\n...\n# Latest user question (answer this)\n\nWhere is the Hillbrook report card for Jules?",
        final_answer="Jules's report card is attached [ref:19a4ca14d351c76d].",
    )
    conn.events.append(
        {
            "session_id": "s_hist",
            "seq": 1,
            "kind": "mcp_tool_call_full",
            "payload": {
                "name": "find_facts",
                "args": {"query": "Hillbrook report card"},
                "response": {"facts": [{"thread_id": "19a4ca14d351c76d", "fact": "Report card"}]},
            },
        }
    )
    stats = backfill_session_leads(conn, user_id="u1", check_jev=False)
    assert stats == {"processed": 1, "eligible": 1}
    plan = conn.sessions["s_hist"]["plan"]
    assert plan["leads"]["question"] == "Where is the Hillbrook report card for Jules?"
    assert plan["leads"]["cited_thread_ids"] == ["19a4ca14d351c76d"]
    assert plan["leads"]["winning_queries"] == [{"tool": "find_facts", "query": "Hillbrook report card"}]
