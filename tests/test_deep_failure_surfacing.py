"""Tests for deep-mode failure surfacing helpers in agents.service.

BUG #2 backstop: when a deep run fails, the turn must leave a visible
assistant bubble in `conversation_messages` (durability on reload) and
emit an `error` SSE frame the front-end renders. These unit-test the
helpers directly with a fake DB connection.
"""

from __future__ import annotations

import json

from gmail_search.agents import service


class _FakeConn:
    """Records executed SQL + params; answers the next-seq SELECT."""

    def __init__(self, *, next_seq: int = 0):
        self._next_seq = next_seq
        self.executed: list[tuple[str, tuple]] = []
        self.committed = False

    def execute(self, sql, params=()):
        self.executed.append((sql, params))

        class _Cur:
            def __init__(self, row):
                self._row = row

            def fetchone(self):
                return self._row

        if "MAX(seq)" in sql:
            return _Cur({"next_seq": self._next_seq})
        return _Cur(None)

    def commit(self):
        self.committed = True


def test_persist_assistant_error_inserts_text_bubble():
    conn = _FakeConn(next_seq=7)
    ok = service._persist_assistant_error(
        conn,
        conversation_id="conv-1",
        session_id="sess-1",
        text="⚠️ boom",
    )
    assert ok is True
    assert conn.committed is True
    inserts = [(s, p) for (s, p) in conn.executed if "INSERT INTO conversation_messages" in s]
    assert len(inserts) == 1
    sql, params = inserts[0]
    conv_id, seq, parts_json = params
    assert conv_id == "conv-1"
    assert seq == 7
    parts = json.loads(parts_json)
    assert parts == [{"type": "text", "text": "⚠️ boom"}]


def test_persist_assistant_error_no_conversation_id():
    conn = _FakeConn()
    assert service._persist_assistant_error(conn, conversation_id=None, session_id="s", text="x") is False
    assert conn.executed == []


def test_surface_deep_failure_emits_error_frame_and_finalizes(monkeypatch):
    conn = _FakeConn(next_seq=0)
    finalized: list[tuple] = []

    def _fake_finalize(c, session_id, *, status, final_answer=None):
        finalized.append((session_id, status, final_answer))

    monkeypatch.setattr(service, "finalize_session", _fake_finalize)

    frame = service._surface_deep_failure(
        conn,
        conversation_id="conv-9",
        session_id="sess-9",
        reason="creds expired. Please refresh.",
        final_answer="blocked: expired",
    )

    # SSE error frame the front-end understands: event:error + payload.message
    assert frame.startswith("event: error\n")
    data = json.loads(frame.split("data: ", 1)[1].strip())
    assert data["session_id"] == "sess-9"
    assert data["payload"]["message"] == "creds expired. Please refresh."

    # Persisted a visible failure bubble.
    inserts = [(s, p) for (s, p) in conn.executed if "INSERT INTO conversation_messages" in s]
    assert len(inserts) == 1
    parts = json.loads(inserts[0][1][2])
    assert parts[0]["type"] == "text"
    assert "Deep analysis failed" in parts[0]["text"]
    assert "creds expired" in parts[0]["text"]

    # Finalized the session as errored.
    assert finalized == [("sess-9", "error", "blocked: expired")]


def test_surface_deep_failure_still_yields_frame_when_finalize_raises(monkeypatch):
    conn = _FakeConn()

    def _boom(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr(service, "finalize_session", _boom)
    frame = service._surface_deep_failure(
        conn,
        conversation_id="c",
        session_id="s",
        reason="r",
        final_answer="f",
    )
    assert frame.startswith("event: error\n")


def test_expired_creds_user_message_includes_local_time():
    msg = service._expired_creds_user_message(1_700_000_000.0)
    assert "Your Claude credentials expired at " in msg
    assert "claude" in msg  # actionable refresh instruction
    assert "retry" in msg


def test_expired_creds_user_message_handles_no_expiry():
    msg = service._expired_creds_user_message(None)
    assert "Your Claude credentials expired." in msg
    assert "retry" in msg
