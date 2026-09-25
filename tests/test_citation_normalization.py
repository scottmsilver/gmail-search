from gmail_search.agents.citations import normalize_citations


class Result:
    def __init__(self, rows):
        self.rows = rows

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return self.rows


class Connection:
    def __init__(self, owner="tenant-a"):
        self.owner = owner
        self.calls = []

    def execute(self, sql, params):
        self.calls.append((sql, params))
        if "agent_sessions" in sql:
            assert params == ("session-one",)
            return Result([{"user_id": self.owner}])
        assert "user_id = %s" in sql
        assert params[0] == "tenant-a"
        # A foreign tenant's message is deliberately absent from scoped results.
        return Result([{"id": "message-one", "thread_id": "thread-one"},
                       {"id": "message-two", "thread_id": "thread-one"}])


def test_normalizes_message_ids_and_preserves_threads_and_artifacts():
    conn = Connection()
    text = "A [ref:message-one] B [ref:thread-one] C [ref:message-two] [art:12]"
    assert normalize_citations(conn, "session-one", text) == "A [ref:thread-one] B [ref:thread-one] C [ref:thread-one] [art:12]"
    assert len(conn.calls) == 2


def test_unknown_and_foreign_ids_do_not_become_links():
    result = normalize_citations(Connection(), "session-one", "A [ref:unknown] B [ref:foreign-message]")
    assert result == "A [source unavailable] B [source unavailable]"


def test_citation_only_inline_code_becomes_clickable():
    assert normalize_citations(Connection(), "session-one", "A `[ref:message-one]`.") == "A [ref:thread-one]."


def test_no_citations_does_not_query():
    conn = Connection()
    assert normalize_citations(conn, "session-one", "Answer [art:1]") == "Answer [art:1]"
    assert conn.calls == []


def test_missing_owner_never_looks_up_mail():
    conn = Connection(owner=None)
    assert normalize_citations(conn, "session-one", "A [ref:message-one]") == "A [source unavailable]"
    assert len(conn.calls) == 1


def test_finish_normalizes_before_events_and_storage(monkeypatch):
    from gmail_search.agents import runtime_pi
    conn = Connection()
    captured = {}
    monkeypatch.setattr(runtime_pi, "emit_retriever_events", lambda *a, **kw: None)
    monkeypatch.setattr(runtime_pi, "emit_analyst_events", lambda *a, **kw: None)
    monkeypatch.setattr(runtime_pi, "sweep_and_extend_final_text", lambda *a, **kw: kw["base_text"])
    monkeypatch.setattr(runtime_pi, "session_elapsed_ms", lambda *a, **kw: 0)
    monkeypatch.setattr(runtime_pi, "emit_writer_and_final", lambda c, sid, text, **kw: captured.update(event=text))
    monkeypatch.setattr(runtime_pi, "finalize_session", lambda *a, **kw: captured.update(saved=kw["final_answer"]))
    runtime_pi._finish_ok(conn, session_id="session-one", workspace="w", conversation_id="c",
                          turn_started_at=0, outcome=runtime_pi.TurnOutcome(final_text="A [ref:message-one]", local_tool_calls=[], usage=None), side_calls=[])
    assert captured == {"event": "A [ref:thread-one]", "saved": "A [ref:thread-one]"}
