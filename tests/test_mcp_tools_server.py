"""Focused tests for the streamable-HTTP MCP tools server.

We test the slice WE own — session registry, tool re-routing,
artifact persistence — without spinning up uvicorn or hitting a
real backend. Underlying primitives (`_search_emails_impl`,
`execute_in_sandbox`, `save_artifact`) are mocked because they're
already covered by their own test files.
"""

from __future__ import annotations

import asyncio

import pytest

from gmail_search.agents import mcp_tools_server as mts


@pytest.fixture(autouse=True)
def _clear_session_registry():
    """Wipe the module-level session dict before AND after each test
    so tests can't leak context into each other."""
    mts._SESSIONS.clear()
    yield
    mts._SESSIONS.clear()


# ── register_session + tool happy path ─────────────────────────────


def test_search_emails_batch_routes_to_underlying_impl(monkeypatch):
    """register_session + a batch search call should reach the
    underlying batch impl with the input list."""
    captured: dict = {}

    async def fake_batch(searches):
        captured["searches"] = searches
        return {"results": [{"input": s, "result": {"results": []}} for s in searches]}

    monkeypatch.setattr(mts, "_search_emails_batch_impl", fake_batch)

    mts.register_session("sess-1", evidence_records=None, db_dsn=None)
    out = asyncio.run(
        mts._tool_search_emails_batch(
            "sess-1",
            searches=[
                {"query": "budget", "date_from": "2026-01-01", "top_k": 5},
                {"query": "delta refund"},
            ],
        )
    )

    assert captured["searches"][0]["query"] == "budget"
    assert captured["searches"][1]["query"] == "delta refund"
    assert len(out["results"]) == 2


def test_query_emails_batch_routes_to_underlying_impl(monkeypatch):
    """The batch structured-filter tool must forward the filters list."""
    captured: dict = {}

    async def fake_batch(filters):
        captured["filters"] = filters
        return {"results": [{"input": f, "result": {"results": []}} for f in filters]}

    monkeypatch.setattr(mts, "_query_emails_batch_impl", fake_batch)
    mts.register_session("sess-2", evidence_records=None, db_dsn=None)

    asyncio.run(
        mts._tool_query_emails_batch(
            "sess-2",
            filters=[{"sender": "@dartmouth.edu", "has_attachment": True, "limit": 50}],
        )
    )

    assert captured["filters"][0]["sender"] == "@dartmouth.edu"
    assert captured["filters"][0]["has_attachment"] is True
    assert captured["filters"][0]["limit"] == 50


# ── unknown session_id ─────────────────────────────────────────────


def test_unknown_session_id_raises():
    """A tool call with no prior register_session must blow up loud
    (not silently return empty / wrong evidence)."""
    with pytest.raises(RuntimeError, match="not registered"):
        asyncio.run(mts._tool_search_emails_batch("never-registered", searches=[{"query": "x"}]))


def test_empty_session_id_raises():
    """Defence-in-depth: even if upstream lets through `""`, we
    refuse to look up a blank id."""
    with pytest.raises(ValueError, match="non-empty"):
        mts.register_session("", evidence_records=None, db_dsn=None)


# ── unregister_session is idempotent ───────────────────────────────


def test_unregister_session_is_idempotent():
    """Calling unregister twice (or on an unknown id) must not raise
    — cleanup paths can't afford to be picky."""
    mts.register_session("sess-5", evidence_records=None, db_dsn=None)
    mts.unregister_session("sess-5")
    mts.unregister_session("sess-5")  # second call: no-op
    mts.unregister_session("never-existed")  # unknown: no-op


# ── build_app surfaces the five tools ──────────────────────────────


def test_build_app_registers_all_tools():
    """Smoke test that the FastMCP app exposes exactly the tool names
    the spec requires."""
    app = mts.build_app(host="127.0.0.1", port=0)
    tools = asyncio.run(app.list_tools())
    names = {t.name for t in tools}
    assert names == {
        "search_emails_batch",
        "query_emails_batch",
        "get_thread_batch",
        "sql_query_batch",
        "describe_schema",
        "get_attachment_batch",
        "publish_artifact_batch",
    }


# ── Side-channel: per-session structured tool-call log ─────────────


@pytest.fixture(autouse=True)
def _clear_call_log():
    """Same isolation contract as the session registry — clear the
    side-channel call log around every test."""
    mts._SESSION_CALLS.clear()
    yield
    mts._SESSION_CALLS.clear()


def test_search_emails_batch_records_full_structured_response(monkeypatch):
    """The side channel must store the COMPLETE response dict, not a
    stringified preview — that's the whole point of bypassing the
    claudebox 2000-char truncation."""
    big_payload = {
        "results": [
            {
                "input": {"query": "hi"},
                "result": {"results": [{"thread_id": "t1", "cite_ref": "abc12345", "preview": "x" * 5000}]},
            },
        ]
    }

    async def fake_batch(searches):
        return big_payload

    monkeypatch.setattr(mts, "_search_emails_batch_impl", fake_batch)
    mts.register_session("sess-call-log", evidence_records=None, db_dsn=None)

    asyncio.run(mts._tool_search_emails_batch("sess-call-log", searches=[{"query": "hi"}]))

    calls = mts.get_session_calls("sess-call-log")
    assert len(calls) == 1
    rec = calls[0]
    assert rec["name"] == "search_emails_batch"
    assert rec["args"]["searches"] == [{"query": "hi"}]
    # FULL response — not truncated, not stringified.
    assert rec["response"] == big_payload
    assert isinstance(rec["ts"], float)


def test_unregister_session_clears_call_log(monkeypatch):
    """unregister_session must wipe the side-channel for the session
    so we don't leak structured payloads across turns."""

    async def fake_batch(searches):
        return {"results": []}

    monkeypatch.setattr(mts, "_search_emails_batch_impl", fake_batch)
    mts.register_session("sess-clear", evidence_records=None, db_dsn=None)
    asyncio.run(mts._tool_search_emails_batch("sess-clear", searches=[{"query": "x"}]))
    assert len(mts.get_session_calls("sess-clear")) == 1

    mts.unregister_session("sess-clear")
    assert mts.get_session_calls("sess-clear") == []


def test_admin_get_calls_requires_bearer_token(monkeypatch):
    """The admin endpoint must reject requests with a missing or
    wrong bearer token. We exercise the route handler via Starlette's
    TestClient against the FastMCP-built ASGI app."""
    from starlette.testclient import TestClient

    monkeypatch.setenv("GMAIL_MCP_ADMIN_TOKEN", "secret-xyz")
    app = mts.build_app(host="127.0.0.1", port=0)
    asgi = app.streamable_http_app()
    client = TestClient(asgi)

    # Wrong token → 401
    r = client.get("/admin/calls/sess-1", headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401

    # Right token, but no calls → empty list
    r = client.get("/admin/calls/sess-1", headers={"Authorization": "Bearer secret-xyz"})
    assert r.status_code == 200
    assert r.json() == {"calls": []}


def test_admin_post_session_registers_via_http(monkeypatch):
    """POST /admin/sessions registers a session so the orchestrator
    (or a colocated process) can wire up state without needing
    in-process imports. The DSN is resolved server-side from
    `DB_DSN` — caller-supplied values are ignored (see security
    test below)."""
    from starlette.testclient import TestClient

    monkeypatch.setenv("GMAIL_MCP_ADMIN_TOKEN", "tok-abc")
    monkeypatch.setenv("DB_DSN", "postgres://server-side")
    app = mts.build_app(host="127.0.0.1", port=0)
    client = TestClient(app.streamable_http_app())

    r = client.post(
        "/admin/sessions",
        headers={"Authorization": "Bearer tok-abc"},
        json={"session_id": "sess-http", "evidence_records": [{"a": 1}]},
    )
    assert r.status_code == 200
    ctx = mts._SESSIONS["sess-http"]
    assert ctx.evidence_records == [{"a": 1}]
    # DSN comes from server-side config, NOT the request body.
    assert ctx.db_dsn == "postgres://server-side"

    # DELETE mirror
    r = client.delete("/admin/sessions/sess-http", headers={"Authorization": "Bearer tok-abc"})
    assert r.status_code == 200
    assert "sess-http" not in mts._SESSIONS


def test_admin_post_session_ignores_caller_supplied_db_dsn(monkeypatch):
    """SECURITY: a caller-supplied `db_dsn` in the POST body must be
    ignored. If the admin token leaks, accepting an attacker-chosen
    DSN would let them point our DB connections at a host they
    control (SSRF / outbound-DB / credential exfil). The server
    resolves DSN exclusively from its own config."""
    from starlette.testclient import TestClient

    monkeypatch.setenv("GMAIL_MCP_ADMIN_TOKEN", "tok-abc")
    monkeypatch.setenv("DB_DSN", "postgres://trusted-server-side")
    app = mts.build_app(host="127.0.0.1", port=0)
    client = TestClient(app.streamable_http_app())

    r = client.post(
        "/admin/sessions",
        headers={"Authorization": "Bearer tok-abc"},
        json={
            "session_id": "sess-evil",
            "evidence_records": None,
            "db_dsn": "postgres://attacker-controlled-host:5432/db",
        },
    )
    assert r.status_code == 200
    ctx = mts._SESSIONS["sess-evil"]
    # The attacker's DSN was discarded; the server-side one was used.
    assert ctx.db_dsn == "postgres://trusted-server-side"
    assert "attacker" not in (ctx.db_dsn or "")


def test_admin_post_session_no_dsn_when_unset(monkeypatch):
    """When no server-side DSN env is set, ctx.db_dsn is None — the
    session still registers successfully (artifact-free turns work
    fine without a DSN)."""
    from starlette.testclient import TestClient

    monkeypatch.setenv("GMAIL_MCP_ADMIN_TOKEN", "tok-abc")
    monkeypatch.delenv("DB_DSN", raising=False)
    monkeypatch.delenv("GMAIL_DB_DSN", raising=False)
    app = mts.build_app(host="127.0.0.1", port=0)
    client = TestClient(app.streamable_http_app())

    r = client.post(
        "/admin/sessions",
        headers={"Authorization": "Bearer tok-abc"},
        json={"session_id": "sess-nodsn", "evidence_records": None, "db_dsn": "postgres://attacker"},
    )
    assert r.status_code == 200
    assert mts._SESSIONS["sess-nodsn"].db_dsn is None


# ── Per-session call-log cap ───────────────────────────────────────


def test_record_call_caps_per_session_log(monkeypatch):
    """Once `_MAX_CALLS_PER_SESSION` is reached, further calls are
    dropped from the log (the tool still runs in the live path; this
    test just exercises the recorder directly). Protects against a
    pathological loop turning the side-channel into a memory leak."""
    # Shrink the cap so the test runs fast and is obviously correct.
    monkeypatch.setattr(mts, "_MAX_CALLS_PER_SESSION", 5)
    sid = "sess-cap"
    for i in range(20):
        mts._record_call(sid, "search_emails", {"q": str(i)}, {"results": []})
    calls = mts.get_session_calls(sid)
    # Exactly the cap — additional 15 calls dropped silently.
    assert len(calls) == 5
    # The five we kept are the FIRST five (we drop the overflow,
    # not evict the oldest — the early calls are usually the
    # important ones for the orchestrator's downstream walkers).
    assert [c["args"]["q"] for c in calls] == ["0", "1", "2", "3", "4"]


def test_record_call_warns_only_once_per_session(monkeypatch, caplog):
    """The cap-reached warning must fire exactly once per session,
    not on every dropped call (otherwise the log itself becomes the
    DoS vector)."""
    import logging as _logging

    monkeypatch.setattr(mts, "_MAX_CALLS_PER_SESSION", 2)
    mts._SESSION_CAP_WARNED.discard("sess-warn")
    with caplog.at_level(_logging.WARNING, logger=mts.__name__):
        for i in range(10):
            mts._record_call("sess-warn", "search_emails", {"q": str(i)}, {})
    cap_warnings = [r for r in caplog.records if "_MAX_CALLS_PER_SESSION" in r.getMessage()]
    assert len(cap_warnings) == 1


def test_clear_session_calls_resets_cap_warning_state():
    """unregister/clear must drop the per-session warned flag so a
    re-used session_id (test reuse, edge orchestrator paths) gets a
    fresh log without losing the cap-warning ability."""
    mts._SESSION_CAP_WARNED.add("sess-reset")
    mts._SESSION_CALLS["sess-reset"] = [{"x": 1}]
    mts.clear_session_calls("sess-reset")
    assert "sess-reset" not in mts._SESSION_CAP_WARNED
    assert mts.get_session_calls("sess-reset") == []


# ── conversation_id threads through to register_session ───────────


def test_register_session_records_conversation_id():
    """register_session must persist the conversation_id onto the
    SessionContext (the in-process path; the admin-HTTP path is
    covered separately)."""
    mts.register_session(
        "sess-conv-r",
        evidence_records=None,
        db_dsn=None,
        conversation_id="conv-foo",
    )
    assert mts._SESSIONS["sess-conv-r"].conversation_id == "conv-foo"


def test_admin_post_session_accepts_conversation_id(monkeypatch):
    """The admin POST endpoint must persist a caller-supplied
    `conversation_id` onto the SessionContext so the orchestrator can
    opt into persistent /work without an in-process import."""
    from starlette.testclient import TestClient

    monkeypatch.setenv("GMAIL_MCP_ADMIN_TOKEN", "tok-conv")
    monkeypatch.delenv("DB_DSN", raising=False)
    monkeypatch.delenv("GMAIL_DB_DSN", raising=False)
    app = mts.build_app(host="127.0.0.1", port=0)
    client = TestClient(app.streamable_http_app())

    r = client.post(
        "/admin/sessions",
        headers={"Authorization": "Bearer tok-conv"},
        json={"session_id": "sess-conv-admin", "conversation_id": "conv-zz"},
    )
    assert r.status_code == 200
    assert mts._SESSIONS["sess-conv-admin"].conversation_id == "conv-zz"


# ── publish_artifact ───────────────────────────────────────────────


def _setup_publish_session(tmp_path, monkeypatch, *, workspace="ws-pub", conversation_id=None):
    """Common harness: register a session, monkeypatch the workspace
    + scratch roots to point at tmp_path, return the workspace dir
    Path so the test can drop a file in it."""
    monkeypatch.setattr(mts, "_PUBLISH_WORKSPACE_ROOT", str(tmp_path / "workspaces"))
    monkeypatch.setattr(mts, "_PUBLISH_SCRATCH_ROOT", str(tmp_path / "scratch"))
    ws_dir = tmp_path / "workspaces" / workspace
    ws_dir.mkdir(parents=True)
    if conversation_id:
        (tmp_path / "scratch" / conversation_id).mkdir(parents=True)
    mts.register_session(
        "sess-pub",
        evidence_records=None,
        db_dsn=None,
        conversation_id=conversation_id,
        workspace=workspace,
    )
    return ws_dir


def test_publish_artifact_reads_workspace_file_and_returns_id(tmp_path, monkeypatch):
    ws_dir = _setup_publish_session(tmp_path, monkeypatch)
    (ws_dir / "plot.png").write_bytes(b"\x89PNG-fake")

    save_calls: list[dict] = []

    def fake_save_artifact(conn, *, session_id, name, mime_type, data, meta=None):
        save_calls.append({"name": name, "mime_type": mime_type, "data": data})
        return 777

    monkeypatch.setattr(mts, "save_artifact", fake_save_artifact)
    monkeypatch.setattr(mts.SessionContext, "get_db_conn", lambda self: object())

    out = asyncio.run(mts._tool_publish_artifact_batch("sess-pub", items=[{"path": "plot.png"}]))
    result = out["results"][0]["result"]
    assert result == {"id": 777, "name": "plot.png", "mime_type": "image/png", "size": 9}
    assert save_calls == [{"name": "plot.png", "mime_type": "image/png", "data": b"\x89PNG-fake"}]
    mts.unregister_session("sess-pub")


def test_publish_artifact_strips_in_container_absolute_prefix(tmp_path, monkeypatch):
    ws_dir = _setup_publish_session(tmp_path, monkeypatch, workspace="ws-abs")
    (ws_dir / "out.csv").write_bytes(b"x,y\n1,2\n")

    monkeypatch.setattr(mts, "save_artifact", lambda *a, **k: 42)
    monkeypatch.setattr(mts.SessionContext, "get_db_conn", lambda self: object())

    out = asyncio.run(mts._tool_publish_artifact_batch("sess-pub", items=[{"path": "/workspaces/ws-abs/out.csv"}]))
    result = out["results"][0]["result"]
    assert result["id"] == 42
    assert result["mime_type"] == "text/csv"
    mts.unregister_session("sess-pub")


def test_publish_artifact_rejects_path_traversal(tmp_path, monkeypatch):
    _setup_publish_session(tmp_path, monkeypatch)
    # Drop a sentinel above the workspace root that the model must NOT reach.
    (tmp_path / "secret.txt").write_bytes(b"don't read me")

    out = asyncio.run(mts._tool_publish_artifact_batch("sess-pub", items=[{"path": "../../secret.txt"}]))
    result = out["results"][0]["result"]
    assert "error" in result and "no such file" in result["error"]
    mts.unregister_session("sess-pub")


def test_publish_artifact_rejects_oversized_file(tmp_path, monkeypatch):
    ws_dir = _setup_publish_session(tmp_path, monkeypatch)
    big = ws_dir / "big.bin"
    monkeypatch.setattr(mts, "MAX_PUBLISH_BYTES", 100)
    big.write_bytes(b"x" * 200)

    out = asyncio.run(mts._tool_publish_artifact_batch("sess-pub", items=[{"path": "big.bin"}]))
    result = out["results"][0]["result"]
    assert "error" in result and "too large" in result["error"]
    mts.unregister_session("sess-pub")


def test_publish_artifact_missing_file_returns_clean_error(tmp_path, monkeypatch):
    _setup_publish_session(tmp_path, monkeypatch)
    out = asyncio.run(mts._tool_publish_artifact_batch("sess-pub", items=[{"path": "nope.png"}]))
    result = out["results"][0]["result"]
    assert "error" in result and "no such file" in result["error"]
    mts.unregister_session("sess-pub")


def test_publish_artifact_falls_back_to_scratch_root(tmp_path, monkeypatch):
    """If a session has BOTH workspace and conversation_id and the
    file isn't in the workspace, the resolver tries the scratch root."""
    ws_dir = _setup_publish_session(tmp_path, monkeypatch, conversation_id="conv-fb")
    scratch_dir = tmp_path / "scratch" / "conv-fb"
    (scratch_dir / "from_sandbox.txt").write_bytes(b"hi")

    monkeypatch.setattr(mts, "save_artifact", lambda *a, **k: 99)
    monkeypatch.setattr(mts.SessionContext, "get_db_conn", lambda self: object())

    out = asyncio.run(mts._tool_publish_artifact_batch("sess-pub", items=[{"path": "from_sandbox.txt"}]))
    result = out["results"][0]["result"]
    assert result["id"] == 99
    assert result["name"] == "from_sandbox.txt"
    mts.unregister_session("sess-pub")
