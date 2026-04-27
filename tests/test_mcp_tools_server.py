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
from gmail_search.agents.sandbox import SandboxArtifact, SandboxResult


@pytest.fixture(autouse=True)
def _clear_session_registry():
    """Wipe the module-level session dict before AND after each test
    so tests can't leak context into each other."""
    mts._SESSIONS.clear()
    yield
    mts._SESSIONS.clear()


# ── register_session + tool happy path ─────────────────────────────


def test_search_emails_routes_to_underlying_impl(monkeypatch):
    """register_session + a search call should reach the existing
    async search_emails function with the args we passed."""
    captured: dict = {}

    async def fake_search_emails(**kwargs):
        captured.update(kwargs)
        return {"results": [{"thread_id": "abc12345xyz", "cite_ref": "abc12345"}]}

    monkeypatch.setattr(mts, "_search_emails_impl", fake_search_emails)

    mts.register_session("sess-1", evidence_records=None, db_dsn=None)
    out = asyncio.run(mts._tool_search_emails("sess-1", query="budget", date_from="2026-01-01", top_k=5))

    assert captured == {"query": "budget", "date_from": "2026-01-01", "date_to": "", "top_k": 5}
    assert out["results"][0]["cite_ref"] == "abc12345"


def test_query_emails_routes_to_underlying_impl(monkeypatch):
    """The structured-filter tool must forward its kwargs verbatim."""
    captured: dict = {}

    async def fake_query_emails(**kwargs):
        captured.update(kwargs)
        return {"results": []}

    monkeypatch.setattr(mts, "_query_emails_impl", fake_query_emails)
    mts.register_session("sess-2", evidence_records=None, db_dsn=None)

    asyncio.run(
        mts._tool_query_emails(
            "sess-2",
            sender="@dartmouth.edu",
            has_attachment=True,
            limit=50,
        )
    )

    assert captured["sender"] == "@dartmouth.edu"
    assert captured["has_attachment"] is True
    assert captured["limit"] == 50


# ── unknown session_id ─────────────────────────────────────────────


def test_unknown_session_id_raises():
    """A tool call with no prior register_session must blow up loud
    (not silently return empty / wrong evidence)."""
    with pytest.raises(RuntimeError, match="not registered"):
        asyncio.run(mts._tool_search_emails("never-registered", query="x"))


def test_empty_session_id_raises():
    """Defence-in-depth: even if upstream lets through `""`, we
    refuse to look up a blank id."""
    with pytest.raises(ValueError, match="non-empty"):
        mts.register_session("", evidence_records=None, db_dsn=None)


# ── run_code: artifacts persisted via save_artifact ────────────────


def test_run_code_persists_artifacts_and_returns_shape(monkeypatch):
    """The run_code tool must (a) call execute_in_sandbox with the
    session's evidence + DSN, (b) call save_artifact for each
    artifact the sandbox produced, and (c) return the exact dict
    shape the orchestrator's `_artifact_ids_from_tool_calls` walker
    expects."""
    fake_result = SandboxResult(
        exit_code=0,
        stdout="hello\n",
        stderr="",
        artifacts=[
            SandboxArtifact(name="plot.png", mime_type="image/png", data=b"\x89PNG..."),
            SandboxArtifact(name="data.csv", mime_type="text/csv", data=b"a,b\n1,2\n"),
        ],
        wall_ms=42,
        timed_out=False,
        oom_killed=False,
    )

    sandbox_calls = []

    def fake_execute(req):
        sandbox_calls.append(req)
        return fake_result

    save_calls = []
    next_id = iter([101, 102])

    def fake_save_artifact(conn, *, session_id, name, mime_type, data, meta=None):
        save_calls.append({"conn": conn, "session_id": session_id, "name": name, "mime_type": mime_type, "data": data})
        return next(next_id)

    # Replace BOTH the import-bound name (used by _persist_sandbox_artifacts)
    # and the original module name. The module imports save_artifact at
    # module load, so patching the local binding is the one that matters.
    monkeypatch.setattr(mts, "execute_in_sandbox", fake_execute)
    monkeypatch.setattr(mts, "save_artifact", fake_save_artifact)

    sentinel_conn = object()

    class _Ctx:
        evidence_records = [{"x": 1}]
        db_dsn = "postgresql://fake"
        conversation_id = None

        def get_db_conn(self):
            return sentinel_conn

        def close(self):
            pass

    # Inject our fake context directly so we don't need a live DB.
    mts._SESSIONS["sess-3"] = _Ctx()

    out = mts._tool_run_code("sess-3", code="print('hi')")

    # Sandbox got the session's evidence + dsn
    assert len(sandbox_calls) == 1
    assert sandbox_calls[0].evidence == [{"x": 1}]
    assert sandbox_calls[0].db_dsn == "postgresql://fake"
    assert sandbox_calls[0].code == "print('hi')"

    # save_artifact was called once per artifact, with the right kwargs
    assert len(save_calls) == 2
    assert save_calls[0]["conn"] is sentinel_conn
    assert save_calls[0]["session_id"] == "sess-3"
    assert save_calls[0]["name"] == "plot.png"
    assert save_calls[0]["mime_type"] == "image/png"
    assert save_calls[1]["name"] == "data.csv"

    # Return shape matches the ADK contract exactly
    assert out["exit_code"] == 0
    assert out["stdout"] == "hello\n"
    assert out["stderr"] == ""
    assert out["wall_ms"] == 42
    assert out["timed_out"] is False
    assert out["oom_killed"] is False
    assert out["artifacts"] == [
        {"id": 101, "name": "plot.png", "mime_type": "image/png"},
        {"id": 102, "name": "data.csv", "mime_type": "text/csv"},
    ]


def test_run_code_truncates_long_stdout(monkeypatch):
    """stdout > 8000 chars must come back with the truncation marker
    so the model knows content was clipped."""
    big_stdout = "x" * 20_000
    fake_result = SandboxResult(
        exit_code=0,
        stdout=big_stdout,
        stderr="",
        artifacts=[],
        wall_ms=1,
    )
    monkeypatch.setattr(mts, "execute_in_sandbox", lambda req: fake_result)
    # No artifacts → save_artifact + db_conn not touched, no need to mock.

    class _Ctx:
        evidence_records = None
        db_dsn = None
        conversation_id = None

        def get_db_conn(self):
            return None

        def close(self):
            pass

    mts._SESSIONS["sess-4"] = _Ctx()

    out = mts._tool_run_code("sess-4", code="print('x' * 20000)")
    assert "truncated" in out["stdout"]
    assert len(out["stdout"]) < 20_000


# ── unregister_session is idempotent ───────────────────────────────


def test_unregister_session_is_idempotent():
    """Calling unregister twice (or on an unknown id) must not raise
    — cleanup paths can't afford to be picky."""
    mts.register_session("sess-5", evidence_records=None, db_dsn=None)
    mts.unregister_session("sess-5")
    mts.unregister_session("sess-5")  # second call: no-op
    mts.unregister_session("never-existed")  # unknown: no-op


# ── build_app surfaces the five tools ──────────────────────────────


def test_build_app_registers_all_seven_tools():
    """Smoke test that the FastMCP app exposes exactly the seven tool
    names the spec requires."""
    app = mts.build_app(host="127.0.0.1", port=0)
    tools = asyncio.run(app.list_tools())
    names = {t.name for t in tools}
    assert names == {
        "search_emails",
        "query_emails",
        "get_thread",
        "sql_query",
        "run_code",
        "get_attachment",
        "publish_artifact",
    }


# ── Side-channel: per-session structured tool-call log ─────────────


@pytest.fixture(autouse=True)
def _clear_call_log():
    """Same isolation contract as the session registry — clear the
    side-channel call log around every test."""
    mts._SESSION_CALLS.clear()
    yield
    mts._SESSION_CALLS.clear()


def test_search_emails_records_full_structured_response(monkeypatch):
    """The side channel must store the COMPLETE response dict, not a
    stringified preview — that's the whole point of bypassing the
    claudebox 2000-char truncation."""
    big_payload = {"results": [{"thread_id": "t1", "cite_ref": "abc12345", "preview": "x" * 5000}]}

    async def fake_search_emails(**kwargs):
        return big_payload

    monkeypatch.setattr(mts, "_search_emails_impl", fake_search_emails)
    mts.register_session("sess-call-log", evidence_records=None, db_dsn=None)

    asyncio.run(mts._tool_search_emails("sess-call-log", query="hi"))

    calls = mts.get_session_calls("sess-call-log")
    assert len(calls) == 1
    rec = calls[0]
    assert rec["name"] == "search_emails"
    assert rec["args"]["query"] == "hi"
    # FULL response — not truncated, not stringified.
    assert rec["response"] == big_payload
    assert isinstance(rec["ts"], float)


def test_run_code_records_artifacts_in_call_log(monkeypatch):
    """run_code's structured response (artifacts list with ids) is
    exactly what the orchestrator's `_artifact_ids_from_tool_calls`
    walker needs — the side channel must preserve it intact."""
    from gmail_search.agents.sandbox import SandboxArtifact, SandboxResult

    fake_result = SandboxResult(
        exit_code=0,
        stdout="ok",
        stderr="",
        artifacts=[SandboxArtifact(name="plot.png", mime_type="image/png", data=b"PNG")],
        wall_ms=10,
    )
    monkeypatch.setattr(mts, "execute_in_sandbox", lambda req: fake_result)
    monkeypatch.setattr(mts, "save_artifact", lambda *a, **kw: 999)

    class _Ctx:
        evidence_records = None
        db_dsn = None
        conversation_id = None

        def get_db_conn(self):
            return object()

        def close(self):
            pass

    mts._SESSIONS["sess-rc"] = _Ctx()
    mts._tool_run_code("sess-rc", code="print('x')")

    calls = mts.get_session_calls("sess-rc")
    assert len(calls) == 1
    assert calls[0]["name"] == "run_code"
    assert calls[0]["response"]["artifacts"] == [{"id": 999, "name": "plot.png", "mime_type": "image/png"}]


def test_unregister_session_clears_call_log(monkeypatch):
    """unregister_session must wipe the side-channel for the session
    so we don't leak structured payloads across turns."""

    async def fake_search_emails(**kw):
        return {"results": []}

    monkeypatch.setattr(mts, "_search_emails_impl", fake_search_emails)
    mts.register_session("sess-clear", evidence_records=None, db_dsn=None)
    asyncio.run(mts._tool_search_emails("sess-clear", query="x"))
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


# ── conversation_id threads through to SandboxRequest ──────────────


def test_run_code_passes_conversation_id_to_sandbox(monkeypatch):
    """A session registered with a conversation_id must thread that id
    into every SandboxRequest the run_code tool builds — that's how the
    sandbox knows to use the per-conversation persistent /work mount."""
    captured: list = []

    def fake_execute(req):
        captured.append(req)
        return SandboxResult(exit_code=0, stdout="", stderr="", artifacts=[], wall_ms=1)

    monkeypatch.setattr(mts, "execute_in_sandbox", fake_execute)

    class _Ctx:
        evidence_records = None
        db_dsn = None
        conversation_id = "conv-xyz-1"

        def get_db_conn(self):
            return None

        def close(self):
            pass

    mts._SESSIONS["sess-conv"] = _Ctx()

    mts._tool_run_code("sess-conv", code="print('hi')")

    assert len(captured) == 1
    assert captured[0].conversation_id == "conv-xyz-1"


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

    result = asyncio.run(mts._tool_publish_artifact("sess-pub", "plot.png"))
    assert result == {"id": 777, "name": "plot.png", "mime_type": "image/png", "size": 9}
    assert save_calls == [{"name": "plot.png", "mime_type": "image/png", "data": b"\x89PNG-fake"}]
    mts.unregister_session("sess-pub")


def test_publish_artifact_strips_in_container_absolute_prefix(tmp_path, monkeypatch):
    ws_dir = _setup_publish_session(tmp_path, monkeypatch, workspace="ws-abs")
    (ws_dir / "out.csv").write_bytes(b"x,y\n1,2\n")

    monkeypatch.setattr(mts, "save_artifact", lambda *a, **k: 42)
    monkeypatch.setattr(mts.SessionContext, "get_db_conn", lambda self: object())

    result = asyncio.run(mts._tool_publish_artifact("sess-pub", "/workspaces/ws-abs/out.csv"))
    assert result["id"] == 42
    assert result["mime_type"] == "text/csv"
    mts.unregister_session("sess-pub")


def test_publish_artifact_rejects_path_traversal(tmp_path, monkeypatch):
    _setup_publish_session(tmp_path, monkeypatch)
    # Drop a sentinel above the workspace root that the model must NOT reach.
    (tmp_path / "secret.txt").write_bytes(b"don't read me")

    result = asyncio.run(mts._tool_publish_artifact("sess-pub", "../../secret.txt"))
    assert "error" in result and "no such file" in result["error"]
    mts.unregister_session("sess-pub")


def test_publish_artifact_rejects_oversized_file(tmp_path, monkeypatch):
    ws_dir = _setup_publish_session(tmp_path, monkeypatch)
    big = ws_dir / "big.bin"
    monkeypatch.setattr(mts, "MAX_PUBLISH_BYTES", 100)
    big.write_bytes(b"x" * 200)

    result = asyncio.run(mts._tool_publish_artifact("sess-pub", "big.bin"))
    assert "error" in result and "too large" in result["error"]
    mts.unregister_session("sess-pub")


def test_publish_artifact_missing_file_returns_clean_error(tmp_path, monkeypatch):
    _setup_publish_session(tmp_path, monkeypatch)
    result = asyncio.run(mts._tool_publish_artifact("sess-pub", "nope.png"))
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

    result = asyncio.run(mts._tool_publish_artifact("sess-pub", "from_sandbox.txt"))
    assert result["id"] == 99
    assert result["name"] == "from_sandbox.txt"
    mts.unregister_session("sess-pub")
