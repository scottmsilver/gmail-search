"""Tests for the deep-analysis agent's retrieval tools.

These tools wrap our existing HTTP endpoints (/api/search, /api/query,
/api/thread/<id>, /api/sql). We test the WRAPPER behavior — clip
logic, cite_ref backfill, build_retrieval_tools assembly — by
stubbing httpx.AsyncClient. No live server needed.

Tools are async because the retriever runs inside the same FastAPI
event loop that serves the retrieval endpoints; a sync httpx.Client
would deadlock (tool waits on the socket, uvicorn can't accept the
new request because the loop is blocked).
"""

from __future__ import annotations

import pytest

try:
    import google.adk  # noqa: F401

    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False


def _stub_httpx_async(monkeypatch, response_json: dict):
    """Patch httpx.AsyncClient so every .get()/.post() returns a
    synthetic response. Covers the _get / _post helpers without
    touching a real network."""
    import httpx

    from gmail_search.agents import tools

    class _R:
        status_code = 200
        text = ""

        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class _C:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

        async def get(self, url, params=None, headers=None):  # noqa: ARG002
            return _R(response_json)

        async def post(self, url, json=None, headers=None):  # noqa: ARG002
            return _R(response_json)

    monkeypatch.setattr(tools.httpx, "AsyncClient", _C)
    monkeypatch.setattr(httpx, "AsyncClient", _C)


@pytest.mark.asyncio
async def test_search_emails_backfills_cite_ref(monkeypatch):
    from gmail_search.agents.tools import search_emails

    _stub_httpx_async(
        monkeypatch,
        {
            "results": [
                {"thread_id": "abcdef0123456789", "subject": "s", "score": 0.9},
                {"thread_id": "1111111122222222", "subject": "t", "score": 0.8, "cite_ref": "preset"},
            ]
        },
    )
    data = await search_emails("how much did we spend?")
    assert data["results"][0]["cite_ref"] == "abcdef0123456789"
    assert data["results"][1]["cite_ref"] == "preset"


@pytest.mark.asyncio
async def test_query_emails_backfills_cite_ref(monkeypatch):
    from gmail_search.agents.tools import query_emails

    _stub_httpx_async(
        monkeypatch,
        {"results": [{"thread_id": "aaabbbcccdddeeee", "subject": "x"}]},
    )
    data = await query_emails(sender="alice@example.com")
    assert data["results"][0]["cite_ref"] == "aaabbbcccdddeeee"


@pytest.mark.asyncio
async def test_get_thread_clips_long_bodies(monkeypatch):
    """Bodies longer than 20k chars should come back clipped with
    `body_text_truncated=True` + `original_chars` set. The chat-mode
    TS tool does the same — keep the wrapper contracts aligned."""
    from gmail_search.agents.tools import THREAD_BODY_CHAR_CAP, get_thread

    long_body = "x" * (THREAD_BODY_CHAR_CAP + 5000)
    _stub_httpx_async(
        monkeypatch,
        {
            "thread_id": "t1",
            "messages": [
                {"id": "m1", "body_text": long_body, "subject": "s"},
                {"id": "m2", "body_text": "short", "subject": "s2"},
            ],
        },
    )
    data = await get_thread("t1")
    first = data["messages"][0]
    assert first.get("body_text_truncated") is True
    assert first["original_chars"] == len(long_body)
    assert len(first["body_text"]) <= THREAD_BODY_CHAR_CAP + 40

    second = data["messages"][1]
    assert "body_text_truncated" not in second
    assert "original_chars" not in second


@pytest.mark.asyncio
async def test_sql_query_clips_oversized_cells(monkeypatch):
    """Long string cells get clipped to 8000 chars so a 500-row
    SELECT body_text can't ship 10MB back to the model."""
    from gmail_search.agents.tools import SQL_CELL_CHAR_CAP, sql_query

    long_cell = "z" * (SQL_CELL_CHAR_CAP + 4000)
    _stub_httpx_async(
        monkeypatch,
        {
            "columns": ["id", "body"],
            "rows": [["m1", long_cell], ["m2", "short"]],
            "row_count": 2,
            "truncated": False,
        },
    )
    data = await sql_query("SELECT id, body FROM messages LIMIT 2")
    assert "truncated: original" in data["rows"][0][1]
    assert len(data["rows"][0][1]) <= SQL_CELL_CHAR_CAP + 80
    assert data["rows"][1][1] == "short"


def _schema_carries_additional_properties(schema) -> bool:
    """True if an ADK proto Schema (or any nested items/properties/anyOf
    branch) still sets `additional_properties`. Gemini's non-Vertex
    function-calling API rejects that field anywhere in a tool's
    parameter schema."""
    if schema is None:
        return False
    if getattr(schema, "additional_properties", None) is not None:
        return True
    if getattr(schema, "items", None) is not None and _schema_carries_additional_properties(schema.items):
        return True
    if getattr(schema, "properties", None):
        if any(_schema_carries_additional_properties(s) for s in schema.properties.values()):
            return True
    if getattr(schema, "defs", None):
        if any(_schema_carries_additional_properties(s) for s in schema.defs.values()):
            return True
    if getattr(schema, "any_of", None):
        if any(_schema_carries_additional_properties(s) for s in schema.any_of):
            return True
    return False


@pytest.mark.skipif(not ADK_AVAILABLE, reason="google-adk not installed")
def test_retrieval_tool_declarations_have_no_additional_properties():
    """The *_batch tools take `list[dict]`, which ADK renders as an
    array whose items carry `additional_properties`. Gemini's
    (non-Vertex) function-calling API rejects that field — nested, so
    genai's own top-level guard misses it — and deep mode dies with a
    400 INVALID_ARGUMENT. Every generated declaration must be clean."""
    from gmail_search.agents.tools import build_retrieval_tools

    offenders = [
        tool.name
        for tool in build_retrieval_tools()
        if _schema_carries_additional_properties(tool._get_declaration().parameters)
    ]
    assert offenders == [], f"declarations still carry additional_properties: {offenders}"


def _capture_request_headers(monkeypatch):
    """Patch httpx.AsyncClient so each GET/POST records the headers it
    was called with. Returns a dict that fills in `headers` after a
    call — lets a test assert the auth headers a tool attaches."""
    import httpx

    from gmail_search.agents import tools

    captured: dict = {}

    class _R:
        status_code = 200
        text = ""

        def json(self):
            return {"results": []}

    class _C:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

        async def get(self, url, params=None, headers=None):  # noqa: ARG002
            captured["headers"] = headers or {}
            return _R()

        async def post(self, url, json=None, headers=None):  # noqa: ARG002
            captured["headers"] = headers or {}
            return _R()

    monkeypatch.setattr(tools.httpx, "AsyncClient", _C)
    monkeypatch.setattr(httpx, "AsyncClient", _C)
    return captured


def _tool_named(tools_list, name):
    return next(t for t in tools_list if t.name == name)


@pytest.mark.skipif(not ADK_AVAILABLE, reason="google-adk not installed")
@pytest.mark.asyncio
async def test_build_retrieval_tools_injects_authenticated_user_id(monkeypatch):
    """Deep mode binds the request's authenticated user_id into every
    tool so the internal /api/* calls carry the service-token +
    X-User-Id pair `require_user_id` needs. Without this the calls hit
    the cookie-less path and 401 ('not signed in'), which is exactly
    what killed deep-mode retrieval."""
    from gmail_search.agents.tools import build_retrieval_tools

    monkeypatch.setenv("GMAIL_MCP_ADMIN_TOKEN", "svc-secret")
    captured = _capture_request_headers(monkeypatch)

    tools = build_retrieval_tools(user_id="u_test123")
    await _tool_named(tools, "search_emails").func(query="staircase")

    assert captured["headers"].get("X-User-Id") == "u_test123"
    assert captured["headers"].get("Authorization") == "Bearer svc-secret"


@pytest.mark.skipif(not ADK_AVAILABLE, reason="google-adk not installed")
def test_build_retrieval_tools_hides_user_id_from_schema():
    """user_id must NOT be a model-visible parameter. With a valid
    service token, require_user_id trusts X-User-Id verbatim, so a
    model-supplied user_id would be a cross-tenant escape vector. The
    authenticated id is injected server-side; the model never sees it."""
    from gmail_search.agents.tools import build_retrieval_tools

    for tool in build_retrieval_tools(user_id="u_test123"):
        params = tool._get_declaration().parameters
        props = getattr(params, "properties", None) or {}
        assert "user_id" not in props, f"{tool.name} exposes user_id to the model"


@pytest.mark.skipif(not ADK_AVAILABLE, reason="google-adk not installed")
def test_build_retrieval_tools_assembles_expected_set():
    """All retrieval tools must always be present — the Retriever
    agent relies on this exact set. A missing tool silently degrades
    retrieval quality."""
    from gmail_search.agents.tools import build_retrieval_tools

    tools = build_retrieval_tools()
    names = sorted(t.name for t in tools)
    assert names == [
        "describe_schema",
        "get_attachment",
        "get_attachment_batch",
        "get_thread",
        "get_thread_batch",
        "query_emails",
        "query_emails_batch",
        "search_emails",
        "search_emails_batch",
        "sql_query",
        "sql_query_batch",
    ]
