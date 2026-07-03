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


def _capture_get_request(monkeypatch, response_json: dict):
    """Patch httpx.AsyncClient so each GET records the url + params it
    was called with, and returns `response_json`. Lets a test assert
    the path/params a tool constructs without a live server."""
    import httpx

    from gmail_search.agents import tools

    captured: dict = {}

    class _R:
        status_code = 200
        text = ""

        def json(self):
            return response_json

    class _C:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

        async def get(self, url, params=None, headers=None):  # noqa: ARG002
            captured["url"] = url
            captured["params"] = params or {}
            return _R()

    monkeypatch.setattr(tools.httpx, "AsyncClient", _C)
    monkeypatch.setattr(httpx, "AsyncClient", _C)
    return captured


@pytest.mark.asyncio
async def test_search_emails_batch_isolates_failures(monkeypatch):
    """A single failing/slow search must NOT nuke the whole batch — it lands
    as a per-item {error} while siblings still return. This is the regression
    that caused multi-item search_emails_batch to error while singles worked
    (one httpx ReadTimeout propagated through a bare asyncio.gather)."""
    from gmail_search.agents import tools

    async def fake_search(**kwargs):
        if kwargs.get("query") == "boom":
            raise RuntimeError("simulated timeout")
        return {"results": [{"thread_id": "t1"}]}

    monkeypatch.setattr(tools, "search_emails", fake_search)

    out = await tools.search_emails_batch([{"query": "ok"}, {"query": "boom"}, {"query": "ok2"}])
    assert len(out["results"]) == 3
    assert out["results"][0]["result"] == {"results": [{"thread_id": "t1"}]}
    assert "RuntimeError" in out["results"][1]["result"]["error"]
    assert out["results"][2]["result"] == {"results": [{"thread_id": "t1"}]}

    # A malformed (non-dict) item is isolated too, not a crash.
    out = await tools.search_emails_batch([{"query": "ok"}, "notadict"])
    assert "error" in out["results"][1]["result"]


@pytest.mark.asyncio
async def test_find_facts_constructs_url_and_params(monkeypatch):
    """find_facts must GET /api/find_facts with q + cap (k) + the
    boolean flags coerced to lowercase strings the FastAPI bool Query
    parses, and pass the response through unchanged."""
    from gmail_search.agents.tools import find_facts

    payload = {"facts": [{"fact": "ABC123 is a plate", "message_id": "m1", "thread_id": "t1"}]}
    captured = _capture_get_request(monkeypatch, payload)

    data = await find_facts("all my license plates", exhaustive=True, k=50)

    assert captured["url"].endswith("/api/find_facts")
    assert captured["params"]["q"] == "all my license plates"
    assert captured["params"]["k"] == 50
    assert captured["params"]["exhaustive"] == "true"
    assert captured["params"]["hybrid"] == "true"
    assert data == payload


@pytest.mark.asyncio
async def test_search_emails_detail_param(monkeypatch):
    """search_emails forwards `detail` as the match_detail query param,
    defaulting to the compact 'snippet' level so agents don't pay for
    per-message summaries/bodies they didn't ask for."""
    from gmail_search.agents.tools import search_emails

    captured = _capture_get_request(monkeypatch, {"results": []})
    await search_emails("flights")
    assert captured["url"].endswith("/api/search")
    assert captured["params"]["match_detail"] == "snippet"

    captured = _capture_get_request(monkeypatch, {"results": []})
    await search_emails("flights", detail="full")
    assert captured["params"]["match_detail"] == "full"


@pytest.mark.asyncio
async def test_search_emails_forwards_refs_detail(monkeypatch):
    """detail='refs' passes straight through as match_detail — the
    one-line-per-thread level for fan-out inventory questions."""
    from gmail_search.agents.tools import search_emails

    captured = _capture_get_request(monkeypatch, {"results": []})
    await search_emails("pledges", detail="refs")
    assert captured["params"]["match_detail"] == "refs"


@pytest.mark.asyncio
async def test_search_emails_compact_defaults(monkeypatch):
    """The agent path always opts out of facets (agents never read
    them) and caps matches per thread at 3 by default — the uncapped
    matches array is what bloats snippet-level payloads."""
    from gmail_search.agents.tools import search_emails

    captured = _capture_get_request(monkeypatch, {"results": []})
    await search_emails("pledges")
    assert captured["params"]["include_facets"] == "false"
    assert captured["params"]["max_matches"] == 3


@pytest.mark.asyncio
async def test_search_emails_max_matches_override(monkeypatch):
    """Callers can raise the per-thread match cap (or lift it with 0)."""
    from gmail_search.agents.tools import search_emails

    captured = _capture_get_request(monkeypatch, {"results": []})
    await search_emails("pledges", max_matches=25)
    assert captured["params"]["max_matches"] == 25

    captured = _capture_get_request(monkeypatch, {"results": []})
    await search_emails("pledges", max_matches=0)
    assert captured["params"]["max_matches"] == 0


@pytest.mark.asyncio
async def test_find_facts_exhaustive_false_lowercased(monkeypatch):
    """exhaustive=False must serialize as the literal 'false' string."""
    from gmail_search.agents.tools import find_facts

    captured = _capture_get_request(monkeypatch, {"facts": []})
    await find_facts("vins", exhaustive=False)
    assert captured["params"]["exhaustive"] == "false"
    assert captured["params"]["k"] == 200  # default cap


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


@pytest.mark.asyncio
async def test_get_attachment_raw_mode_by_reference(monkeypatch):
    """raw mode hits /raw by-reference; inline flag is threaded; bad mode rejected."""
    import httpx

    from gmail_search.agents import tools

    captured = {}

    class _R:
        status_code = 200
        text = ""

        def raise_for_status(self):
            return None

        def json(self):
            return {"attachment_id": 7, "fetch_url": "/api/attachment/7", "base64": None}

    class _C:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

        async def get(self, url, params=None, headers=None):  # noqa: ARG002
            captured["url"] = url
            captured["params"] = params
            return _R()

        async def post(self, url, json=None, headers=None):  # noqa: ARG002
            return _R()

    monkeypatch.setattr(tools.httpx, "AsyncClient", _C)
    monkeypatch.setattr(httpx, "AsyncClient", _C)

    await tools.get_attachment(7, mode="raw", user_id="u1")
    assert captured["url"].endswith("/api/attachment/7/raw")
    assert captured["params"]["inline"] == "true"  # bytes inlined by default (only usable delivery)

    await tools.get_attachment(7, mode="raw", inline=False, user_id="u1")
    assert captured["params"]["inline"] == "false"  # explicit reference-only

    bad = await tools.get_attachment(7, mode="bogus", user_id="u1")
    assert "error" in bad
