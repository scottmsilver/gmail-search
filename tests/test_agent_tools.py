"""Tests for the deep-analysis agent's retrieval tools.

These tools wrap our existing HTTP endpoints (/api/search, /api/query,
/api/thread/<id>, /api/sql). We test the WRAPPER behavior — clip
logic, cite_ref backfill, batching — by
stubbing httpx.AsyncClient. No live server needed.

Tools are async because the retriever runs inside the same FastAPI
event loop that serves the retrieval endpoints; a sync httpx.Client
would deadlock (tool waits on the socket, uvicorn can't accept the
new request because the loop is blocked).
"""

from __future__ import annotations

import pytest


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
async def test_sql_query_rejects_stale_calls(monkeypatch):
    from gmail_search.agents.tools import sql_query

    data = await sql_query("SELECT id, body FROM messages LIMIT 2")
    assert data["code"] == "raw_sql_disabled"
    assert data["status"] == 403
    assert data["rows"] == []



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
