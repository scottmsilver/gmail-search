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

        async def get(self, url, params=None):  # noqa: ARG002
            return _R(response_json)

        async def post(self, url, json=None):  # noqa: ARG002
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
