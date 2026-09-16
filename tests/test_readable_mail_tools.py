import copy

import pytest

from gmail_search.agents import tools


@pytest.fixture
def thread(monkeypatch):
    original = {
        "thread_id": "thread-one",
        "messages": [
            {"id": "m1", "body_text": "", "body_html": "<style>noise</style><table><tr><td>Flight</td><td>UA2356</td></tr><tr><td>Date</td><td>2026-09-10</td></tr></table>", "date": "2026-09-10", "attachments": [{"id": 5}]},
            {"id": "m2", "body_text": "Correction: departure is 10:16 AM.", "body_html": ""},
        ],
    }
    async def get(path, *, user_id=None):
        assert path == "/api/thread/thread-one"
        assert user_id == "tenant-a"
        return copy.deepcopy(original)
    monkeypatch.setattr(tools, "_get", get)
    return original


@pytest.mark.asyncio
async def test_default_thread_read_is_readable_and_citable(thread):
    result = await tools.get_thread("thread-one", user_id="tenant-a")
    message = result["messages"][0]
    assert "UA2356" in message["body_text"] and "2026-09-10" in message["body_text"]
    assert "body_html" not in message
    assert "noise" not in message["body_text"]
    assert message["body_format"] == "markdown"
    assert message["body_source"] == "html"
    assert message["cite_ref"] == result["cite_ref"] == "thread-one"
    assert message["attachments"] == [{"id": 5}]


@pytest.mark.asyncio
async def test_selected_messages_and_continuation_preserve_exact_content(thread):
    pages = []
    offset = 0
    # Body limit is intentionally small to exercise continuation in a short fixture.
    while True:
        result = await tools.get_thread_batch(["thread-one"], user_id="tenant-a", message_ids=["m2"], body_offset=offset, body_limit=12)
        result = result["results"][0]["result"]
        assert result["thread_message_count"] == 2
        assert [m["id"] for m in result["messages"]] == ["m2"]
        m = result["messages"][0]
        pages.append(m["body_text"])
        offset = m["body_next_offset"]
        if offset is None:
            break
    assert "".join(pages) == thread["messages"][1]["body_text"]


@pytest.mark.asyncio
async def test_raw_fallback_explicit_and_bounded(thread):
    result = await tools.get_thread("thread-one", user_id="tenant-a", body_format="raw", body_limit=20)
    m = result["messages"][0]
    assert m["body_html"] == thread["messages"][0]["body_html"][:20]
    assert m["body_html_truncated"] is True
    assert m["body_html_next_offset"] == 20
    assert m["body_format"] == "raw"


@pytest.mark.asyncio
@pytest.mark.parametrize("kwargs", [{"body_format": "summary"}, {"body_offset": -1}, {"body_limit": 0}, {"body_limit": 100001}, {"message_ids": []}])
async def test_invalid_read_options_rejected_before_fetch(monkeypatch, kwargs):
    async def unexpected(*args, **kw):
        pytest.fail("invalid options must not fetch")
    monkeypatch.setattr(tools, "_get", unexpected)
    result = await tools.get_thread_batch(["thread-one"], **kwargs)
    assert "error" in result
