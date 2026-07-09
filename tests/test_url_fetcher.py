"""Per-page fetch behaviour of the URL crawler's HTTP-first path.

The expensive failure mode this file pins down: falling back to the
Chromium browser when the HTTP outcome was already definitive. A 404,
a 5xx, a non-HTML body, or a network timeout looks exactly the same
through a browser — re-fetching it through Chromium just re-pays a
~10s page load for a guaranteed-identical result. The browser fallback
is reserved for the two cases where it can actually win: a thin 200
(JS shell that needs rendering) and a 403 (anti-bot walls sometimes
pass a real browser fingerprint).

Uses httpx.MockTransport so no test touches the network.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from gmail_search.gmail import url_fetcher as uf

RICH_HTML = (
    "<html><head><title>Recipe</title></head><body><article>"
    + "Slow-roasted tomato soup with basil and garlic croutons. " * 20
    + "</article></body></html>"
)
THIN_HTML = "<html><head><title>App</title></head><body><div id='root'></div></body></html>"


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False)


def _fetch(handler, url="https://pub.example.com/page", monkeypatch=None):
    """Run _fetch_via_http against a MockTransport with the SSRF guard
    stubbed open (mock hosts don't resolve)."""

    async def go():
        async with _client(handler) as client:
            return await uf._fetch_via_http(url, 5.0, client=client)

    return asyncio.run(go())


@pytest.fixture(autouse=True)
def _open_ssrf_guard(monkeypatch):
    monkeypatch.setattr(uf, "_ssrf_guard", lambda url: "blocked.internal" not in url)


# ─── outcome classification ────────────────────────────────────────────


def test_rich_html_page_is_ok():
    kind, payload = _fetch(lambda req: httpx.Response(200, html=RICH_HTML))
    assert kind == "ok"
    title, text = payload
    assert title == "Recipe"
    assert "tomato soup" in text


def test_thin_200_goes_to_browser():
    # browser payload carries the guarded terminal URL (here, the original).
    kind, payload = _fetch(lambda req: httpx.Response(200, html=THIN_HTML))
    assert kind == "browser"
    assert payload == "https://pub.example.com/page"


def test_403_goes_to_browser():
    kind, payload = _fetch(lambda req: httpx.Response(403, text="forbidden"))
    assert kind == "browser"
    assert payload == "https://pub.example.com/page"


def test_browser_payload_is_guarded_terminal_url():
    # After a redirect chain, the browser must start at the FINAL guarded
    # URL, not the original — else Chromium replays the chain unguarded.
    def handler(req):
        if req.url.host == "pub.example.com":
            return httpx.Response(302, headers={"location": "https://final.example.com/app"})
        return httpx.Response(200, html=THIN_HTML)  # thin → browser

    kind, payload = _fetch(handler)
    assert kind == "browser"
    assert payload == "https://final.example.com/app"


@pytest.mark.parametrize("status", [401, 404, 410, 429, 500, 503])
def test_definitive_statuses_fail_without_browser(status):
    kind, payload = _fetch(lambda req: httpx.Response(status, text="nope"))
    assert (kind, payload) == ("fail", None)


def test_non_html_200_fails_without_browser():
    handler = lambda req: httpx.Response(
        200, content=b"%PDF-1.4", headers={"content-type": "application/pdf"}
    )  # noqa: E731
    assert _fetch(handler) == ("fail", None)


def test_network_timeout_fails_without_browser():
    def handler(req):
        raise httpx.ReadTimeout("tarpit")

    assert _fetch(handler) == ("fail", None)


def test_redirect_without_location_fails_without_browser():
    # A malformed 3xx must NOT reach Chromium: the browser would follow
    # whatever the body navigates to with no per-hop SSRF guard.
    assert _fetch(lambda req: httpx.Response(302)) == ("fail", None)


# ─── redirects + SSRF ──────────────────────────────────────────────────


def test_redirect_is_followed_to_ok():
    def handler(req):
        if req.url.host == "pub.example.com":
            return httpx.Response(302, headers={"location": "https://dest.example.com/article"})
        return httpx.Response(200, html=RICH_HTML)

    assert _fetch(handler)[0] == "ok"


def test_redirect_loop_fails_without_browser():
    handler = lambda req: httpx.Response(302, headers={"location": str(req.url)})  # noqa: E731
    assert _fetch(handler) == ("fail", None)


def test_redirect_to_blocked_host_raises():
    def handler(req):
        return httpx.Response(302, headers={"location": "https://blocked.internal/admin"})

    with pytest.raises(uf._SSRFBlocked):
        _fetch(handler)


# ─── streaming byte cap ────────────────────────────────────────────────


def test_body_read_is_capped():
    # A "page" far past the cap: the fetch must succeed using only the
    # first _MAX_FETCH_BYTES instead of buffering the whole body.
    huge = RICH_HTML + ("x" * (uf._MAX_FETCH_BYTES * 3))
    kind, payload = _fetch(lambda req: httpx.Response(200, html=huge))
    assert kind == "ok"
    assert "tomato soup" in payload[1]


# ─── browser fallback wiring ───────────────────────────────────────────


class _CountingCrawler:
    calls = 0


def _run_markdown(monkeypatch, outcome):
    async def fake_http(url, timeout_s, client=None):
        return outcome

    async def fake_browser(crawler, url, timeout_s):
        crawler.calls += 1
        return ("from-browser", "rendered text")

    monkeypatch.setattr(uf, "_fetch_via_http", fake_http)
    monkeypatch.setattr(uf, "_fetch_via_crawl4ai", fake_browser)
    monkeypatch.setattr(uf, "_ssrf_guard", lambda url: True)
    crawler = _CountingCrawler()
    crawler.calls = 0
    result = asyncio.run(uf.fetch_url_markdown(crawler, "https://pub.example.com/x"))
    return crawler.calls, result


def test_fail_outcome_skips_chromium(monkeypatch):
    calls, result = _run_markdown(monkeypatch, ("fail", None))
    assert calls == 0
    assert result is None


def test_browser_outcome_invokes_chromium(monkeypatch):
    calls, result = _run_markdown(monkeypatch, ("browser", "https://pub.example.com/x"))
    assert calls == 1
    assert result == ("from-browser", "rendered text")


def test_browser_starts_at_guarded_terminal_url(monkeypatch):
    # The browser must be invoked with the terminal URL from the outcome,
    # not the original — closing the unguarded-redirect-replay gap.
    seen = {}

    async def fake_http(url, timeout_s, client=None):
        return ("browser", "https://terminal.example.com/app")

    async def fake_browser(crawler, url, timeout_s):
        seen["url"] = url
        return ("t", "body")

    monkeypatch.setattr(uf, "_fetch_via_http", fake_http)
    monkeypatch.setattr(uf, "_fetch_via_crawl4ai", fake_browser)
    monkeypatch.setattr(uf, "_ssrf_guard", lambda url: True)
    asyncio.run(uf.fetch_url_markdown(object(), "https://orig.example.com/x"))
    assert seen["url"] == "https://terminal.example.com/app"


def test_browser_fallback_reguards_terminal_url(monkeypatch):
    # If the terminal URL re-resolves to a blocked target (DNS changed),
    # the browser fallback must be skipped (fail closed).
    async def fake_http(url, timeout_s, client=None):
        return ("browser", "https://rebind.internal/app")

    async def fake_browser(crawler, url, timeout_s):
        raise AssertionError("browser must not run on a blocked terminal URL")

    monkeypatch.setattr(uf, "_fetch_via_http", fake_http)
    monkeypatch.setattr(uf, "_fetch_via_crawl4ai", fake_browser)
    # First guard (original URL) passes; terminal guard fails.
    monkeypatch.setattr(uf, "_ssrf_guard", lambda url: "rebind.internal" not in url)
    result = asyncio.run(uf.fetch_url_markdown(object(), "https://orig.example.com/x"))
    assert result is None


def test_ok_outcome_returns_payload_without_chromium(monkeypatch):
    calls, result = _run_markdown(monkeypatch, ("ok", ("T", "body")))
    assert calls == 0
    assert result == ("T", "body")


# ─── curl_cffi primary engine ──────────────────────────────────────────


class _CffiStreamResponse:
    """Models a curl_cffi streamed response from session.get(stream=True):
    status/headers available immediately, body via aiter_content(), torn
    down by _abort_cffi_stream (cancels astream_task, then awaits aclose)."""

    def __init__(self, status_code, *, body=b"", content_type="text/html", location=None, encoding="utf-8"):
        self.status_code = status_code
        self.headers = {"content-type": content_type}
        if location is not None:
            self.headers["location"] = location
        self._body = body if isinstance(body, bytes) else body.encode(encoding)
        self.charset = encoding
        self.astream_task = None  # a fully-buffered fake has no live transfer
        self.closed = False

    async def aiter_content(self, chunk_size=65536):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i : i + chunk_size]

    async def aclose(self):
        self.closed = True


class _FakeCffiSession:
    """Records every .get() call so tests can assert allow_redirects /
    discard_cookies and walk a scripted redirect chain. `script` maps
    url -> _CffiStreamResponse."""

    def __init__(self, script):
        self.script = script
        self.calls = []

    async def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        resp = self.script.get(url)
        if resp is None:
            raise AssertionError(f"unscripted url {url}")
        return resp


def _stub(status, *, html="", content_type="text/html", location=None):
    return _CffiStreamResponse(status, body=html, content_type=content_type, location=location)


def _cffi_fetch(script, url="https://pub.example.com/p"):
    session = _FakeCffiSession(script)

    async def go():
        return await uf._fetch_via_cffi(url, 5.0, session)

    return asyncio.run(go()), session


def test_cffi_rich_page_ok():
    (kind, payload), _ = _cffi_fetch({"https://pub.example.com/p": _stub(200, html=RICH_HTML)})
    assert kind == "ok"
    assert payload[0] == "Recipe"


def test_cffi_403_goes_to_browser():
    (kind, payload), _ = _cffi_fetch({"https://pub.example.com/p": _stub(403, html="no")})
    assert kind == "browser"
    assert payload == "https://pub.example.com/p"  # guarded terminal URL


def test_cffi_thin_200_goes_to_browser():
    (kind, payload), _ = _cffi_fetch({"https://pub.example.com/p": _stub(200, html=THIN_HTML)})
    assert kind == "browser"
    assert payload == "https://pub.example.com/p"


@pytest.mark.parametrize("status", [404, 500, 503])
def test_cffi_definitive_statuses_fail(status):
    (kind, payload), _ = _cffi_fetch({"https://pub.example.com/p": _stub(status)})
    assert (kind, payload) == ("fail", None)


def test_cffi_body_read_is_capped():
    # Decompression-bomb guard: a body far past the cap must still classify
    # using only the first _MAX_FETCH_BYTES, never buffering all of it.
    huge = RICH_HTML + ("x" * (uf._MAX_FETCH_BYTES * 3))
    (kind, payload), _ = _cffi_fetch({"https://pub.example.com/p": _stub(200, html=huge)})
    assert kind == "ok"
    assert "tomato soup" in payload[1]


def test_cffi_never_lets_libcurl_follow_redirects():
    # SECURITY: libcurl must NOT follow redirects itself — that would
    # connect to attacker-chosen hops in C with no SSRF guard.
    script = {
        "https://pub.example.com/p": _stub(302, location="https://dest.example.com/a"),
        "https://dest.example.com/a": _stub(200, html=RICH_HTML),
    }
    (kind, _payload), session = _cffi_fetch(script)
    assert kind == "ok"
    assert session.calls, "no requests made"
    assert all(kw.get("allow_redirects") is False for _u, kw in session.calls)
    # Each fetch must be cold — no cross-URL cookie replay (codex finding).
    assert all(kw.get("discard_cookies") is True for _u, kw in session.calls)


def test_cffi_browser_payload_is_terminal_url():
    # A redirect chain ending in a thin page → browser must target the
    # final guarded URL, not the original.
    script = {
        "https://pub.example.com/p": _stub(302, location="https://final.example.com/app"),
        "https://final.example.com/app": _stub(200, html=THIN_HTML),
    }
    (kind, payload), _ = _cffi_fetch(script)
    assert kind == "browser"
    assert payload == "https://final.example.com/app"


def test_cffi_redirect_hop_is_ssrf_guarded(monkeypatch):
    # A redirect to a blocked host must raise _SSRFBlocked, never fetch it.
    monkeypatch.setattr(uf, "_ssrf_guard", lambda url: "blocked.internal" not in url)
    script = {"https://pub.example.com/p": _stub(302, location="https://blocked.internal/x")}
    with pytest.raises(uf._SSRFBlocked):
        _cffi_fetch(script)


def test_cffi_engine_error_falls_back_to_httpx(monkeypatch):
    # If curl_cffi throws an engine-level error, the fetch must degrade to
    # the httpx path rather than losing the URL.
    class _BoomSession:
        async def get(self, *a, **k):
            raise RuntimeError("libcurl boom")

    async def fake_http(url, timeout_s, client=None):
        return ("ok", ("via-httpx", "body"))

    monkeypatch.setattr(uf, "_ssrf_guard", lambda url: True)
    monkeypatch.setattr(uf, "_fetch_via_http", fake_http)
    result = asyncio.run(uf.fetch_url_markdown(None, "https://pub.example.com/p", cffi_session=_BoomSession()))
    assert result == ("via-httpx", "body")


def test_cffi_ssrf_block_does_not_fall_back(monkeypatch):
    # An SSRF block is terminal — it must NOT degrade to httpx (which would
    # just hit the same blocked target) and must return None.
    class _BlockingSession:
        async def get(self, *a, **k):
            raise AssertionError("should not fetch after guard fails")

    monkeypatch.setattr(uf, "_ssrf_guard", lambda url: False)
    result = asyncio.run(uf.fetch_url_markdown(None, "https://blocked.internal/p", cffi_session=_BlockingSession()))
    assert result is None


def test_abort_cancels_live_transfer_before_close():
    # _abort_cffi_stream must cancel the background transfer task BEFORE
    # awaiting aclose — else curl_cffi drains the whole (possibly huge)
    # body. Models a still-running transfer task.
    class _LiveTask:
        def __init__(self):
            self.cancelled = False

        def done(self):
            return False

        def cancel(self):
            self.cancelled = True

    class _Resp:
        def __init__(self):
            self.astream_task = _LiveTask()
            self.closed = False

        async def aclose(self):
            # close must only be awaited AFTER the transfer was cancelled.
            assert self.astream_task.cancelled, "aclose ran before cancel"
            self.closed = True

    resp = _Resp()
    asyncio.run(uf._abort_cffi_stream(resp))
    assert resp.astream_task.cancelled and resp.closed


def test_cffi_browser_outcome_retries_httpx_before_browser(monkeypatch):
    # Reconciliation: when curl_cffi walls (browser) but plain-UA httpx
    # gets the content (substack/condenast trackers), recover via httpx
    # and NEVER spend a browser pass.
    async def fake_cffi(url, timeout_s, session):
        return ("browser", url)

    async def fake_http(url, timeout_s, client=None):
        return ("ok", ("via-httpx", "real article"))

    async def fake_browser(crawler, url, timeout_s):
        raise AssertionError("browser must not run when httpx recovers the page")

    monkeypatch.setattr(uf, "_ssrf_guard", lambda url: True)
    monkeypatch.setattr(uf, "_fetch_via_cffi", fake_cffi)
    monkeypatch.setattr(uf, "_fetch_via_http", fake_http)
    monkeypatch.setattr(uf, "_fetch_via_crawl4ai", fake_browser)
    result = asyncio.run(uf.fetch_url_markdown(object(), "https://sub.example.com/p", cffi_session=object()))
    assert result == ("via-httpx", "real article")


def test_cffi_browser_then_httpx_browser_reaches_chromium(monkeypatch):
    # If BOTH cffi and httpx wall, only then does Chromium run — at the
    # guarded terminal URL.
    seen = {}

    async def fake_cffi(url, timeout_s, session):
        return ("browser", url)

    async def fake_http(url, timeout_s, client=None):
        return ("browser", "https://terminal.example.com/x")

    async def fake_browser(crawler, url, timeout_s):
        seen["url"] = url
        return ("t", "rendered")

    monkeypatch.setattr(uf, "_ssrf_guard", lambda url: True)
    monkeypatch.setattr(uf, "_fetch_via_cffi", fake_cffi)
    monkeypatch.setattr(uf, "_fetch_via_http", fake_http)
    monkeypatch.setattr(uf, "_fetch_via_crawl4ai", fake_browser)
    result = asyncio.run(uf.fetch_url_markdown(object(), "https://sub.example.com/p", cffi_session=object()))
    assert result == ("t", "rendered")
    assert seen["url"] == "https://terminal.example.com/x"


def test_readable_text_falls_back_to_body_on_empty_main():
    # The amazon trap: role=main is an empty mount, real text is in <body>.
    html = (
        "<html><head><title>Item</title></head><body>"
        '<div role="main"></div>'
        "<div>" + ("Product details and a long description. " * 20) + "</div>"
        "</body></html>"
    )
    text, title = uf._readable_text(html)
    assert title == "Item"
    assert "Product details" in text


# ─── DNS cache ─────────────────────────────────────────────────────────


def test_positive_dns_results_are_not_cached(monkeypatch):
    # SECURITY: a cached "safe" answer would widen the DNS-rebinding
    # TOCTOU window from one guard-to-connect race to the whole TTL.
    # Successful resolutions must re-resolve every time.
    uf._dns_cache.clear()
    resolved = []

    def fake_getaddrinfo(host, port):
        resolved.append(host)
        return [(None, None, None, None, ("93.184.216.34", 0))]

    monkeypatch.setattr(uf.socket, "getaddrinfo", fake_getaddrinfo)
    assert uf._resolve_all_ips("live.example.com") == ["93.184.216.34"]
    assert uf._resolve_all_ips("live.example.com") == ["93.184.216.34"]
    assert resolved == ["live.example.com", "live.example.com"]


def test_dns_failures_are_cached_too(monkeypatch):
    uf._dns_cache.clear()
    resolved = []

    def fake_getaddrinfo(host, port):
        resolved.append(host)
        raise OSError("NXDOMAIN")

    monkeypatch.setattr(uf.socket, "getaddrinfo", fake_getaddrinfo)
    assert uf._resolve_all_ips("dead.example.com") == []
    assert uf._resolve_all_ips("dead.example.com") == []
    assert resolved == ["dead.example.com"]


def test_negative_dns_cache_expires(monkeypatch):
    uf._dns_cache.clear()

    def nxdomain(host, port):
        raise OSError("NXDOMAIN")

    monkeypatch.setattr(uf.socket, "getaddrinfo", nxdomain)
    uf._resolve_all_ips("ttl.example.com")
    # Force the entry past its TTL, then confirm a re-resolve happens.
    expiry, ips = uf._dns_cache["ttl.example.com"]
    uf._dns_cache["ttl.example.com"] = (expiry - uf._DNS_NEG_TTL_S - 1, ips)
    resolved = []

    def counting(host, port):
        resolved.append(host)
        raise OSError("NXDOMAIN")

    monkeypatch.setattr(uf.socket, "getaddrinfo", counting)
    uf._resolve_all_ips("ttl.example.com")
    assert resolved == ["ttl.example.com"]


# ─── embedded-JSON rescue (SPA shells without a browser) ───────────────


def test_readable_text_rescues_next_data_shell():
    # A Next.js-style shell: empty <div id=__next>, content only in __NEXT_DATA__.
    article = "Scale AI founder Alexandr Wang discusses the future of data labeling. " * 8
    html = (
        "<html><head><title>Newcomer</title></head><body>"
        '<div id="__next"></div>'
        '<script id="__NEXT_DATA__" type="application/json">'
        '{"props":{"pageProps":{"post":{"title":"x","body":"' + article + '"}}}}'
        "</script></body></html>"
    )
    text, title = uf._readable_text(html)
    assert title == "Newcomer"
    assert "Alexandr Wang" in text
    assert len(text) >= uf._MIN_USABLE_CHARS


def test_readable_text_rescues_ld_json_article():
    body = "This is the full article body with lots of real prose content to index. " * 6
    html = (
        "<html><head><title>Doc</title>"
        '<script type="application/ld+json">'
        '{"@type":"Article","headline":"H","articleBody":"' + body + '"}'
        "</script></head><body><div id=root></div></body></html>"
    )
    text, _ = uf._readable_text(html)
    assert "full article body" in text
    assert len(text) >= uf._MIN_USABLE_CHARS


def test_readable_text_prefers_real_dom_over_json():
    # When the DOM has real content, the JSON path must NOT override it.
    html = (
        "<html><head><title>T</title>"
        '<script type="application/json">{"config":"some setting value here ok"}</script>'
        "</head><body><article>" + ("Real rendered article text. " * 20) + "</article></body></html>"
    )
    text, _ = uf._readable_text(html)
    assert "Real rendered article text" in text


def test_readable_text_no_json_still_thin():
    # A genuine empty shell with no embedded content stays thin → browser.
    html = "<html><head><title>App</title></head><body><div id='root'></div></body></html>"
    text, _ = uf._readable_text(html)
    assert len(text) < uf._MIN_USABLE_CHARS


# ─── split-pool scheduling (HTTP vs browser) ───────────────────────────


def test_resolve_via_http_returns_tristate(monkeypatch):
    monkeypatch.setattr(uf, "_ssrf_guard", lambda url: True)

    async def fake_http(url, timeout_s, client=None):
        return ("ok", ("T", "body"))

    monkeypatch.setattr(uf, "_fetch_via_http", fake_http)
    kind, data = asyncio.run(uf.resolve_via_http("https://x.example.com/a"))
    assert kind == "ok"
    assert data == ("T", "body")


def test_resolve_via_http_browser_returns_guarded_url(monkeypatch):
    monkeypatch.setattr(uf, "_ssrf_guard", lambda url: True)

    async def fake_http(url, timeout_s, client=None):
        return ("browser", "https://term.example.com/p")

    # No cffi session → no second-chance; browser url passes straight through.
    monkeypatch.setattr(uf, "_fetch_via_http", fake_http)
    kind, data = asyncio.run(uf.resolve_via_http("https://x.example.com/a"))
    assert kind == "browser"
    assert data == "https://term.example.com/p"


def test_process_one_browser_uses_browser_sem_not_http_sem(monkeypatch):
    # The browser render must run under browser_sem, AFTER the http sem is
    # released — so a slow render can't hold an HTTP slot.
    import gmail_search.store.queries as q  # noqa: F401

    class _FakeResult:
        def fetchall(self):
            return []

    class _FakeConn:
        def execute(self, *a, **k):
            return _FakeResult()

        def commit(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(uf, "get_connection", lambda db_path: _FakeConn())
    monkeypatch.setattr(uf, "_ssrf_guard", lambda url: True)
    monkeypatch.setattr(uf, "fill_url_attachment", lambda *a, **k: None)

    async def fake_resolve(url, timeout_s, http_client=None, cffi_session=None):
        return ("browser", "https://term.example.com/p")

    monkeypatch.setattr(uf, "resolve_via_http", fake_resolve)

    state = {"http_held_during_browser": None}
    http_sem = asyncio.Semaphore(1)
    browser_sem = asyncio.Semaphore(1)

    async def fake_browser(crawler, url, timeout_s):
        # While rendering, the HTTP sem must be FREE (locked == False).
        state["http_held_during_browser"] = http_sem.locked()
        return ("bt", "rendered")

    monkeypatch.setattr(uf, "_fetch_via_crawl4ai", fake_browser)
    stub = {"id": 1, "url": "https://x.example.com/a", "filename": "URL: https://x.example.com/a"}
    ok = asyncio.run(uf._process_one(None, stub, http_sem, None, 5.0, browser_sem=browser_sem))
    assert ok is True
    assert state["http_held_during_browser"] is False  # HTTP slot released before render


# ─── continuous worker-pool ────────────────────────────────────────────


def test_run_continuous_processes_all_outcomes(monkeypatch):
    # End-to-end of run_continuous with everything below the orchestration
    # mocked: a producer batch of stubs with ok / fail / browser outcomes;
    # assert counters are right and the browser stub was rendered + persisted.
    stubs = [
        {"id": i, "url": f"https://s{i}.example.com/p", "filename": f"URL: https://s{i}.example.com/p"}
        for i in range(6)
    ]
    served = {"done": False}

    def fake_pending(conn, n):
        if served["done"]:
            return []
        served["done"] = True
        return list(stubs)

    monkeypatch.setattr(uf, "pending_url_stubs", fake_pending)
    monkeypatch.setattr(uf, "get_connection", lambda db_path: type("C", (), {"close": lambda self: None})())
    monkeypatch.setattr(uf, "build_http_client", lambda timeout_s=10.0: type("X", (), {"aclose": _noop_async})())
    monkeypatch.setattr(uf, "build_cffi_session", lambda: None)
    monkeypatch.setattr(uf, "_ssrf_guard", lambda url: "s5" not in url)  # s5 → abandon (fail)
    monkeypatch.setattr(uf, "_mark_attempt_sync", lambda db, fn: None)
    monkeypatch.setattr(uf, "_abandon_sync", lambda db, fn: None)

    persisted = []

    async def fake_persist(db_path, stub, result):
        if result is None:
            return False
        persisted.append(stub["id"])
        return True

    monkeypatch.setattr(uf, "_persist_result", fake_persist)

    async def fake_resolve(url, timeout_s, http_client=None, cffi_session=None):
        if "s0" in url or "s1" in url:
            return ("ok", ("t", "body"))
        if "s2" in url:
            return ("fail", None)
        return ("browser", url)  # s3, s4

    monkeypatch.setattr(uf, "resolve_via_http", fake_resolve)

    async def fake_browser(crawler, url, timeout_s):
        return ("bt", "rendered")

    monkeypatch.setattr(uf, "_fetch_via_crawl4ai", fake_browser)

    # Stub the crawl4ai AsyncWebCrawler context manager.
    class _FakeCrawler:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    import sys as _sys
    import types as _types

    fake_mod = _types.ModuleType("crawl4ai")
    fake_mod.AsyncWebCrawler = lambda **k: _FakeCrawler()
    fake_mod.BrowserConfig = lambda **k: None
    monkeypatch.setitem(_sys.modules, "crawl4ai", fake_mod)

    r = asyncio.run(uf.run_continuous(None, http_concurrency=3, browser_cap=2, target=6))
    # s0,s1 ok via http; s3,s4 ok via browser → 4 done. s2 fail, s5 abandon → 2 fail.
    assert r["done"] == 4
    assert r["failed"] == 2
    assert set(persisted) == {0, 1, 3, 4}


async def _noop_async(*a, **k):
    return None
