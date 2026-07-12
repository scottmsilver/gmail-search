"""Crawl URL stubs and fill `attachments.extracted_text` with page text.

Shape mirrors the Drive extract flow in `cli.py`:
  1. `pending_url_stubs(conn, limit)` → rows needing a fetch.
  2. For each row: SSRF-check, fetch via crawl4ai (or fall through to
     the PDF extractor for application/pdf responses), and fill the
     attachment via `fill_url_attachment`.
  3. Progress surfaces through the same `JobProgress` table the
     summarizer uses so the Settings UI can display "crawling N/M".

Concurrency is bounded with an asyncio.Semaphore. crawl4ai's
`AsyncWebCrawler` is opened once per `run()` call so we don't pay
Chromium startup per URL.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
import re
import socket
import time  # noqa: F401 — used in _resolve_all_ips; formatter must not strip
from pathlib import Path
from urllib.parse import urlparse

from gmail_search.store.db import JobProgress, get_connection
from gmail_search.store.queries import fill_url_attachment, pending_url_stubs

logger = logging.getLogger(__name__)

# Keep crawled content bounded so downstream prompts don't blow up.
# 8000 chars ≈ 2000 tokens — enough for a summarizer to get the gist.
_MAX_CRAWL_CHARS = 8000

# Top-level per-URL budget. Now that curl_cffi/httpx have their own tight
# per-op cap (_HTTP_OP_TIMEOUT_S), this primarily bounds the BROWSER tier
# (crawl4ai page_timeout, in ms) and the PDF download. Measured (2026-06-10):
# every SUCCESSFUL browser render finished under 2.4s (hits median 1.9s,
# max 2.4s; dead pages max 1.6s), so 10s only let hung/dead browser pages
# burn the clock. 6s keeps a ~2.5x margin over the observed max while
# halving worst-case hung-browser time. (Was 30s, then 10s.)
_DEFAULT_TIMEOUT_S = 6.0

# Cap for the PDF-routed path. Binary download, not char count.
_MAX_PDF_BYTES = 20 * 1024 * 1024


# NEGATIVE-ONLY DNS cache. Dead domains (very common in old mail) repeat
# across many stubs, and each NXDOMAIN costs a full resolver timeout —
# caching those is the big win and is fail-closed by construction (a
# cached [] can only cause a skip). Positive results are deliberately
# NOT cached: the SSRF guard's answer must stay as close as possible to
# the IP httpx's own connect-time resolution will get, and a "safe"
# answer cached for minutes would widen the DNS-rebinding TOCTOU window
# from one guard-to-connect race to the whole TTL (codex audit finding).
_DNS_NEG_TTL_S = 600.0
_DNS_CACHE_MAX = 4096
_dns_cache: dict[str, tuple[float, list[str]]] = {}


def _resolve_all_ips(host: str) -> list[str]:
    """DNS-resolve a hostname to every A / AAAA record. We reject on
    ANY private IP — if an attacker controls DNS they could return
    `[public-ip, 127.0.0.1]` and win a race with Chromium's own
    resolver, so checking just the first record isn't enough.

    Returns `[]` on resolution failure; the caller treats that as
    "skip this URL" (fail closed). Only failures are cached (see the
    cache comment above) so repeated stubs of a dead domain don't
    re-pay the resolver timeout.
    """
    now = time.monotonic()
    hit = _dns_cache.get(host)
    if hit is not None and hit[0] > now:
        return hit[1]
    try:
        # getaddrinfo covers both A and AAAA records.
        infos = socket.getaddrinfo(host, None)
    except Exception:
        infos = []
    out: list[str] = []
    for info in infos:
        sockaddr = info[4]
        if sockaddr:
            out.append(sockaddr[0])
    if not out:
        if len(_dns_cache) >= _DNS_CACHE_MAX:
            _dns_cache.clear()  # crude eviction; refilling is cheap at this size
        _dns_cache[host] = (now + _DNS_NEG_TTL_S, out)
    return out


def _is_private_ip(ip: str) -> bool:
    """True if the IP is in a range we must not fetch — RFC1918,
    loopback, link-local, multicast, or reserved. Guards against an
    attacker-controlled email URL pointing at 127.0.0.1 / 169.254.x.x
    / metadata.google.internal / internal k8s service IPs.
    """
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return True
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_multicast
        or addr.is_reserved
        or addr.is_unspecified
    )


def _ssrf_guard(url: str) -> bool:
    """Pre-flight check before the HTTP fetch. Returns True if the URL
    is safe to fetch. Pure function on the host → IP mapping at call
    time; TOCTOU is real (attacker could change DNS between this
    check and the fetch) but for v1 this is acceptable — crawl4ai
    runs inside a Chromium sandbox, and the content caps below limit
    what a late-binding redirect could smuggle out.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme not in {"http", "https"}:
        return False
    host = parsed.hostname
    if not host:
        return False
    # Host might already be a numeric IP literal — handle that case
    # without a DNS round-trip.
    try:
        literal = ipaddress.ip_address(host)
        return not _is_private_ip(str(literal))
    except ValueError:
        pass
    ips = _resolve_all_ips(host)
    if not ips:
        return False
    # Reject if ANY resolved address is private — an attacker
    # controlling DNS could otherwise smuggle a loopback IP into an
    # otherwise-public-looking response set.
    return all(not _is_private_ip(ip) for ip in ips)


def _truncate_markdown(text: str, max_chars: int = _MAX_CRAWL_CHARS) -> str:
    """Trim Markdown to `max_chars` with a visible boundary marker so
    downstream prompts know they got a truncation rather than a short
    page.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n\n[…truncated]"


async def _fetch_via_crawl4ai(crawler, url: str, timeout_s: float) -> tuple[str, str] | None:
    """Single URL fetch through an already-open `AsyncWebCrawler`.
    Returns (title, markdown) or None. Never raises — all exceptions
    are caught and logged, and a failure just means the row stays
    pending for the next pass.
    """
    try:
        from crawl4ai import CrawlerRunConfig
        from crawl4ai.content_filter_strategy import PruningContentFilter
        from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
    except Exception as e:
        logger.warning(f"crawl4ai import failed: {e}")
        return None

    # page_timeout is in ms inside crawl4ai; keep our top-level
    # asyncio.wait_for in seconds so the API matches our other code.
    #
    # The PruningContentFilter is what turns raw HTML into the
    # "article body only" markdown we actually want. Without it
    # `fit_markdown` is empty and we fall through to `raw_markdown`,
    # which is the whole page (nav + footer + sidebar), burning the
    # 8k char cap on chrome. Threshold 0.48 is crawl4ai's own
    # recommended default.
    run_config = CrawlerRunConfig(
        page_timeout=int(timeout_s * 1000),
        verbose=False,
        # Don't eat our RAM chasing rich media when we only care about text.
        exclude_all_images=True,
        remove_overlay_elements=True,
        remove_forms=True,
        markdown_generator=DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(threshold=0.48, threshold_type="fixed"),
        ),
    )
    try:
        result_container = await asyncio.wait_for(crawler.arun(url=url, config=run_config), timeout=timeout_s + 5.0)
    except asyncio.TimeoutError:
        logger.info(f"url_fetcher: timeout after {timeout_s}s fetching {url}")
        return None
    except Exception as e:
        logger.info(f"url_fetcher: crawl4ai error on {url}: {type(e).__name__}: {str(e)[:200]}")
        return None

    # crawl4ai 0.8+ returns a CrawlResultContainer; take the first.
    result = None
    try:
        if hasattr(result_container, "__iter__"):
            for r in result_container:
                result = r
                break
        else:
            result = result_container
    except Exception:
        result = result_container

    if result is None or not getattr(result, "success", False):
        reason = getattr(result, "error_message", "no result") if result else "no result"
        logger.info(f"url_fetcher: crawl failed for {url}: {str(reason)[:200]}")
        return None

    markdown = ""
    md_obj = getattr(result, "markdown", None)
    if md_obj is not None:
        # crawl4ai wraps markdown in a `StringCompatibleMarkdown`
        # object whose `str()` value is the RAW markdown. That class
        # subclasses `str`, so a naive `isinstance(md_obj, str)`
        # check returns True and we'd silently pick the unfiltered
        # output — losing the whole point of PruningContentFilter.
        #
        # Order of preference:
        #   1. `fit_markdown` (post content-filter — the good stuff)
        #   2. `raw_markdown` (pre-filter full-page)
        #   3. `str(md_obj)` only if the object is actually a plain
        #      string with neither attribute.
        fit = getattr(md_obj, "fit_markdown", None)
        raw = getattr(md_obj, "raw_markdown", None)
        if fit:
            markdown = fit
        elif raw:
            markdown = raw
        elif type(md_obj) is str:  # noqa: E721 — exact type, not a subclass
            markdown = md_obj
    if not markdown:
        return None
    title = (getattr(result, "metadata", None) or {}).get("title") or ""
    return title, _truncate_markdown(markdown)


# Neutral browser UA. Deliberately NOT self-identifying: the old value
# ("gmail-search/1.0; +https://oursilverfamily.com") told every crawled
# server that this specific tool — and its owner's personal domain — had
# processed the mail, a direct privacy leak to senders/trackers. A generic
# Chrome UA leaks nothing about the mailbox or its owner and also trips
# fewer bot-walls. Matches the Chrome family the curl_cffi engine
# impersonates so the two engines present consistently.
_HTTP_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
_MIN_USABLE_CHARS = 250  # below this we assume a JS shell / empty page → browser
_MAX_HTML_CHARS = 3_000_000  # cap parsed HTML so a giant page can't blow RAM
_MAX_HTML_BYTES = 20_000_000  # reject responses larger than this (Content-Length)
_MAX_FETCH_BYTES = 1_000_000  # stream cap: article text virtually always fits in the first MB
_MAX_REDIRECTS = 6
_REDIRECT_CODES = (301, 302, 303, 307, 308)

# Per-operation (connect / read / write) timeout for the HTTP path.
# Measured (2026-06-10) UNDER CONCURRENCY 8 (the daemon's real load — NOT
# sequential, which understates latency): good fetches run median 285ms,
# p99 1865ms, max 2870ms. A 3.0s cap loses 0/105 good fetches; 2.5s lost
# ~1% (one contended outlier at 2870ms). 3.0s caps tarpit / hung hosts at
# 3s (was 5s) with zero good-fetch loss — hung hosts otherwise sit in a
# concurrency slot blocking live URLs. Lesson: tune timeouts against the
# CONCURRENT distribution, not a sequential probe.
_HTTP_OP_TIMEOUT_S = 3.0


class _SSRFBlocked(Exception):
    """A redirect hop resolved to a blocked (private/loopback/link-local)
    target. Raised so the caller does NOT fall back to crawl4ai, which would
    follow the same attacker-controlled redirect WITHOUT the per-hop guard."""


_JSON_PROSE_RE = re.compile(r'"([^"\\]{40,})"')
_LDJSON_FIELD_RE = re.compile(
    r'"(?:articleBody|description|headline|text|name|caption|abstract|reviewBody|snippet)":\s*"([^"\\]{15,})"'
)
_EMBEDDED_STATE_MARKERS = ("__NEXT_DATA__", "__NUXT__", "__APOLLO_STATE__", "__INITIAL_STATE__", "__remixContext")


def _extract_embedded_json_text(soup) -> str:
    """Pull human-readable text out of embedded SPA state / structured data
    in the RAW html — WITHOUT a browser. Many "JS shell" pages (yelp,
    substack, github docs, Next.js/Nuxt sites) ship their full content as
    JSON in a `<script>` (`__NEXT_DATA__`, ld+json, Apollo/Redux state);
    rendering them in Chromium is a ~2s pass that often fails anyway, while
    the data is right there in the HTML we already fetched. Measured
    (2026-06-10): rescues ~32% of browser-bound pages at HTTP speed.

    Heuristic by design: prefers schema.org content fields, then any prose-
    looking string (has a space, not a URL/path) from JSON `<script>`s. Some
    config noise can leak in, but for SEARCH indexing recall beats precision
    and the result is capped downstream."""
    texts: list[str] = []
    seen: set[str] = set()

    def add(s: str):
        s = s.strip()
        if len(s) >= 15 and s not in seen:
            seen.add(s)
            texts.append(s)

    for tag in soup.find_all("script"):
        s = tag.string or ""
        if len(s) < 80:
            continue
        type_attr = (tag.get("type") or "").lower()
        is_ldjson = "ld+json" in type_attr
        is_json = "json" in type_attr
        is_state = any(mk in s for mk in _EMBEDDED_STATE_MARKERS)
        if is_ldjson:
            for m in _LDJSON_FIELD_RE.finditer(s):
                add(_unescape_json_str(m.group(1)))
        if is_json or is_state:
            for m in _JSON_PROSE_RE.finditer(s):
                v = m.group(1)
                if " " in v and not v.startswith(("http", "/", "data:", "#", "\\u", "@")):
                    add(_unescape_json_str(v))
    return "\n".join(texts)


def _unescape_json_str(s: str) -> str:
    """Best-effort unescape of a raw JSON string value (\\n, \\", \\uXXXX)."""
    try:
        import json as _json  # noqa: PLC0415

        return _json.loads('"' + s.replace('"', '\\"') + '"')
    except Exception:
        return s


def _readable_text(html: str) -> tuple[str, str]:
    """(text, title) from raw HTML. Strips script/nav/footer/etc. boilerplate
    and prefers the main-content container (<article>/<main>) so embeddings
    aren't diluted by menus. No new dependency (bs4 + lxml); good enough for
    search even if cruder than a dedicated readability library."""
    import warnings  # noqa: PLC0415

    from bs4 import BeautifulSoup  # noqa: PLC0415

    try:
        from bs4 import XMLParsedAsHTMLWarning  # noqa: PLC0415

        warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
    except Exception:
        pass
    soup = BeautifulSoup(html, "lxml")
    title = (soup.title.get_text(strip=True) if soup.title else "") or ""
    # Pull embedded JSON/SPA-state text BEFORE decomposing <script> tags —
    # this is what rescues "JS shell" pages without a browser. Computed
    # lazily-ish: we always extract it (cheap regex over scripts) but only
    # USE it if the DOM text comes back thin.
    json_text = _extract_embedded_json_text(soup)
    for tag in soup(
        ["script", "style", "noscript", "template", "svg", "head", "nav", "footer", "header", "form", "aside", "iframe"]
    ):
        tag.decompose()
    main = soup.find("article") or soup.find("main") or soup.find(attrs={"role": "main"}) or soup.body or soup
    text = main.get_text(separator="\n", strip=True)
    if len(text) < _MIN_USABLE_CHARS and main is not soup.body and soup.body is not None:
        # The "main content" node can be an empty JS mount point (or, e.g. on
        # amazon, a container whose content lived in <form> tags we removed)
        # while the rest of <body> holds real server-rendered text. Falling
        # back to body turns those false "thin" pages into HTTP successes
        # instead of pointless ~10s Chromium passes (measured on amazon /dp:
        # role=main div is 0 chars, body is ~2.5k chars).
        body_text = soup.body.get_text(separator="\n", strip=True)
        if len(body_text) > len(text):
            text = body_text
    # Still thin after DOM extraction? The page is a JS shell — use the
    # embedded JSON content if it's richer (yelp/substack/Next.js/etc.).
    if len(text) < _MIN_USABLE_CHARS and len(json_text) > len(text):
        return json_text, title
    return text, title


def _crawl_proxy() -> str | None:
    """Optional egress proxy for ALL crawl traffic, read from the
    `GMAIL_CRAWL_PROXY` env var (e.g. `http://user:pass@1.2.3.4:3128` or
    `socks5://127.0.0.1:9050`). Unset/blank → direct connection (no
    behavior change), so this is a safe no-op until an egress is stood up.

    Purpose: route fetches through a cloud/datacenter egress so crawled
    sites see that IP, not the home/residential IP — the home identity is
    never exposed to senders/trackers. Never hard-code a proxy URL here;
    it's deployment config, so it lives in the environment."""
    p = os.environ.get("GMAIL_CRAWL_PROXY", "").strip()
    return p or None


def _crawl_proxy_config() -> dict | None:
    """The same egress proxy as `_crawl_proxy()`, in the dict shape crawl4ai's
    `BrowserConfig(proxy_config=...)` wants — the bare `proxy=` kwarg is
    deprecated and slated for removal (after which the browser tier would
    silently stop using the proxy). None when GMAIL_CRAWL_PROXY is unset."""
    url = _crawl_proxy()
    if not url:
        return None
    from urllib.parse import urlsplit  # noqa: PLC0415

    parts = urlsplit(url)
    server = f"{parts.scheme}://{parts.hostname}"
    if parts.port:
        server += f":{parts.port}"
    cfg: dict = {"server": server}
    if parts.username:
        cfg["username"] = parts.username
    if parts.password:
        cfg["password"] = parts.password
    return cfg


def build_http_client(timeout_s: float = _DEFAULT_TIMEOUT_S):
    """Pooled httpx client for the HTTP-first path. Built ONCE per crawl
    pass and shared across every URL: keep-alive + HTTP/2 mean repeated
    hosts (tracking domains, google.com, evite.com, …) skip the TCP+TLS
    handshake that dominated per-page latency when each URL built its
    own client. Caller is responsible for `await client.aclose()`."""
    import httpx  # noqa: PLC0415

    per_op = min(_HTTP_OP_TIMEOUT_S, timeout_s)
    return httpx.AsyncClient(
        follow_redirects=False,  # hops are followed manually, SSRF-guarded each
        http2=True,
        proxy=_crawl_proxy(),  # None → direct; set GMAIL_CRAWL_PROXY to route egress
        headers={"User-Agent": _HTTP_UA},
        timeout=httpx.Timeout(connect=per_op, read=per_op, write=per_op, pool=timeout_s),
        limits=httpx.Limits(max_connections=64, max_keepalive_connections=32, keepalive_expiry=30.0),
    )


async def _read_capped(resp, cap: int = _MAX_FETCH_BYTES) -> str:
    """Stream the body and stop after `cap` bytes. We only keep
    _MAX_CRAWL_CHARS of extracted text, so downloading a page past the
    first MB is pure waste — and on slow/tarpit hosts it's where the
    old full-body read spent its time."""
    chunks: list[bytes] = []
    read = 0
    async for chunk in resp.aiter_bytes():
        chunks.append(chunk)
        read += len(chunk)
        if read >= cap:
            break
    encoding = resp.charset_encoding or "utf-8"
    return b"".join(chunks).decode(encoding, errors="replace")


async def _classify_response(resp, final_url: str) -> tuple[str, object]:
    """Turn a terminal (non-redirect) response into a fetch outcome:

    ("ok", (title, text)) — usable server-rendered page.
    ("browser", final_url) — worth the Chromium fallback: a thin 200
                            (JS shell) or a 403 (anti-bot walls
                            sometimes pass a real browser fingerprint).
                            `final_url` is the SSRF-GUARDED terminal URL
                            of the redirect chain — the browser must
                            start there, NOT at the original URL, so it
                            doesn't replay the whole chain unguarded.
    ("fail", None)        — DEFINITIVE for this cycle: 404/410/5xx,
                            non-HTML 200s, oversized bodies. Chromium
                            would see the same status/bytes, so the
                            fallback is a guaranteed-wasted ~10s.
    """
    status = resp.status_code
    if status == 403:
        return ("browser", final_url)
    if status != 200:
        return ("fail", None)
    if "html" not in resp.headers.get("content-type", "").lower():
        return ("fail", None)
    try:
        if int(resp.headers.get("content-length", "0")) > _MAX_HTML_BYTES:
            return ("fail", None)  # oversized — don't buffer/parse it
    except ValueError:
        pass
    html = await _read_capped(resp)
    text, title = _readable_text(html[:_MAX_HTML_CHARS])
    if len(text) < _MIN_USABLE_CHARS:
        return ("browser", final_url)  # JS shell / near-empty → browser
    return ("ok", (title, _truncate_markdown(text)))


async def _fetch_via_http(url: str, timeout_s: float, client=None) -> tuple[str, tuple[str, str] | None]:
    """Browser-free fetch with manual redirect-following. Each hop is SSRF-
    checked BEFORE we fetch it (the caller's guard only covered the first URL),
    so a redirect can't smuggle us to a private IP.

    Returns an outcome tuple (see `_classify_response`); network errors —
    timeouts, refused connects, TLS failures — are ("fail", None): the
    browser shares the same network path, so retrying them through
    Chromium just re-pays the timeout. A 3xx with no Location header is
    also ("fail", None) — Chromium would follow the body's navigation
    with no per-hop SSRF guard.

    `client` is the shared pooled client from `build_http_client`; when
    None (tests, one-off callers) an ephemeral one is built per call.
    """
    from urllib.parse import urljoin  # noqa: PLC0415

    own_client = None
    if client is None:
        own_client = client = build_http_client(timeout_s)
    cur = url
    try:
        for _ in range(_MAX_REDIRECTS):
            if not _ssrf_guard(cur):
                # Blocked target: raise (don't return a fallback outcome) so
                # the caller won't hand the original URL to crawl4ai, which
                # would follow this same redirect with no guard.
                raise _SSRFBlocked(cur)
            async with client.stream("GET", cur) as resp:
                if resp.status_code in _REDIRECT_CODES:
                    loc = resp.headers.get("location")
                    if not loc:
                        # Malformed 3xx. Don't hand it to Chromium — the
                        # browser follows whatever the body navigates to
                        # with no per-hop SSRF guard (codex audit finding).
                        # Real meta-refresh pages are 200s and still get
                        # the browser via the thin-200 path.
                        return ("fail", None)
                    cur = urljoin(cur, loc)
                    continue
                return await _classify_response(resp, cur)
        return ("fail", None)  # redirect loop / too many hops
    except _SSRFBlocked:
        raise
    except Exception:
        return ("fail", None)
    finally:
        if own_client is not None:
            await own_client.aclose()


# crawl4ai impersonation profile for curl_cffi: a recent Chrome the
# library ships a matching TLS+HTTP2 fingerprint for. Env-overridable so
# we can bump it as the library adds newer profiles without a code change.
_CFFI_IMPERSONATE = os.environ.get("GMAIL_CRAWL_IMPERSONATE", "chrome")


def build_cffi_session():
    """Shared curl_cffi AsyncSession — the PRIMARY fetch engine. libcurl +
    BoringSSL perform a real Chrome TLS+HTTP2 handshake, so anti-bot walls
    that fingerprint the TLS ClientHello (Cloudflare/Akamai JA3) see a
    browser, not Python. Measured (2026-06-09) both faster than httpx
    (253ms vs 433ms median) and able to fetch ~15-20% of pages httpx got
    403/blocked on. One session per crawl pass; caller closes it.

    Returns None if curl_cffi can't be imported, so the caller falls back
    to the httpx engine rather than crashing the pass."""
    try:
        from curl_cffi.requests import AsyncSession  # noqa: PLC0415
    except Exception as e:  # noqa: BLE001
        logger.warning(f"curl_cffi unavailable, falling back to httpx engine: {e}")
        return None
    proxy = _crawl_proxy()  # None → direct; keeps the home IP off the wire when set
    proxies = {"http": proxy, "https": proxy} if proxy else None
    return AsyncSession(proxies=proxies)


async def _cffi_read_capped(resp, cap: int = _MAX_FETCH_BYTES) -> str:
    """Stream a curl_cffi response body and stop after `cap` bytes —
    AFTER libcurl's transparent gzip/br decompression, so a compressed
    decompression bomb can't expand unbounded before we cap (codex audit
    finding). Mirrors the httpx `_read_capped`."""
    chunks: list[bytes] = []
    read = 0
    async for chunk in resp.aiter_content():
        chunks.append(chunk)
        read += len(chunk)
        if read >= cap:
            break
    encoding = getattr(resp, "charset", None) or getattr(resp, "encoding", None) or "utf-8"
    try:
        return b"".join(chunks).decode(encoding, errors="replace")
    except LookupError:  # bogus server-supplied charset must not force a fail
        return b"".join(chunks).decode("utf-8", errors="replace")


async def _abort_cffi_stream(resp) -> None:
    """Tear down a curl_cffi streamed response. CRITICAL: curl_cffi's
    `aclose()` AWAITS the background transfer task, which keeps downloading
    the WHOLE body — so a capped read alone does NOT stop a large/slow body
    (measured: closing a 100MB stream after a 1MB cap drained for 3-6s; on a
    slow link the parallel agent saw 266s). Cancel the transfer task first,
    then close → teardown is instant. A fully-read small body has an
    already-done task, so cancel is a harmless no-op there."""
    task = getattr(resp, "astream_task", None)
    if task is not None and not task.done():
        task.cancel()
    try:
        await resp.aclose()
    except asyncio.CancelledError:
        # aclose() awaits the child transfer task we just cancelled, so it
        # re-raises that CancelledError. Swallow it — UNLESS our own coroutine
        # is the cancel target (then honour the cancellation and propagate).
        current = asyncio.current_task()
        if current is not None and current.cancelling() > 0:
            raise
    except Exception:  # noqa: BLE001 — dead connection during teardown
        pass


async def _fetch_via_cffi(url: str, timeout_s: float, session) -> tuple[str, object]:
    """Primary fetch via curl_cffi with manual, per-hop SSRF-guarded
    redirect following. CRITICAL: `allow_redirects=False` — libcurl must
    NOT follow redirects itself, or it would connect to attacker-chosen
    hops in C with no SSRF check. Same tri-state outcome contract as
    `_fetch_via_http` (browser payload is the guarded terminal URL); same
    fail-closed-on-error policy. Raises `_SSRFBlocked` on a blocked hop so
    the caller won't hand the URL to the unguarded Chromium fallback.

    Uses `stream=True` + a byte cap so a small compressed body can't
    decompress to huge RAM before the cap, and `_abort_cffi_stream` so the
    uncapped remainder of a large body is never drained (see that helper).
    `discard_cookies=True` keeps each fetch cold — URL A's Set-Cookie must
    not replay onto unrelated URL B in the shared session (codex finding)."""
    from urllib.parse import urljoin  # noqa: PLC0415

    per_op = min(_HTTP_OP_TIMEOUT_S, timeout_s)
    cur = url
    for _ in range(_MAX_REDIRECTS):
        if not _ssrf_guard(cur):
            raise _SSRFBlocked(cur)
        resp = await session.get(
            cur,
            impersonate=_CFFI_IMPERSONATE,
            timeout=per_op,
            allow_redirects=False,
            stream=True,
            discard_cookies=True,
            headers={"Accept-Encoding": "gzip, deflate, br"},
        )
        try:
            if resp.status_code in _REDIRECT_CODES:
                loc = resp.headers.get("location")
                if not loc:
                    return ("fail", None)  # malformed 3xx — see _fetch_via_http
                cur = urljoin(cur, loc)
                continue
            return await _classify_cffi_response(resp, cur)
        finally:
            await _abort_cffi_stream(resp)
    return ("fail", None)  # redirect loop / too many hops


async def _classify_cffi_response(resp, final_url: str) -> tuple[str, object]:
    """Tri-state outcome from a terminal curl_cffi (streamed) response.
    Mirrors `_classify_response`; the browser payload is the guarded
    terminal URL so Chromium starts at a vetted target."""
    status = resp.status_code
    if status == 403:
        return ("browser", final_url)  # anti-bot — a full browser may pass
    if status != 200:
        return ("fail", None)
    if "html" not in resp.headers.get("content-type", "").lower():
        return ("fail", None)
    try:
        if int(resp.headers.get("content-length", "0")) > _MAX_HTML_BYTES:
            return ("fail", None)  # oversized declared body — reject
    except (ValueError, TypeError):
        pass
    html = await _cffi_read_capped(resp)
    text, title = _readable_text(html[:_MAX_HTML_CHARS])
    if len(text) < _MIN_USABLE_CHARS:
        return ("browser", final_url)  # JS shell → let the browser render it
    return ("ok", (title, _truncate_markdown(text)))


async def resolve_via_http(
    url: str,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    http_client=None,
    cffi_session=None,
) -> tuple[str, object]:
    """HTTP-only resolution phase of the fetch ladder. Returns:
      ("ok",   (title, markdown))  — content obtained without a browser.
      ("fail", None)               — definitive: skip, no browser would help.
      ("browser", browser_url)     — needs Chromium; `browser_url` is the
                                     SSRF-GUARDED terminal URL to render.

    This is everything in the old `fetch_url_markdown` EXCEPT the final
    crawl4ai call, split out so the orchestrator can hold a cheap HTTP
    concurrency slot here and switch to a SEPARATE small browser pool for
    the render — a 2s Chromium pass must not block fast HTTP fetches. All
    the SSRF logic (per-hop guard, terminal-URL guard, no-fallback-on-
    block) is preserved verbatim.

    Engine ladder: curl_cffi (Chrome TLS impersonation) → httpx (plain-UA
    backup + second chance for trackers that wall impersonation) → caller's
    browser. PDF URLs route to the local extractor (an "ok")."""
    if not _ssrf_guard(url):
        logger.info(f"url_fetcher: skipping {url} (SSRF guard)")
        return ("fail", None)

    if url.lower().split("?", 1)[0].endswith(".pdf"):
        pdf = await asyncio.to_thread(_fetch_pdf_url, url)
        return ("ok", pdf) if pdf else ("fail", None)

    try:
        if cffi_session is not None:
            try:
                kind, payload = await _fetch_via_cffi(url, timeout_s, cffi_session)
            except _SSRFBlocked:
                raise
            except Exception as e:  # noqa: BLE001 — engine-level failure → fall back to httpx
                logger.info(f"url_fetcher: cffi engine error on {url}, trying httpx: {type(e).__name__}")
                kind, payload = await _fetch_via_http(url, timeout_s, client=http_client)
        else:
            kind, payload = await _fetch_via_http(url, timeout_s, client=http_client)
    except _SSRFBlocked as e:
        # A redirect pointed at a blocked address — give up; do NOT fall back
        # to crawl4ai (Chromium would follow the same redirect unguarded).
        logger.info(f"url_fetcher: skipping {url} (SSRF redirect target {e})")
        return ("fail", None)
    if kind == "ok":
        return ("ok", payload)
    if kind == "fail":
        # Definitive HTTP outcome (404/5xx/non-HTML/timeout) — Chromium
        # would see the same thing, so don't burn a browser pass on it.
        return ("fail", None)

    # kind == "browser": payload is the SSRF-GUARDED terminal URL of the
    # redirect chain (or the original URL if there were no redirects).
    browser_url = payload if isinstance(payload, str) and payload else url

    # Second chance before the expensive browser: if curl_cffi (Chrome
    # fingerprint) hit a wall but we haven't yet tried a plain-UA httpx
    # GET, try it — some trackers serve honest clients the real content
    # while walling impersonated ones. Cheap, and saves a browser pass.
    if cffi_session is not None:
        try:
            hk, hpayload = await _fetch_via_http(browser_url, timeout_s, client=http_client)
        except _SSRFBlocked as e:
            logger.info(f"url_fetcher: skipping {url} (SSRF redirect target {e})")
            return ("fail", None)
        if hk == "ok":
            return ("ok", hpayload)
        if hk == "fail":
            return ("fail", None)
        if isinstance(hpayload, str) and hpayload:
            browser_url = hpayload  # httpx's guarded terminal url

    # Hand back the guarded terminal URL, NOT the original — crawl4ai follows
    # redirects unguarded, so the original would let it replay the whole chain
    # to an attacker-chosen private target (codex finding).
    if not _ssrf_guard(browser_url):
        # Terminal URL re-resolves to a blocked target (DNS changed since
        # the in-loop guard) — fail closed rather than hand it to Chromium.
        logger.info(f"url_fetcher: skipping browser fallback for {url} (terminal SSRF)")
        return ("fail", None)
    return ("browser", browser_url)


async def fetch_url_markdown(
    crawler,
    url: str,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    http_client=None,
    cffi_session=None,
) -> tuple[str, str] | None:
    """Fetch one URL through the full ladder, return `(title, markdown)` or
    None. Thin wrapper over `resolve_via_http` + the crawl4ai browser phase,
    for callers (tests, one-offs) that don't need the split-pool scheduling
    the daemon uses in `_process_one`."""
    kind, data = await resolve_via_http(url, timeout_s, http_client=http_client, cffi_session=cffi_session)
    if kind == "ok":
        return data  # type: ignore[return-value]
    if kind == "fail":
        return None
    return await _fetch_via_crawl4ai(crawler, data, timeout_s)  # data = guarded browser_url


def _fetch_pdf_url(url: str) -> tuple[str, str] | None:
    """Download a PDF with stdlib + run it through the existing PDF
    extractor. Synchronous because `extract_pdf` is — the caller
    routes us through `asyncio.to_thread` to stay off the event loop.

    SSRF was checked by the caller for THIS url. We also refuse to follow
    redirects here, so a `…/a.pdf` that 30x-redirects to a private IP can't
    smuggle us past the guard (urllib follows redirects by default).
    """
    import tempfile
    import urllib.request

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):  # noqa: ARG002
            return None  # don't follow — an unguarded redirect could be SSRF

    try:
        opener = urllib.request.build_opener(_NoRedirect)
        req = urllib.request.Request(url, headers={"User-Agent": "gmail-search/1.0"})
        with opener.open(req, timeout=_DEFAULT_TIMEOUT_S) as resp:
            data = resp.read(_MAX_PDF_BYTES)
    except Exception as e:
        logger.info(f"url_fetcher: pdf download failed for {url}: {e}")
        return None

    from gmail_search.extract.pdf import extract_pdf

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as f:
        f.write(data)
        f.flush()
        result = extract_pdf(Path(f.name), {"max_pdf_pages": 20})
    if result is None or not result.text:
        return None
    title = urlparse(url).path.rsplit("/", 1)[-1] or urlparse(url).hostname or url
    return title, _truncate_markdown(result.text)


# The id-ordered FOR UPDATE in the three writers below is load-bearing: a
# plain multi-row UPDATE locks rows in arbitrary physical order, and
# concurrent workers touching the same filename (one URL fans out to
# thousands of copies) deadlocked ~5.5k times/day (2026-07-08 storm).
# Deterministic lock order makes the writers serialize instead.
_LOCKED_UNFILLED_COPIES = (
    "SELECT id FROM attachments WHERE filename = %s AND extracted_text IS NULL ORDER BY id FOR UPDATE"
)


def _mark_attempt_sync(db_path, filename: str) -> None:
    """Stamp +1 attempt on ALL unfilled copies of this URL (dedup fan-out) so
    duplicate stubs back off / abandon together instead of each being retried.
    Deliberately cross-user: URL reachability is a property of the URL, not
    of the mailbox that linked it — a dead domain is dead for everyone."""
    conn = get_connection(db_path)
    try:
        conn.execute(
            "UPDATE attachments SET crawl_attempts = crawl_attempts + 1, "
            f"crawl_last_attempt = now() WHERE id IN ({_LOCKED_UNFILLED_COPIES})",
            (filename,),
        )
        conn.commit()
    finally:
        conn.close()


def _abandon_sync(db_path, filename: str) -> None:
    """PERMANENT-failure fast path: jump all copies of this URL straight to the
    retry cap (NEVER-succeeds: private/reserved IP, NXDOMAIN dead domain) so we
    don't burn repeated browser fetches over the ~3 weeks of exponential
    backoff a normal-failing URL would take to reach the cap."""
    from gmail_search.store.queries import _MAX_CRAWL_ATTEMPTS  # noqa: PLC0415

    conn = get_connection(db_path)
    try:
        conn.execute(
            "UPDATE attachments SET crawl_attempts = %s, crawl_last_attempt = now() "
            f"WHERE id IN ({_LOCKED_UNFILLED_COPIES})",
            (_MAX_CRAWL_ATTEMPTS, filename),
        )
        conn.commit()
    finally:
        conn.close()


def _write_result_sync(db_path, stub: dict, title: str, markdown: str) -> None:
    """Persist the fetched content once PER USER that has unfilled copies of
    this URL, then resolve each user's remaining duplicates out of the queue
    without content (the identical page shouldn't be embedded once per
    linking message — duplicate vectors + reindex churn).

    Per-user because search indexes are per-user: until 2026-07-08 this
    filled ONE global representative and capped every other user's copies,
    so only one user's index ever saw the page."""
    from gmail_search.store.queries import _MAX_CRAWL_ATTEMPTS  # noqa: PLC0415

    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT id, user_id FROM attachments"
            " WHERE filename = %s AND extracted_text IS NULL"
            " ORDER BY id FOR UPDATE",
            (stub["filename"],),
        ).fetchall()
        reps: dict = {}
        for r in rows:
            reps.setdefault(r["user_id"], r["id"])
        # The stub we actually fetched stays its owner's representative.
        stub_owner = next((r["user_id"] for r in rows if r["id"] == stub["id"]), None)
        if stub_owner is not None:
            reps[stub_owner] = stub["id"]
        for rep_id in reps.values():
            fill_url_attachment(conn, attachment_id=rep_id, title=title, text=markdown, url=stub["url"])
        dup_ids = [r["id"] for r in rows if r["id"] not in set(reps.values())]
        if dup_ids:
            conn.execute(
                "UPDATE attachments SET crawl_attempts = %s, crawl_last_attempt = now() " "WHERE id = ANY(%s)",
                (_MAX_CRAWL_ATTEMPTS, dup_ids),
            )
        conn.commit()
    finally:
        conn.close()


async def _persist_result(db_path, stub: dict, result) -> bool:
    """Write a (title, markdown) result for `stub`, or return False on a None
    result / DB error. Shared by the batch and continuous orchestrators."""
    if result is None:
        return False
    title, markdown = result
    try:
        await asyncio.to_thread(_write_result_sync, db_path, stub, title, markdown)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"url_fetcher: db write failed for {stub['url']}: {e}")
        return False
    return True


async def _process_one(
    crawler,
    stub: dict,
    sem: asyncio.Semaphore,
    db_path: Path,
    timeout_s: float,
    http_client=None,
    cffi_session=None,
    browser_sem: asyncio.Semaphore | None = None,
) -> bool:
    """Fetch a single stub and write the result. Returns True on success.

    Two-pool scheduling: the HTTP resolution phase runs under `sem` (large —
    HTTP is I/O-bound and cheap), then the slot is RELEASED before the Chromium
    render acquires the separate, small `browser_sem`. When `browser_sem` is
    None the render runs under `sem` (single-pool; tests / one-off callers).

    NOTE: this is the BATCH path (one task per stub, awaited together in
    `run`). The daemon uses `run_continuous` instead, which keeps the HTTP pool
    saturated across the slow browser tail; this is kept for tests + one-shots.
    """
    async with sem:
        # Denylisted URLs are filtered upstream in pending_url_stubs; here we
        # only abandon URLs that can never resolve (private IP / NXDOMAIN).
        if not await asyncio.to_thread(_ssrf_guard, stub["url"]):
            await asyncio.to_thread(_abandon_sync, db_path, stub["filename"])
            return False
        # Stamp the attempt BEFORE fetching so a timeout / block still counts
        # against the retry budget (else a failing link is re-crawled forever).
        await asyncio.to_thread(_mark_attempt_sync, db_path, stub["filename"])
        kind, data = await resolve_via_http(
            stub["url"], timeout_s=timeout_s, http_client=http_client, cffi_session=cffi_session
        )

    if kind == "ok":
        return await _persist_result(db_path, stub, data)
    if kind == "fail":
        return False
    # Browser phase — HTTP slot released; acquire the small browser pool.
    bsem = browser_sem if browser_sem is not None else sem
    async with bsem:
        result = await _fetch_via_crawl4ai(crawler, data, timeout_s)
    return await _persist_result(db_path, stub, result)


async def run(
    db_path: Path,
    *,
    concurrency: int = 3,
    limit: int | None = None,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> dict[str, int]:
    """Fetch pending URL stubs, fill `attachments.extracted_text`.

    Returns `{total, done, failed}` so the CLI can report progress.
    Uses a single shared `AsyncWebCrawler` to keep Chromium hot.
    """
    from crawl4ai import AsyncWebCrawler, BrowserConfig

    conn = get_connection(db_path)
    try:
        pending = pending_url_stubs(conn, limit or 10_000)
    finally:
        conn.close()

    total = len(pending)
    if total == 0:
        return {"total": 0, "done": 0, "failed": 0}

    progress = JobProgress(db_path, "crawl_urls")
    progress.update("crawling", 0, total, f"0/{total}")

    # Two pools (see _process_one): a LARGE HTTP pool (I/O-bound, ~no memory)
    # and a SMALL browser pool (Chromium is CPU/memory-heavy). Measured: HTTP
    # throughput ~doubles from concurrency 8→16 with flat latency, then
    # plateaus; the browser stays bounded so it can't starve HTTP or OOM the
    # box. `concurrency` sizes the HTTP pool; the browser cap is small and
    # fixed (overridable via $GMAIL_CRAWL_BROWSER_CONCURRENCY).
    sem = asyncio.Semaphore(max(1, concurrency))
    _browser_cap = max(1, int(os.environ.get("GMAIL_CRAWL_BROWSER_CONCURRENCY", "3")))
    browser_sem = asyncio.Semaphore(min(_browser_cap, max(1, concurrency)))
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        light_mode=True,
        text_mode=True,
        proxy_config=_crawl_proxy_config(),  # None → direct; route browser tier through egress
        extra_args=[
            # Chromium refuses to run as root inside containers / some
            # CI — these flags match what crawl4ai's own docs recommend.
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu",
        ],
    )

    done = 0
    failed = 0
    http_client = build_http_client(timeout_s)
    cffi_session = build_cffi_session()  # primary engine; None → httpx-only
    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            tasks = [
                asyncio.create_task(
                    _process_one(
                        crawler,
                        stub,
                        sem,
                        db_path,
                        timeout_s,
                        http_client=http_client,
                        cffi_session=cffi_session,
                        browser_sem=browser_sem,
                    )
                )
                for stub in pending
            ]
            for coro in asyncio.as_completed(tasks):
                ok = await coro
                if ok:
                    done += 1
                else:
                    failed += 1
                if (done + failed) % 5 == 0 or (done + failed) == total:
                    progress.update(
                        "crawling",
                        done + failed,
                        total,
                        f"{done} ok / {failed} failed",
                    )
    except Exception as e:
        logger.exception(f"url_fetcher.run crashed: {e}")
        progress.finish("error", f"{type(e).__name__}: {str(e)[:120]}")
        raise
    finally:
        await http_client.aclose()
        if cffi_session is not None:
            try:
                await cffi_session.close()
            except Exception:  # noqa: BLE001 — best-effort cleanup
                pass

    progress.finish("done", f"{done} ok / {failed} failed of {total}")
    return {"total": total, "done": done, "failed": failed}


async def run_continuous(
    db_path: Path,
    *,
    http_concurrency: int = 16,
    browser_cap: int = 3,
    target: int = 500,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> dict[str, int]:
    """Continuous worker-pool crawl — the daemon's path. Unlike `run` (which
    creates one task per stub and BARRIERS on the whole batch, so the HTTP
    pool sits idle ~75% of the wall while the slow browser tail finishes),
    this keeps the HTTP pool SATURATED: a producer refills a queue from the
    DB, `http_concurrency` HTTP workers pull continuously, and stubs that need
    a render are handed to `browser_cap` background browser workers WITHOUT
    blocking HTTP. The only barrier is the final drain of the last few renders
    — negligible amortised over `target` stubs.

    Returns `{total, done, failed}`. `target` is a soft cap on stubs pulled
    this call (the CLI loops it, refreshing progress + the memory-aware browser
    cap between calls)."""
    from crawl4ai import AsyncWebCrawler, BrowserConfig

    progress = JobProgress(db_path, "crawl_urls")
    counters = {"done": 0, "failed": 0, "pulled": 0}
    # Bounded queues give natural backpressure: the producer blocks when the
    # HTTP pool is saturated, so we never pull more than we can chew.
    stub_q: asyncio.Queue = asyncio.Queue(maxsize=http_concurrency * 2)
    browser_q: asyncio.Queue = asyncio.Queue(maxsize=max(8, browser_cap * 4))
    http_client = build_http_client(timeout_s)
    cffi_session = build_cffi_session()

    def _fetch_pending(n: int):
        conn = get_connection(db_path)
        try:
            return pending_url_stubs(conn, n)
        finally:
            conn.close()

    async def _producer():
        """Refill `stub_q` from the DB until `target` stubs are enqueued. A
        marked stub (attempts+1, last_attempt=now) is excluded from the next
        `pending_url_stubs` by its backoff window, so we don't re-pull — but we
        pull a fresh DB chunk only when the queue is low, giving workers time
        to stamp attempts first."""
        while counters["pulled"] < target:
            want = min(http_concurrency * 2, target - counters["pulled"])
            rows = await asyncio.to_thread(_fetch_pending, want)
            if not rows:
                break  # backlog drained
            for stub in rows:
                if counters["pulled"] >= target:
                    break
                await stub_q.put(stub)
                counters["pulled"] += 1

    async def _http_worker():
        while True:
            stub = await stub_q.get()
            try:
                if stub is None:
                    return  # poison pill
                if not await asyncio.to_thread(_ssrf_guard, stub["url"]):
                    await asyncio.to_thread(_abandon_sync, db_path, stub["filename"])
                    counters["failed"] += 1
                    continue
                await asyncio.to_thread(_mark_attempt_sync, db_path, stub["filename"])
                kind, data = await resolve_via_http(
                    stub["url"], timeout_s=timeout_s, http_client=http_client, cffi_session=cffi_session
                )
                if kind == "ok":
                    counters["done" if await _persist_result(db_path, stub, data) else "failed"] += 1
                elif kind == "fail":
                    counters["failed"] += 1
                else:
                    await browser_q.put((stub, data))  # hand off; do NOT block HTTP
            except Exception as e:  # noqa: BLE001 — one bad stub must not kill the worker
                logger.warning(f"url_fetcher: http worker error on {stub.get('url') if stub else '?'}: {e}")
                counters["failed"] += 1
            finally:
                stub_q.task_done()

    async def _browser_worker(crawler):
        while True:
            item = await browser_q.get()
            try:
                if item is None:
                    return  # poison pill
                # Broad guard: a browser worker must NEVER die on a bad item —
                # if all browser workers died, HTTP workers would block forever
                # on `browser_q.put(...)` during shutdown (codex audit finding).
                try:
                    stub, burl = item
                    try:
                        result = await _fetch_via_crawl4ai(crawler, burl, timeout_s)
                    except Exception as e:  # noqa: BLE001
                        logger.warning(f"url_fetcher: browser worker error on {burl}: {e}")
                        result = None
                    counters["done" if await _persist_result(db_path, stub, result) else "failed"] += 1
                except Exception as e:  # noqa: BLE001 — never let the worker die
                    logger.warning(f"url_fetcher: browser worker loop error: {e}")
                    counters["failed"] += 1
            finally:
                browser_q.task_done()

    async def _progress_ticker():
        while True:
            await asyncio.sleep(2.0)
            n = counters["done"] + counters["failed"]
            progress.update("crawling", n, target, f"{counters['done']} ok / {counters['failed']} failed")

    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        light_mode=True,
        text_mode=True,
        proxy_config=_crawl_proxy_config(),  # None → direct; route browser tier through egress
        extra_args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
    )
    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            http_workers = [asyncio.create_task(_http_worker()) for _ in range(max(1, http_concurrency))]
            browser_workers = [asyncio.create_task(_browser_worker(crawler)) for _ in range(max(1, browser_cap))]
            ticker = asyncio.create_task(_progress_ticker())

            await _producer()
            # Drain HTTP: poison each worker, wait for the queue + tasks to finish.
            for _ in http_workers:
                await stub_q.put(None)
            await asyncio.gather(*http_workers)
            # Now drain the browser tail (the only barrier, amortised over target).
            for _ in browser_workers:
                await browser_q.put(None)
            await asyncio.gather(*browser_workers)
            ticker.cancel()
    except Exception as e:
        logger.exception(f"url_fetcher.run_continuous crashed: {e}")
        progress.finish("error", f"{type(e).__name__}: {str(e)[:120]}")
        raise
    finally:
        await http_client.aclose()
        if cffi_session is not None:
            try:
                await cffi_session.close()
            except Exception:  # noqa: BLE001
                pass

    total = counters["done"] + counters["failed"]
    progress.finish("done", f"{counters['done']} ok / {counters['failed']} failed of {total}")
    return {"total": total, "done": counters["done"], "failed": counters["failed"]}


def _self_test() -> None:
    """Ad-hoc smoke test. Run with `python -m gmail_search.gmail.url_fetcher`.
    Fetches a known-good URL and prints the Markdown so we can sanity-
    check that crawl4ai is wired up correctly on this machine.
    """
    import asyncio as _aio

    from crawl4ai import AsyncWebCrawler, BrowserConfig

    async def _go() -> None:
        browser_config = BrowserConfig(
            headless=True,
            verbose=False,
            light_mode=True,
            text_mode=True,
            proxy_config=_crawl_proxy_config(),  # None → direct; route browser tier through egress
            extra_args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
        )
        async with AsyncWebCrawler(config=browser_config) as crawler:
            r = await fetch_url_markdown(crawler, "https://example.com", timeout_s=30.0)
            if r is None:
                print("FETCH FAILED")
                return
            title, markdown = r
            print(f"TITLE: {title!r}")
            print("--- MARKDOWN ---")
            print(markdown)
            print("--- END ---")

    _aio.run(_go())


if __name__ == "__main__":
    _self_test()
