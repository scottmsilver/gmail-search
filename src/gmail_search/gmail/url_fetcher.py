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
import socket
from pathlib import Path
from urllib.parse import urlparse

from gmail_search.store.db import JobProgress, get_connection
from gmail_search.store.queries import fill_url_attachment, pending_url_stubs

logger = logging.getLogger(__name__)

# Keep crawled content bounded so downstream prompts don't blow up.
# 8000 chars ≈ 2000 tokens — enough for a summarizer to get the gist.
_MAX_CRAWL_CHARS = 8000

# Hard cap per URL. crawl4ai's page_timeout is in ms. Kept short: most
# uncrawled stubs are old dead / anti-bot links that would otherwise burn the
# full timeout each; a live page loads well under this. (Was 30s — far too
# long when a cycle has hundreds of dead links to try.)
_DEFAULT_TIMEOUT_S = 10.0

# Cap for the PDF-routed path. Binary download, not char count.
_MAX_PDF_BYTES = 20 * 1024 * 1024


def _resolve_all_ips(host: str) -> list[str]:
    """DNS-resolve a hostname to every A / AAAA record. We reject on
    ANY private IP — if an attacker controls DNS they could return
    `[public-ip, 127.0.0.1]` and win a race with Chromium's own
    resolver, so checking just the first record isn't enough.

    Returns `[]` on resolution failure; the caller treats that as
    "skip this URL" (fail closed).
    """
    try:
        # getaddrinfo covers both A and AAAA records.
        infos = socket.getaddrinfo(host, None)
    except Exception:
        return []
    out: list[str] = []
    for info in infos:
        sockaddr = info[4]
        if sockaddr:
            out.append(sockaddr[0])
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


_HTTP_UA = "Mozilla/5.0 (compatible; gmail-search/1.0; +https://oursilverfamily.com)"
_MIN_USABLE_CHARS = 250  # below this we assume a JS shell / empty page → browser
_MAX_HTML_CHARS = 3_000_000  # cap parsed HTML so a giant page can't blow RAM
_MAX_HTML_BYTES = 20_000_000  # reject responses larger than this (Content-Length)
_MAX_REDIRECTS = 6


class _SSRFBlocked(Exception):
    """A redirect hop resolved to a blocked (private/loopback/link-local)
    target. Raised so the caller does NOT fall back to crawl4ai, which would
    follow the same attacker-controlled redirect WITHOUT the per-hop guard."""


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
    for tag in soup(
        ["script", "style", "noscript", "template", "svg", "head", "nav", "footer", "header", "form", "aside", "iframe"]
    ):
        tag.decompose()
    main = soup.find("article") or soup.find("main") or soup.find(attrs={"role": "main"}) or soup.body or soup
    return main.get_text(separator="\n", strip=True), title


async def _fetch_via_http(url: str, timeout_s: float) -> tuple[str, str] | None:
    """Browser-free fetch with manual redirect-following. Each hop is SSRF-
    checked BEFORE we fetch it (the caller's guard only covered the first URL),
    so a redirect can't smuggle us to a private IP. Returns (title, text) on a
    usable HTML page, else None → caller falls back to crawl4ai for JS pages."""
    from urllib.parse import urljoin  # noqa: PLC0415

    import httpx  # noqa: PLC0415

    cur = url
    try:
        async with httpx.AsyncClient(
            follow_redirects=False, timeout=min(timeout_s, 10.0), headers={"User-Agent": _HTTP_UA}
        ) as client:
            for _ in range(_MAX_REDIRECTS):
                if not _ssrf_guard(cur):
                    # Blocked target: raise (don't return None) so the caller
                    # won't hand the original URL to crawl4ai, which would
                    # follow this same redirect with no guard.
                    raise _SSRFBlocked(cur)
                resp = await client.get(cur)
                if resp.status_code in (301, 302, 303, 307, 308):
                    loc = resp.headers.get("location")
                    if not loc:
                        return None
                    cur = urljoin(cur, loc)
                    continue
                break
            else:
                return None  # redirect loop / too many hops
    except _SSRFBlocked:
        raise
    except Exception:
        return None
    if resp.status_code != 200 or "html" not in resp.headers.get("content-type", "").lower():
        return None
    try:
        if int(resp.headers.get("content-length", "0")) > _MAX_HTML_BYTES:
            return None  # oversized — don't buffer/parse it
    except ValueError:
        pass
    text, title = _readable_text(resp.text[:_MAX_HTML_CHARS])
    if len(text) < _MIN_USABLE_CHARS:
        return None  # JS shell / near-empty → let the browser try
    return title, _truncate_markdown(text)


async def fetch_url_markdown(
    crawler,
    url: str,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> tuple[str, str] | None:
    """Fetch one URL, return `(title, markdown)` or None on failure.

    - Pre-flight SSRF guard against private / loopback / link-local IPs.
    - PDF URLs (by URL suffix) route through the local PDF extractor.
    - HTTP-FIRST: a redirect-following httpx GET + readability extraction
      handles server-rendered pages (incl. tracking redirects) with no browser
      — far faster, no Chromium memory, no slot-holding. crawl4ai is the
      fallback for JS-rendered / empty pages.

    Known SSRF limitations (documented, accepted for this self-hosted app —
    fetched content lands only in the user's own private index, never back to
    an attacker):
      * crawl4ai (Chromium) fallback follows redirects itself, unguarded —
        pre-existing. The HTTP-first path (the common case) IS guarded per hop,
        and a guard-rejected redirect now short-circuits before crawl4ai.
      * _ssrf_guard resolves DNS separately from the connect, so DNS-rebinding
        (TOCTOU) can still slip past. Fully closing it needs connect-time IP
        pinning; tracked as a follow-up.
    """
    if not _ssrf_guard(url):
        logger.info(f"url_fetcher: skipping {url} (SSRF guard)")
        return None

    if url.lower().split("?", 1)[0].endswith(".pdf"):
        return await asyncio.to_thread(_fetch_pdf_url, url)

    try:
        light = await _fetch_via_http(url, timeout_s)
    except _SSRFBlocked as e:
        # A redirect pointed at a blocked address — give up; do NOT fall back
        # to crawl4ai (Chromium would follow the same redirect unguarded).
        logger.info(f"url_fetcher: skipping {url} (SSRF redirect target {e})")
        return None
    if light is not None:
        return light
    return await _fetch_via_crawl4ai(crawler, url, timeout_s)


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


async def _process_one(
    crawler,
    stub: dict,
    sem: asyncio.Semaphore,
    db_path: Path,
    timeout_s: float,
) -> bool:
    """Fetch a single stub and write the result. Returns True on
    success (attachment filled), False otherwise.

    Commits its own transaction so a slow crawl can't hold a writer
    lock across concurrent fetches.
    """

    def _mark_attempt():
        conn = get_connection(db_path)
        try:
            # Mark ALL unfilled copies of this URL (dedup fan-out) so duplicate
            # stubs back off / abandon together instead of each being retried.
            conn.execute(
                "UPDATE attachments SET crawl_attempts = crawl_attempts + 1, "
                "crawl_last_attempt = now() WHERE filename = %s AND extracted_text IS NULL",
                (stub["filename"],),
            )
            conn.commit()
        finally:
            conn.close()

    def _abandon():
        # PERMANENT failure (URL resolves to a private/reserved IP, or doesn't
        # resolve at all — a dead domain, very common in old mail). It will
        # NEVER succeed, so jump all copies straight to the retry cap instead
        # of burning 4 Chromium fetches over ~6h of backoff. _ssrf_guard fails
        # closed on NXDOMAIN, so this also clears dead links cheaply (DNS only,
        # no browser).
        # Inline import: the formatter strips this from module scope as
        # "unused" (it's only referenced in this nested fn), so import it here.
        from gmail_search.store.queries import _MAX_CRAWL_ATTEMPTS  # noqa: PLC0415

        conn = get_connection(db_path)
        try:
            conn.execute(
                "UPDATE attachments SET crawl_attempts = %s, crawl_last_attempt = now() "
                "WHERE filename = %s AND extracted_text IS NULL",
                (_MAX_CRAWL_ATTEMPTS, stub["filename"]),
            )
            conn.commit()
        finally:
            conn.close()

    async with sem:
        # Permanent-failure fast path: skip the browser entirely for URLs that
        # can never succeed and abandon them in one DNS check.
        if not await asyncio.to_thread(_ssrf_guard, stub["url"]):
            await asyncio.to_thread(_abandon)
            return False
        # Stamp the attempt BEFORE fetching so a timeout / crash / anti-bot
        # block still counts against the retry budget (_MAX_CRAWL_ATTEMPTS).
        # Otherwise a perpetually-failing link never accrues attempts and is
        # re-crawled every cycle forever — the head-of-line blocking that left
        # 517k stubs thrashing while live URLs starved.
        await asyncio.to_thread(_mark_attempt)
        result = await fetch_url_markdown(crawler, stub["url"], timeout_s=timeout_s)
    if result is None:
        return False
    title, markdown = result

    def _write():
        from gmail_search.store.queries import _MAX_CRAWL_ATTEMPTS  # noqa: PLC0415

        conn = get_connection(db_path)
        try:
            # Embed THIS copy of the URL once...
            fill_url_attachment(
                conn,
                attachment_id=stub["id"],
                title=title,
                text=markdown,
                url=stub["url"],
            )
            # ...and resolve the duplicate stubs of the same URL out of the
            # crawl queue WITHOUT content, so the identical page isn't embedded
            # again for every message that links it (avoids thousands of
            # duplicate vectors + the reindex churn that creates). They still
            # match by the original `URL: <url>` filename; the one just filled
            # was renamed + has extracted_text, so it's excluded here.
            conn.execute(
                "UPDATE attachments SET crawl_attempts = %s, crawl_last_attempt = now() "
                "WHERE filename = %s AND extracted_text IS NULL",
                (_MAX_CRAWL_ATTEMPTS, stub["filename"]),
            )
            conn.commit()
        finally:
            conn.close()

    try:
        await asyncio.to_thread(_write)
    except Exception as e:
        logger.warning(f"url_fetcher: db write failed for {stub['url']}: {e}")
        return False
    return True


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

    sem = asyncio.Semaphore(max(1, concurrency))
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        light_mode=True,
        text_mode=True,
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
    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            tasks = [asyncio.create_task(_process_one(crawler, stub, sem, db_path, timeout_s)) for stub in pending]
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

    progress.finish("done", f"{done} ok / {failed} failed of {total}")
    return {"total": total, "done": done, "failed": failed}


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
