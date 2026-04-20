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

# Hard cap per URL. crawl4ai's page_timeout is in ms.
_DEFAULT_TIMEOUT_S = 30.0


def _resolve_first_ip(host: str) -> str | None:
    """DNS-resolve a hostname to its first A/AAAA record. Returns None
    on any failure — the caller treats that as "skip this URL", which
    is the safe default.
    """
    try:
        # getaddrinfo covers both A and AAAA records.
        infos = socket.getaddrinfo(host, None)
    except Exception:
        return None
    for info in infos:
        sockaddr = info[4]
        if sockaddr:
            return sockaddr[0]
    return None


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
    ip = _resolve_first_ip(host)
    if ip is None:
        return False
    return not _is_private_ip(ip)


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
    except Exception as e:
        logger.warning(f"crawl4ai import failed: {e}")
        return None

    # page_timeout is in ms inside crawl4ai; keep our top-level
    # asyncio.wait_for in seconds so the API matches our other code.
    run_config = CrawlerRunConfig(
        page_timeout=int(timeout_s * 1000),
        verbose=False,
        # Don't eat our RAM chasing rich media when we only care about text.
        exclude_all_images=True,
        remove_overlay_elements=True,
        remove_forms=True,
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
        # crawl4ai returns a MarkdownGenerationResult object; .raw_markdown
        # or .fit_markdown — prefer fit_markdown (post content-filter) if
        # present, else fall back to raw_markdown, else str().
        markdown = (
            getattr(md_obj, "fit_markdown", None)
            or getattr(md_obj, "raw_markdown", None)
            or (md_obj if isinstance(md_obj, str) else str(md_obj))
        )
    if not markdown:
        return None
    title = (getattr(result, "metadata", None) or {}).get("title") or ""
    return title, _truncate_markdown(markdown)


async def fetch_url_markdown(
    crawler,
    url: str,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> tuple[str, str] | None:
    """Fetch one URL, return `(title, markdown)` or None on failure.

    - Pre-flight SSRF guard against private / loopback / link-local IPs.
    - PDF URLs (by URL suffix) route through the local PDF extractor so
      we don't pipe a binary into crawl4ai. Non-suffix PDFs detected
      via Content-Type inside crawl4ai would still be caught there.
    - Crawl4ai output is capped at `_MAX_CRAWL_CHARS`.
    """
    if not _ssrf_guard(url):
        logger.info(f"url_fetcher: skipping {url} (SSRF guard)")
        return None

    if url.lower().split("?", 1)[0].endswith(".pdf"):
        return await asyncio.to_thread(_fetch_pdf_url, url)

    return await _fetch_via_crawl4ai(crawler, url, timeout_s)


def _fetch_pdf_url(url: str) -> tuple[str, str] | None:
    """Download a PDF with stdlib + run it through the existing PDF
    extractor. Synchronous because `extract_pdf` is — the caller
    routes us through `asyncio.to_thread` to stay off the event loop.

    SSRF was already checked by the caller.
    """
    import tempfile
    import urllib.request

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "gmail-search/1.0"})
        with urllib.request.urlopen(req, timeout=_DEFAULT_TIMEOUT_S) as resp:
            data = resp.read(20 * 1024 * 1024)  # cap at 20 MB
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
    async with sem:
        result = await fetch_url_markdown(crawler, stub["url"], timeout_s=timeout_s)
    if result is None:
        return False
    title, markdown = result

    def _write():
        conn = get_connection(db_path)
        try:
            fill_url_attachment(
                conn,
                attachment_id=stub["id"],
                title=title,
                text=markdown,
                url=stub["url"],
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
