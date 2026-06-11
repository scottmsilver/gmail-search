"""Per-stage crawl latency profiler — the measurement half of the
measure → fix → measure loop for per-page crawl speed.

Single URL (the anchor case):
    python -m gmail_search.gmail.crawl_profile <url> [--repeat 3] [--browser]

Sample of pending stubs (same instrument, scaled up):
    python -m gmail_search.gmail.crawl_profile --sample 16

Stages reported per URL:
    guard   — _ssrf_guard (DNS resolve + private-IP check)
    headers — request sent → status/headers available (TCP+TLS+TTFB)
    body    — streamed, capped body read
    parse   — _readable_text extraction
    real    — one uninstrumented `_fetch_via_http` call on the same URL,
              so the stage breakdown can be sanity-checked against the
              path production actually runs. If real ≉ guard+http stages,
              the instrument is lying — fix it before trusting it.

`--browser` additionally times the Chromium fallback on the URL, so the
cost of a "browser" outcome is a measured number, not folklore.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from urllib.parse import urljoin

from gmail_search.gmail import url_fetcher as uf


async def profile_url(url: str, client) -> dict:
    """Stage-timed fetch of one URL via url_fetcher's building blocks,
    plus an uninstrumented `_fetch_via_http` for cross-checking."""
    out: dict = {"url": url}

    t0 = time.perf_counter()
    guard_ok = await asyncio.to_thread(uf._ssrf_guard, url)
    out["guard_ms"] = (time.perf_counter() - t0) * 1000
    if not guard_ok:
        out["outcome"] = "ssrf/dead-dns"
        return out

    cur, hops = url, 0
    try:
        for _ in range(uf._MAX_REDIRECTS):
            if hops and not await asyncio.to_thread(uf._ssrf_guard, cur):
                out["outcome"] = "ssrf-redirect"
                return out
            t0 = time.perf_counter()
            async with client.stream("GET", cur) as resp:
                out["headers_ms"] = out.get("headers_ms", 0.0) + (time.perf_counter() - t0) * 1000
                if resp.status_code in uf._REDIRECT_CODES:
                    loc = resp.headers.get("location")
                    if not loc:
                        out["outcome"] = "fail:3xx-no-location"
                        return out
                    cur = urljoin(cur, loc)
                    hops += 1
                    continue
                out["status"] = resp.status_code
                out["hops"] = hops
                if resp.status_code == 403:
                    out["outcome"] = "browser:403"
                    return out
                if resp.status_code != 200 or "html" not in resp.headers.get("content-type", "").lower():
                    out["outcome"] = f"fail:{resp.status_code}"
                    return out
                t0 = time.perf_counter()
                html = await uf._read_capped(resp)
                out["body_ms"] = (time.perf_counter() - t0) * 1000
                out["body_kb"] = len(html) / 1024
                break
        else:
            out["outcome"] = "fail:redirect-loop"
            return out
    except Exception as e:
        out["outcome"] = f"fail:{type(e).__name__}"
        return out

    t0 = time.perf_counter()
    text, _title = uf._readable_text(html[: uf._MAX_HTML_CHARS])
    out["parse_ms"] = (time.perf_counter() - t0) * 1000
    out["text_chars"] = len(text)
    out["outcome"] = "ok" if len(text) >= uf._MIN_USABLE_CHARS else "browser:thin"
    return out


async def cross_check(url: str, client) -> float:
    """Wall time of the REAL production fetch on the same URL."""
    t0 = time.perf_counter()
    try:
        await uf._fetch_via_http(url, uf._DEFAULT_TIMEOUT_S, client=client)
    except uf._SSRFBlocked:
        pass
    return (time.perf_counter() - t0) * 1000


async def profile_browser(url: str) -> float:
    """Measured cost of the Chromium fallback for this URL."""
    from crawl4ai import AsyncWebCrawler, BrowserConfig

    config = BrowserConfig(
        headless=True,
        verbose=False,
        light_mode=True,
        text_mode=True,
        extra_args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
    )
    t0 = time.perf_counter()
    async with AsyncWebCrawler(config=config) as crawler:
        t_open = time.perf_counter()
        await uf._fetch_via_crawl4ai(crawler, url, uf._DEFAULT_TIMEOUT_S)
        t_fetch = time.perf_counter()
    print(f"  browser: startup {(t_open - t0) * 1000:.0f}ms  fetch {(t_fetch - t_open) * 1000:.0f}ms")
    return (t_fetch - t0) * 1000


def _fmt_row(p: dict) -> str:
    stages = "  ".join(
        f"{k.split('_')[0]}={p[k]:.0f}ms" for k in ("guard_ms", "headers_ms", "body_ms", "parse_ms") if k in p
    )
    extra = f"  real={p['real_ms']:.0f}ms" if "real_ms" in p else ""
    return f"{p['outcome']:<18} {stages}{extra}  {p['url'][:70]}"


async def _run_single(url: str, repeat: int, with_browser: bool) -> None:
    client = uf.build_http_client()
    try:
        for i in range(repeat):
            p = await profile_url(url, client)
            p["real_ms"] = await cross_check(url, client)
            label = "cold" if i == 0 else f"warm{i}"
            print(f"[{label}] {_fmt_row(p)}")
    finally:
        await client.aclose()
    if with_browser:
        await profile_browser(url)


async def _run_sample(n: int) -> None:
    from gmail_search.store.db import get_connection

    conn = get_connection(None)
    rows = conn.execute(
        """
        SELECT substring(filename from 6) AS url FROM attachments
        WHERE filename LIKE 'URL: %%' AND extracted_text IS NULL AND crawl_attempts = 0
        ORDER BY random() LIMIT %s
        """,
        (n,),
    ).fetchall()
    conn.close()

    client = uf.build_http_client()
    profiles = []
    try:
        for r in rows:
            p = await profile_url(r["url"], client)
            profiles.append(p)
            print(_fmt_row(p))
    finally:
        await client.aclose()

    print(f"\n— aggregate over {len(profiles)} URLs —")
    outcomes: dict[str, int] = {}
    for p in profiles:
        key = p["outcome"].split(":")[0]
        outcomes[key] = outcomes.get(key, 0) + 1
    print(f"outcomes: {outcomes}")
    for stage in ("guard_ms", "headers_ms", "body_ms", "parse_ms"):
        vals = [p[stage] for p in profiles if stage in p]
        if vals:
            print(
                f"{stage:<11} median {statistics.median(vals):>6.0f}ms   p90 {sorted(vals)[max(0, int(0.9 * len(vals)) - 1)]:>6.0f}ms   max {max(vals):>6.0f}ms   (n={len(vals)})"
            )


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("url", nargs="?", help="single URL to profile (the anchor case)")
    ap.add_argument("--repeat", type=int, default=3, help="repetitions for the single-URL case")
    ap.add_argument("--sample", type=int, help="profile N random pending stubs instead")
    ap.add_argument("--browser", action="store_true", help="also time the Chromium fallback")
    args = ap.parse_args()

    if args.sample:
        asyncio.run(_run_sample(args.sample))
    elif args.url:
        asyncio.run(_run_single(args.url, args.repeat, args.browser))
    else:
        ap.error("give a URL or --sample N")


if __name__ == "__main__":
    main()
