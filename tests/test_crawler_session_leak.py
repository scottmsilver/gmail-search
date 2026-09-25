"""The crawler session must reclaim the Playwright driver when STARTUP fails.

Regression test for the 2026-09-15 outage: a Playwright upgrade left
chromium-1208 undownloaded, so `AsyncWebCrawler.__aenter__` spawned the node
driver subprocess and *then* raised at `BrowserType.launch`. Python does not
call `__aexit__` when `__aenter__` raises, so each of the crawl daemon's
2-second retries orphaned a driver holding two pipe fds. 504 of them
exhausted the process's 1024-fd limit in ~19 minutes and wedged crawling for
four days.
"""

import pytest

from gmail_search.gmail import url_fetcher


class _FakeCrawler:
    """Mimics crawl4ai: `start()` spawns the driver, then may fail."""

    def __init__(self, *, fail_on_start: bool):
        self._fail_on_start = fail_on_start
        self.driver_spawned = False
        self.closed = False

    async def start(self):
        self.driver_spawned = True  # the node subprocess now exists
        if self._fail_on_start:
            raise RuntimeError("BrowserType.launch: Executable doesn't exist at chromium-1208")
        return self

    async def close(self):
        self.closed = True

    # crawl4ai's own protocol: __aenter__ delegates to start(), __aexit__ to
    # close(). Modelling it exactly is what makes this test able to catch the
    # bug — under `async with`, a start() failure skips __aexit__ entirely.
    async def __aenter__(self):
        return await self.start()

    async def __aexit__(self, *_exc):
        await self.close()
        return False


@pytest.fixture
def fake_crawler(monkeypatch):
    """Install a fake AsyncWebCrawler; returns a factory keyed on failure mode."""
    made = []

    def _install(*, fail_on_start):
        import crawl4ai

        def _factory(*_args, **_kwargs):
            c = _FakeCrawler(fail_on_start=fail_on_start)
            made.append(c)
            return c

        monkeypatch.setattr(crawl4ai, "AsyncWebCrawler", _factory)
        monkeypatch.setattr(url_fetcher, "_browser_config", lambda: object())
        return made

    return _install


@pytest.mark.asyncio
async def test_driver_is_closed_when_start_fails(fake_crawler):
    made = fake_crawler(fail_on_start=True)

    with pytest.raises(RuntimeError, match="BrowserType.launch"):
        async with url_fetcher._crawler_session():
            pytest.fail("body must not run when startup fails")

    assert len(made) == 1
    assert made[0].driver_spawned, "precondition: the driver was spawned before the failure"
    assert made[0].closed, "startup failure leaked the Playwright driver — the 2026-09-15 fd exhaustion"


@pytest.mark.asyncio
async def test_driver_is_closed_on_the_happy_path(fake_crawler):
    made = fake_crawler(fail_on_start=False)

    async with url_fetcher._crawler_session() as crawler:
        assert crawler.driver_spawned

    assert made[0].closed


@pytest.mark.asyncio
async def test_driver_is_closed_when_the_body_raises(fake_crawler):
    made = fake_crawler(fail_on_start=False)

    with pytest.raises(ValueError):
        async with url_fetcher._crawler_session():
            raise ValueError("crawl blew up mid-batch")

    assert made[0].closed


@pytest.mark.asyncio
async def test_repeated_startup_failures_do_not_accumulate_drivers(fake_crawler):
    """The daemon retries forever; every attempt must clean up after itself."""
    made = fake_crawler(fail_on_start=True)

    for _ in range(50):
        with pytest.raises(RuntimeError):
            async with url_fetcher._crawler_session():
                pass

    assert len(made) == 50
    assert all(c.closed for c in made), "a retry loop still leaks drivers"


def test_crawl_backoff_grows_then_caps():
    """A doomed crawl loop must idle, not hot-spin at the 2s happy-path rate."""
    from gmail_search.cli import _CRAWL_BACKOFF_CAP_S, _crawl_backoff_s

    assert _crawl_backoff_s(1, 2) == 2
    assert _crawl_backoff_s(2, 2) == 4
    assert _crawl_backoff_s(5, 2) == 32
    assert _crawl_backoff_s(99, 2) == _CRAWL_BACKOFF_CAP_S

    # The four-day outage retried every 2s. Under backoff, one hour of a
    # permanent failure costs a couple of dozen attempts, not 1,800.
    t, attempts = 0.0, 0
    while t < 3600:
        attempts += 1
        t += _crawl_backoff_s(attempts, 2)
    assert attempts < 30, f"{attempts} attempts/hour is still a hot loop"


def test_backoff_never_beats_the_healthy_interval():
    """A failing loop must never retry faster than a healthy one."""
    from gmail_search.cli import _crawl_backoff_s

    # interval above the cap: naive min() would retry 2x faster when broken.
    assert _crawl_backoff_s(1, 600) >= 600
    # degenerate operator input must not become a hot spin or a ValueError.
    assert _crawl_backoff_s(1, 0) > 0
    assert _crawl_backoff_s(3, -5) > 0
    assert _crawl_backoff_s(0, 2) > 0
    for failures in range(0, 200):
        assert _crawl_backoff_s(failures, 2) > 0


def test_failure_detail_is_redacted_before_it_reaches_the_ui():
    """`job_progress.detail` renders in SettingsView; exceptions quote secrets."""
    from gmail_search.cli import _record_crawl_failure

    class _Progress:
        detail = None

        def finish(self, status, detail):
            self.status, self.detail = status, detail

    p = _Progress()
    _record_crawl_failure(p, RuntimeError("proxy http://crawler:s3cr3t@egress.internal:8080 refused"))

    assert p.status == "error"
    assert "s3cr3t" not in p.detail
    assert "[REDACTED]" in p.detail


def test_failure_record_survives_a_dead_database():
    """Postgres down is one of the failures this path reports — it must not
    raise out of the handler and skip the backoff sleep."""
    from gmail_search.cli import _record_crawl_failure

    class _DeadProgress:
        def finish(self, status, detail):
            raise ConnectionError("connection refused")

    _record_crawl_failure(_DeadProgress(), RuntimeError("boom"))  # must not raise


@pytest.mark.asyncio
async def test_teardown_survives_cancellation_redelivered_mid_close(monkeypatch):
    """Teardown must finish stopping the driver even if the caller is
    cancelled *again* while the close is in flight.

    One cancel is survivable for free — asyncio delivers it once, so a
    cleanup await still runs. The dangerous case is a second cancel landing
    while `close()` is awaiting: without shielding it aborts teardown before
    `playwright.stop()`, which is exactly how the driver gets orphaned.
    """
    import asyncio

    import crawl4ai

    close_started = asyncio.Event()
    closed = asyncio.Event()

    class _SlowClosingCrawler:
        async def start(self):
            return self

        async def close(self):
            close_started.set()
            await asyncio.sleep(0.05)
            closed.set()

    monkeypatch.setattr(crawl4ai, "AsyncWebCrawler", lambda *a, **k: _SlowClosingCrawler())
    monkeypatch.setattr(url_fetcher, "_browser_config", lambda: object())

    async def _use():
        async with url_fetcher._crawler_session():
            await asyncio.sleep(10)

    task = asyncio.create_task(_use())
    await asyncio.sleep(0.01)
    task.cancel()
    await asyncio.wait_for(close_started.wait(), timeout=2)
    task.cancel()  # re-deliver while teardown is mid-flight

    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.wait_for(closed.wait(), timeout=2)
    assert closed.is_set(), "re-delivered cancellation aborted teardown — the driver leaked"


@pytest.mark.asyncio
async def test_teardown_gives_up_on_a_wedged_driver_instead_of_hanging(monkeypatch):
    """Playwright's stop() can block forever on a wedged node driver."""
    import asyncio

    import crawl4ai

    class _HangingCrawler:
        async def start(self):
            return self

        async def close(self):
            await asyncio.Event().wait()  # never returns

    monkeypatch.setattr(crawl4ai, "AsyncWebCrawler", lambda *a, **k: _HangingCrawler())
    monkeypatch.setattr(url_fetcher, "_browser_config", lambda: object())
    monkeypatch.setattr(url_fetcher, "_CRAWLER_CLOSE_TIMEOUT_S", 0.05)

    async def _use():
        async with url_fetcher._crawler_session():
            pass

    await asyncio.wait_for(_use(), timeout=2)  # must not hang


def test_backoff_cannot_overflow_after_days_of_failure():
    """At the 300s cap, 1025 consecutive failures is ~85h — shorter than the
    93h outage this backoff exists for — and `float(2 ** 1024)` raises
    OverflowError from inside the daemon's own error handler."""
    from gmail_search.cli import _CRAWL_BACKOFF_CAP_S, _crawl_backoff_s

    for failures in (1024, 1025, 10_000, 1_000_000):
        assert _crawl_backoff_s(failures, 2) == _CRAWL_BACKOFF_CAP_S


@pytest.mark.asyncio
async def test_timed_out_teardown_does_not_leave_a_pending_task(monkeypatch):
    """`wait_for` cancels the shield, not the task under it — a wedged close
    left pending trips 'Task was destroyed but it is pending' when
    `asyncio.run()` tears the loop down at the end of the batch."""
    import asyncio

    import crawl4ai

    entered = asyncio.Event()

    class _HangingCrawler:
        async def start(self):
            return self

        async def close(self):
            entered.set()
            await asyncio.Event().wait()  # never returns

    monkeypatch.setattr(crawl4ai, "AsyncWebCrawler", lambda *a, **k: _HangingCrawler())
    monkeypatch.setattr(url_fetcher, "_browser_config", lambda: object())
    monkeypatch.setattr(url_fetcher, "_CRAWLER_CLOSE_TIMEOUT_S", 0.05)

    before = set(asyncio.all_tasks())

    async def _use():
        async with url_fetcher._crawler_session():
            pass

    await asyncio.wait_for(_use(), timeout=2)
    await asyncio.sleep(0.05)  # let the cancellation land

    leaked = [t for t in asyncio.all_tasks() - before if not t.done()]
    assert entered.is_set(), "precondition: the close actually started"
    assert not leaked, f"teardown left {len(leaked)} pending task(s) behind"
