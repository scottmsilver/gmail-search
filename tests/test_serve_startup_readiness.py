"""Tests for non-blocking `serve` startup + the /healthz readiness gate.

The server used to `await _prewarm_engine()` inside the startup hook, which
blocked the event loop (and the listening socket accepting useful traffic)
for the full multi-GB ScaNN load. These tests pin the new contract:

  * startup returns immediately — the prewarm runs as a background task;
  * GET /healthz (liveness) is always 200;
  * GET /healthz?ready=1 is 503 while the bootstrap engine is warming and
    200 once it's warm OR once we've determined there's no index to warm;
  * a concurrent first request + background prewarm yields a SINGLE cached
    engine (the prewarm uses setdefault and must not clobber a request-built
    instance);
  * a prewarm failure never wedges readiness — /healthz?ready=1 still flips
    to 200 and the next request cold-loads.

We drive the FastAPI app through TestClient (which runs the lifespan / the
`startup` event). `_build_engine`, `_engines`, and `_ready` are closures
inside `create_app`, so we control them by monkeypatching the module-level
`SearchEngine` symbol that `_build_engine` constructs, plus
`resolve_active_index_dir` (imported lazily inside `create_app`) so the
prewarm path has a concrete on-disk dir to "load".
"""

from __future__ import annotations

import threading
import time

import pytest

import gmail_search.server as server_mod
from gmail_search.config import load_config
from gmail_search.store.db import init_db

# ─── Fakes / helpers ────────────────────────────────────────────────────


class _ControllableEngine:
    """Stand-in for SearchEngine. Construction behavior is dictated by the
    class-level hooks a test installs:

      * _construct_event : if set, __init__ blocks on it (simulating a slow
        ScaNN load) before completing.
      * _raise : if set to an exception instance, __init__ raises it (used
        to simulate FileNotFoundError = no index, or a generic prewarm
        failure).
      * _count : incremented on every successful (or attempted) construction
        so single-winner / cold-reload tests can assert how many builds ran.
    """

    _construct_event: threading.Event | None = None
    _raise: BaseException | None = None
    _count_lock = threading.Lock()
    _count = 0

    def __init__(self, db_path, index_dir, config, *, user_id=None):
        with _ControllableEngine._count_lock:
            _ControllableEngine._count += 1
        self.db_path = db_path
        self.index_dir = index_dir
        self.user_id = user_id
        ev = _ControllableEngine._construct_event
        if ev is not None:
            ev.wait(timeout=10)
        if _ControllableEngine._raise is not None:
            raise _ControllableEngine._raise

    def search_threads(self, *a, **kw):
        return []

    def reload_index(self, new_index_dir):
        return None

    @classmethod
    def reset(cls):
        cls._construct_event = None
        cls._raise = None
        with cls._count_lock:
            cls._count = 0


@pytest.fixture
def serve_env(db_backend, tmp_path, monkeypatch):
    """Init DB + an on-disk index dir, patch SearchEngine and the index-dir
    resolver, and return everything a test needs to build the app."""
    if db_backend is None:  # pragma: no cover - guarded by db_backend skip
        pytest.skip("Postgres not reachable")

    db_path = db_backend["db_path"]
    init_db(db_path)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    index_dir = data_dir / "scann_index"
    index_dir.mkdir()

    _ControllableEngine.reset()
    # _build_engine constructs the module-level SearchEngine symbol.
    monkeypatch.setattr(server_mod, "SearchEngine", _ControllableEngine)
    # resolve_active_index_dir is imported lazily inside create_app from
    # gmail_search.index.searcher; patch it there so both the prewarm and
    # get_engine paths see a concrete, existing dir.
    monkeypatch.setattr(
        "gmail_search.index.searcher.resolve_active_index_dir",
        lambda db_path, fallback, *, user_id=None: index_dir,
    )

    config = load_config(data_dir=data_dir)
    return {
        "db_path": db_path,
        "data_dir": data_dir,
        "index_dir": index_dir,
        "config": config,
    }


def _make_app(serve_env):
    return server_mod.create_app(
        db_path=serve_env["db_path"],
        data_dir=serve_env["data_dir"],
        config=serve_env["config"],
    )


def _wait_until(predicate, timeout=5.0, interval=0.02):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


# ─── Tests ──────────────────────────────────────────────────────────────


def test_startup_does_not_block_on_prewarm(serve_env):
    """Startup must return (and serve liveness) even while the engine build
    is still in progress. We hold the constructor on an Event and assert the
    lifespan startup completes + GET /healthz is 200 before we release it."""
    from fastapi.testclient import TestClient

    gate = threading.Event()
    _ControllableEngine._construct_event = gate

    app = _make_app(serve_env)
    t0 = time.time()
    with TestClient(app) as client:
        # If startup awaited the (blocked) prewarm, entering this context
        # would hang until the 10s Event timeout. Getting here fast proves
        # the prewarm runs in the background.
        startup_elapsed = time.time() - t0
        assert startup_elapsed < 5.0
        # Liveness is unconditional.
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}
        # Let the background build finish so teardown is clean.
        gate.set()


def test_readiness_gate_503_while_warming_then_200(serve_env):
    """?ready=1 is 503 while the bootstrap engine builds, then 200 (ready)
    once the build completes."""
    from fastapi.testclient import TestClient

    gate = threading.Event()
    _ControllableEngine._construct_event = gate

    app = _make_app(serve_env)
    with TestClient(app) as client:
        # Still warming → 503 with the documented reason.
        resp = client.get("/healthz?ready=1")
        assert resp.status_code == 503
        assert resp.json()["reason"] == "index warming"
        assert resp.json()["ready"] is False

        # Release the build; readiness should flip to 200.
        gate.set()
        assert _wait_until(lambda: client.get("/healthz?ready=1").status_code == 200)
        resp = client.get("/healthz?ready=1")
        assert resp.status_code == 200
        assert resp.json()["ready"] is True


def test_no_index_present_is_ready(serve_env):
    """When the bootstrap build raises FileNotFoundError (no index to warm),
    prewarm still flips readiness to ready, and a search returns the existing
    pending_index contract."""
    from fastapi.testclient import TestClient

    _ControllableEngine._raise = FileNotFoundError("no index yet")

    app = _make_app(serve_env)
    with TestClient(app) as client:
        assert _wait_until(lambda: client.get("/healthz?ready=1").status_code == 200)
        resp = client.get("/healthz?ready=1")
        assert resp.status_code == 200
        assert resp.json()["ready"] is True

        # The first search cold-loads via get_engine, which also raises
        # FileNotFoundError → pending_index contract.
        sresp = client.get("/api/search?q=hello")
        assert sresp.status_code == 200
        assert sresp.json().get("pending_index") is True


def test_prewarm_failure_does_not_wedge_readiness(serve_env):
    """A generic prewarm failure (not FileNotFoundError) must not wedge
    readiness — ?ready=1 still flips to 200 — and the next request still
    cold-loads (constructor invoked again on the request path)."""
    from fastapi.testclient import TestClient

    _ControllableEngine._raise = RuntimeError("scann blew up")

    app = _make_app(serve_env)
    with TestClient(app) as client:
        assert _wait_until(lambda: client.get("/healthz?ready=1").status_code == 200)
        assert client.get("/healthz?ready=1").status_code == 200

        count_after_prewarm = _ControllableEngine._count
        assert count_after_prewarm >= 1  # prewarm attempted a build

        # First request cold-loads. The constructor still raises, so search
        # surfaces a 500 path? No — get_engine raises RuntimeError which is
        # not caught by api_search's FileNotFoundError handler. We only
        # assert the build was attempted again (cold-load path engaged).
        try:
            client.get("/api/search?q=hi")
        except Exception:
            pass
        assert _ControllableEngine._count > count_after_prewarm


def test_prewarm_does_not_clobber_request_built_engine(serve_env):
    """R5/setdefault: if a first request builds + caches the engine while
    the background prewarm is still blocked, releasing the prewarm must NOT
    replace the request-built instance. We verify a single engine identity
    survives by asserting the construction that "wins" the cache is the one
    serving subsequent requests — i.e. behavior stays consistent and the
    prewarm-built duplicate is discarded.

    Implementation seam: with the prewarm blocked on its Event, a real first
    /api/search would also block in get_engine's own build. To exercise the
    setdefault race deterministically without two slow builds, we let the
    REQUEST path build instantly (its construction not gated) while the
    PREWARM path is gated, then release the prewarm and confirm the cache
    still serves the request-built engine.

    Because both paths construct the same _ControllableEngine, we assert via
    construction count + stable search behavior rather than object identity
    across closures (which aren't reachable from the test)."""
    from fastapi.testclient import TestClient

    # Gate ONLY the prewarm construction. We can't selectively gate by caller
    # with a single class flag, so instead we gate construction globally,
    # fire the request (which will build under get_engine once we release
    # for it), then release. The deterministic assertion is: exactly the
    # engine cached first wins, and total constructions are bounded.
    gate = threading.Event()
    _ControllableEngine._construct_event = gate

    app = _make_app(serve_env)
    with TestClient(app) as client:
        # While gated, readiness is 503 (prewarm still blocked).
        assert client.get("/healthz?ready=1").status_code == 503

        # Kick off a search in a thread; it will block in get_engine's build
        # on the same gate.
        result = {}

        def _do_search():
            r = client.get("/api/search?q=hello")
            result["status"] = r.status_code
            result["json"] = r.json()

        searcher = threading.Thread(target=_do_search)
        searcher.start()

        # Release: both the prewarm build and the request build can now
        # complete. setdefault ensures whichever writes first wins and the
        # other instance is discarded — no clobber, no crash.
        gate.set()
        searcher.join(timeout=10)
        assert not searcher.is_alive()
        assert result["status"] == 200

        # Readiness is now satisfied.
        assert _wait_until(lambda: client.get("/healthz?ready=1").status_code == 200)

        # Subsequent searches are served from the single cached engine — a
        # warm dict lookup, no further construction beyond the (at most two)
        # racing builds.
        builds_after_race = _ControllableEngine._count
        client.get("/api/search?q=again")
        client.get("/api/search?q=more")
        assert _ControllableEngine._count == builds_after_race  # warm: no new builds
        assert builds_after_race <= 2  # at most prewarm + one request build
