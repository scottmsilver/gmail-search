"""Tests for the /healthz?ready=1 search canary.

2026-07 incident: a deploy-skewed serve process 500'd on EVERY search for
nine days while readiness stayed green, because readiness only meant "the
index loaded", not "queries succeed". The canary closes that gap: readiness
now runs a REAL (rate-limited) search through the bootstrap engine and goes
503 when it throws — so any probe watching readiness sees search breakage.
"""

from __future__ import annotations

import time

import pytest

import gmail_search.server as server_mod
from gmail_search.config import load_config
from gmail_search.store.db import get_connection, init_db


class _CanaryEngine:
    """SearchEngine stand-in with a controllable search_threads."""

    _search_error: BaseException | None = None
    _search_calls = 0

    def __init__(self, db_path, index_dir, config, *, user_id=None):
        pass

    def search_threads(self, *a, **kw):
        type(self)._search_calls += 1
        if type(self)._search_error is not None:
            raise type(self)._search_error
        return []

    def reload_index(self, new_index_dir):
        return None

    @classmethod
    def reset(cls):
        cls._search_error = None
        cls._search_calls = 0


def _wait_until(predicate, timeout=5.0, interval=0.02):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


@pytest.fixture
def canary_env(db_backend, tmp_path, monkeypatch):
    if db_backend is None:  # pragma: no cover - guarded by db_backend skip
        pytest.skip("Postgres not reachable")

    db_path = db_backend["db_path"]
    init_db(db_path)

    # The canary probes the bootstrap user's engine — create that user and
    # clear the per-process resolution cache so this test's schema wins.
    from gmail_search.auth import write_user as wu

    monkeypatch.setenv("GMS_BOOTSTRAP_EMAIL", "canary@example.com")
    wu._BOOTSTRAP_CACHE.clear()
    conn = get_connection(db_path)
    conn.execute(
        "INSERT INTO users (id, email) VALUES (%s, %s) ON CONFLICT (email) DO NOTHING",
        ("u_canary", "canary@example.com"),
    )
    conn.commit()
    conn.close()

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    index_dir = data_dir / "scann_index"
    index_dir.mkdir()

    _CanaryEngine.reset()
    monkeypatch.setattr(server_mod, "SearchEngine", _CanaryEngine)
    monkeypatch.setattr(
        "gmail_search.index.searcher.resolve_active_index_dir",
        lambda db_path, fallback, *, user_id=None: index_dir,
    )

    config = load_config(data_dir=data_dir)

    def make_app(interval="0"):
        monkeypatch.setenv("GMAIL_SEARCH_CANARY_INTERVAL", interval)
        return server_mod.create_app(db_path=db_path, data_dir=data_dir, config=config)

    yield make_app
    wu._BOOTSTRAP_CACHE.clear()


def _ready_probe(client):
    assert _wait_until(lambda: client.get("/healthz?ready=1").status_code in (200, 503))
    return client.get("/healthz?ready=1")


def test_ready_includes_passing_search_canary(canary_env):
    """Healthy engine → readiness 200 with search_ok, and the canary
    actually ran a search."""
    from fastapi.testclient import TestClient

    with TestClient(canary_env()) as client:
        assert _wait_until(lambda: client.get("/healthz?ready=1").status_code == 200)
        resp = client.get("/healthz?ready=1")
        assert resp.json()["search_ok"] is True
        assert _CanaryEngine._search_calls >= 1


def test_ready_503_when_search_throws(canary_env):
    """The 2026-07 failure mode: engine loads fine but every search
    raises. Readiness must go 503 with a canary reason — not stay green."""
    from fastapi.testclient import TestClient

    _CanaryEngine._search_error = IndexError("index 1401113 is out of bounds")
    with TestClient(canary_env()) as client:
        assert _wait_until(lambda: client.get("/healthz?ready=1").status_code == 503)
        resp = client.get("/healthz?ready=1")
        body = resp.json()
        assert body["ready"] is False
        assert body["search_ok"] is False
        assert "canary" in body["reason"]
        assert "IndexError" in body["reason"]


def test_canary_is_rate_limited(canary_env):
    """Back-to-back probes within the interval run ONE real search —
    readiness stays cheap enough for aggressive polling."""
    from fastapi.testclient import TestClient

    with TestClient(canary_env(interval="3600")) as client:
        assert _wait_until(lambda: client.get("/healthz?ready=1").status_code == 200)
        calls_after_first = _CanaryEngine._search_calls
        for _ in range(5):
            assert client.get("/healthz?ready=1").status_code == 200
        assert _CanaryEngine._search_calls == calls_after_first


def test_canary_recovers_after_fix(canary_env):
    """A failing canary flips back to 200 once searches succeed again
    (e.g. after the deploy-skew restart)."""
    from fastapi.testclient import TestClient

    _CanaryEngine._search_error = RuntimeError("boom")
    with TestClient(canary_env()) as client:
        assert _wait_until(lambda: client.get("/healthz?ready=1").status_code == 503)
        _CanaryEngine._search_error = None
        assert _wait_until(lambda: client.get("/healthz?ready=1").status_code == 200)
        assert client.get("/healthz?ready=1").json()["search_ok"] is True


def test_no_bootstrap_user_keeps_ready_green(canary_env, monkeypatch):
    """A fresh install with no users yet must not fail readiness — the
    canary skips when there is nothing to probe."""
    from fastapi.testclient import TestClient

    from gmail_search.auth import write_user as wu

    monkeypatch.setenv("GMS_BOOTSTRAP_EMAIL", "nobody@example.com")
    wu._BOOTSTRAP_CACHE.clear()
    with TestClient(canary_env()) as client:
        assert _wait_until(lambda: client.get("/healthz?ready=1").status_code == 200)
        assert client.get("/healthz?ready=1").status_code == 200


def test_reason_hides_exception_message(canary_env):
    """Codex: /healthz is unauthenticated — the 503 reason must carry only
    the exception CLASS, never the message (which can leak paths, DB
    diagnostics, API details). Full text goes to the server log only."""
    from fastapi.testclient import TestClient

    _CanaryEngine._search_error = IndexError("/home/secret/path leaked 42")
    with TestClient(canary_env()) as client:
        assert _wait_until(lambda: client.get("/healthz?ready=1").status_code == 503)
        body = client.get("/healthz?ready=1").json()
        assert "IndexError" in body["reason"]
        assert "/home/secret/path" not in body["reason"]
        assert "leaked" not in body["reason"]


def test_db_failure_fails_canary(canary_env, monkeypatch):
    """Codex: a DB outage breaks real searches too — it must FAIL the
    canary, not silently pass as 'no bootstrap user yet'."""
    from fastapi.testclient import TestClient

    def _boom(conn):
        raise ConnectionError("could not connect to server: Connection refused")

    monkeypatch.setattr("gmail_search.auth.write_user.get_bootstrap_user_id", _boom)
    with TestClient(canary_env()) as client:
        assert _wait_until(lambda: client.get("/healthz?ready=1").status_code == 503)
        body = client.get("/healthz?ready=1").json()
        assert body["search_ok"] is False
        assert "canary" in body["reason"]
        assert "Connection refused" not in body["reason"]  # class only


def test_concurrent_probes_single_flight(canary_env):
    """Codex: concurrent probes at interval expiry must not stampede the
    engine — exactly one real search runs; the rest see the cached state."""
    from concurrent.futures import ThreadPoolExecutor

    from fastapi.testclient import TestClient

    with TestClient(canary_env(interval="3600")) as client:
        # Wait for _ready without consuming the first canary run: liveness.
        assert _wait_until(lambda: client.get("/healthz").status_code == 200)
        with ThreadPoolExecutor(max_workers=5) as pool:
            statuses = list(pool.map(lambda _: client.get("/healthz?ready=1").status_code, range(5)))
        assert all(s == 200 for s in statuses)
        assert _CanaryEngine._search_calls == 1
