"""Tests for compact /api/search output controls aimed at agent callers.

A 38-query batch at the cheapest existing detail level ("snippet") still
ships an uncapped matches array (200-char snippet per matching message),
the full participants list, and a facets block per search — hundreds of
KB for an inventory question whose answer is one line per thread. These
tests pin the three knobs that fix that:

  * match_detail="refs" — one line per thread: {thread_id, subject,
    date_last, from, score}. No matches array, no participants.
  * max_matches=N — caps the per-thread matches array at the N highest
    scoring, reporting the number dropped as matches_truncated. 0 (the
    default) keeps today's unlimited behavior for the web UI.
  * include_facets=false — omits the facets block agents never read.
"""

from __future__ import annotations

import pytest

import gmail_search.server as server_mod
from gmail_search.config import load_config
from gmail_search.search.engine import ThreadMatch, ThreadResult
from gmail_search.store.db import init_db


class _CannedEngine:
    """SearchEngine stand-in returning a fixed result set."""

    results: list[ThreadResult] = []

    def __init__(self, db_path, index_dir, config, *, user_id=None):
        pass

    def search_threads(self, *a, **kw):
        return list(type(self).results)

    def reload_index(self, new_index_dir):
        return None


def _thread(thread_id: str = "t1", n_matches: int = 3) -> ThreadResult:
    return ThreadResult(
        thread_id=thread_id,
        score=0.91,
        similarity=0.85,
        subject="Pledge receipt",
        participants=["donor@example.org", "charity@example.org"],
        message_count=5,
        date_first="2024-01-01",
        date_last="2024-06-01",
        user_replied=True,
        matches=[
            ThreadMatch(
                message_id=f"m{i}",
                score=0.9 - i * 0.1,
                from_addr="charity@example.org",
                date="2024-06-01",
                snippet="thank you for your pledge " * 5,
                match_type="semantic",
            )
            for i in range(n_matches)
        ],
    )


@pytest.fixture
def search_app(db_backend, tmp_path, monkeypatch):
    if db_backend is None:  # pragma: no cover - guarded by db_backend skip
        pytest.skip("Postgres not reachable")

    db_path = db_backend["db_path"]
    init_db(db_path)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    index_dir = data_dir / "scann_index"
    index_dir.mkdir()

    _CannedEngine.results = [_thread()]
    monkeypatch.setattr(server_mod, "SearchEngine", _CannedEngine)
    monkeypatch.setattr(
        "gmail_search.index.searcher.resolve_active_index_dir",
        lambda db_path, fallback, *, user_id=None: index_dir,
    )

    config = load_config(data_dir=data_dir)
    return server_mod.create_app(db_path=db_path, data_dir=data_dir, config=config)


def _client(app):
    from fastapi.testclient import TestClient

    return TestClient(app)


def test_refs_detail_returns_one_line_per_thread(search_app):
    """match_detail=refs collapses each thread to a fixed 5-key row —
    no matches array, no participants, no topic_ids."""
    with _client(search_app) as client:
        resp = client.get("/api/search?q=pledge&match_detail=refs")
    assert resp.status_code == 200
    rows = resp.json()["results"]
    assert len(rows) == 1
    assert rows[0] == {
        "thread_id": "t1",
        "subject": "Pledge receipt",
        "date_last": "2024-06-01",
        "from": "charity@example.org",
        "score": pytest.approx(0.91),
    }


def test_refs_detail_no_matches_yields_empty_from(search_app):
    """A refs row for a thread with no matches has from='' rather than
    crashing on matches[0]."""
    _CannedEngine.results = [_thread(n_matches=0)]
    with _client(search_app) as client:
        resp = client.get("/api/search?q=pledge&match_detail=refs")
    assert resp.status_code == 200
    assert resp.json()["results"][0]["from"] == ""


def test_max_matches_caps_and_reports_truncation(search_app):
    """max_matches=1 keeps only the top-scoring match and reports the
    number dropped as matches_truncated."""
    with _client(search_app) as client:
        resp = client.get("/api/search?q=pledge&max_matches=1")
    assert resp.status_code == 200
    row = resp.json()["results"][0]
    assert len(row["matches"]) == 1
    assert row["matches"][0]["message_id"] == "m0"  # highest score kept
    assert row["matches_truncated"] == 2


def test_max_matches_default_is_unlimited(search_app):
    """Without max_matches the full matches array comes back (web UI
    contract unchanged) and no matches_truncated key appears."""
    with _client(search_app) as client:
        resp = client.get("/api/search?q=pledge")
    row = resp.json()["results"][0]
    assert len(row["matches"]) == 3
    assert "matches_truncated" not in row


def test_max_matches_larger_than_matches_adds_no_marker(search_app):
    """A cap that doesn't bite leaves the row unmarked."""
    with _client(search_app) as client:
        resp = client.get("/api/search?q=pledge&max_matches=10")
    row = resp.json()["results"][0]
    assert len(row["matches"]) == 3
    assert "matches_truncated" not in row


def test_include_facets_false_omits_facets(search_app):
    """include_facets=false drops the facets block; the default keeps
    it (web UI contract unchanged)."""
    with _client(search_app) as client:
        resp = client.get("/api/search?q=pledge&include_facets=false")
        assert "facets" not in resp.json()
        resp = client.get("/api/search?q=pledge")
        assert "facets" in resp.json()
