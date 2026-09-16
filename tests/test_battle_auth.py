"""Deep battle voting must use the authenticated user, including stats."""

import pytest
from fastapi.testclient import TestClient

from gmail_search.auth import require_user_id
from gmail_search.config import load_config
from gmail_search import server
from gmail_search.store.db import init_db


@pytest.fixture
def app(db_backend, tmp_path, monkeypatch):
    if db_backend is None:
        pytest.skip("Postgres unavailable")
    db_path = db_backend["db_path"]
    init_db(db_path)
    index = tmp_path / "scann_index"
    index.mkdir()
    monkeypatch.setattr(server, "SearchEngine", lambda *a, **kw: object())
    monkeypatch.setattr(
        "gmail_search.index.searcher.resolve_active_index_dir", lambda *a, **kw: index
    )
    app = server.create_app(
        db_path=db_path, data_dir=tmp_path, config=load_config(data_dir=tmp_path)
    )
    app.dependency_overrides[require_user_id] = lambda: "battle-owner"
    return app


def test_vote_uses_authenticated_owner(app, monkeypatch):
    calls = []

    class Conn:
        def execute(self, sql, args=()):
            calls.append((sql, args))
            return self

        def fetchone(self):
            return {"id": 42}

        def commit(self):
            pass

        def close(self):
            pass

    endpoint = next(
        route.endpoint for route in app.routes if route.path == "/api/battle/vote"
    )
    cells = dict(zip(endpoint.__code__.co_freevars, endpoint.__closure__))
    cells["get_connection"].cell_contents = lambda _: Conn()
    response = TestClient(app).post(
        "/api/battle/vote",
        json={
            "question": "q",
            "variant_a": {"backend": "pi", "model": "muse"},
            "variant_b": {"backend": "adk", "model": "gemini"},
            "winner": "a",
        },
    )
    assert response.status_code == 200
    insert = next(args for sql, args in calls if "INSERT INTO model_battles" in sql)
    assert insert[-1] == "battle-owner"


def test_stats_scope_owner_and_distinguish_backends(app):
    calls = []

    class Conn:
        def execute(self, sql, args=()):
            calls.append((sql, args))
            return self

        def fetchall(self):
            import json

            return [
                {
                    "variant_a": json.dumps(
                        {"backend": "claude_code", "model": "sonnet"}
                    ),
                    "variant_b": json.dumps(
                        {"backend": "claude_native", "model": "sonnet"}
                    ),
                    "winner": "a",
                }
            ]

        def close(self):
            pass

    endpoint = next(
        route.endpoint for route in app.routes if route.path == "/api/battle/stats"
    )
    cells = dict(zip(endpoint.__code__.co_freevars, endpoint.__closure__))
    cells["get_connection"].cell_contents = lambda _: Conn()
    response = TestClient(app).get("/api/battle/stats")
    assert response.status_code == 200
    assert "claude_code" in response.text
    assert "claude_native" in response.text
    assert calls[0][1] == ("battle-owner",)
    assert "WHERE user_id" in calls[0][0]
