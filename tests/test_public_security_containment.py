"""Security regressions using synthetic tenants in an isolated PG schema."""

import asyncio

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from gmail_search import server
from gmail_search.auth import require_user_id
from gmail_search.config import load_config
from gmail_search.store.db import get_connection


@pytest.fixture
def secured_app(db_backend, tmp_path, monkeypatch):
    conn = get_connection(db_backend["db_path"])
    for owner in ("owner-a", "owner-b"):
        conn.execute("INSERT INTO users (id, email) VALUES (%s, %s)", (owner, f"{owner}@example.test"))
    conn.commit()
    conn.close()
    monkeypatch.setattr(server, "SearchEngine", lambda *a, **kw: object())
    app = server.create_app(db_backend["db_path"], tmp_path, load_config(data_dir=tmp_path))

    def synthetic_user(request: Request):
        return request.headers.get("x-test-user", "owner-a")

    app.dependency_overrides[require_user_id] = synthetic_user
    return app


@pytest.mark.parametrize("query", [
    "SELECT 1",
    "SELECT set_config('app.user_id', 'victim', true)",
    "SELECT set_config('role', session_user, true)",
    'SELECT * FROM "pg_stat_activity"',
])
def test_sql_disabled_before_any_query_runs(secured_app, monkeypatch, query):
    def forbidden(*args, **kwargs):
        pytest.fail("Untrusted SQL reached the database executor")

    monkeypatch.setattr(server, "_run_sql_with_timeout", forbidden)
    response = TestClient(secured_app).post("/api/sql", json={"query": query})
    assert response.status_code == 403
    assert response.json()["code"] == "raw_sql_disabled"


def test_foreign_conversation_put_preserves_history(secured_app):
    client = TestClient(secured_app)
    original = {"title": "Private conversation", "messages": [
        {"role": "user", "parts": [{"type": "text", "text": "private synthetic mail"}]},
    ]}
    assert client.put("/api/conversations/owned", json=original).status_code == 200
    before = client.get("/api/conversations/owned").json()
    response = client.put("/api/conversations/owned", headers={"x-test-user": "owner-b"},
                          json={"title": "Poison", "messages": []})
    assert response.status_code == 404
    assert client.get("/api/conversations/owned").json() == before


def test_owner_can_replace_own_conversation(secured_app):
    client = TestClient(secured_app)
    for title in ("First", "Second"):
        assert client.put("/api/conversations/owned", json={"title": title, "messages": []}).status_code == 200
    assert client.get("/api/conversations/owned").json()["title"] == "Second"


def test_global_progress_is_not_anonymous(secured_app):
    response = TestClient(secured_app).get("/api/progress")
    assert response.status_code == 401


def test_sql_tool_does_not_make_backend_request(monkeypatch):
    from gmail_search.agents import tools

    async def forbidden(*args, **kwargs):
        pytest.fail("Disabled SQL tool contacted the backend")

    monkeypatch.setattr(tools, "_post", forbidden)
    result = asyncio.run(tools.sql_query("SELECT set_config('role', session_user, true)", user_id="owner-a"))
    assert result["status"] == 403
    assert result["code"] == "raw_sql_disabled"


def test_progress_filters_tenants_before_limit(db_backend):
    from gmail_search.store.db import JobProgress

    conn = get_connection(db_backend["db_path"])
    ids = ["backfill:owner-a", "frontfill:owner-a", "global", "backfill:owner-a:extra"]
    ids += [f"stage{i}:owner-b" for i in range(12)]
    for index, job_id in enumerate(ids):
        conn.execute(
            "INSERT INTO job_progress (job_id, started_at, updated_at) VALUES (%s, %s, %s)",
            (job_id, "2026-09-14", f"2026-09-14T00:00:{index:02d}"),
        )
    conn.commit()
    conn.close()
    rows = JobProgress.get_for_user(db_backend["db_path"], "owner-a")
    assert {row["job_id"] for row in rows} == {"backfill:owner-a", "frontfill:owner-a"}


@pytest.mark.parametrize("mime, disposition", [("text/html", "attachment"), ("image/svg+xml", "attachment"), ("image/png", "inline")])
def test_artifact_content_is_sandboxed(secured_app, monkeypatch, mime, disposition):
    from gmail_search.agents import service

    monkeypatch.setattr(service, "get_artifact", lambda *a: ('test"\r\n.html', mime, b"synthetic"))
    response = TestClient(secured_app).get("/api/artifact/123")
    assert response.status_code == 200
    assert response.headers["content-disposition"].startswith(disposition + ";")
    assert "\r" not in response.headers["content-disposition"]
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["content-security-policy"] == "sandbox; default-src 'none'"
    assert response.headers["cache-control"] == "private, no-store"
