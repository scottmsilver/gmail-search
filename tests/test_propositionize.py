"""Live proposition-extraction daemon: OpenRouterBackend + propositionize_pending."""

import json
from datetime import datetime

import httpx
import pytest

from gmail_search.llm.openrouter import OpenRouterBackend

# ── OpenRouterBackend (no DB / no network) ────────────────────────────────


class _Resp:
    def __init__(self, status=200, content='{"facts":["f1"]}'):
        self.status_code = status
        self._content = content
        self.request = None

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=self)

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


class _Client:
    def __init__(self):
        self.calls = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        return _Resp()


def test_openrouter_requires_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        OpenRouterBackend()


def test_openrouter_chat_auth_model_json(monkeypatch):
    monkeypatch.setenv("OPENROUTER_KEY", "sekret")
    monkeypatch.delenv("PROP_LLM_MODEL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    b = OpenRouterBackend()
    assert b.model_id == "amazon/nova-lite-v1"
    cl = _Client()
    out = b.chat(cl, [{"role": "user", "content": "hi"}], max_tokens=100, json_format=True)
    assert out == '{"facts":["f1"]}'
    call = cl.calls[0]
    assert call["headers"]["Authorization"] == "Bearer sekret"
    assert call["json"]["model"] == "amazon/nova-lite-v1"
    assert call["json"]["response_format"] == {"type": "json_object"}
    assert call["url"].startswith("https://openrouter.ai/api/v1")


def test_openrouter_base_url_override_and_model(monkeypatch):
    monkeypatch.setenv("OPENROUTER_KEY", "k")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://proxy.local/v1")
    monkeypatch.setenv("PROP_LLM_MODEL", "custom/model")
    b = OpenRouterBackend()
    assert b.model_id == "custom/model"
    cl = _Client()
    b.chat(cl, [{"role": "user", "content": "x"}], max_tokens=10)  # json_format default False
    assert cl.calls[0]["url"] == "https://proxy.local/v1/chat/completions"
    assert "response_format" not in cl.calls[0]["json"]


# ── propositionize_pending (Postgres fixture) ─────────────────────────────


class FakeBackend:
    model_id = "fake"

    def __init__(self):
        self.calls = 0

    def chat(self, client, messages, *, max_tokens, json_format=False):
        self.calls += 1
        content = messages[1]["content"]
        if "FAIL" in content:
            raise RuntimeError("boom")
        if "SUBJ_A_V2" in content:
            return json.dumps({"facts": ["fact A v2"]})
        if "SUBJ_A" in content:
            return json.dumps({"facts": ["fact A v1"]})
        return json.dumps({"facts": ["generic fact"]})


class FakeEmbedder:
    model = "fake-embed"

    def embed_texts_batch(self, texts):
        return [[0.1] * 8 for _ in texts]


def _seed_msg(conn, mid, subject, body, hist):
    from gmail_search.store.models import Message
    from gmail_search.store.queries import upsert_message

    upsert_message(
        conn,
        Message(
            id=mid,
            thread_id="t_" + mid,
            from_addr="a@b.com",
            to_addr="me@test.com",
            subject=subject,
            body_text=body,
            body_html="",
            date=datetime(2025, 1, 1),
            labels=["INBOX"],
            history_id=hist,
            raw_json="{}",
        ),
    )


def _count_facts(conn, uid):
    return conn.execute("SELECT count(*) AS n FROM propositions WHERE user_id=%s", (uid,)).fetchone()["n"]


def _setup(db_backend):
    from gmail_search import propositions as P
    from gmail_search.auth.write_user import resolve_write_user_id
    from gmail_search.store.db import get_connection, init_db

    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)
    uid = resolve_write_user_id(conn)
    P.ensure_table(conn)
    P.ensure_processed_table(conn)
    return conn, uid


def test_propositionize_pending_idempotent(db_backend, tmp_path):
    from gmail_search import propositions as P

    conn, uid = _setup(db_backend)
    _seed_msg(conn, "m1", "SUBJ_A", "body one", 1)
    conn.commit()
    be, emb = FakeBackend(), FakeEmbedder()

    s1 = P.propositionize_pending(conn, None, be, emb, user_id=uid, owner="Owner (o@x.com)", batch=100)
    assert (s1["messages"], s1["facts"], s1["errors"]) == (1, 1, 0)
    assert _count_facts(conn, uid) == 1
    assert be.calls == 1

    # second pass: nothing unprocessed -> no work, no extra LLM calls
    s2 = P.propositionize_pending(conn, None, be, emb, user_id=uid, owner="Owner (o@x.com)", batch=100)
    assert s2["messages"] == 0
    assert be.calls == 1


def test_propositionize_atomic_replace(db_backend, tmp_path):
    from gmail_search import propositions as P

    conn, uid = _setup(db_backend)
    _seed_msg(conn, "m1", "SUBJ_A", "body one", 1)
    conn.commit()
    be, emb = FakeBackend(), FakeEmbedder()
    P.propositionize_pending(conn, None, be, emb, user_id=uid, owner="o", batch=100)
    assert [r["text"] for r in conn.execute("SELECT text FROM propositions WHERE user_id=%s", (uid,)).fetchall()] == [
        "fact A v1"
    ]

    # mutate the message + clear marker -> reprocess replaces the stale fact
    conn.execute("UPDATE messages SET subject=%s WHERE id=%s AND user_id=%s", ("SUBJ_A_V2", "m1", uid))
    conn.execute("DELETE FROM prop_processed WHERE user_id=%s AND message_id=%s", (uid, "m1"))
    conn.commit()
    P.propositionize_pending(conn, None, be, emb, user_id=uid, owner="o", batch=100)
    texts = [r["text"] for r in conn.execute("SELECT text FROM propositions WHERE user_id=%s", (uid,)).fetchall()]
    assert texts == ["fact A v2"]  # old fact gone, no duplicate


def test_propositionize_failure_leaves_message_unstamped(db_backend, tmp_path):
    from gmail_search import propositions as P

    conn, uid = _setup(db_backend)
    _seed_msg(conn, "ok1", "SUBJ_A", "fine", 1)
    _seed_msg(conn, "bad1", "FAIL please", "nope", 2)
    conn.commit()
    be, emb = FakeBackend(), FakeEmbedder()

    s = P.propositionize_pending(conn, None, be, emb, user_id=uid, owner="o", batch=100)
    assert s["errors"] == 1
    stamped = {
        r["message_id"]
        for r in conn.execute("SELECT message_id FROM prop_processed WHERE user_id=%s", (uid,)).fetchall()
    }
    assert "ok1" in stamped and "bad1" not in stamped  # failed msg retries next pass


def test_owner_string_for_user_multitenant(db_backend, tmp_path):
    from gmail_search import propositions as P
    from gmail_search.store.db import get_connection, init_db

    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)
    conn.execute(
        "INSERT INTO users (id, email, name) VALUES (%s,%s,%s) ON CONFLICT (id) DO NOTHING",
        ("u_named", "named@x.com", "Named Person"),
    )
    conn.execute(
        "INSERT INTO users (id, email) VALUES (%s,%s) ON CONFLICT (id) DO NOTHING",
        ("u_bare", "bare@x.com"),
    )
    conn.commit()
    assert P.owner_string_for_user(conn, "u_named") == "Named Person (named@x.com)"
    assert P.owner_string_for_user(conn, "u_bare") == "bare@x.com"
