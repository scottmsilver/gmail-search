"""Tests for the deep-analysis agent's session + event persistence.

Phase 1 locks in the DB helpers before real sub-agents land on top.
These cover the contract the HTTP layer and future agent code will
rely on: events have monotonic seq within a session, terminated
sessions round-trip, artifacts save and retrieve by id.
"""

from __future__ import annotations

from gmail_search.agents.session import (
    append_event,
    create_session,
    fetch_events_after,
    finalize_session,
    get_artifact,
    new_session_id,
    save_artifact,
    session_elapsed_ms,
)
from gmail_search.store.db import get_connection, init_db


def test_new_session_id_is_unique_and_short():
    ids = {new_session_id() for _ in range(256)}
    assert len(ids) == 256
    assert all(len(s) == 16 for s in ids)


def test_create_session_persists_row(db_backend):
    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)

    sid = new_session_id()
    create_session(conn, session_id=sid, conversation_id="conv-1", mode="deep", question="hello?")
    row = conn.execute(
        "SELECT conversation_id, mode, question, status FROM agent_sessions WHERE id = %s",
        (sid,),
    ).fetchone()
    assert row["conversation_id"] == "conv-1"
    assert row["mode"] == "deep"
    assert row["question"] == "hello?"
    assert row["status"] == "running"
    conn.close()


def test_append_event_assigns_monotonic_seq(db_backend):
    """Several appends in a row — seq must be 1, 2, 3 and never repeat.
    The HTTP layer relies on this for SSE resume-after-seq."""
    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)

    sid = new_session_id()
    create_session(conn, session_id=sid, conversation_id=None, mode="deep", question="q")
    seqs = [append_event(conn, session_id=sid, agent_name="planner", kind="plan", payload={"n": i}) for i in range(5)]
    assert seqs == [1, 2, 3, 4, 5]

    # Sanity: two sessions don't share a counter — seq restarts at 1.
    sid2 = new_session_id()
    create_session(conn, session_id=sid2, conversation_id=None, mode="deep", question="q2")
    s = append_event(conn, session_id=sid2, agent_name="planner", kind="plan", payload={})
    assert s == 1
    conn.close()


def test_fetch_events_after_returns_new_only(db_backend):
    """fetch_events_after emulates the SSE resume path: caller passes
    last-seen seq, gets everything strictly newer."""
    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)

    sid = new_session_id()
    create_session(conn, session_id=sid, conversation_id=None, mode="deep", question="q")
    for i in range(4):
        append_event(conn, session_id=sid, agent_name="analyst", kind="code_run", payload={"i": i})

    initial = list(fetch_events_after(conn, sid, after_seq=0))
    assert [e.seq for e in initial] == [1, 2, 3, 4]
    assert [e.payload["i"] for e in initial] == [0, 1, 2, 3]

    rest = list(fetch_events_after(conn, sid, after_seq=2))
    assert [e.seq for e in rest] == [3, 4]
    conn.close()


def test_finalize_session_flips_status_and_stamps_finish_time(db_backend):
    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)

    sid = new_session_id()
    create_session(conn, session_id=sid, conversation_id=None, mode="deep", question="q")
    finalize_session(conn, sid, status="done", final_answer="the answer")

    row = conn.execute(
        "SELECT status, final_answer, finished_at FROM agent_sessions WHERE id = %s",
        (sid,),
    ).fetchone()
    assert row["status"] == "done"
    assert row["final_answer"] == "the answer"
    assert row["finished_at"] is not None
    conn.close()


def test_session_elapsed_ms_measures_against_started_at(db_backend):
    """issue #74: the UI's cost/time footer reads this. It must be
    callable — and give a real number — WHILE the session is still
    'running', before `finalize_session` ever runs: every backend
    calls it, then writes the `final` event, then finalizes, in that
    order (see `session_elapsed_ms`'s docstring for why that order is
    load-bearing, not incidental)."""
    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)

    sid = new_session_id()
    create_session(conn, session_id=sid, conversation_id=None, mode="deep", question="q")
    elapsed_ms = session_elapsed_ms(conn, sid)

    assert elapsed_ms is not None
    assert elapsed_ms >= 0

    row = conn.execute(
        "SELECT status FROM agent_sessions WHERE id = %s",
        (sid,),
    ).fetchone()
    assert row["status"] == "running"
    conn.close()


def test_session_elapsed_ms_missing_row_returns_none(db_backend):
    """A session id that was never created (shouldn't happen —
    create_session always runs first) must not raise; the caller's
    elapsed_ms just comes back None and the UI shows no time."""
    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)

    elapsed_ms = session_elapsed_ms(conn, "does-not-exist")

    assert elapsed_ms is None
    conn.close()


def test_save_and_get_artifact_roundtrip(db_backend):
    """Analyst saves a PNG (say); Writer cites it as [art:<id>] and the
    UI fetches via /api/artifact/<id>. This test covers just the DB
    layer, INCLUDING the owner scoping that closes the IDOR: a caller
    only sees artifacts owned (via the session) by their own user_id."""
    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)

    # Two tenants, each with a deep-mode session.
    conn.execute("INSERT INTO users (id, email) VALUES ('uA', 'a@t.local') ON CONFLICT DO NOTHING")
    conn.execute("INSERT INTO users (id, email) VALUES ('uB', 'b@t.local') ON CONFLICT DO NOTHING")
    conn.commit()

    sid = new_session_id()
    create_session(conn, session_id=sid, conversation_id=None, mode="deep", question="q", user_id="uA")
    payload = b"\x89PNG\r\n\x1a\nfake-png-bytes"
    art_id = save_artifact(
        conn,
        session_id=sid,
        name="spending_by_month.png",
        mime_type="image/png",
        data=payload,
        meta={"rows": 12, "summary": "monthly spend"},
    )
    assert isinstance(art_id, int)

    # Owner sees it.
    name, mime, blob = get_artifact(conn, art_id, "uA")
    assert name == "spending_by_month.png"
    assert mime == "image/png"
    assert bytes(blob) == payload
    # Another tenant does NOT — the IDOR fix. Returns None (→ 404), so the
    # endpoint can't be enumerated across users.
    assert get_artifact(conn, art_id, "uB") is None
    # Unknown id → None regardless of user.
    assert get_artifact(conn, 999999, "uA") is None
    conn.close()
