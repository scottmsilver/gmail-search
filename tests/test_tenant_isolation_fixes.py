"""Regression tests for the 2026-07-08 tenant-isolation fixes, covering
the specific holes an adversarial review flagged:

- conversation ownership gate: NULL-owner rows must NOT be treated as
  "brand new", and absent ids must be atomically claimed (no TOCTOU).
- session/artifact ownership helpers fail closed across tenants.
- summary upsert refreshes a stale owner instead of preserving it.
"""

from __future__ import annotations

import pytest
from gmail_search.agents.session import conversation_owner, create_session, new_session_id, session_owner
from gmail_search.store.db import get_connection, init_db


@pytest.fixture
def db(db_backend):
    init_db(db_backend["db_path"])
    conn = get_connection(db_backend["db_path"])
    conn.execute("INSERT INTO users (id, email) VALUES ('uA','a@t.local') ON CONFLICT DO NOTHING")
    conn.execute("INSERT INTO users (id, email) VALUES ('uB','b@t.local') ON CONFLICT DO NOTHING")
    conn.commit()
    yield conn
    conn.close()


def _claim_and_check(conn, conversation_id: str, user_id: str) -> bool:
    """Replicates the POST /api/agent/analyze gate: claim-if-absent then
    verify ownership. Returns True if the caller is allowed."""
    conn.execute(
        "INSERT INTO conversations (id, user_id) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING",
        (conversation_id, user_id),
    )
    conn.commit()
    return conversation_owner(conn, conversation_id) == user_id


class TestConversationGate:
    def test_absent_id_is_claimed_by_first_caller(self, db):
        assert _claim_and_check(db, "c-new", "uA") is True
        assert conversation_owner(db, "c-new") == "uA"

    def test_second_caller_cannot_steal_claimed_id(self, db):
        assert _claim_and_check(db, "c-race", "uA") is True
        # uB tries the same id — claim is a no-op, ownership check rejects.
        assert _claim_and_check(db, "c-race", "uB") is False

    def test_existing_foreign_conversation_rejected(self, db):
        db.execute("INSERT INTO conversations (id, user_id) VALUES ('c-b', 'uB')")
        db.commit()
        assert _claim_and_check(db, "c-b", "uA") is False

    def test_null_owner_row_not_treated_as_new(self, db):
        # The conflation bug: a legacy row with user_id NULL must NOT pass
        # the gate for an arbitrary caller.
        db.execute("INSERT INTO conversations (id, user_id) VALUES ('c-legacy', NULL)")
        db.commit()
        assert _claim_and_check(db, "c-legacy", "uA") is False
        # And it stays NULL-owned (claim INSERT was a no-op on conflict).
        assert conversation_owner(db, "c-legacy") is None

    def test_owner_can_reuse_their_own_conversation(self, db):
        db.execute("INSERT INTO conversations (id, user_id) VALUES ('c-a', 'uA')")
        db.commit()
        assert _claim_and_check(db, "c-a", "uA") is True


class TestConversationIdValidation:
    def test_rejects_path_traversal_and_junk(self):
        from gmail_search.agents.service import _is_valid_conversation_id

        for bad in [
            "x/../deep-conv-victim",  # path traversal
            "a/b",
            "..",
            "with space",
            "semi;colon",
            "",  # empty
            "z" * 65,  # too long
            "unicode‮",
        ]:
            assert _is_valid_conversation_id(bad) is False, bad

    def test_accepts_valid_slugs(self):
        from gmail_search.agents.service import _is_valid_conversation_id

        for ok in ["conv-123", "abc_DEF-456", "a", "z" * 64]:
            assert _is_valid_conversation_id(ok) is True, ok

    def test_workspace_slug_falls_back_on_bad_id(self):
        from gmail_search.agents.service import _claudebox_workspace_for

        # A traversal id must NOT reach the directory name; fall back to
        # per-session naming instead.
        ws = _claudebox_workspace_for("x/../victim", "sess-1")
        assert ws == "deep-sess-1"
        assert _claudebox_workspace_for("conv-9", "sess-1") == "deep-conv-conv-9"


class TestSessionOwnerFailsClosed:
    def test_absent_session_returns_none(self, db):
        assert session_owner(db, "nope") is None

    def test_returns_owner(self, db):
        sid = new_session_id()
        create_session(db, session_id=sid, conversation_id=None, mode="deep", question="q", user_id="uA")
        assert session_owner(db, sid) == "uA"


class TestSummaryOwnerRefresh:
    def test_upsert_refreshes_stale_owner(self, db):
        from gmail_search.summarize import _store_summary

        # Seed a message owned by uB and a STALE summary row misattributed
        # to uA (simulating a pre-fix write).
        db.execute(
            "INSERT INTO messages (id, thread_id, from_addr, to_addr, date, user_id)"
            " VALUES ('m1','t1','a@b.c','d@e.f','2026-01-01','uB')"
        )
        db.execute(
            "INSERT INTO message_summaries (message_id, summary, model, user_id)" " VALUES ('m1','old','mdl','uA')"
        )
        db.commit()

        _store_summary(db, "m1", "fresh summary", "mdl")
        db.commit()

        row = db.execute("SELECT summary, user_id FROM message_summaries WHERE message_id='m1'").fetchone()
        assert row["summary"] == "fresh summary"
        # Owner corrected to the message's true owner (uB), not left at uA.
        assert row["user_id"] == "uB"
