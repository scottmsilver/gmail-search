"""Tests for the Analyst sub-agent wiring.

We test the pieces WE own — the `run_code` tool closure, artifact
persistence round-trip, instruction + skills injection — without
relying on live Gemini calls. A full end-to-end run with the real
LlmAgent would require GEMINI_API_KEY + network, so that's a
separate guarded test.

Skips cleanly on machines where the sandbox image isn't built (same
guard as test_agent_sandbox.py) OR ADK isn't importable.
"""

from __future__ import annotations

import pytest

try:
    import google.adk  # noqa: F401

    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False

from gmail_search.agents.sandbox import image_available

pytestmark = pytest.mark.skipif(
    not (ADK_AVAILABLE and image_available()),
    reason="requires google-adk installed + gmail-search-analyst image built",
)


# ── The run_code tool closure ─────────────────────────────────────


def test_run_code_tool_executes_and_returns_stdout(db_backend):
    """Happy path: the tool closure bound to a session runs a snippet
    in the sandbox and returns the stdout + exit_code we expect.
    Proves the closure path works end-to-end (FunctionTool wrapping,
    sandbox call, result shape)."""
    from gmail_search.agents.analyst import build_run_code_tool
    from gmail_search.agents.session import create_session, new_session_id
    from gmail_search.store.db import get_connection, init_db

    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)
    sid = new_session_id()
    create_session(conn, session_id=sid, conversation_id=None, mode="deep", question="q")

    tool = build_run_code_tool(
        evidence_records={"x": [1, 2, 3]},
        db_dsn=None,
        session_id=sid,
        db_conn=conn,
    )
    # FunctionTool stores the Python function on `.func`.
    result = tool.func(code="print('sum:', sum(evidence['x']))")
    assert result["exit_code"] == 0, result["stderr"]
    assert "sum: 6" in result["stdout"]
    assert result["artifacts"] == []
    assert result["wall_ms"] > 0
    conn.close()


def test_run_code_tool_persists_artifacts_to_db(db_backend):
    """When the snippet calls save_artifact, the tool persists the
    bytes to `agent_artifacts` BEFORE returning, and the returned
    artifacts list carries the DB ids the Writer will later cite as
    [art:<id>]."""
    from gmail_search.agents.analyst import build_run_code_tool
    from gmail_search.agents.session import create_session, get_artifact, new_session_id
    from gmail_search.store.db import get_connection, init_db

    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)
    sid = new_session_id()
    create_session(conn, session_id=sid, conversation_id=None, mode="deep", question="q")

    tool = build_run_code_tool(
        evidence_records=None,
        db_dsn=None,
        session_id=sid,
        db_conn=conn,
    )
    code = (
        "import matplotlib.pyplot as plt\n"
        "fig = plt.figure()\n"
        "plt.plot([1,2,3])\n"
        "save_artifact('trend.png', fig)\n"
        "print('saved')\n"
    )
    result = tool.func(code=code)
    assert result["exit_code"] == 0, result["stderr"]
    assert len(result["artifacts"]) == 1
    art = result["artifacts"][0]
    assert art["name"] == "trend.png"
    assert art["mime_type"] == "image/png"
    assert isinstance(art["id"], int) and art["id"] > 0

    # Round-trip the bytes through the DB to prove they were persisted
    # faithfully — not just that we got an id back.
    row = get_artifact(conn, art["id"])
    assert row is not None
    name, mime, data = row
    assert name == "trend.png"
    assert mime == "image/png"
    assert data[:4] == b"\x89PNG"
    conn.close()


def test_run_code_tool_truncates_oversized_stdout(db_backend):
    """A snippet that prints megabytes of junk should come back
    capped, with an explicit "truncated" marker — otherwise the next
    LLM turn wastes tokens or blows the context window."""
    from gmail_search.agents.analyst import build_run_code_tool
    from gmail_search.agents.session import create_session, new_session_id
    from gmail_search.store.db import get_connection, init_db

    db_path = db_backend["db_path"]
    init_db(db_path)
    conn = get_connection(db_path)
    sid = new_session_id()
    create_session(conn, session_id=sid, conversation_id=None, mode="deep", question="q")

    tool = build_run_code_tool(evidence_records=None, db_dsn=None, session_id=sid, db_conn=conn)
    # Emits ~100k chars — well over the 8000-char cap.
    result = tool.func(code="print('A' * 100000)")
    assert result["exit_code"] == 0
    assert len(result["stdout"]) <= 8100
    assert "truncated" in result["stdout"]
    conn.close()


# ── Agent factory ─────────────────────────────────────────────────


def test_build_analyst_agent_has_run_code_tool():
    """build_analyst_agent should return an ADK agent with exactly one
    tool named `run_code`. Guarantees the Analyst's tool surface is
    the one we designed — no accidental additions that'd confuse the
    model."""
    from gmail_search.agents.analyst import build_analyst_agent

    # DB conn can be None for this test — we're not running a snippet.
    agent = build_analyst_agent(
        evidence_records=None,
        db_dsn=None,
        session_id="test",
        db_conn=None,
    )
    assert agent.name == "analyst"
    tool_names = [t.name for t in agent.tools]
    assert tool_names == ["run_code"]


# ── Skills integration ───────────────────────────────────────────


def test_instruction_with_skills_injects_matching_skill_body(tmp_path):
    """A SKILL.md in `<project>/.claude/skills/<name>/` whose
    description mentions keywords from the question should end up
    appended to the Analyst's instruction. Proves the Phase-3
    stopgap path (body injection) works end-to-end before we swap
    to ADK SkillToolset in Phase 4."""
    from gmail_search.agents.analyst import instruction_with_skills

    # Create a project-scoped skill dir.
    skill_dir = tmp_path / ".claude" / "skills" / "spending-analyzer"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: spending-analyzer\n"
        "description: guidance for analysing monthly spending totals\n"
        "agent: analyst\n"
        "---\n"
        "Prefer YTD sums over per-month breakdowns.\n",
        encoding="utf-8",
    )

    base = "You are the Analyst."
    out = instruction_with_skills(
        base,
        question="summarise my monthly spending",
        project_root=tmp_path,
    )
    assert out.startswith(base)
    assert "<skills>" in out
    assert "spending-analyzer" in out
    assert "Prefer YTD sums" in out


def test_instruction_with_skills_noop_when_no_skills_dir(tmp_path):
    """No `.claude/skills/` dir → base instruction unchanged. Keeps
    the hot path zero-overhead for users who don't author skills."""
    from gmail_search.agents.analyst import instruction_with_skills

    base = "You are the Analyst."
    out = instruction_with_skills(base, question="anything", project_root=tmp_path)
    assert out == base
