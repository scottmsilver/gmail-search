"""Surface tests for the Planner / Retriever / Writer / Critic
sub-agents. We're NOT running them against a live model here — that
belongs in the Phase-6 integration suite. What we test is the factory
output: the agent has the name, tool set, and instruction we designed,
so accidental edits get caught by CI.

Skips when google-adk isn't installed.
"""

from __future__ import annotations

import pytest

try:
    import google.adk  # noqa: F401

    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False

pytestmark = pytest.mark.skipif(not ADK_AVAILABLE, reason="google-adk not installed")


def test_build_planner_agent_has_no_tools():
    """The Planner outputs JSON; giving it tools would just tempt the
    model to skip the planning step and act immediately."""
    from gmail_search.agents.planner import PLANNER_INSTRUCTION, build_planner_agent

    agent = build_planner_agent()
    assert agent.name == "planner"
    assert list(agent.tools) == []
    assert "question_type" in agent.instruction
    assert agent.instruction == PLANNER_INSTRUCTION


def test_build_retriever_agent_carries_all_four_retrieval_tools():
    """Retriever is the ONLY agent that talks to the data layer; if
    any of the four retrieval tools goes missing, whole classes of
    questions fail silently."""
    from gmail_search.agents.retriever import build_retriever_agent

    agent = build_retriever_agent()
    assert agent.name == "retriever"
    names = sorted(t.name for t in agent.tools)
    assert names == ["get_thread", "query_emails", "search_emails", "sql_query"]


def test_build_writer_agent_has_no_tools_and_mentions_citation_rules():
    """The Writer composes from handed-in context; it must NOT have
    retrieval tools (that's the Retriever's job; mixing them invites
    fresh retrieval to patch hallucinations)."""
    from gmail_search.agents.writer import build_writer_agent

    agent = build_writer_agent()
    assert agent.name == "writer"
    assert list(agent.tools) == []
    assert "[ref:" in agent.instruction
    assert "[art:" in agent.instruction


def test_build_critic_agent_mentions_all_violation_categories():
    """The Critic's four categories (ungrounded, invented_citation,
    numerical, overreach) are a contract with the orchestrator — it
    dispatches on the `kind` field. Prompt edits that drop a category
    quietly break the verdict loop."""
    from gmail_search.agents.critic import build_critic_agent

    agent = build_critic_agent()
    assert agent.name == "critic"
    assert list(agent.tools) == []
    for kind in ("ungrounded", "invented_citation", "numerical", "overreach"):
        assert kind in agent.instruction, f"Critic instruction missing {kind!r}"


def test_model_overrides_use_env_and_kwarg(monkeypatch):
    """Model selection priority: explicit kwarg > env var > default.
    Lets ops tune which agent runs on which tier without code
    changes."""
    from gmail_search.agents.planner import build_planner_agent

    # Default
    default_agent = build_planner_agent()
    assert "gemini" in default_agent.model

    # Env override
    monkeypatch.setenv("GMAIL_PLANNER_MODEL", "gemini-2.5-pro")
    env_agent = build_planner_agent()
    assert env_agent.model == "gemini-2.5-pro"

    # Explicit kwarg beats env
    explicit = build_planner_agent(model="gemini-2.0-flash")
    assert explicit.model == "gemini-2.0-flash"
