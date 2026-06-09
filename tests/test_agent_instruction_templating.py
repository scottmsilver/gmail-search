"""Regression test for ADK instruction templating crashing a stage.

ADK runs `{var}` session-state substitution over an agent's *string*
instruction. The Analyst injects matched SKILL.md bodies into its
instruction, and real skills contain literal braces (e.g. a gstack
skill prints `v{to}`). A plain-string instruction makes ADK treat
`{to}` as a missing state variable and crash the whole deep-mode turn
with `KeyError: Context variable not found: to`.

The fix: the Analyst's instruction is an ADK InstructionProvider (a
callable), which ADK resolves with `bypass_state_injection=True` — no
templating. We don't use ADK session state anyway (each sub-agent gets
a one-shot session; context is curated through prompts).
"""

from __future__ import annotations

import asyncio

import pytest

try:
    import google.adk  # noqa: F401

    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False


@pytest.mark.skipif(not ADK_AVAILABLE, reason="google-adk not installed")
def test_analyst_instruction_bypasses_adk_state_injection():
    from gmail_search.agents.analyst import build_analyst_agent

    # A skill body with a literal `{to}` — exactly what crashed deep mode.
    instruction = 'Analyze per the plan.\n\n<skills>\nprint "Running gstack v{to}"\n</skills>'
    agent = build_analyst_agent(
        evidence_records=None,
        db_dsn=None,
        session_id="s1",
        db_conn=None,
        instruction=instruction,
    )

    # Must be an InstructionProvider (callable), not a raw string.
    assert callable(agent.instruction)

    # ADK resolves it with bypass_state_injection=True and never tries
    # to substitute `{to}` — no KeyError, braces preserved verbatim.
    text, bypass = asyncio.run(agent.canonical_instruction(None))
    assert bypass is True
    assert "{to}" in text
