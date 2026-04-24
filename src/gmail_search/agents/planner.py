"""Planner sub-agent — reads the user's question, emits a short
structured plan the orchestrator (and human debugger) can follow.

Intentionally lightweight: no tools, one LLM call, output is JSON.
The plan is a communication channel between the Planner and the
Retriever + Analyst; it's NOT fed back into the LLMs downstream as a
prompt contract, just used by Python code to decide what to do next
and displayed in the UI's "deep-mode" panel for transparency.
"""

from __future__ import annotations

import os

PLANNER_INSTRUCTION = """\
You are the Planner sub-agent for a deep-analysis pipeline over a
user's personal Gmail archive. Given one question, emit a JSON plan:

{
  "question_type": "<factual | synthesis | analytical | exploratory>",
  "retrieval": [
    {"tool": "search_emails"|"query_emails"|"sql_query",
     "args": {...},
     "why": "<one sentence>"}
  ],
  "analysis": [
    {"step": "<what Python snippet should do>", "expected_output":
     "<chart|table|scalar|none>"}
  ],
  "answer_shape": "<one short sentence>"
}

Rules:
- Keep retrieval to 1-3 steps; the Retriever can re-plan if the
  first round is thin.
- Keep analysis to 0-3 steps; 0 is the RIGHT answer when the question
  is purely factual and a search hit answers it.
- The plan is advisory for the downstream agents, not a binding
  contract. They may diverge when the retrieved evidence suggests
  a different path.
- Output ONLY the JSON object. No prose, no markdown fences.
"""


def build_planner_agent(*, model: str | None = None):
    """Build the Planner LlmAgent. No tools; one call, JSON out.
    Model default is flash — planning is cheap and doesn't need
    pro-tier reasoning."""
    from google.adk import Agent

    model_name = model or os.environ.get("GMAIL_PLANNER_MODEL", "gemini-2.5-flash")
    return Agent(
        name="planner",
        model=model_name,
        instruction=PLANNER_INSTRUCTION,
        tools=[],
    )
