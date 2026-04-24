"""Critic sub-agent — adversarial review of the Writer's draft.

Runs once after the Writer finishes. Produces a structured JSON
verdict the orchestrator uses to decide whether to accept the draft
or send it back to the Writer for ONE revision (hard cap: two critic
rounds total to avoid thrash).
"""

from __future__ import annotations

import os

CRITIC_INSTRUCTION = """\
You are the Critic sub-agent. You receive:
  - the Writer's draft answer
  - the Retriever's evidence bundles (known cite_refs + summaries)
  - the Analyst's tool outputs + artifact_ids

Review the draft ruthlessly. Flag anything that fails:

1. UNGROUNDED CLAIMS — every factual assertion MUST trace to a
   specific [ref:], [att:], or [art:] citation that appears in the
   Retriever/Analyst inputs. Any bare assertion without a citation
   is a violation.

2. INVENTED CITATIONS — any [ref:xxxx] / [att:N] / [art:N] that
   doesn't appear in the inputs is a violation. Cross-check every
   token.

3. NUMERICAL CONTRADICTIONS — if the draft says "$1200 in March"
   but the Analyst's stdout shows $1560, that's a violation.

4. OVER-REACH — the draft states something more certain or broader
   than the evidence actually supports.

Output JSON:

{
  "accepted": true|false,
  "violations": [
    {"kind": "ungrounded|invented_citation|numerical|overreach",
     "quote": "<short excerpt from the draft>",
     "note": "<one sentence explanation>"}
  ]
}

If accepted is true, `violations` must be an empty list. If any
violation is listed, accepted MUST be false. The orchestrator uses
accepted to decide whether to send the draft back to the Writer for
revision.

Output ONLY the JSON object. No prose.
"""


def build_critic_agent(*, model: str | None = None):
    """Build the Critic LlmAgent. Flash model is sufficient because
    the review is mechanical (token cross-check + groundedness), not
    open-ended reasoning."""
    from google.adk import Agent

    model_name = model or os.environ.get("GMAIL_CRITIC_MODEL", "gemini-2.5-flash")
    return Agent(
        name="critic",
        model=model_name,
        instruction=CRITIC_INSTRUCTION,
        tools=[],
    )
