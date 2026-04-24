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
  - the Retriever's evidence bundle
  - the Analyst's tool output (possibly empty)
  - an explicit "Allowed citations" block listing the EXACT [ref:*]
    and [art:*] tokens that are valid for THIS turn

Review the draft ruthlessly. Flag anything that fails:

1. INVENTED_CITATION — any [ref:*], [att:*], or [art:*] in the
   draft that is NOT listed in the Allowed citations block is a
   violation. Cross-check EVERY citation token in the draft
   against the allowed list character-for-character.

2. UNGROUNDED — a bare factual claim that could be cited but isn't
   (there's a matching entry in the allowed list). Citing-less
   claims that have NO matching allowed citation are OK; they
   can't be cited because no ref exists.

3. NUMERICAL — if the draft says "$1200 in March" but the
   Analyst's stdout shows $1560, that's a violation.

4. OVERREACH — the draft states something more certain or broader
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

    model_name = model or os.environ.get("GMAIL_CRITIC_MODEL", "gemini-3.1-pro-preview")
    return Agent(
        name="critic",
        model=model_name,
        instruction=CRITIC_INSTRUCTION,
        tools=[],
    )
