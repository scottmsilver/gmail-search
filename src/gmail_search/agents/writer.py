"""Writer sub-agent — composes the final markdown answer from the
Retriever's evidence bundles + the Analyst's tool outputs. No tools,
one call, markdown out.

Citation rules mirror the chat-mode prompt (web/lib/systemPrompt.ts):
  [ref:<cite_ref>]       — thread citation, 8-hex id from a retrieval
                           result in THIS turn
  [att:<attachment_id>]  — attachment citation
  [art:<artifact_id>]    — NEW for deep mode: reference an Analyst-
                           produced plot / CSV (served by
                           /api/artifact/<id>)
"""

from __future__ import annotations

import os

WRITER_INSTRUCTION = """\
You are the Writer sub-agent. You receive a structured context:
  - the original user question
  - the Retriever's evidence bundles (each row has a cite_ref)
  - the Analyst's tool outputs (stdout + artifact_ids)

Your job: compose the final markdown answer to the user. Ground
EVERY factual claim in a specific piece of evidence, and cite it:

  [ref:<cite_ref>]  for a thread / message fact
  [att:<attachment_id>]  for a quote from an attachment
  [art:<artifact_id>]   for a chart or CSV the Analyst produced

Rules:
- Do NOT invent a cite_ref, attachment_id, or artifact_id. If the
  input doesn't carry it, you can't cite it.
- Do NOT paraphrase the archive into a confident claim when the
  evidence is thin. Say "I found X but couldn't verify Y" and point
  to the relevant refs.
- Match length + structure to the question. A factual ask gets one
  sentence with one citation. A trend question might warrant 3-5
  bullets + a chart citation.
- If the Analyst produced a chart, always cite it inline where the
  reader would want to see it: "Spending ramped in Q2 [art:42]".
- Never apologise for being an AI, summarise your tools, or
  describe what you just did.

Output markdown only. No front matter, no code fences.
"""


def build_writer_agent(*, model: str | None = None):
    """Build the Writer LlmAgent. Default model is pro-tier because
    the Writer's job (synthesise with grounded citations) benefits
    from the extra reasoning budget; a flash model misses citations
    more often."""
    from google.adk import Agent

    model_name = model or os.environ.get("GMAIL_WRITER_MODEL", "gemini-2.5-pro")
    return Agent(
        name="writer",
        model=model_name,
        instruction=WRITER_INSTRUCTION,
        tools=[],
    )
