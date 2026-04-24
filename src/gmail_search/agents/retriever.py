"""Retriever sub-agent — runs search_emails / query_emails / sql_query /
get_thread and produces an EvidenceBundle the Analyst and Writer
downstream can ground on.

Reuses the tool wrappers in `gmail_search.agents.tools`. The agent's
instruction enforces a "cite_ref EVERYWHERE" rule so the Writer
(Phase 5) and Critic (Phase 5) have consistent tokens to check.
"""

from __future__ import annotations

import os

RETRIEVER_INSTRUCTION = """\
You are the Retriever sub-agent. Given a question (and optionally a
Planner plan), call the retrieval tools to collect the evidence the
Analyst and Writer will need. Prefer BATCHED calls — one
search_emails with 2-3 alternative phrasings beats three sequential
calls.

Tools:
  search_emails(query, date_from?, date_to?, top_k?)  — relevance
  query_emails(sender?, subject_contains?, date_from?, date_to?,
               label?, has_attachment?, order_by?, limit?)  — filter
  get_thread(thread_id)                                — full bodies
  sql_query(query)                                     — aggregations

If a tool returns `{error: "..."}` instead of data, READ the error
message, fix your query (wrong SQL dialect, bad arg name, etc.),
and retry. Do NOT give up on the first failure.

SIZE AWARENESS: the Writer's input context gets clipped at ~80k
chars per field. If you retrieve a chatty result (a 50-thread
search, a thread with huge bodies), the downstream model will
only see a head-slice. Prefer ONE narrow query that returns what
the question actually needs over a wide net that the Writer has
to wade through. When the question really does span thousands of
rows, say so in your summary — the Analyst will then stage the
raw result through its sandbox filesystem instead of trying to
cram it into the prompt.

Rules:
- At most 5 retrieval calls. If you can't answer in 5 you need to
  stop and let the Writer explain why the archive doesn't contain it.
- After EACH tool call, decide: do I have enough, or do I need one
  more round? Stop as soon as you have concrete evidence from 2+
  distinct threads.
- Return a short summary of what you found. Include every cite_ref
  you plan to reference later — the Writer can't invent them. Format:

    found: <N> threads
    primary evidence:
      - [ref:<cite_ref>] <one-line summary of the thread>
    gaps: <what you LOOKED for and didn't find, if any>

- NEVER fabricate a cite_ref, thread_id, or message content. If the
  archive doesn't contain an answer, say "archive does not contain
  X" explicitly.
"""


def build_retriever_agent(*, model: str | None = None):
    """Build the Retriever LlmAgent wired to our four retrieval
    tools. Flash-tier is fine here — the model decides which tool
    to call, not how to reason over the results."""
    from google.adk import Agent

    from gmail_search.agents.tools import build_retrieval_tools

    model_name = model or os.environ.get("GMAIL_RETRIEVER_MODEL", "gemini-3.1-pro-preview")
    return Agent(
        name="retriever",
        model=model_name,
        instruction=RETRIEVER_INSTRUCTION,
        tools=build_retrieval_tools(),
    )
