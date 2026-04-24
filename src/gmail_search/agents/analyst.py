"""The Analyst sub-agent — ADK LlmAgent with one tool (`run_code`) that
executes Python in the Docker sandbox and returns stdout / stderr /
artifact ids.

Flow for a deep-mode turn:
  1. Orchestrator seeds `evidence_records` (list of message / thread
     rows the retriever produced) + opens a DB connection the tool
     will use to persist artifacts.
  2. `build_analyst_agent(...)` returns a closure-bound LlmAgent so
     tool invocations land on THIS session's sandbox input + artifact
     sink — no module-level shared state, safe for concurrent turns.
  3. The model emits `run_code({code: "..."})`. The tool dispatches
     to `execute_in_sandbox()`, persists any artifacts it produced to
     `agent_artifacts`, and returns a compact result dict.
  4. The model reads stdout / stderr / artifact ids, decides to iterate
     (write more code) or hand off. The caller sees all of this as
     `tool_call` / `tool_result` events on the session transcript.

Deliberately minimal: one tool, one model call per turn. Planner,
Retriever, Writer, Critic come online in phase 4+ and REUSE this
tool via ADK's multi-agent orchestration.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from gmail_search.agents.sandbox import SandboxRequest, execute_in_sandbox
from gmail_search.agents.session import save_artifact

logger = logging.getLogger(__name__)


# Instruction injected into the Analyst's LLM call. Kept separate
# from the orchestration code so prompt edits are just a string
# change — no imports, no functions. Mirrors the phrasing we use in
# web/lib/systemPrompt.ts but scoped to the code-execution path.
ANALYST_INSTRUCTION = """\
You are the Analyst sub-agent. Your job is to answer a data question
by writing small Python snippets that run in a sandboxed container
with a pre-seeded `evidence` DataFrame and a read-only `db`
psycopg connection (Postgres: tables `messages`, `attachments`,
`message_summaries`, `thread_summary`, `topics`, `message_topics`,
`contact_frequency`, `embeddings`, `term_aliases`).

Call `run_code` with ONE snippet per call. Read stdout + stderr in
the result, decide whether to iterate (write more code) or to
summarise what you found. Iterate at most 4 times; more than that
usually means the plan is off.

Available in every snippet (via the runtime preamble):
  evidence       — pandas DataFrame from the retriever's results
  db             — psycopg connection, autocommit + read-only
  pd, np, plt, sns, sklearn  — imported for you
  save_artifact(name, obj, mime_type=None)  — persist a plot, CSV,
      or text blob; returns the filename. The orchestrator uploads
      these to the database and tells you the artifact_ids so you
      can cite them.

Rules:
  - Always PRINT what you want the caller to see. Return values from
    the snippet are discarded.
  - Do NOT attempt network access (it's blocked) or try to mutate the
    database (connection is read-only).
  - Keep snippets small — one logical step at a time. If you need
    multiple artifacts, save each with a distinct filename.
  - If an error comes back in stderr, read it carefully before
    retrying; repeated identical failures are wasteful.

Final output: a short natural-language summary of what you found,
with explicit references to the artifact filenames or ids you
produced. Do NOT dump raw DataFrames inline — point to a saved CSV.
"""


def _truncate(s: str, cap: int) -> str:
    """Clip a string to a ceiling, with a marker so the model knows
    content was cut. Important for tool results: stdout/stderr from
    a runaway snippet can be megabytes, and shipping all of that back
    to the LLM wastes tokens and can blow the context window."""
    if len(s) <= cap:
        return s
    return s[: cap - 16] + f"\n... (truncated, original {len(s)} chars)"


def build_run_code_tool(
    *,
    evidence_records: list[dict] | dict | None,
    db_dsn: str | None,
    session_id: str,
    db_conn,
    timeout_seconds: int = 60,
):
    """Return an ADK FunctionTool bound to THIS session's sandbox
    inputs + artifact sink.

    The returned callable's signature is `run_code(code: str) -> dict`.
    ADK introspects the Python signature + docstring to produce the
    schema the LLM sees, so those are the tool's real contract.

    Artifacts produced by the snippet are persisted to
    `agent_artifacts` before the tool returns, and their ids are
    included in the result so the model can cite them later.
    """
    from google.adk.tools import FunctionTool

    def run_code(code: str) -> dict:
        """Execute a Python snippet in the analysis sandbox.

        Args:
            code: A self-contained Python snippet. Has access to
                `evidence` (pandas DataFrame from retrieval), `db`
                (read-only psycopg connection), `pd`, `np`, `plt`,
                `sns`, `sklearn`, and `save_artifact(name, obj)`.

        Returns:
            A dict with `exit_code`, `stdout`, `stderr`, `wall_ms`,
            `timed_out`, `oom_killed`, and `artifacts`: a list of
            `{id, name, mime_type}` rows for every artifact the
            snippet persisted via save_artifact. The `id` is the row
            in `agent_artifacts` — cite in the final answer as
            `[art:<id>]`.
        """
        req = SandboxRequest(
            code=code,
            evidence=evidence_records,
            db_dsn=db_dsn,
            timeout_seconds=timeout_seconds,
        )
        result = execute_in_sandbox(req)

        persisted: list[dict[str, Any]] = []
        for art in result.artifacts:
            try:
                art_id = save_artifact(
                    db_conn,
                    session_id=session_id,
                    name=art.name,
                    mime_type=art.mime_type,
                    data=art.data,
                )
                persisted.append({"id": art_id, "name": art.name, "mime_type": art.mime_type})
            except Exception as e:
                logger.warning(f"save_artifact failed for {art.name}: {e}")

        return {
            "exit_code": result.exit_code,
            "stdout": _truncate(result.stdout, 8000),
            "stderr": _truncate(result.stderr, 4000),
            "wall_ms": result.wall_ms,
            "timed_out": result.timed_out,
            "oom_killed": result.oom_killed,
            "artifacts": persisted,
        }

    return FunctionTool(run_code)


def build_analyst_agent(
    *,
    evidence_records: list[dict] | dict | None,
    db_dsn: str | None,
    session_id: str,
    db_conn,
    model: str | None = None,
    instruction: str | None = None,
):
    """Assemble the Analyst LlmAgent ready to run.

    Model defaults to $GMAIL_ANALYST_MODEL or gemini-2.5-flash — the
    Analyst doesn't need pro-tier reasoning for most questions; the
    Writer / Critic are where we spend the bigger model. The
    `instruction` override lets the Planner inject question-specific
    context without the Analyst having to know about the Planner.
    """
    from google.adk import Agent

    run_code = build_run_code_tool(
        evidence_records=evidence_records,
        db_dsn=db_dsn,
        session_id=session_id,
        db_conn=db_conn,
    )
    model_name = model or os.environ.get("GMAIL_ANALYST_MODEL", "gemini-2.5-flash")
    return Agent(
        name="analyst",
        model=model_name,
        instruction=instruction or ANALYST_INSTRUCTION,
        tools=[run_code],
    )


# ── Local skills discovery ─────────────────────────────────────────
#
# Honors the SKILL.md convention so users can drop analysis-specific
# guidance into `<repo>/.claude/skills/<name>/SKILL.md` and have it
# reach the Analyst. In Phase 3 we inject matched skill bodies into
# the instruction (via our custom loader); Phase 4 will swap this
# for `SkillToolset` once the full multi-agent wiring lands and we
# care about progressive loading.


def instruction_with_skills(
    base_instruction: str,
    *,
    question: str,
    project_root: Path | None = None,
) -> str:
    """Load project + personal SKILL.md files and append any that
    match `question` for this sub-agent. Returns `base_instruction`
    unchanged when no skills match — zero overhead when the feature
    isn't used."""
    from gmail_search.agents.skills import default_skill_roots, inject_skill_instructions, load_skills, match_skills

    roots = default_skill_roots(project_root=project_root)
    if not roots:
        return base_instruction
    skills = load_skills(roots)
    matched = match_skills(skills, question, agent_name="analyst")
    return inject_skill_instructions(base_instruction, matched)
