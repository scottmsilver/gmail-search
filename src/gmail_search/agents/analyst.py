"""The Analyst stage of the deep pipeline: its instruction (keyed to the
schema shape this process talks to, plus any matching SKILL.md bodies) and
its StageAgent. The runtime adapter (claudebox) supplies the tools.
"""

from __future__ import annotations

from pathlib import Path


# Instruction injected into the Analyst's LLM call. Kept separate
# from the orchestration code so prompt edits are just a string
# change — no imports, no functions. Mirrors the phrasing we use in
# web/lib/systemPrompt.ts but scoped to the code-execution path.
_ANALYST_INSTRUCTION_TEMPLATE = """\
You are the Analyst sub-agent. You have access to a Python sandbox
(via the `run_code` tool) pre-seeded with an `evidence` DataFrame
and a read-only `db` psycopg connection to Postgres (tables:
`messages`, `attachments`, `message_summaries`, `thread_summary`,
`topics`, `message_topics`, `contact_frequency`, `embeddings`,
`term_aliases`).

PERF: BM25 search (`messages.{bm25_key} @@@ 'field:term'`) is fast, but per-row text
processing over `body_text` (`regexp_replace`/`~*`/`substring`) across a
broad match set is slow. Prefer the precomputed `message_summaries.summary`
over re-deriving text from raw `body_text`; if you must touch `body_text`,
filter + LIMIT first, then process only that small final set.

Use the tool when the question actually needs computation —
aggregations, plots, clustering, ratios, trend detection. Skip it
when the question is already answered by the evidence handed to
you. That's a judgment call; make it deliberately.

Call `run_code` with ONE snippet per call. Read stdout + stderr
in the result, decide whether to iterate or summarise. Iterate at
most 4 times.

Your final text response must:
1. If you ran code: summarise what the snippet output showed, with
   real numbers from stdout (don't approximate).
2. Reference any artifact_ids the tool actually returned (e.g.
   "saved as artifact 42"). NEVER invent an artifact_id. If you
   didn't save one, don't mention one.
3. If you didn't run code: say why and give whatever answer the
   evidence supports.

SCALE (when the Planner flagged a large question):
- The filesystem `/work/` is writable scratch. Use it to stage big
  intermediate data:
    cur = db.execute("SELECT ... FROM messages WHERE ...")
    df = pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])
    df.to_parquet('/work/raw.parquet')   # or to_csv for smaller data
- Then process incrementally. For analyses that don't fit in
  RAM, use `pd.read_sql(..., chunksize=1000)` or
  `pd.read_parquet('/work/raw.parquet', columns=[...])` to pull
  just what you need per iteration.
- `print()` ONLY the compact summary the Writer needs (totals,
  top-N, key statistics). Writer's input gets clipped at 80k
  chars — don't emit long tables into stdout; save them as a CSV
  artifact instead and reference the artifact id.
- If a single operation risks OOM (>512 MB), split it — the
  sandbox kills at 512 MB hard.

## Your `/work` directory persists.
Files you write under `/work` (or `/work/anything/`) survive across
`run_code` calls in this turn AND across turns in this conversation.
Use this to save intermediate data — fitted models, parquets, large
dataframes — anywhere outside `/work/artifacts/` that you want to
reuse later. `/work/run.py` and `/work/inputs.json` are overwritten
by the orchestrator on every call; don't rely on those being yours.

Packages installed via `pip` do NOT persist (they live in the
container's system site-packages, not `/work`) — and `pip install`
won't work anyway because the sandbox has no network. The standard
stack is pre-installed: `pandas`, `numpy`, `matplotlib`, `seaborn`,
`sklearn`, `scipy`. If you genuinely need something else, check the
existing import surface first; don't try to install.

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


def analyst_instruction() -> str:
    """The Analyst instruction, keyed to the schema shape this process talks to.

    The template names the message BM25 key, and an example naming a column the
    database does not have is how the sub-agent gets handed SQL that cannot run.
    """
    from gmail_search.store.schema_profile import selected_bm25_key

    return _ANALYST_INSTRUCTION_TEMPLATE.replace("{bm25_key}", selected_bm25_key())


def build_analyst_agent(*, model: str | None = None, instruction: str | None = None):
    """Analyst: computation over the retrieved evidence. `instruction`
    defaults to analyst_instruction(); callers pass the skill-matched one."""
    from gmail_search.agents.orchestration import stage_agent

    return stage_agent("analyst", instruction or analyst_instruction(), model=model, model_env="GMAIL_ANALYST_MODEL")


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
