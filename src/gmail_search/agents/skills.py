"""SKILL.md loader for the deep-analysis agents.

Mirrors the Claude Code / Claude Agent SDK `SKILL.md` convention so
users can drop project-local (or personal) guidance into a standard
path and have it automatically injected into the relevant sub-agent's
prompt. The Claude Agent SDK doesn't ship a built-in loader; we
implement the subset we need for ADK.

Path conventions (walked in this order, later paths override earlier):
  1. ~/.claude/skills/<name>/SKILL.md                 (personal)
  2. <repo_root>/.claude/skills/<name>/SKILL.md       (project)

SKILL.md frontmatter (YAML):
  name:        slug the skill is addressed by
  description: natural-language summary used for semantic matching
  when_to_use: extra trigger phrases (concatenated with description
               when scoring)
  paths:       comma-separated glob(s); if set, skill only activates
               when the task text mentions a matching path/file
  agent:       which sub-agent this skill is for. One of:
               'planner' | 'retriever' | 'analyst' | 'writer' |
               'critic' | 'all'. Default 'all'.

Anything else in the frontmatter is kept but ignored by this loader —
forward-compatible with Claude Code's richer shape (allowed-tools,
model, effort, etc.) which doesn't apply to an ADK-hosted agent.

Matching: given a task description and the target sub-agent, we score
every loaded skill by counting keyword overlap between the task and
the skill's description + when_to_use, AND requiring the skill's
`agent` to be 'all' or match the target. No ML; deterministic so the
tests are stable.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


# Sub-agents the deep-analysis pipeline knows about. A skill with
# `agent: planner` only injects into the Planner's prompt; `agent: all`
# (the default) injects everywhere.
KNOWN_AGENTS = frozenset({"planner", "retriever", "analyst", "writer", "critic", "all"})

# Per-body cap: any single SKILL.md body longer than this gets clipped.
# A 200KB SKILL.md could otherwise blow Gemini's 1M-token cap by itself.
# orchestration.STAGE_FIELD_CHAR_CAP clips evidence/analysis but not
# the instruction, so we enforce a separate budget here.
SKILL_BODY_CHAR_CAP = 8_000

# Sum cap across ALL matched skill bodies in a single injection.
# Even if individual bodies are under the per-body cap, three or four
# verbose skills could still dominate the prompt — this is the
# total-injection ceiling.
SKILL_BODIES_TOTAL_CHAR_CAP = 24_000

# Marker appended after a clipped body so the LLM knows content was
# cut. Explicit text rather than a silent "..." so the agent can
# decide whether to ask the user for the missing detail.
_TRUNCATION_MARKER = "\n\n[...truncated by skill loader; see SKILL.md for full content]"


@dataclass
class Skill:
    """One loaded skill. `body` is the markdown content after the
    frontmatter — that's what gets injected into the agent prompt."""

    name: str
    description: str
    when_to_use: str
    agent: str  # one of KNOWN_AGENTS
    paths: list[str] = field(default_factory=list)
    body: str = ""
    source_path: Path | None = None
    # Everything else from frontmatter; unused by matching but kept so
    # downstream code (logging / debugging) can peek.
    extra: dict[str, str] = field(default_factory=dict)


# ── Frontmatter parsing ─────────────────────────────────────────────
#
# We intentionally do NOT import pyyaml — the frontmatter shape we
# care about is a flat `key: value` map with no nesting, no lists, no
# block scalars. A 20-line parser is easier to reason about than a
# transitive YAML dep, and trivially covers the SKILL.md shape.

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", re.DOTALL)
_SPACE_SPLIT_RE = re.compile(r"\s*[,\s]\s*")


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split a SKILL.md into (frontmatter dict, body). Returns ({}, text)
    if no frontmatter is present — a headless SKILL.md is treated as
    pure content under default-empty metadata."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    raw_fm, body = m.group(1), m.group(2)
    fm: dict[str, str] = {}
    for line in raw_fm.splitlines():
        line = line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        fm[key.strip()] = value.strip().strip('"').strip("'")
    return fm, body


def _parse_skill_file(path: Path) -> Skill | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning(f"skill load: cannot read {path}: {e}")
        return None
    fm, body = _parse_frontmatter(text)
    # Name defaults to the parent directory (the <skill-name>/ folder
    # convention) so authors don't have to keep name in two places.
    name = fm.get("name") or path.parent.name
    agent = fm.get("agent", "all").strip().lower()
    if agent not in KNOWN_AGENTS:
        logger.warning(f"skill {name}: agent={agent!r} is not one of {sorted(KNOWN_AGENTS)}; treating as 'all'")
        agent = "all"
    paths = [p for p in _SPACE_SPLIT_RE.split(fm.get("paths", "").strip()) if p]
    description = fm.get("description", "").strip()
    when_to_use = fm.get("when_to_use", "").strip()
    # Keep the unrecognized frontmatter keys around for debug logging.
    known_keys = {"name", "description", "when_to_use", "agent", "paths"}
    extra = {k: v for k, v in fm.items() if k not in known_keys}
    return Skill(
        name=name,
        description=description,
        when_to_use=when_to_use,
        agent=agent,
        paths=paths,
        body=body.strip(),
        source_path=path,
        extra=extra,
    )


def default_skill_roots(project_root: Path | None = None) -> list[Path]:
    """Where to look for SKILL.md files. Personal first, then project —
    matches Claude Code's precedence (project overrides personal on
    name collision, handled by `load_skills` via an insertion-order
    dict)."""
    roots: list[Path] = []
    home = Path(os.environ.get("HOME", "~")).expanduser()
    roots.append(home / ".claude" / "skills")
    if project_root is not None:
        roots.append(project_root / ".claude" / "skills")
    return [r for r in roots if r.exists()]


def load_skills(roots: Iterable[Path] | None = None) -> list[Skill]:
    """Walk each root for `<name>/SKILL.md` and return the parsed
    skills. Later-root skills with the same name override earlier
    ones — same precedence rule as Claude Code."""
    if roots is None:
        roots = default_skill_roots()
    by_name: dict[str, Skill] = {}
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            skill_file = child / "SKILL.md"
            if not skill_file.is_file():
                continue
            skill = _parse_skill_file(skill_file)
            if skill is None:
                continue
            by_name[skill.name] = skill
    return list(by_name.values())


# ── Matching ────────────────────────────────────────────────────────


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]{2,}")


def _tokens(text: str) -> set[str]:
    """Lowercase content words of length >= 3. Intentionally dumb —
    no stemming, no stopwords — so matches are predictable in tests.
    This is a semantic HINT, not a classifier."""
    return {m.group(0).lower() for m in _WORD_RE.finditer(text or "")}


def _score(skill: Skill, task_tokens: set[str]) -> int:
    """Raw keyword-overlap count between the task and the skill's
    description + when_to_use. Zero means 'not relevant'. We DON'T
    weight description higher than when_to_use — if the author
    duplicates a phrase, let it count twice; it's cheap and gives
    them a way to boost."""
    skill_tokens = _tokens(skill.description) | _tokens(skill.when_to_use)
    return len(task_tokens & skill_tokens)


def match_skills(
    skills: list[Skill],
    task_description: str,
    *,
    agent_name: str,
    min_score: int = 1,
    top_k: int = 3,
) -> list[Skill]:
    """Return up to `top_k` skills that (a) target this `agent_name`
    (or 'all') and (b) share at least `min_score` content words with
    the task. Sorted by score desc then name for stable ordering.
    """
    task_tokens = _tokens(task_description)
    eligible = [s for s in skills if s.agent in {agent_name, "all"}]
    scored = [(s, _score(s, task_tokens)) for s in eligible]
    scored = [(s, n) for s, n in scored if n >= min_score]
    scored.sort(key=lambda it: (-it[1], it[0].name))
    return [s for s, _ in scored[:top_k]]


def _clip_skill_body(body: str, cap: int) -> str:
    """Clip a single skill body to `cap` chars, appending the
    truncation marker so the LLM sees content was cut. Bodies
    already within budget pass through unchanged."""
    if len(body) <= cap:
        return body
    return body[:cap].rstrip() + _TRUNCATION_MARKER


def _budget_skill_body(body: str, remaining: int) -> str | None:
    """Trim a skill body so it fits in the remaining sum-budget.
    Returns None if there's no room left at all (caller should stop
    appending further skills)."""
    if remaining <= 0:
        return None
    if len(body) <= remaining:
        return body
    return body[:remaining].rstrip() + _TRUNCATION_MARKER


def inject_skill_instructions(base_prompt: str, matched: list[Skill]) -> str:
    """Append matched skill bodies to a sub-agent's base prompt under
    a dedicated section the LLM can reference. Returns the base prompt
    unchanged when no skills match — keeps the no-skills path free
    of noise.

    Bodies are capped per-skill (`SKILL_BODY_CHAR_CAP`) and across all
    skills (`SKILL_BODIES_TOTAL_CHAR_CAP`) so a runaway SKILL.md cannot
    blow Gemini's context budget — `orchestration.STAGE_FIELD_CHAR_CAP`
    only protects evidence and analysis, not the instruction itself.
    Truncated bodies carry an explicit marker so the agent knows
    content was cut.
    """
    if not matched:
        return base_prompt
    parts = [base_prompt.rstrip(), "", "<skills>"]
    remaining_budget = SKILL_BODIES_TOTAL_CHAR_CAP
    for skill in matched:
        clipped = _clip_skill_body(skill.body, SKILL_BODY_CHAR_CAP)
        budgeted = _budget_skill_body(clipped, remaining_budget)
        if budgeted is None:
            # Out of room across the whole injection — stop adding
            # more skills rather than emit half-empty headers.
            break
        header = f"## Skill: {skill.name}"
        if skill.description:
            header += f" — {skill.description}"
        parts.extend([header, budgeted, ""])
        remaining_budget -= len(budgeted)
    parts.append("</skills>")
    return "\n".join(parts)
