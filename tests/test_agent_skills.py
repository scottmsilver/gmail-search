"""Tests for the SKILL.md loader + matcher used by the deep-analysis
agents. Covers frontmatter parsing, multi-root precedence, keyword
matching with the per-agent scoping rule, and prompt injection.
"""

from __future__ import annotations

from pathlib import Path

from gmail_search.agents.skills import Skill, inject_skill_instructions, load_skills, match_skills


def _write_skill(base: Path, name: str, frontmatter: dict[str, str], body: str) -> Path:
    """Helper: write a SKILL.md into `base/<name>/SKILL.md`."""
    skill_dir = base / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    fm_lines = ["---"] + [f"{k}: {v}" for k, v in frontmatter.items()] + ["---", ""]
    (skill_dir / "SKILL.md").write_text("\n".join(fm_lines) + body, encoding="utf-8")
    return skill_dir / "SKILL.md"


def test_load_skills_parses_frontmatter_and_body(tmp_path):
    _write_skill(
        tmp_path,
        "receipts-analyzer",
        {
            "name": "receipts-analyzer",
            "description": "Preferred totals + taxes for spending questions",
            "when_to_use": "receipt total spending",
            "agent": "analyst",
            "paths": "",
        },
        "Always prefer YTD totals over per-month breakdowns.",
    )

    skills = load_skills([tmp_path])
    assert len(skills) == 1
    s = skills[0]
    assert s.name == "receipts-analyzer"
    assert s.description.startswith("Preferred totals")
    assert s.when_to_use == "receipt total spending"
    assert s.agent == "analyst"
    assert "YTD totals" in s.body


def test_load_skills_name_defaults_to_directory(tmp_path):
    """Name can be omitted — falls back to the containing directory.
    Matches Claude Code behaviour."""
    _write_skill(tmp_path, "my-skill", {"description": "desc"}, "body text")
    skills = load_skills([tmp_path])
    assert skills[0].name == "my-skill"


def test_load_skills_headless_file_is_body_only(tmp_path):
    """A SKILL.md with no --- frontmatter is treated as pure body.
    Covers the "drop a markdown note" path without requiring authors
    to learn frontmatter first."""
    skill_dir = tmp_path / "headless"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("just some instructions", encoding="utf-8")
    skills = load_skills([tmp_path])
    assert skills[0].name == "headless"
    assert skills[0].body == "just some instructions"
    assert skills[0].agent == "all"


def test_load_skills_project_overrides_personal(tmp_path):
    """Same-named skill in project scope wins over personal. Order of
    roots in the call is personal-first, project-last."""
    personal = tmp_path / "personal"
    project = tmp_path / "project"
    _write_skill(personal, "shared", {"description": "old"}, "personal body")
    _write_skill(project, "shared", {"description": "new"}, "project body")

    skills = load_skills([personal, project])
    assert len(skills) == 1
    assert skills[0].body == "project body"


def test_unknown_agent_falls_back_to_all(tmp_path):
    """Typos in `agent:` shouldn't cause the skill to disappear —
    warn and treat as 'all' so the content still reaches every
    sub-agent."""
    _write_skill(
        tmp_path,
        "typo",
        {"description": "x", "agent": "plarnner"},
        "body",
    )
    skills = load_skills([tmp_path])
    assert skills[0].agent == "all"


def test_match_skills_scopes_to_agent():
    """A skill with agent=analyst must NOT match when we're prompting
    the planner."""
    skills = [
        Skill(name="a", description="spending totals receipts", when_to_use="", agent="analyst", body="x"),
        Skill(name="b", description="spending totals receipts", when_to_use="", agent="planner", body="y"),
        Skill(name="c", description="irrelevant weather", when_to_use="", agent="all", body="z"),
    ]
    matched = match_skills(skills, "Summarize my spending totals on receipts", agent_name="analyst")
    names = [s.name for s in matched]
    assert "a" in names
    assert "b" not in names
    assert "c" not in names


def test_match_skills_returns_top_k_by_overlap():
    """Higher keyword overlap sorts first; ties break by name."""
    skills = [
        Skill(name="zeta", description="monthly spending receipt totals", when_to_use="", agent="all", body="z"),
        Skill(name="alpha", description="monthly receipts", when_to_use="", agent="all", body="a"),
        Skill(name="beta", description="unrelated content", when_to_use="", agent="all", body="b"),
    ]
    matched = match_skills(skills, "monthly spending and receipt totals please", agent_name="analyst", top_k=2)
    assert [s.name for s in matched] == ["zeta", "alpha"]


def test_match_skills_returns_empty_when_nothing_relevant():
    skills = [Skill(name="x", description="cooking recipes", when_to_use="", agent="all", body="b")]
    assert match_skills(skills, "analyze stock prices", agent_name="analyst") == []


def test_inject_skill_instructions_appends_section():
    base = "You are the Analyst. Do analysis."
    matched = [
        Skill(name="s1", description="Skill one", when_to_use="", agent="analyst", body="instruction one"),
        Skill(name="s2", description="", when_to_use="", agent="analyst", body="instruction two"),
    ]
    out = inject_skill_instructions(base, matched)
    assert out.startswith(base.rstrip())
    assert "<skills>" in out
    assert "## Skill: s1 — Skill one" in out
    assert "instruction one" in out
    assert "## Skill: s2" in out  # no description → no em-dash
    assert "instruction two" in out
    assert out.rstrip().endswith("</skills>")


def test_inject_skill_instructions_noop_when_no_matches():
    """No matched skills → base prompt unchanged (no orphan <skills>
    tags). Keeps the hot path clean."""
    out = inject_skill_instructions("hello", [])
    assert out == "hello"
