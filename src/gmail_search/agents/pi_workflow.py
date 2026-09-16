"""Optional Pi workflow configuration and private child telemetry.

Each turn owns its agent settings and trace directory. Concurrent evaluations
never rewrite another conversation's models, extensions, or execution limits.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from gmail_search.agents.pi_protocol import UsageStats

EXTENSION = "/opt/pi-workflow/index.ts"
SKILLS = "/opt/pi-workflow/skills"
_SLUG = re.compile(r"^[A-Za-z0-9_-]{1,80}$")

INSTRUCTION = """
# Optional workflow capabilities

You choose how to investigate: use Gmail tools directly, use mcpScript to
batch/filter tool results, or delegate independent investigations with subagent.
Use todo when tracking outstanding work helps. None of these is a mandatory step.
The mail-researcher and mail-verifier agents have Gmail access and fresh contexts.
Give children focused questions, relevant source IDs, and a concise output
contract: findings, source message/thread IDs, and unresolved contradictions.
You may run them in the background, inspect or steer them, and compose workflows.
Use subagent action="guide" for its exact workflow syntax when needed.
Continue useful independent work while children run. Their completions will
return to you; incorporate their evidence before giving the final answer.
Stop or redirect investigations that are no longer needed. A small lookup may
need no delegation at all. Avoid fetching the same evidence repeatedly.
The selected model is shared by parent and children. All work belongs to this
turn and must finish within its deadline. Files under .workflow are runtime
state, never deliverables. Optional Gmail technique skills are available on demand.
"""


def resolve_profile(profile: str | None) -> str:
    value = profile or os.environ.get("GMAIL_PI_WORKFLOW_PROFILE", "baseline")
    if value not in ("baseline", "workflow"):
        raise ValueError(f"Unknown Pi workflow profile: {value!r}")
    return value


@dataclass(frozen=True)
class WorkflowFiles:
    directory: Path
    env: dict[str, str]


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_text(json.dumps(value, indent=2) + "\n")
    path.chmod(0o600)


def prepare_workflow(
    workspaces_root: Path,
    workspace: str,
    session_id: str,
    *,
    models_path: Path = Path("deploy/pi/models.json"),
    mcp_path: Path = Path("deploy/pi/mcp.json"),
    timeout: float = 900,
) -> WorkflowFiles:
    for value in (workspace, session_id):
        if not _SLUG.fullmatch(value):
            raise ValueError("Invalid workflow workspace/session identifier")
    directory = workspaces_root / workspace / ".workflow" / session_id
    directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    remote = f"/workspaces/{workspace}/.workflow/{session_id}"
    config = json.loads(mcp_path.read_text())
    # Absolute path is essential: children can select another working directory.
    config["mcpServers"]["gmail"]["bearerToken"] = f"!cat /workspaces/{workspace}/.session-token"
    _write_json(directory / "mcp.json", config)
    # Native child mcp: selectors use this discovery path and the adapter's
    # metadata cache; they do not read its explicit --mcp-config argument.
    _write_json(directory / "agent/mcp.json", config)
    _write_json(directory / "agent/models.json", json.loads(models_path.read_text()))
    _write_json(directory / "agent/settings.json", {
        "subagents": {
            "disableBuiltins": True,
            "agentScanDirs": ["/opt/pi-workflow/agents"],
            "defaultModel": "inherit",
            "defaultExtensions": ["/opt/pi-workflow/gmail-mcp.ts", "/opt/pi-workflow/telemetry.ts"],
            "modelScope": {"enforce": True, "strict": True, "allow": ["inherit"]},
        },
    })
    _write_json(directory / "agent/extensions/subagent/config.json", {
        "maxActiveAsyncRunsPerSession": 3,
        "maxSubagentSpawnsPerSession": 12,
        "maxSubagentSpawnsPerRun": 12,
        "maxSubagentDepth": 1,
        "timeoutMs": int(timeout * 1000),
        "toolDescriptionMode": "compact",
    })
    return WorkflowFiles(directory, {
        "PI_CODING_AGENT_DIR": f"{remote}/agent",
        "GMS_MCP_CONFIG": f"{remote}/mcp.json",
        "GMS_WORKFLOW_TRACE": f"{remote}/events.jsonl",
    })


def read_trace(path: Path) -> tuple[list[dict], bool]:
    if not path.exists():
        return [], False
    records = []
    seen = set()
    complete = True
    for line in path.read_text().splitlines():
        try:
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError("Non-object trace event")
        except ValueError:
            complete = False
            continue
        key = event.get("event_id")
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        records.append(event)
    opened = {r.get("pi_session_id") for r in records if r.get("type") == "session_start"}
    closed = {r.get("pi_session_id") for r in records if r.get("type") == "session_end"}
    observed = {r.get("pi_session_id") for r in records}
    complete = complete and bool(opened) and None not in observed and opened == closed == observed
    complete = complete and not any(r.get("type") == "accounting_gap" for r in records)
    return records, complete


def parent_usage(events: list[dict]) -> list[tuple[str, UsageStats]]:
    """Per-turn root usage; excludes usage projected into subagent tool results."""
    return _role_usage(events, parent=True)


def child_usage(events: list[dict]) -> list[tuple[str, UsageStats]]:
    return _role_usage(events, parent=False)


def _role_usage(events: list[dict], *, parent: bool) -> list[tuple[str, UsageStats]]:
    grouped: dict[str, list[dict]] = {}
    seen = set()
    for event in events:
        if event.get("type") not in ("message_end", "summary_usage") or event.get("parent") is not parent:
            continue
        identity = event.get("event_id")
        if identity and identity in seen:
            continue
        seen.add(identity)
        usage = event.get("usage")
        if not isinstance(usage, dict):
            continue
        model = f"{event.get('provider', '')}/{event.get('model', '')}"
        grouped.setdefault(model, []).append(usage)
    result = []
    for model, rows in grouped.items():
        costs = [r["cost"].get("total") if isinstance(r.get("cost"), dict) else None for r in rows]
        result.append((model, UsageStats(
            input_tokens=sum(int(r.get("input") or 0) for r in rows),
            output_tokens=sum(int(r.get("output") or 0) for r in rows),
            cache_read_tokens=sum(int(r.get("cacheRead") or 0) for r in rows),
            cache_write_tokens=sum(int(r.get("cacheWrite") or 0) for r in rows),
            cost_usd=sum(costs) if all(isinstance(c, (int, float)) for c in costs) else None,
        )))
    return result
