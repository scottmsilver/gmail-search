"""A/B compare deep-agent backends by replaying canonical queries.

Runs each query against `GMAIL_DEEP_BACKEND=adk` and
`GMAIL_DEEP_BACKEND=claude_code`, captures the SSE event stream, and
emits a JSON report with per-stage metrics.

Assumes:
- Postgres is reachable (project default config).
- `GEMINI_API_KEY` is set (ADK path).
- The claudebox container is running on :8765 (claude_code path).
- The MCP tools server is running on :7878 (claude_code path).
- Workspace dir `deploy/claudebox/workspaces/` exists.

Usage:
    python scripts/run_deep_compare.py                 # 3 default queries
    python scripts/run_deep_compare.py "my query"      # custom queries
    python scripts/run_deep_compare.py --backends adk  # subset
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any

DEFAULT_QUERIES = [
    "Plot my emails per month for 2024 as a PNG.",
    "tell me about how much we've paid landmarks west for our renovation in park city",
    "in 2003 in april what was i up to? who did i talk to a bunch?",
]


def _parse_sse_frame(frame: str) -> tuple[str, dict] | None:
    """Pull `event:` and `data:` lines out of one SSE frame."""
    event_match = re.search(r"^event: (.+)$", frame, re.MULTILINE)
    data_match = re.search(r"^data: (.+)$", frame, re.MULTILINE)
    if not event_match or not data_match:
        return None
    try:
        return event_match.group(1), json.loads(data_match.group(1))
    except json.JSONDecodeError:
        return None


def _new_metrics() -> dict[str, Any]:
    return {
        "events": [],
        "events_by_kind": {},
        "tool_calls": 0,
        "artifacts": [],
        "cite_refs": 0,
        "critic_rounds": 0,
        "critic_accepted": None,
        "final_text": None,
        "total_cost_usd": 0.0,
        "errors": [],
    }


def _consume_event(metrics: dict[str, Any], event_type: str, data: dict) -> None:
    """Update metrics from one SSE event."""
    metrics["events"].append({"kind": event_type, "data": data})
    metrics["events_by_kind"][event_type] = metrics["events_by_kind"].get(event_type, 0) + 1

    payload = data.get("payload", {}) if isinstance(data, dict) else {}

    if event_type == "tool_call":
        metrics["tool_calls"] += 1
    elif event_type == "evidence":
        metrics["cite_refs"] = max(metrics["cite_refs"], len(payload.get("cite_refs") or []))
    elif event_type == "analysis":
        metrics["artifacts"].extend(payload.get("artifact_ids") or [])
    elif event_type == "critique":
        metrics["critic_rounds"] = max(metrics["critic_rounds"], payload.get("round", 0) + 1)
        if "accepted" in payload:
            metrics["critic_accepted"] = payload["accepted"]
    elif event_type == "final":
        metrics["final_text"] = payload.get("text") or payload.get("answer")
    elif event_type == "cost":
        metrics["total_cost_usd"] = max(metrics["total_cost_usd"], payload.get("turn_total_usd", 0.0))
    elif event_type == "error":
        metrics["errors"].append(data)


_PER_STAGE_MODEL_VARS = (
    "GMAIL_PLANNER_MODEL",
    "GMAIL_RETRIEVER_MODEL",
    "GMAIL_ANALYST_MODEL",
    "GMAIL_WRITER_MODEL",
    "GMAIL_CRITIC_MODEL",
)


_CLAUDE_MODEL_DEFAULT = "sonnet"


def _scope_model_envs(backend: str) -> None:
    """ADK and claude_code accept different model name spaces. Per-stage
    overrides like `sonnet` work for claudebox but blow up ADK. Strip
    them before invoking ADK and ensure they're set when invoking
    claudebox."""
    if backend == "adk":
        for k in _PER_STAGE_MODEL_VARS:
            os.environ.pop(k, None)
    elif backend == "claude_code":
        for k in _PER_STAGE_MODEL_VARS:
            os.environ.setdefault(k, _CLAUDE_MODEL_DEFAULT)


async def _run_one_turn(question: str, backend: str, timeout_s: float) -> dict[str, Any]:
    """Run one deep-mode turn end-to-end and capture metrics."""
    os.environ["GMAIL_DEEP_BACKEND"] = backend
    os.environ["GMAIL_DEEP_REAL"] = "1"
    _scope_model_envs(backend)

    from gmail_search.agents.service import _real_run
    from gmail_search.agents.session import create_session, new_session_id
    from gmail_search.store.db import get_connection

    db_path = Path.cwd() / "data" / "gmail_search.db"

    session_id = new_session_id()
    conn = get_connection(db_path)
    try:
        create_session(
            conn,
            session_id=session_id,
            conversation_id=None,
            mode="deep",
            question=question,
        )
    finally:
        conn.close()

    metrics = _new_metrics()
    metrics["session_id"] = session_id
    metrics["backend"] = backend
    metrics["question"] = question

    started_at = time.monotonic()
    try:
        async with asyncio.timeout(timeout_s):
            async for frame in _real_run(db_path, session_id, question, default_model=None):
                parsed = _parse_sse_frame(frame)
                if parsed is None:
                    continue
                event_type, data = parsed
                _consume_event(metrics, event_type, data)
    except asyncio.TimeoutError:
        metrics["errors"].append({"kind": "timeout", "after_s": timeout_s})
    except Exception as exc:
        metrics["errors"].append({"kind": "exception", "type": type(exc).__name__, "msg": str(exc)})
    metrics["wall_s"] = round(time.monotonic() - started_at, 2)
    return metrics


def _summarize(result: dict[str, Any]) -> dict[str, Any]:
    """Flat row for human consumption."""
    return {
        "backend": result["backend"],
        "wall_s": result["wall_s"],
        "cost_usd": round(result["total_cost_usd"], 4),
        "tool_calls": result["tool_calls"],
        "cite_refs": result["cite_refs"],
        "artifacts": len(result["artifacts"]),
        "critic_rounds": result["critic_rounds"],
        "critic_accepted": result["critic_accepted"],
        "final_chars": len(result["final_text"] or ""),
        "errors": len(result["errors"]),
        "session_id": result["session_id"],
    }


async def _run_compare(queries: list[str], backends: list[str], timeout_s: float) -> list[dict[str, Any]]:
    runs = []
    for question in queries:
        print(f"\n=== Q: {question}", flush=True)
        for backend in backends:
            print(f"  -> {backend} ...", end=" ", flush=True)
            result = await _run_one_turn(question, backend, timeout_s)
            print(
                f"done in {result['wall_s']}s, ${result['total_cost_usd']:.4f}, "
                f"{result['tool_calls']} tools, {len(result['artifacts'])} artifacts, "
                f"errors={len(result['errors'])}",
                flush=True,
            )
            runs.append(result)
    return runs


def _print_table(runs: list[dict[str, Any]]) -> None:
    rows = [_summarize(r) for r in runs]
    cols = [
        "backend",
        "wall_s",
        "cost_usd",
        "tool_calls",
        "cite_refs",
        "artifacts",
        "critic_rounds",
        "critic_accepted",
        "final_chars",
        "errors",
    ]
    widths = {c: max(len(c), max((len(str(r.get(c, ""))) for r in rows), default=0)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print("\n" + header)
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


def _save_report(runs: list[dict[str, Any]], out_path: Path) -> None:
    out_path.write_text(json.dumps(runs, indent=2, default=str))
    print(f"\nFull report -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("queries", nargs="*", help="Override the default canonical queries.")
    parser.add_argument("--backends", default="adk,claude_code", help="Comma-separated backend list.")
    parser.add_argument("--timeout", type=float, default=300.0, help="Per-turn timeout in seconds.")
    parser.add_argument("--out", default="scripts/deep_compare_report.json", help="JSON output path.")
    args = parser.parse_args()

    queries = args.queries or DEFAULT_QUERIES
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]

    runs = asyncio.run(_run_compare(queries, backends, args.timeout))
    _print_table(runs)
    _save_report(runs, Path(args.out))


if __name__ == "__main__":
    main()
