"""Pure helpers over pi's RPC/JSONL records.

Everything here is a function of its inputs: no subprocess, no DB,
no clock. `pi_rpc.py` owns the process; `runtime_pi.py` owns the turn.
Shapes follow packages/coding-agent/docs/rpc.md in the pi repo.
"""

from __future__ import annotations

from dataclasses import dataclass

RESPONSE_CLIP_CHARS = 4000

_PI_ISOLATION_FLAGS = ("--no-extensions", "--no-skills", "--no-context-files", "--no-prompt-templates")


def _as_dict(value) -> dict:
    return dict(value) if isinstance(value, dict) else {"value": value}


def _text_of_content(content) -> str:
    if not isinstance(content, list):
        return ""
    return "".join(str(b.get("text") or "") for b in content if isinstance(b, dict) and b.get("type") == "text")


def tool_call_args_entry(ev: dict) -> dict:
    return {"name": str(ev.get("toolName") or ""), "args": _as_dict(ev.get("args") or {})}


def tool_call_response_entry(ev: dict, *, clip: int = RESPONSE_CLIP_CHARS) -> dict:
    result = ev.get("result") if isinstance(ev.get("result"), dict) else {}
    text = _text_of_content(result.get("content"))[:clip]
    return {"name": str(ev.get("toolName") or ""), "response": {"text": text, "is_error": bool(ev.get("isError"))}}


def assistant_text(ev: dict) -> str | None:
    if ev.get("type") != "message_end":
        return None
    message = ev.get("message") if isinstance(ev.get("message"), dict) else {}
    if message.get("role") != "assistant":
        return None
    text = _text_of_content(message.get("content"))
    return text or None


def bash_as_run_code(start: dict, end: dict, *, clip: int = RESPONSE_CLIP_CHARS) -> list[dict]:
    """Present a pi `bash` call in the `run_code` shape the analyst
    panel understands. `artifacts` is always empty: files reach the
    UI through `publish_artifact_batch` or the auto-publish sweep."""
    command = str(_as_dict(start.get("args") or {}).get("command") or "")
    stdout = tool_call_response_entry(end, clip=clip)["response"]["text"]
    return [
        {"name": "run_code", "args": {"code": command}},
        {"name": "run_code", "response": {"stdout": stdout, "artifacts": []}},
    ]


@dataclass(frozen=True)
class UsageStats:
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost_usd: float | None


def usage_from_stats_response(resp: dict) -> UsageStats:
    data = resp.get("data") if isinstance(resp.get("data"), dict) else {}
    tokens = data.get("tokens") if isinstance(data.get("tokens"), dict) else {}
    cost = data.get("cost")
    return UsageStats(
        input_tokens=int(tokens.get("input") or 0),
        output_tokens=int(tokens.get("output") or 0),
        cache_read_tokens=int(tokens.get("cacheRead") or 0),
        cache_write_tokens=int(tokens.get("cacheWrite") or 0),
        cost_usd=float(cost) if isinstance(cost, (int, float)) else None,
    )


def _session_flags(session_path: str | None) -> list[str]:
    return ["--session", session_path] if session_path else ["--no-session"]


def _thinking_flags(thinking: str | None) -> list[str]:
    return ["--thinking", thinking] if thinking else []


def _builtin_tools_flags(builtin_tools: bool) -> list[str]:
    return [] if builtin_tools else ["--no-builtin-tools"]


def build_pi_argv(
    *,
    container: str,
    session_id: str,
    workspace: str,
    session_path: str | None,
    extension_path: str,
    model: str,
    thinking: str | None,
    system_prompt: str,
    builtin_tools: bool = True,
) -> list[str]:
    """argv for one turn. Secrets are NOT here — the service token and
    provider key live in the container's own environment."""
    return [
        "docker",
        "exec",
        "-i",
        "-e",
        f"GMS_SESSION_ID={session_id}",
        "-w",
        f"/workspaces/{workspace}",
        container,
        "pi",
        "--mode",
        "rpc",
        *_PI_ISOLATION_FLAGS,
        *_builtin_tools_flags(builtin_tools),
        "-e",
        extension_path,
        *_session_flags(session_path),
        "--model",
        model,
        *_thinking_flags(thinking),
        "--system-prompt",
        system_prompt,
    ]
