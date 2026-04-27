"""Tail a claudebox per-session JSONL transcript and stream events.

Claudebox 2.1.90 writes one transcript file per `/run` session at
`/home/claude/.claude/projects/<encoded-workspace>/<sessionId>.jsonl`
inside the container. The same directory is host-mounted at
`deploy/claudebox/claude-config/projects/<encoded-workspace>/`, so we
can `tail -f` the file from the FastAPI side and surface tool_use
events to the UI as soon as the model emits them — instead of waiting
for `/run` to return and parsing the post-hoc message stream.

This module is deliberately backend-agnostic: no orchestrator, service,
or DB imports — it just polls a directory, finds the newest JSONL file,
and calls a user-supplied async callback per parsed line. That makes it
testable with `tmp_path` alone.

The encoding scheme: claudebox derives the project subdir from the
workspace path by replacing every `/` with `-`. Example:
`/workspaces/deep-XYZ` → `-workspaces-deep-XYZ`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)


def encode_workspace_path(workspace: str) -> str:
    """Map a container workspace path to claudebox's projects subdir name.

    Claudebox writes session transcripts under
    `~/.claude/projects/<encoded>/`, where `<encoded>` is the workspace
    path with every `/` replaced by `-`. Example:
    `/workspaces/deep-XYZ` -> `-workspaces-deep-XYZ`.
    """
    return workspace.replace("/", "-")


def map_jsonl_event_to_tool_calls(jsonl_event: dict) -> list[dict]:
    """Pull tool_use entries out of one JSONL line.

    Returns a list of `{"name", "args"}` dicts in the orchestrator's
    canonical tool_call shape. Empty list for non-relevant lines (text,
    tool_result, queue-operation, last-prompt, malformed).

    For v1 we ONLY emit `tool_use` blocks. `tool_result` and free-text
    blocks are intentionally NOT mapped — that mapping is deferred to
    a v2 that adds matching UI handlers."""
    if not isinstance(jsonl_event, dict):
        return []
    if jsonl_event.get("type") != "assistant":
        return []
    message = jsonl_event.get("message")
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [_to_tool_call_entry(b) for b in content if _is_tool_use_block(b)]


def _is_tool_use_block(block: object) -> bool:
    return isinstance(block, dict) and block.get("type") == "tool_use"


def _to_tool_call_entry(block: dict) -> dict:
    """Coerce a tool_use content block into the orchestrator shape."""
    name = str(block.get("name") or "")
    args = block.get("input") or {}
    if not isinstance(args, dict):
        args = {"value": args}
    return {"name": name, "args": dict(args)}


async def tail_session_events(
    workspace_dir: Path,
    on_event: Callable[[dict], Awaitable[None]],
    *,
    stop_event: asyncio.Event,
    poll_interval: float = 0.2,
    file_appearance_timeout: float = 10.0,
) -> None:
    """Watch `workspace_dir` for the newest .jsonl file and tail it.

    Once a file appears, every newline-terminated JSON line is parsed
    and forwarded to `on_event`. Partial trailing lines (no newline)
    are buffered until the newline arrives. Malformed JSON is logged
    and skipped, never raised.

    Returns when `stop_event` is set (after draining the current
    poll's batch), OR if no file appears within
    `file_appearance_timeout`. We never raise — the goal is "best
    effort streaming"; the caller still has the post-hoc message
    parser as a backstop."""
    started_at = time.monotonic()
    jsonl_path = await _wait_for_jsonl_file(
        workspace_dir,
        deadline=started_at + file_appearance_timeout,
        stop_event=stop_event,
        poll_interval=poll_interval,
    )
    if jsonl_path is None:
        logger.warning(
            "tail_session_events: no .jsonl appeared in %s within %.1fs — streaming disabled for this turn",
            workspace_dir,
            file_appearance_timeout,
        )
        return
    await _tail_file_until_stop(
        jsonl_path,
        on_event,
        stop_event=stop_event,
        poll_interval=poll_interval,
    )


async def _wait_for_jsonl_file(
    workspace_dir: Path,
    *,
    deadline: float,
    stop_event: asyncio.Event,
    poll_interval: float,
) -> Path | None:
    """Block until the first `.jsonl` file appears in `workspace_dir`,
    or until `deadline` / `stop_event` fires. Returns the file path or
    None on timeout."""
    while time.monotonic() < deadline:
        if stop_event.is_set():
            return None
        candidate = _newest_jsonl_in(workspace_dir)
        if candidate is not None:
            return candidate
        await asyncio.sleep(poll_interval)
    return _newest_jsonl_in(workspace_dir)


def _newest_jsonl_in(workspace_dir: Path) -> Path | None:
    """Return the most-recently-modified .jsonl in `workspace_dir`,
    or None if the dir is missing or empty. Robust to the dir not
    existing yet — claudebox creates it on first session."""
    try:
        entries = list(workspace_dir.glob("*.jsonl"))
    except OSError:
        return None
    if not entries:
        return None
    return max(entries, key=lambda p: p.stat().st_mtime)


async def _tail_file_until_stop(
    jsonl_path: Path,
    on_event: Callable[[dict], Awaitable[None]],
    *,
    stop_event: asyncio.Event,
    poll_interval: float,
) -> None:
    """Open the file and read forward, polling for new bytes between
    iterations. Maintains a position cursor so each poll only sees new
    content. Holds partial trailing lines until their newline lands."""
    pending = ""
    try:
        with jsonl_path.open("r", encoding="utf-8", errors="replace") as f:
            while True:
                pending = await _drain_available_lines(f, pending, on_event)
                if stop_event.is_set():
                    # One final drain in case writer flushed between
                    # the previous read and the stop signal.
                    pending = await _drain_available_lines(f, pending, on_event)
                    return
                await asyncio.sleep(poll_interval)
    except FileNotFoundError:
        logger.warning("tail_session_events: file vanished mid-tail: %s", jsonl_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("tail_session_events: tail loop failed for %s: %s", jsonl_path, exc)


async def _drain_available_lines(
    f,
    pending: str,
    on_event: Callable[[dict], Awaitable[None]],
) -> str:
    """Read everything currently available, split on newlines, forward
    each complete line to `on_event`, and return whatever partial
    suffix is left for the next poll."""
    chunk = f.read()
    if not chunk:
        return pending
    buf = pending + chunk
    lines = buf.split("\n")
    # Last element is whatever sits after the final newline — empty if
    # the chunk ended on `\n`, partial otherwise. Hold it for next time.
    leftover = lines.pop()
    for line in lines:
        await _safe_dispatch_line(line, on_event)
    return leftover


async def _safe_dispatch_line(
    line: str,
    on_event: Callable[[dict], Awaitable[None]],
) -> None:
    """Parse one JSONL line and call `on_event`. Empty lines are
    skipped silently. Malformed JSON or callback failures are logged
    and swallowed — one bad line must not abort the whole tail."""
    stripped = line.strip()
    if not stripped:
        return
    try:
        parsed = json.loads(stripped)
    except ValueError as exc:
        logger.debug("tail_session_events: skipping malformed JSONL line: %s", exc)
        return
    try:
        await on_event(parsed)
    except Exception as exc:  # noqa: BLE001
        logger.warning("tail_session_events: on_event raised (non-fatal): %s", exc)
