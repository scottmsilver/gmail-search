"""Tests for the claudebox per-session JSONL tailer.

Uses tmp_path + a real on-disk file so we exercise the actual file
polling loop (no httpx, no claudebox). The tailer is asyncio-driven
but each iteration is bounded by `poll_interval=0.02` to keep tests
fast.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from gmail_search.agents.jsonl_tail import encode_workspace_path, map_jsonl_event_to_tool_calls, tail_session_events


def _write_lines(path: Path, lines: list[str]) -> None:
    """Append each line + newline; create the file if absent."""
    with path.open("a", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")
        f.flush()


def _assistant_with_tool_use(name: str, input_dict: dict, tool_id: str = "tu_1") -> dict:
    return {
        "type": "assistant",
        "message": {
            "content": [
                {"type": "text", "text": f"calling {name}"},
                {"type": "tool_use", "id": tool_id, "name": name, "input": input_dict},
            ]
        },
    }


def _tool_result(tool_use_id: str, content: str) -> dict:
    return {
        "type": "user",
        "message": {
            "content": [
                {"type": "tool_result", "tool_use_id": tool_use_id, "content": content, "is_error": False},
            ]
        },
    }


# ── encode_workspace_path ─────────────────────────────────────────


def test_encode_workspace_path_simple():
    assert encode_workspace_path("/workspaces/deep-XYZ") == "-workspaces-deep-XYZ"


def test_encode_workspace_path_nested():
    assert encode_workspace_path("/workspaces/foo/bar") == "-workspaces-foo-bar"


# ── map_jsonl_event_to_tool_calls ─────────────────────────────────


def test_map_assistant_with_tool_use_returns_one_entry():
    ev = _assistant_with_tool_use("search_emails", {"q": "x"})
    out = map_jsonl_event_to_tool_calls(ev)
    assert out == [{"name": "search_emails", "args": {"q": "x"}}]


def test_map_user_tool_result_returns_empty():
    ev = _tool_result("tu_1", '{"ok": true}')
    assert map_jsonl_event_to_tool_calls(ev) == []


def test_map_queue_operation_returns_empty():
    assert map_jsonl_event_to_tool_calls({"type": "queue-operation", "data": {}}) == []


def test_map_assistant_text_only_returns_empty():
    ev = {"type": "assistant", "message": {"content": [{"type": "text", "text": "hi"}]}}
    assert map_jsonl_event_to_tool_calls(ev) == []


def test_map_assistant_with_multiple_blocks_returns_only_tool_use():
    ev = {
        "type": "assistant",
        "message": {
            "content": [
                {"type": "text", "text": "thinking"},
                {"type": "tool_use", "id": "a", "name": "search_emails", "input": {"q": "x"}},
                {"type": "text", "text": "done"},
                {"type": "tool_use", "id": "b", "name": "run_code", "input": {"code": "print(1)"}},
            ]
        },
    }
    out = map_jsonl_event_to_tool_calls(ev)
    assert out == [
        {"name": "search_emails", "args": {"q": "x"}},
        {"name": "run_code", "args": {"code": "print(1)"}},
    ]


def test_map_missing_keys_returns_empty():
    assert map_jsonl_event_to_tool_calls({}) == []
    assert map_jsonl_event_to_tool_calls({"type": "assistant"}) == []
    assert map_jsonl_event_to_tool_calls({"type": "assistant", "message": "not a dict"}) == []


# ── tail_session_events ───────────────────────────────────────────


async def _collect_events(path_dir: Path, *, write_after_start: list[dict] | None = None) -> list[dict]:
    """Run the tailer against `path_dir`, collect every emitted event,
    return them. Spawns a writer task that drops `write_after_start`
    rows into the .jsonl file 30ms after the tailer kicks off."""
    captured: list[dict] = []
    stop_event = asyncio.Event()

    async def on_event(parsed: dict) -> None:
        captured.append(parsed)

    async def writer() -> None:
        if write_after_start is None:
            return
        await asyncio.sleep(0.03)
        target = path_dir / "session1.jsonl"
        _write_lines(target, [json.dumps(ev) for ev in write_after_start])
        # let the tailer poll once more, then stop
        await asyncio.sleep(0.1)
        stop_event.set()

    async def stopper_after_grace() -> None:
        await asyncio.sleep(0.5)
        stop_event.set()

    if write_after_start is not None:
        await asyncio.gather(
            tail_session_events(
                path_dir, on_event, stop_event=stop_event, poll_interval=0.02, file_appearance_timeout=2.0
            ),
            writer(),
        )
    else:
        await asyncio.gather(
            tail_session_events(
                path_dir, on_event, stop_event=stop_event, poll_interval=0.02, file_appearance_timeout=2.0
            ),
            stopper_after_grace(),
        )
    return captured


def test_existing_file_lines_are_read(tmp_path):
    """When a .jsonl already exists at start, the tailer reads every
    complete line through to EOF before exiting on stop_event."""
    target = tmp_path / "abc.jsonl"
    events = [
        _assistant_with_tool_use("search_emails", {"q": "foo"}),
        _tool_result("tu_1", '{"ok": true}'),
    ]
    _write_lines(target, [json.dumps(e) for e in events])

    captured = asyncio.run(_collect_events(tmp_path))
    assert len(captured) == 2
    assert captured[0]["type"] == "assistant"
    assert captured[1]["type"] == "user"


def test_lines_appended_after_tail_starts_are_emitted(tmp_path):
    """The tailer must pick up appends to the file after it opened it."""
    # Pre-create the file so it's the chosen one.
    target = tmp_path / "session1.jsonl"
    target.write_text("")  # touch
    appended = [
        _assistant_with_tool_use("search_emails", {"q": "later"}),
        {"type": "queue-operation", "id": 1},
    ]
    captured = asyncio.run(_collect_events(tmp_path, write_after_start=appended))
    types = [c["type"] for c in captured]
    assert "assistant" in types
    assert "queue-operation" in types


def test_partial_line_held_until_newline_arrives(tmp_path):
    """If a chunk ends mid-line (no trailing newline), the tailer must
    NOT call on_event yet — it must buffer until the newline lands."""
    target = tmp_path / "s.jsonl"
    # Write a partial line first (no trailing newline)
    target.write_text('{"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "x"')

    captured: list[dict] = []
    stop_event = asyncio.Event()

    async def on_event(parsed: dict) -> None:
        captured.append(parsed)

    async def runner() -> None:
        async def append_rest() -> None:
            await asyncio.sleep(0.05)
            with target.open("a", encoding="utf-8") as f:
                f.write(', "name": "search_emails", "input": {"q": "x"}}]}}\n')
                f.flush()
            await asyncio.sleep(0.1)
            stop_event.set()

        await asyncio.gather(
            tail_session_events(
                tmp_path, on_event, stop_event=stop_event, poll_interval=0.02, file_appearance_timeout=2.0
            ),
            append_rest(),
        )

    asyncio.run(runner())
    assert len(captured) == 1
    assert captured[0]["type"] == "assistant"


def test_malformed_json_is_skipped(tmp_path):
    """A bad line must NOT crash the tailer; valid surrounding lines
    must still be emitted."""
    target = tmp_path / "s.jsonl"
    _write_lines(
        target, [json.dumps(_assistant_with_tool_use("a", {})), "{not json", json.dumps(_tool_result("tu_1", "ok"))]
    )

    captured = asyncio.run(_collect_events(tmp_path))
    # First and third lines parse; second is dropped silently.
    assert len(captured) == 2
    assert captured[0]["type"] == "assistant"
    assert captured[1]["type"] == "user"


def test_returns_cleanly_when_file_never_appears(tmp_path):
    """When `workspace_dir` is empty for the entire timeout, the
    tailer logs a warning and returns — no exception."""
    captured: list[dict] = []

    async def on_event(parsed: dict) -> None:
        captured.append(parsed)

    async def runner() -> None:
        stop_event = asyncio.Event()
        await tail_session_events(
            tmp_path, on_event, stop_event=stop_event, poll_interval=0.02, file_appearance_timeout=0.1
        )

    asyncio.run(runner())
    assert captured == []


def test_stop_event_fires_promptly_during_tail(tmp_path):
    """Setting `stop_event` mid-tail must return promptly without
    hanging (the next poll iteration drains and exits)."""
    target = tmp_path / "s.jsonl"
    _write_lines(target, [json.dumps(_assistant_with_tool_use("a", {}))])

    captured: list[dict] = []
    stop_event = asyncio.Event()

    async def on_event(parsed: dict) -> None:
        captured.append(parsed)

    async def runner() -> None:
        async def stop_quick() -> None:
            # Give the tailer a moment to read the existing line.
            await asyncio.sleep(0.05)
            stop_event.set()

        await asyncio.gather(
            tail_session_events(
                tmp_path, on_event, stop_event=stop_event, poll_interval=0.02, file_appearance_timeout=2.0
            ),
            stop_quick(),
        )

    asyncio.run(runner())
    assert len(captured) == 1


def test_picks_newest_jsonl_when_multiple_present(tmp_path):
    """When several .jsonl files exist, the tailer follows the most
    recently modified one (each `/run` is a new session)."""
    older = tmp_path / "old.jsonl"
    newer = tmp_path / "new.jsonl"
    _write_lines(older, [json.dumps({"type": "assistant", "message": {"content": []}})])
    # Bump newer's mtime ahead of older.
    import time as _time

    _time.sleep(0.01)
    _write_lines(newer, [json.dumps(_assistant_with_tool_use("search_emails", {"q": "newest"}))])

    captured = asyncio.run(_collect_events(tmp_path))
    # Should have read only newer.jsonl
    names_seen = [c.get("message", {}).get("content", [{}])[-1].get("name") for c in captured]
    assert "search_emails" in names_seen
