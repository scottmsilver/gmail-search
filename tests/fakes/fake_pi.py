"""Scripted stand-in for `pi --mode rpc`, driven by FAKE_PI_SCRIPT."""

from __future__ import annotations

import json
import os
import sys
import time


def _load_script() -> dict:
    with open(os.environ["FAKE_PI_SCRIPT"]) as f:
        return json.load(f)


def _emit(record: dict) -> None:
    sys.stdout.write(json.dumps(record) + "\n")
    sys.stdout.flush()


def _handle_prompt(cmd: dict, script: dict) -> None:
    _emit({"id": cmd.get("id"), "type": "response", "command": "prompt", "success": True})
    stderr_bytes = script.get("stderr_bytes", 0)
    if stderr_bytes:
        sys.stderr.write("x" * stderr_bytes)
        sys.stderr.flush()
    for ev in script.get("events", []):
        if ev.get("type") == "agent_end":
            time.sleep(float(script.get("delay_before_end", 0.0)))
        _emit(ev)


def _handle_stats(cmd: dict, script: dict) -> None:
    _emit(
        {
            "id": cmd.get("id"),
            "type": "response",
            "command": "get_session_stats",
            "success": True,
            "data": script.get("stats", {}),
        }
    )


def main() -> None:
    script = _load_script()
    for line in sys.stdin:
        line = line.rstrip("\r\n")
        if not line:
            continue
        cmd = json.loads(line)
        kind = cmd.get("type")
        if kind == "prompt":
            _handle_prompt(cmd, script)
        elif kind == "get_session_stats":
            _handle_stats(cmd, script)
        elif kind == "abort":
            _emit({"id": cmd.get("id"), "type": "response", "command": "abort", "success": True})
            _emit({"type": "agent_end", "messages": []})
            return
    return


if __name__ == "__main__":
    main()
