"""PiRpcClient against the scripted fake pi process."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from gmail_search.agents import pi_rpc

FAKE_PI = Path(__file__).parent / "fakes" / "fake_pi.py"


def _write_script(tmp_path: Path, events: list[dict], stats: dict | None = None, delay: float = 0.0) -> Path:
    p = tmp_path / "script.json"
    p.write_text(json.dumps({"events": events, "stats": stats or {}, "delay_before_end": delay}))
    return p


def _argv() -> list[str]:
    return [sys.executable, str(FAKE_PI)]


def test_prompt_streams_events_until_agent_end(tmp_path, monkeypatch):
    script = _write_script(tmp_path, [{"type": "agent_start"}, {"type": "agent_end", "messages": []}])
    monkeypatch.setenv("FAKE_PI_SCRIPT", str(script))

    async def run():
        client = await pi_rpc.PiRpcClient.spawn(_argv())
        await client.send({"id": "p1", "type": "prompt", "message": "hi"})
        seen = []
        while True:
            rec = await client.read_record(timeout=5.0)
            assert rec is not None
            seen.append(rec["type"])
            if rec["type"] == "agent_end":
                break
        await client.close()
        return seen

    assert asyncio.run(run()) == ["response", "agent_start", "agent_end"]


def test_request_returns_matching_response_and_stashes_strays(tmp_path, monkeypatch):
    script = _write_script(tmp_path, [{"type": "agent_end", "messages": []}], stats={"cost": 0.5})
    monkeypatch.setenv("FAKE_PI_SCRIPT", str(script))

    async def run():
        client = await pi_rpc.PiRpcClient.spawn(_argv())
        await client.send({"type": "prompt", "message": "hi"})  # response + agent_end become strays
        resp = await client.request({"type": "get_session_stats"}, timeout=5.0)
        await client.close()
        return resp, [r["type"] for r in client.stray]

    resp, strays = asyncio.run(run())
    assert resp["command"] == "get_session_stats" and resp["data"] == {"cost": 0.5}
    assert strays == ["response", "agent_end"]


def test_read_record_times_out_when_idle(tmp_path, monkeypatch):
    script = _write_script(tmp_path, [{"type": "agent_end", "messages": []}], delay=2.0)
    monkeypatch.setenv("FAKE_PI_SCRIPT", str(script))

    async def run():
        client = await pi_rpc.PiRpcClient.spawn(_argv())
        await client.send({"type": "prompt", "message": "hi"})
        await client.read_record(timeout=5.0)  # prompt response
        with pytest.raises(asyncio.TimeoutError):
            await client.read_record(timeout=0.2)
        await client.abort_and_close(grace=3.0)
        return client.returncode

    assert asyncio.run(run()) == 0


def test_read_record_returns_none_on_eof(tmp_path, monkeypatch):
    script = _write_script(tmp_path, [])
    monkeypatch.setenv("FAKE_PI_SCRIPT", str(script))

    async def run():
        client = await pi_rpc.PiRpcClient.spawn(_argv())
        await client.close()  # closes stdin → fake exits
        return await client.read_record(timeout=5.0)

    assert asyncio.run(run()) is None


def test_malformed_line_is_skipped(tmp_path, monkeypatch):
    script = _write_script(tmp_path, [{"type": "agent_end", "messages": []}])
    monkeypatch.setenv("FAKE_PI_SCRIPT", str(script))

    async def run():
        client = await pi_rpc.PiRpcClient.spawn(_argv())
        client._inject_line(b"not json\n")
        await client.send({"type": "prompt", "message": "hi"})
        first = await client.read_record(timeout=5.0)
        await client.close()
        return first["type"]

    assert asyncio.run(run()) == "response"
