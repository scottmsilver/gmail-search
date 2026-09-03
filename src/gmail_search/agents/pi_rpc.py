"""Minimal client for `pi --mode rpc` over a subprocess's stdin/stdout.

Protocol (pi repo, packages/coding-agent/docs/rpc.md): one JSON object
per line each way, LF-delimited only. Commands may carry an `id`; the
matching `response` record echoes it. Agent events stream on stdout
in between. asyncio's `readline()` splits on b"\\n" only, which is
exactly what the protocol requires.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging

logger = logging.getLogger(__name__)

_ids = itertools.count(1)


class PiRpcError(RuntimeError):
    """The pi process died or answered a request with success=false."""


class PiRpcClient:
    def __init__(self, proc: asyncio.subprocess.Process) -> None:
        self._proc = proc
        self._pending: list[bytes] = []  # test seam for injected lines
        self.stray: list[dict] = []

    @classmethod
    async def spawn(cls, argv: list[str]) -> "PiRpcClient":
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        return cls(proc)

    @property
    def returncode(self) -> int | None:
        return self._proc.returncode

    def _inject_line(self, line: bytes) -> None:
        self._pending.append(line)

    async def send(self, command: dict) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(command).encode() + b"\n")
        await self._proc.stdin.drain()

    async def _next_line(self, timeout: float) -> bytes:
        if self._pending:
            return self._pending.pop(0)
        assert self._proc.stdout is not None
        return await asyncio.wait_for(self._proc.stdout.readline(), timeout)

    async def read_record(self, timeout: float) -> dict | None:
        """Next parsed record, None on EOF. Malformed lines are logged
        and skipped. Raises asyncio.TimeoutError when nothing arrives
        within `timeout` seconds."""
        while True:
            line = await self._next_line(timeout)
            if not line:
                return None
            record = _parse_line(line)
            if record is not None:
                return record

    async def request(self, command: dict, *, timeout: float) -> dict:
        req_id = f"req-{next(_ids)}"
        await self.send({**command, "id": req_id})
        while True:
            rec = await self.read_record(timeout)
            if rec is None:
                raise PiRpcError(f"pi exited before answering {command.get('type')}")
            if rec.get("type") == "response" and rec.get("id") == req_id:
                if not rec.get("success", False):
                    raise PiRpcError(f"pi rejected {command.get('type')}: {rec.get('error')}")
                return rec
            self.stray.append(rec)

    async def close(self) -> None:
        if self._proc.stdin is not None and not self._proc.stdin.is_closing():
            self._proc.stdin.close()
        try:
            await asyncio.wait_for(self._proc.wait(), 5.0)
        except asyncio.TimeoutError:
            self._proc.kill()
            await self._proc.wait()

    async def abort_and_close(self, *, grace: float = 5.0) -> None:
        """Ask pi to stop, wait up to `grace` for it to exit, then kill."""
        try:
            await self.send({"type": "abort"})
        except (BrokenPipeError, ConnectionResetError):
            pass
        try:
            await asyncio.wait_for(self._proc.wait(), grace)
        except asyncio.TimeoutError:
            self._proc.kill()
            await self._proc.wait()
        await self._drain_stderr()

    async def _drain_stderr(self) -> None:
        if self._proc.stderr is None:
            return
        try:
            tail = await asyncio.wait_for(self._proc.stderr.read(4000), 1.0)
        except asyncio.TimeoutError:
            return
        if tail:
            logger.warning("pi stderr tail: %s", tail.decode(errors="replace"))


def _parse_line(line: bytes) -> dict | None:
    text = line.rstrip(b"\r\n")
    if not text:
        return None
    try:
        record = json.loads(text)
    except ValueError:
        logger.warning("pi rpc: skipping malformed line: %r", text[:200])
        return None
    return record if isinstance(record, dict) else None
