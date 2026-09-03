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
from collections import deque

logger = logging.getLogger(__name__)

_ids = itertools.count(1)
STDOUT_LINE_LIMIT = 64 * 1024 * 1024  # 64 MiB; agent_end embeds all messages


class PiRpcError(RuntimeError):
    """The pi process died or answered a request with success=false."""


class PiRpcClient:
    def __init__(self, proc: asyncio.subprocess.Process, pump_task: asyncio.Task) -> None:  # noqa: F821
        self._proc = proc
        self._pending: list[bytes] = []  # test seam for injected lines
        self.stray: list[dict] = []
        self._pump_task = pump_task
        self._stderr_tail: deque = deque(maxlen=1)  # last 4000 bytes

    @classmethod
    async def spawn(cls, argv: list[str]) -> "PiRpcClient":
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=STDOUT_LINE_LIMIT,
        )
        client = cls.__new__(cls)
        client._proc = proc
        client._pending = []
        client.stray = []
        client._stderr_tail = deque(maxlen=1)
        client._pump_task = asyncio.create_task(client._stderr_pump())
        return client

    @property
    def returncode(self) -> int | None:
        return self._proc.returncode

    def _inject_line(self, line: bytes) -> None:
        self._pending.append(line)

    async def _stderr_pump(self) -> None:
        """Read stderr in background, keep last 4000 bytes."""
        if self._proc.stderr is None:
            return
        tail = bytearray()
        try:
            while True:
                chunk = await self._proc.stderr.read(4096)
                if not chunk:
                    break
                tail.extend(chunk)
                if len(tail) > 4000:
                    tail = tail[-4000:]
        except asyncio.CancelledError:
            pass
        if tail:
            self._stderr_tail.append(bytes(tail))

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
        self._pump_task.cancel()
        try:
            await self._pump_task
        except asyncio.CancelledError:
            pass
        await self._log_stderr_tail()

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
        self._pump_task.cancel()
        try:
            await self._pump_task
        except asyncio.CancelledError:
            pass
        await self._log_stderr_tail()

    async def _log_stderr_tail(self) -> None:
        if self._stderr_tail:
            tail = self._stderr_tail[0]
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
