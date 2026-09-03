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
import time
from collections import deque

from gmail_search.agents.pi_protocol import redact_secrets

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
        self.killed = False

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
        client.killed = False
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
        within `timeout` seconds — an overall deadline, so a stream of
        malformed lines cannot keep resetting the clock."""
        return await self._read_record_by_deadline(time.monotonic() + timeout)

    async def _read_record_by_deadline(self, deadline: float) -> dict | None:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise asyncio.TimeoutError()
            line = await self._next_line(remaining)
            if not line:
                return None
            record = _parse_line(line)
            if record is not None:
                return record

    async def request(self, command: dict, *, timeout: float) -> dict:
        """Send `command` and wait for its matching response. `timeout`
        is an overall deadline covering every stray/malformed record
        skipped along the way, not a per-line timeout."""
        req_id = f"req-{next(_ids)}"
        await self.send({**command, "id": req_id})
        deadline = time.monotonic() + timeout
        while True:
            rec = await self._read_record_by_deadline(deadline)
            if rec is None:
                raise PiRpcError(f"pi exited before answering {command.get('type')}")
            if rec.get("type") == "response" and rec.get("id") == req_id:
                if not rec.get("success", False):
                    raise PiRpcError(f"pi rejected {command.get('type')}: {rec.get('error')}")
                return rec
            self.stray.append(rec)

    async def _drain_stdout(self) -> None:
        """Read stdout to EOF and discard it. Run concurrently with
        `proc.wait()` during shutdown: if pi writes enough
        events/messages after its last response to fill the pipe, it
        blocks on that write until someone reads — without this, that
        blocks pi from ever reaching the stdin-EOF check and deadlocks
        shutdown."""
        if self._proc.stdout is None:
            return
        try:
            while True:
                chunk = await self._proc.stdout.read(65536)
                if not chunk:
                    break
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            pass

    async def _wait_or_kill(self, timeout: float) -> None:
        """Wait up to `timeout` for the process to exit; kill it and
        record `self.killed = True` if it doesn't."""
        try:
            await asyncio.wait_for(self._proc.wait(), timeout)
        except asyncio.TimeoutError:
            self._proc.kill()
            await self._proc.wait()
            self.killed = True

    async def _finish_shutdown(self, drain_task: asyncio.Task) -> None:  # noqa: F821
        drain_task.cancel()
        try:
            await drain_task
        except asyncio.CancelledError:
            pass
        self._pump_task.cancel()
        try:
            await self._pump_task
        except asyncio.CancelledError:
            pass
        await self._log_stderr_tail()

    async def close(self) -> None:
        drain_task = asyncio.create_task(self._drain_stdout())
        if self._proc.stdin is not None and not self._proc.stdin.is_closing():
            self._proc.stdin.close()
        await self._wait_or_kill(5.0)
        await self._finish_shutdown(drain_task)

    async def abort_and_close(self, *, grace: float = 5.0) -> None:
        """Ask pi to stop, then close stdin so pi's RPC loop actually
        exits (`abort` only stops the current operation — pi stays
        alive reading stdin until it sees EOF). Wait up to `grace`,
        then kill."""
        try:
            await self.send({"type": "abort"})
        except (BrokenPipeError, ConnectionResetError):
            pass
        drain_task = asyncio.create_task(self._drain_stdout())
        if self._proc.stdin is not None and not self._proc.stdin.is_closing():
            try:
                self._proc.stdin.close()
            except (BrokenPipeError, ConnectionResetError):
                pass
        await self._wait_or_kill(grace)
        await self._finish_shutdown(drain_task)

    async def _log_stderr_tail(self) -> None:
        if self._stderr_tail:
            tail = self._stderr_tail[0]
            logger.warning("pi stderr tail: %s", redact_secrets(tail.decode(errors="replace")))


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
