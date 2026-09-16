#!/usr/bin/env python3
"""One-shot guest tool invocation; stdin carries arguments, never configuration.

Run inside the qualified guest with python -I. Trusted startup installs the
versioned short-lived capabilities and workspace under /tmp/gms-run. Neither this
wrapper nor its tool core belongs on the mail host's model-execution path.
"""
import asyncio
import io
import json
import os
from pathlib import Path
import signal
import stat
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from guest_tool_config import RAW_PROFILE, parse_tool_config
from guest_mail_tools import GuestMailTools, ToolError, _object, _invalid_constant, _drain

RUN_ROOT = Path('/tmp/gms-run')
MAX_INPUT = 256 * 1024
MAX_OUTPUT = 9 * 1024**2


def _json(raw):
    try:
        return json.loads(raw, object_pairs_hook=_object, parse_constant=_invalid_constant)
    except (ValueError, UnicodeError, RecursionError):
        raise ToolError('Invalid guest invocation.') from None


def _capabilities():
    directory = fd = None
    try:
        directory = os.open(RUN_ROOT, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        info = os.fstat(directory)
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise ToolError('Invalid guest configuration.')
        fd = os.open('capabilities.json', os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600 or info.st_size > 4096):
            raise ToolError('Invalid guest configuration.')
        raw = os.read(fd, 4097)
        if len(raw) > 4096:
            raise ToolError('Invalid guest configuration.')
        return parse_tool_config(_json(raw))
    except (OSError, ValueError):
        raise ToolError('Invalid guest configuration.') from None
    finally:
        if fd is not None:
            os.close(fd)
        if directory is not None:
            os.close(directory)


async def invoke(stream):
    raw = stream.read(MAX_INPUT + 1)
    if type(raw) is not bytes or len(raw) > MAX_INPUT:
        raise ToolError('Guest invocation exceeds its byte limit.')
    request = _json(raw)
    if (type(request) is not dict or set(request) != {'name', 'arguments'}
            or type(request['name']) is not str or type(request['arguments']) is not dict):
        raise ToolError('Invalid guest invocation.')
    config=_capabilities()
    if config.profile==RAW_PROFILE:
        raise ToolError('The raw attachment profile requires persistent MCP; one-shot CLI is unavailable.')
    tools = GuestMailTools(RUN_ROOT/'work', config, port=18080)
    try:
        return await tools.dispatch(request['name'], request['arguments'])
    finally:
        await _drain(tools.aclose())


async def _ready(fd, *, write=False):
    loop = asyncio.get_running_loop()
    ready = loop.create_future()
    def wake():
        if not ready.done():
            ready.set_result(None)
    add = loop.add_writer if write else loop.add_reader
    remove = loop.remove_writer if write else loop.remove_reader
    add(fd, wake)
    try:
        await ready
    finally:
        remove(fd)


async def _input():
    os.set_blocking(0, False)
    chunks, size = [], 0
    async with asyncio.timeout(5):
        while True:
            try:
                chunk = os.read(0, min(65536, MAX_INPUT + 1 - size))
            except BlockingIOError:
                await _ready(0)
                continue
            if not chunk:
                return b''.join(chunks)
            size += len(chunk)
            if size > MAX_INPUT:
                raise ToolError('Guest invocation exceeds its byte limit.')
            chunks.append(chunk)


async def _output(result):
    valid = True
    try:
        data = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode() + b'\n'
    except (ValueError, TypeError, UnicodeError, RecursionError):
        data, valid = b'{"error":"Invalid guest tool result."}\n', False
    if len(data) > MAX_OUTPUT:
        data, valid = b'{"error":"Guest result exceeds its byte limit."}\n', False
    os.set_blocking(1, False)
    view = memoryview(data)
    async with asyncio.timeout(5):
        while view:
            try:
                written = os.write(1, view)
            except BlockingIOError:
                await _ready(1, write=True)
                continue
            if written <= 0:
                raise OSError('closed output')
            view = view[written:]
    return valid


async def main():
    loop, task = asyncio.get_running_loop(), asyncio.current_task()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, task.cancel)
    code = 0
    try:
        result = await invoke(io.BytesIO(await _input()))
    except asyncio.CancelledError:
        result, code = {'error':'Guest invocation cancelled; operation outcome may be unknown. No retry was attempted.'}, 130
    except (ToolError, OSError, TimeoutError, ValueError, TypeError):
        result, code = {'error':'Guest invocation failed.'}, 1
    try:
        if not await _output(result):
            code = 1
    except (OSError, TimeoutError, asyncio.CancelledError):
        return 1
    return code


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
