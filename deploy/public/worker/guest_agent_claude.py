#!/usr/bin/env python3
"""Fixed, guest-only native Claude Code runner. No host commands or provider credentials.

The counterpart of guest_agent_pi.py for the same `agent_full` guest. The outer
trusted controller owns run authority, worker lease and stop ACK; this runner's
events are untrusted display output, including terminal status.

The invocation is the one qualified in GUEST_MAIL_MCP_QUALIFICATION.md, where
native Claude Code ran inside a Firecracker guest against the gateway: --bare,
no setting sources, a strict MCP config naming only the guest `mail` server,
and permission prompts skipped because the VM is the permission boundary. The
tool surface is that qualified one -- Bash plus the typed mail tools -- rather
than a broader set the gateway's request compiler has never been shown.

The prompt never appears in argv, where /proc/*/cmdline would expose it to
every same-uid process; `claude -p` reads it from stdin, as Pi's RPC mode does.
"""
import asyncio
import json
import os
from pathlib import Path
import signal
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from guest_agent_bootstrap import CLAUDE_PROFILE, receive, validate_config
from guest_agent_pi import (MAIL_GUIDANCE, MAX_RPC_OUTPUT, MAX_RPC_RECORD, MODEL, ROOT, RUN,
                            RUN_SECONDS, EventSink, RunnerError, bound_event, make_run_dirs,
                            start_process, stop_process, unprivileged)
from guest_mail_tools import _drain, _invalid_constant, _object

MAX_TURNS = 30
MAIL_SERVER = {'mcpServers': {'mail': {'type': 'stdio', 'command': '/usr/bin/env',
    'args': ['-i', 'PATH=/usr/bin:/bin', 'LANG=C.UTF-8', '/usr/bin/python3', '-I',
             str(ROOT / 'guest_mail_mcp.py')]}}}


def claude_argv():
    return [str(ROOT / 'bin/claude'), '--bare', '--setting-sources', '', '--strict-mcp-config',
            '--mcp-config', json.dumps(MAIL_SERVER, separators=(',', ':')),
            '--disable-slash-commands', '--no-session-persistence', '--model', MODEL,
            '--tools', 'Bash', '--allowedTools', 'Bash,mcp__mail__*', '--dangerously-skip-permissions',
            '--max-turns', str(MAX_TURNS), '--output-format', 'stream-json', '--verbose',
            '--append-system-prompt', MAIL_GUIDANCE, '-p']


# The gateway rejects, never clamps, a request above its profile's output cap
# (invited_runtime.OUTPUT_TOKEN_LIMIT). Claude Code asks for 32000 by default.
OUTPUT_TOKEN_LIMIT = 4096


def claude_env(home, inference_capability):
    """The qualified environment: gateway as the only provider, every optional
    network feature off, and a private config dir so no host state is read."""
    return {'HOME': str(home), 'PATH': '/tmp/runtime/bin:/usr/bin:/bin', 'LANG': 'C.UTF-8', 'TERM': 'dumb',
            'ANTHROPIC_API_KEY': inference_capability, 'ANTHROPIC_BASE_URL': 'http://127.0.0.1:18080',
            'DISABLE_PROMPT_CACHING': '1', 'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC': '1',
            'CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS': '1', 'CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK': '1',
            'CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL': '1', 'CLAUDE_CODE_DISABLE_AUTO_MEMORY': '1',
            'CLAUDE_CODE_DISABLE_THINKING': '1', 'CLAUDE_CODE_MAX_OUTPUT_TOKENS': str(OUTPUT_TOKEN_LIMIT),
            'CLAUDE_CONFIG_DIR': str(home / 'claude')}


def normalize(record, names, secrets=()):
    """Claude stream-json records to the runner's bounded Events.

    `names` maps tool_use ids to tool names as they arrive, so a tool_result --
    which carries only the id -- can be labelled the way Pi's results are.
    """
    kind = record.get('type')
    message = record.get('message') if isinstance(record.get('message'), dict) else {}
    blocks = message.get('content') if isinstance(message.get('content'), list) else []
    events = []
    if kind == 'assistant':
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if block.get('type') == 'text' and isinstance(block.get('text'), str) and block['text']:
                events.append({'type': 'text', 'text': block['text']})
            elif block.get('type') == 'tool_use':
                name = block.get('name')
                if type(name) is not str or len(name) > 128:
                    raise RunnerError()
                names[block.get('id')] = name
                events.append({'type': 'tool_start', 'name': name, 'args': block.get('input', {})})
    elif kind == 'user':
        for block in blocks:
            if isinstance(block, dict) and block.get('type') == 'tool_result':
                events.append({'type': 'tool_result', 'name': names.get(block.get('tool_use_id')),
                               'result': block.get('content'), 'is_error': block.get('is_error') is True})
    return [bound_event(event, secrets) for event in events]


async def drive(proc, prompt, emit, *, deadline, secrets=()):
    total, answered, finished, names = 0, False, False, {}
    try:
        async with asyncio.timeout_at(deadline):
            proc.stdin.write(prompt.encode('utf-8'))
            await proc.stdin.drain()
            proc.stdin.close()
            while True:
                raw = await proc.stdout.readline()
                if not raw:
                    break
                total += len(raw)
                if len(raw) > MAX_RPC_RECORD or total > MAX_RPC_OUTPUT:
                    raise RunnerError()
                record = json.loads(raw, object_pairs_hook=_object, parse_constant=_invalid_constant)
                if type(record) is not dict:
                    raise RunnerError()
                if record.get('type') == 'result':
                    if record.get('is_error') is True or record.get('subtype') != 'success':
                        raise RunnerError()
                    finished = True
                    break
                for event in normalize(record, names, secrets):
                    if event['type'] == 'text':
                        answered = True
                    await emit(event)
        if not (finished and answered):
            raise RunnerError()
    finally:
        await _drain(stop_process(proc))


def prepare(config):
    config = validate_config(config)
    if config['profile'] != CLAUDE_PROFILE:
        raise RunnerError()
    home, cwd = make_run_dirs(config)
    claude_dir = home / 'claude'
    claude_dir.mkdir(mode=0o700)
    os.chown(claude_dir, 1000, 1000)
    return cwd, claude_env(home, config['inference_capability'])


async def run(config):
    config = validate_config(config)
    cwd, env = prepare(config)
    sink = EventSink(config['events_capability'])
    secrets = tuple(config['tool_config']['capabilities'].values()) + (
        config['inference_capability'], config['events_capability'])
    bridge = None
    try:
        bridge = await start_process('/usr/bin/python3', str(ROOT / 'guest-agent-vsock-bridge.py'),
            stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            preexec_fn=unprivileged, start_new_session=True, env={'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8'})
        await asyncio.sleep(.3)
        await sink({'type': 'status', 'state': 'running'})
        proc = await start_process(*claude_argv(), cwd=cwd, env=env,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
            preexec_fn=unprivileged, start_new_session=True, limit=MAX_RPC_RECORD)
        await drive(proc, config['prompt'], sink, deadline=asyncio.get_running_loop().time() + RUN_SECONDS,
                    secrets=secrets)
        await sink({'type': 'status', 'state': 'runner_completed'})
    finally:
        async def cleanup():
            try:
                if bridge is not None:
                    await stop_process(bridge)
            finally:
                (RUN / 'capabilities.json').unlink(missing_ok=True)
        await _drain(cleanup())


async def main():
    config = receive()
    task = asyncio.current_task()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, task.cancel)
    await run(config)


if __name__ == '__main__':
    try:
        asyncio.run(main())
        print('GMS_AGENT_RUNNER_COMPLETE', flush=True)
    except BaseException:
        print('GMS_AGENT_RUNNER_FAILED', flush=True)
        raise SystemExit(1)
