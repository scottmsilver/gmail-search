"""Native Claude Code runner for the agent_full guest; synthetic processes only.

Nothing here calls a provider. The invocation mirrors the one qualified inside a
Firecracker guest in GUEST_MAIL_MCP_QUALIFICATION.md, and these tests pin the
properties that made it safe to run there.
"""
import asyncio
import importlib
import json
from pathlib import Path
import sys

import pytest

WORKER = Path(__file__).parents[1] / 'deploy/public/worker'
sys.path.insert(0, str(WORKER))


def envelope(profile='mail-agent-claude-v1'):
    return {'version': 1, 'profile': profile, 'prompt': 'Find invoices; $(touch /tmp/no)\nΔ',
            'tool_config': {'version': 3, 'tool_profile': 'mail-raw-mcp-v3', 'capabilities': {
                'sql': '1' * 64, 'retrieval': '2' * 64, 'artifact': '3' * 64, 'attachment': '4' * 64}},
            'inference_capability': '5' * 64, 'events_capability': '6' * 64}


def module(name):
    return importlib.import_module(name)


# ── which agent runs ─────────────────────────────────────────────────────────

def test_the_bootstrap_accepts_exactly_the_agent_profiles():
    boot = module('guest_agent_bootstrap')
    assert boot.PROFILES == {'mail-agent-pi-v1', 'mail-agent-pi-gemini-v1', 'mail-agent-claude-v1'}
    for profile in boot.PROFILES:
        assert boot.validate_config(envelope(profile))['profile'] == profile
    for profile in ('legacy-mail-v1', 'mail-agent-claude-v2', 'claude', ''):
        with pytest.raises(ValueError):
            boot.validate_config(envelope(profile))


def test_the_dispatcher_picks_the_runner_from_the_validated_profile():
    entry = module('guest_agent')
    assert entry.runner_for({'profile': 'mail-agent-pi-v1'}) is module('guest_agent_pi').run
    assert entry.runner_for({'profile': 'mail-agent-claude-v1'}) is module('guest_agent_claude').run
    with pytest.raises(KeyError):
        entry.runner_for({'profile': 'something-else'})


def test_each_runner_refuses_the_other_runners_profile(tmp_path, monkeypatch):
    """The dispatcher routes correctly, but a runner handed the wrong envelope
    must not quietly run the wrong agent with it."""
    claude, pi = module('guest_agent_claude'), module('guest_agent_pi')
    monkeypatch.setattr(claude, 'RUN', tmp_path / 'c'); monkeypatch.setattr(pi, 'RUN', tmp_path / 'p')
    with pytest.raises(ValueError):
        claude.prepare(envelope('mail-agent-pi-v1'))
    with pytest.raises(ValueError):
        pi.prepare(envelope('mail-agent-claude-v1'))


# ── the invocation ───────────────────────────────────────────────────────────

def test_the_prompt_never_enters_argv():
    """/proc/*/cmdline is readable by every same-uid process in the guest."""
    argv = module('guest_agent_claude').claude_argv()
    assert envelope()['prompt'] not in ' '.join(argv)
    assert argv[-1] == '-p', 'the prompt must come from stdin, so nothing may follow -p'


def test_the_invocation_is_the_qualified_one():
    runner = module('guest_agent_claude')
    argv = runner.claude_argv()
    for flag in ('--bare', '--strict-mcp-config', '--disable-slash-commands',
                 '--no-session-persistence', '--dangerously-skip-permissions', '--verbose'):
        assert flag in argv, flag
    assert argv[argv.index('--setting-sources') + 1] == ''
    assert argv[argv.index('--model') + 1] == runner.MODEL
    assert argv[argv.index('--output-format') + 1] == 'stream-json'
    assert argv[argv.index('--tools') + 1] == 'Bash'
    assert argv[argv.index('--allowedTools') + 1] == 'Bash,mcp__mail__*'
    servers = json.loads(argv[argv.index('--mcp-config') + 1])['mcpServers']
    assert set(servers) == {'mail'}, 'only the guest mail server may be configured'
    assert servers['mail']['args'][-1].endswith('/guest_mail_mcp.py')


def test_the_environment_routes_only_to_the_gateway(tmp_path):
    env = module('guest_agent_claude').claude_env(tmp_path, '5' * 64)
    assert env['ANTHROPIC_BASE_URL'] == 'http://127.0.0.1:18080'
    assert env['ANTHROPIC_API_KEY'] == '5' * 64, 'the run capability is the only credential'
    assert env['CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC'] == '1'
    assert env['CLAUDE_CONFIG_DIR'] == str(tmp_path / 'claude')
    assert env['HOME'] == str(tmp_path)


def test_claude_is_told_the_gateways_output_cap(tmp_path):
    """Claude Code asks for 32000 output tokens by default and the gateway
    rejects -- not clamps -- anything over its cap. Every run failed on its
    first inference call until the two agreed."""
    from gmail_search.invited_runtime import OUTPUT_TOKEN_LIMIT
    env = module('guest_agent_claude').claude_env(tmp_path, '5' * 64)
    assert env['CLAUDE_CODE_MAX_OUTPUT_TOKENS'] == str(OUTPUT_TOKEN_LIMIT)


# ── the event stream ─────────────────────────────────────────────────────────

def test_stream_json_becomes_the_runners_events_with_tool_names_carried():
    runner, names = module('guest_agent_claude'), {}
    start = runner.normalize({'type': 'assistant', 'message': {'content': [
        {'type': 'text', 'text': 'Looking.'},
        {'type': 'tool_use', 'id': 't1', 'name': 'mcp__mail__search', 'input': {'q': 'invoice'}}]}}, names)
    assert start == [{'type': 'text', 'text': 'Looking.'},
                     {'type': 'tool_start', 'name': 'mcp__mail__search', 'args': {'q': 'invoice'}}]
    result = runner.normalize({'type': 'user', 'message': {'content': [
        {'type': 'tool_result', 'tool_use_id': 't1', 'content': 'found 3'}]}}, names)
    assert result == [{'type': 'tool_result', 'name': 'mcp__mail__search', 'result': 'found 3', 'is_error': False}]


def test_credentials_are_redacted_from_events():
    runner = module('guest_agent_claude')
    event = runner.normalize({'type': 'assistant', 'message': {'content': [
        {'type': 'tool_use', 'id': 't', 'name': 'Bash', 'input': {'cmd': 'echo ' + '5' * 64}}]}}, {}, ['5' * 64])
    assert '5' * 64 not in json.dumps(event)


async def _drive(lines, prompt='hello'):
    runner = module('guest_agent_claude')
    code = ('import sys\nseen=sys.stdin.read()\n'
            'sys.stderr.write(seen)\n'
            'for line in sys.argv[1:]:\n    sys.stdout.write(line+"\\n")\nsys.stdout.flush()\n')
    proc = await asyncio.create_subprocess_exec(sys.executable, '-c', code, *lines,
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        start_new_session=True, limit=65536)
    events = []
    async def emit(event): events.append(event)
    await runner.drive(proc, prompt, emit, deadline=asyncio.get_running_loop().time() + 5)
    return events, proc


@pytest.mark.asyncio
async def test_a_successful_run_emits_its_answer_and_completes():
    events, proc = await _drive([
        json.dumps({'type': 'system', 'subtype': 'init'}),
        json.dumps({'type': 'assistant', 'message': {'content': [{'type': 'text', 'text': 'Three invoices.'}]}}),
        json.dumps({'type': 'result', 'subtype': 'success', 'is_error': False, 'result': 'Three invoices.'})])
    assert events == [{'type': 'text', 'text': 'Three invoices.'}]
    assert proc.returncode is not None, 'the process must be reaped'


@pytest.mark.asyncio
@pytest.mark.parametrize('lines', [
    [json.dumps({'type': 'result', 'subtype': 'error_max_turns', 'is_error': True})],
    [json.dumps({'type': 'assistant', 'message': {'content': [{'type': 'text', 'text': 'x'}]}})],
    [json.dumps({'type': 'result', 'subtype': 'success', 'is_error': False})],
    ['{not json}'],
])
async def test_failed_unfinished_or_answerless_runs_are_refused(lines):
    with pytest.raises((ValueError, asyncio.IncompleteReadError)):
        await _drive(lines)
