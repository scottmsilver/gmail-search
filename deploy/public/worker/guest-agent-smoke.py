#!/usr/bin/env python3
"""Immutable synthetic CLI exercise inside the inner VM; all writes on tmpfs."""
import json
import os
from pathlib import Path
import resource
import subprocess
import time
import urllib.request

MODEL = 'claude-sonnet-4-6'
PROMPT = 'SYNTHETIC_CLI_SPIKE: Use the shell tool to run Python, write synthetic-result.txt containing 42, and report SYNTHETIC_CLI_COMPLETE.'
ROOT = Path('/tmp/runtime')
WORK = Path('/tmp/agent-smoke')


def unprivileged():
    os.setgroups([])
    os.setgid(1000)
    os.setuid(1000)
    resource.setrlimit(resource.RLIMIT_FSIZE, (8*1024**2, 8*1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
    resource.setrlimit(resource.RLIMIT_NPROC, (128, 128))


def main():
    print('INNER_AGENT_SMOKE_START', flush=True)
    WORK.mkdir(mode=0o755)
    bridge = subprocess.Popen(['/usr/bin/python3', str(ROOT/'guest-vsock-bridge.py')], stdin=subprocess.DEVNULL,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=unprivileged)
    time.sleep(.3)
    summary = []
    try:
        for runtime in ('pi', 'claude'):
            base = WORK/runtime
            home, cwd = base/'home', base/'work'
            home.mkdir(parents=True)
            cwd.mkdir()
            for path in (base, home, cwd):
                os.chown(path, 1000, 1000)
            endpoint = 'http://127.0.0.1:' + ('18080' if runtime == 'pi' else '18081')
            env = {'HOME': str(home), 'PATH': '/tmp/runtime/bin:/usr/bin:/bin', 'LANG': 'C.UTF-8', 'TERM': 'dumb',
                   'ANTHROPIC_API_KEY': 'synthetic-not-a-real-key', 'ANTHROPIC_BASE_URL': endpoint,
                   'DISABLE_PROMPT_CACHING': '1', 'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC': '1',
                   'CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS': '1', 'CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK': '1',
                   'CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL': '1', 'CLAUDE_CODE_DISABLE_AUTO_MEMORY': '1',
                   'CLAUDE_CODE_DISABLE_THINKING': '1', 'CLAUDE_CONFIG_DIR': str(home/'claude'),
                   'PI_CODING_AGENT_DIR': str(home/'pi')}
            if runtime == 'pi':
                (home/'pi').mkdir()
                (home/'pi/models.json').write_text(json.dumps({'providers': {'synthetic': {
                    'baseUrl': endpoint, 'api': 'anthropic-messages', 'apiKey': 'synthetic-not-a-real-key',
                    'models': [{'id': MODEL, 'reasoning': False, 'input': ['text'], 'contextWindow': 200000,
                                'maxTokens': 1024, 'compat': {'supportsEagerToolInputStreaming': False, 'supportsCacheControlOnTools': False}}]}}}))
                for path in (home/'pi', home/'pi/models.json'):
                    os.chown(path, 1000, 1000)
                argv = ['/tmp/runtime/bin/node', '/tmp/runtime/lib/pi-coding-agent/dist/bundle/cli.js', '--provider', 'synthetic',
                        '--model', MODEL, '--thinking', 'off', '--tools', 'bash', '--no-session', '--no-extensions',
                        '--no-skills', '--no-context-files', '--no-themes', '--no-prompt-templates', '-p', PROMPT]
            else:
                argv = ['/tmp/runtime/bin/claude', '--bare', '--setting-sources', '', '--strict-mcp-config', '--mcp-config',
                        '{"mcpServers":{}}', '--disable-slash-commands', '--no-session-persistence', '--model', MODEL,
                        '--tools', 'Bash', '--allowedTools', 'Bash', '--dangerously-skip-permissions', '--max-turns', '3',
                        '--output-format', 'json', '--system-prompt', 'You are a synthetic local tool test.', '-p', PROMPT]
            with (base/'stdout').open('wb') as stdout, (base/'stderr').open('wb') as stderr:
                try:
                    proc = subprocess.run(argv, cwd=cwd, env=env, stdout=stdout, stderr=stderr,
                                          preexec_fn=unprivileged, timeout=65)
                    result = {'runtime': runtime, 'returncode': proc.returncode,
                              'file': (cwd/'synthetic-result.txt').read_text() if (cwd/'synthetic-result.txt').exists() else None,
                              'completion': b'SYNTHETIC_CLI_COMPLETE' in (base/'stdout').read_bytes()}
                except subprocess.TimeoutExpired:
                    result = {'runtime': runtime, 'timeout': True}
            result['stdout_tail'] = (base/'stdout').read_text(errors='replace')[-1500:]
            result['stderr_tail'] = (base/'stderr').read_text(errors='replace')[-1500:]
            result_file = cwd/'synthetic-result.txt'
            if result_file.exists():
                payload = result_file.read_bytes()
                request = urllib.request.Request(endpoint + '/v1/artifacts?filename=' + runtime + '-result.txt', data=payload,
                    headers={'Content-Type': 'application/octet-stream', 'Authorization': 'Bearer synthetic'}, method='POST')
                with urllib.request.urlopen(request, timeout=10) as response:
                    result['artifact_status'] = response.status
                    result['artifact_receipt'] = response.read(4096).decode()
            summary.append(result)
            print('INNER_RUNTIME_RESULT ' + json.dumps(result), flush=True)
        assert all(item.get('file') == '42' and item.get('completion') and item.get('returncode') == 0 and item.get('artifact_status') == 201 for item in summary)
        print('INNER_AGENT_SMOKE_PASS', flush=True)
    finally:
        bridge.terminate()
        bridge.wait(timeout=5)


if __name__ == '__main__':
    main()
