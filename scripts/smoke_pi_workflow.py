#!/usr/bin/env python3
"""Real Pi + pi-subagents smoke with synthetic HTTP services; no provider billing.

Run from the repo root: PYTHONPATH=src .venv/bin/python scripts/smoke_pi_workflow.py
No database, production container, real credentials, or mailbox files are used.
"""
from __future__ import annotations

import argparse
from contextlib import suppress
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
from uuid import uuid4
from unittest.mock import patch

TOKEN = 'synthetic-smoke-session-token'
MODEL = 'meta/muse-spark-1.3'
RAW_TOOLS = ['search_emails_batch', 'query_emails_batch', 'sql_query_batch', 'find_facts',
             'get_thread_batch', 'get_attachment_batch', 'describe_schema', 'publish_artifact_batch']


class FakeServices:
    def __init__(self):
        self.requests = []
        self.gmail_calls = []
        self.parent_requests = 0
        self.child_finished = False
        self.lock = threading.Lock()
        self.child_started = threading.Event()
        self.slow_child = False
        self.release_child = threading.Event()

    def completion(self, body):
        with self.lock:
            self.requests.append(body)
            names = [t['function']['name'] for t in body.get('tools', [])]
            parent = 'subagent' in names
            if parent:
                self.parent_requests += 1
                if self.parent_requests == 1:
                    return self.tool('subagent', {'agent': 'mail-researcher', 'task': 'Call gmail_describe_schema, then report SMOKE_SOURCE_1.', 'async': True, 'context': 'fresh', 'timeoutMs': 45000})
                return {'content': 'FINAL_SMOKE_SOURCE_1' if self.parent_requests > 2 and self.child_finished else 'Waiting for the delegated synthetic evidence.'}
            self.child_started.set()
            if not any(m.get('role') == 'tool' for m in body.get('messages', [])):
                return self.tool('gmail_describe_schema', {})
            self.child_finished = True
            return {'content': 'Supported: SMOKE_SOURCE_1 from the synthetic Gmail schema.'}

    @staticmethod
    def tool(name, args):
        return {'tool_calls': [{'index': 0, 'id': f'call_{uuid4().hex[:8]}', 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}]}

    def handler(self):
        services = self
        class Handler(BaseHTTPRequestHandler):
            def handle(self):
                try:
                    super().handle()
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *_):
                pass

            def json(self, status, value):
                data = json.dumps(value).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self.json(405, {'error': 'Use POST'})

            def do_DELETE(self):
                self.json(200, {})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', '0'))))
                if self.path == '/v1/chat/completions':
                    if self.headers.get('Authorization') != 'Bearer smoke-only-dummy-key':
                        self.json(401, {'error': 'Expected dummy model credential'})
                        return
                    delta = services.completion(body)
                    if services.slow_child and not any(t['function']['name'] == 'subagent' for t in body.get('tools', [])):
                        services.release_child.wait(30)
                    # Delay child completion enough to exercise the parent's initial agent_end.
                    if delta.get('content', '').startswith('Supported:'):
                        time.sleep(2)
                    base = {'id': 'chatcmpl-smoke', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': MODEL}
                    chunks = [dict(base, choices=[{'index': 0, 'delta': {'role': 'assistant', **delta}, 'finish_reason': None}]),
                              dict(base, choices=[{'index': 0, 'delta': {}, 'finish_reason': 'tool_calls' if 'tool_calls' in delta else 'stop'}]),
                              dict(base, choices=[], usage={'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15})]
                    data = ''.join(f'data: {json.dumps(chunk)}\n\n' for chunk in chunks) + 'data: [DONE]\n\n'
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream')
                    self.send_header('Content-Length', str(len(data.encode())))
                    self.end_headers()
                    try:
                        self.wfile.write(data.encode())
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
                if self.path.rstrip('/') != '/mcp':
                    self.json(404, {})
                    return
                if self.headers.get('Authorization') != f'Bearer {TOKEN}':
                    self.json(401, {'error': 'Incorrect synthetic Gmail session token'})
                    return
                method = body.get('method')
                if 'id' not in body:
                    self.send_response(202)
                    self.send_header('Content-Length', '0')
                    self.end_headers()
                    return
                if method == 'initialize':
                    result = {'protocolVersion': body['params']['protocolVersion'], 'capabilities': {'tools': {}}, 'serverInfo': {'name': 'synthetic-gmail', 'version': '1'}}
                elif method == 'tools/list':
                    result = {'tools': [{'name': name, 'description': 'Synthetic Gmail smoke fixture', 'inputSchema': {'type': 'object', 'properties': {}}} for name in RAW_TOOLS]}
                elif method == 'tools/call':
                    services.gmail_calls.append({'name': body['params']['name'], 'authorized': True})
                    result = {'content': [{'type': 'text', 'text': 'SMOKE_SOURCE_1: synthetic schema evidence'}]}
                else:
                    result = {}
                self.json(200, {'jsonrpc': '2.0', 'id': body['id'], 'result': result})
        return Handler


async def smoke(image, timeout, *, cancel=False):
    from gmail_search.agents.pi_rpc import PiRpcClient
    from gmail_search.agents.pi_workflow import prepare_workflow, read_trace
    from gmail_search.agents.runtime_pi import drive_turn, _workflow_control, _kill_workflow_processes

    subprocess.run(['docker', 'image', 'inspect', image], check=True, stdout=subprocess.DEVNULL)
    services = FakeServices()
    services.slow_child = cancel
    server = ThreadingHTTPServer(('0.0.0.0', 0), services.handler())
    server.daemon_threads = False  # server_close joins every synthetic request worker.
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    name = f'gmail-workflow-smoke-{uuid4().hex[:10]}'
    client = None
    try:
        with tempfile.TemporaryDirectory(prefix='gmail-workflow-smoke-') as tmp:
            scratch = Path(tmp)
            workspace = scratch / 'workspaces' / 'eval-smoke'
            workspace.mkdir(parents=True)
            (scratch / 'sessions').mkdir()
            (workspace / '.session-token').write_text(TOKEN)
            (workspace / '.session-token').chmod(0o600)
            url = f'http://host.docker.internal:{server.server_port}'
            models = scratch / 'models.json'
            models.write_text(json.dumps({'providers': {'openrouter': {'baseUrl': url + '/v1', 'api': 'openai-completions', 'apiKey': 'smoke-only-dummy-key', 'models': [{'id': MODEL, 'name': 'Synthetic Spark', 'reasoning': False, 'input': ['text'], 'contextWindow': 100000, 'maxTokens': 2048, 'cost': {'input': 0, 'output': 0, 'cacheRead': 0, 'cacheWrite': 0}}]}}}))
            mcp = scratch / 'mcp.json'
            mcp.write_text(json.dumps({'settings': {'directTools': True, 'disableProxyTool': True, 'scriptMode': True}, 'mcpServers': {'gmail': {'url': url + '/mcp', 'auth': 'bearer', 'bearerToken': TOKEN, 'lifecycle': 'eager', 'directTools': True}}}))
            files = prepare_workflow(scratch / 'workspaces', 'eval-smoke', 'smoke-turn', models_path=models, mcp_path=mcp, timeout=timeout)
            subprocess.run(['docker', 'run', '-d', '--rm', '--name', name, '--add-host', 'host.docker.internal:host-gateway',
                            '--mount', f'type=bind,src={scratch / "workspaces"},dst=/workspaces',
                            '--mount', f'type=bind,src={scratch / "sessions"},dst=/sessions', image], check=True, stdout=subprocess.DEVNULL)
            argv = ['docker', 'exec', '-i', '-w', '/workspaces/eval-smoke', '-e', 'GMS_SESSION_ID=smoke-turn']
            for key, value in files.env.items():
                argv += ['-e', f'{key}={value}']
            argv += [name, 'pi', '--mode', 'rpc', '--provider', 'openrouter', '--model', MODEL, '--thinking', 'off',
                     '--session', '/sessions/smoke.jsonl', '--no-extensions', '-e', '/opt/pi-workflow/index.ts', '--no-skills']
            client = await PiRpcClient.spawn(argv)
            observed = []
            async def capture(kind, payload):
                observed.append({'kind': kind, 'payload': payload})
            try:
                turn = asyncio.create_task(drive_turn(client, 'Delegate the synthetic schema lookup and incorporate the child evidence.', on_tool_event=capture, hard_timeout=timeout, idle_timeout=20, workflow=True))
                if cancel:
                    deadline = time.monotonic() + timeout
                    while not services.child_started.is_set():
                        if turn.done():
                            await turn
                            raise AssertionError('Parent ended before child started')
                        if time.monotonic() > deadline:
                            raise TimeoutError('Child did not start')
                        await asyncio.sleep(0.05)
                    # Marker processes prove the fallback cleanup is session scoped.
                    for marker in ('smoke-turn', 'unrelated-smoke-turn'):
                        subprocess.run(['docker', 'exec', '-d', '-e', f'GMS_SESSION_ID={marker}', name, 'sleep', '120'], check=True)
                    turn.cancel()
                    with suppress(asyncio.CancelledError):
                        await turn
                    await _workflow_control(client, 'stop', timeout=10)
                    await client.abort_and_close(grace=5)
                    with patch.dict(os.environ, {'GMAIL_PI_CONTAINER': name}):
                        await _kill_workflow_processes('smoke-turn')
                    check = "import pathlib,json; counts={m:0 for m in ['smoke-turn','unrelated-smoke-turn']};\nfor p in pathlib.Path('/proc').glob('[0-9]*/environ'):\n try:\n  env=p.read_bytes().split(b'\\0'); counts={m:n+int(('GMS_SESSION_ID='+m).encode() in env) for m,n in counts.items()}\n except OSError: pass\nprint(json.dumps(counts))"
                    counts = json.loads(subprocess.check_output(['docker', 'exec', name, 'python3', '-c', check], text=True))
                    assert counts['smoke-turn'] == 0, counts
                    assert counts['unrelated-smoke-turn'] >= 1, counts
                else:
                    outcome = await turn
                    status = await _workflow_control(client, 'status')
                    assert status['ready'] and not status['active'], status
                    assert outcome.final_text == 'FINAL_SMOKE_SOURCE_1', outcome.final_text
                    assert services.parent_requests >= 3, services.parent_requests
                    assert services.gmail_calls == [{'name': 'describe_schema', 'authorized': True}], services.gmail_calls
                    assert services.requests and all(r['model'] == MODEL for r in services.requests)
                    assert not any(e['payload'].get('response', {}).get('is_error') for e in observed if e['kind'] == 'tool_call'), observed
            except Exception:
                print(json.dumps({"diagnostic": "synthetic smoke failure", "model_requests": len(services.requests), "gmail_calls": services.gmail_calls, "tool_events": observed[-8:], "model_tool_names": [[t["function"]["name"] for t in r.get("tools", [])] for r in services.requests], "model_tool_results": [m for r in services.requests for m in r.get("messages", []) if m.get("role") == "tool"]}))
                raise
            finally:
                if client.returncode is None:
                    await client.close()
                services.release_child.set()
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)
            records, complete = read_trace(files.directory / 'events.jsonl')
            messages = [r for r in records if r.get('type') == 'message_end']
            assert any(r.get('parent') is True and r.get('usage') for r in messages), messages
            if not cancel:
                assert any(r.get('parent') is False and r.get('usage') for r in messages), messages
                assert complete, 'Missing session lifecycle evidence'
            print(json.dumps({'status': 'passed', 'scenario': 'cancellation' if cancel else 'completion', 'model_requests': len(services.requests), 'parent_requests': services.parent_requests,
                              'gmail_calls': services.gmail_calls, 'trace_events': len(records), 'trace_complete': complete, 'provider_billing': False}))
    finally:
        subprocess.run(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        services.release_child.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', default='gmail-search-pi:workflow-eval-test')
    parser.add_argument('--timeout', type=float, default=90)
    args = parser.parse_args()
    asyncio.run(smoke(args.image, args.timeout))
    asyncio.run(smoke(args.image, args.timeout, cancel=True))


if __name__ == '__main__':
    main()
