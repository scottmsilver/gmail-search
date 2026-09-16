"""Pure synthetic fixture checks; does not start Docker or call providers."""
import importlib.util
import json
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

spec = importlib.util.spec_from_file_location('workflow_smoke', Path(__file__).parents[1] / 'scripts/smoke_pi_workflow.py')
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


def test_fake_model_requires_child_result_before_final_synthesis():
    fake = smoke.FakeServices()
    parent = {'tools': [{'function': {'name': 'subagent'}}], 'messages': []}
    child = {'tools': [{'function': {'name': 'gmail_describe_schema'}}], 'messages': []}
    launch = fake.completion(parent)
    assert launch['tool_calls'][0]['function']['name'] == 'subagent'
    assert 'Waiting' in fake.completion(parent)['content']
    assert fake.completion(child)['tool_calls'][0]['function']['name'] == 'gmail_describe_schema'
    child['messages'].append({'role': 'tool', 'content': 'SMOKE_SOURCE_1'})
    assert 'SMOKE_SOURCE_1' in fake.completion(child)['content']
    assert fake.completion(parent)['content'] == 'FINAL_SMOKE_SOURCE_1'


def test_fake_gmail_rejects_incorrect_session_token():
    fake = smoke.FakeServices()
    server = smoke.ThreadingHTTPServer(('127.0.0.1', 0), fake.handler())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    request = Request(f'http://127.0.0.1:{server.server_port}/mcp',
                      data=json.dumps({'id': 1, 'method': 'tools/list'}).encode(),
                      headers={'Content-Type': 'application/json', 'Authorization': 'Bearer incorrect'})
    try:
        with pytest.raises(HTTPError) as exc:
            urlopen(request, timeout=2)
        assert exc.value.code == 401
        assert fake.gmail_calls == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
