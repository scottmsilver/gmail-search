"""OpenRouter chat-completions boundary: Pi's real request, a real Opus 5 stream."""
import json
from pathlib import Path

import httpx
import pytest

from gmail_search.gateway.openrouter import (OpenRouterProfile, OpenRouterRunService,
                                             compile_openrouter_request)
from gmail_search.gateway.openrouter_http import OpenRouterHTTPTransport
from gmail_search.gateway.registry import AccessDenied
from test_gateway_gemini import Stream
from test_gateway_provider import setup as setup_fixture, spend

setup = setup_fixture
FIXTURES = Path(__file__).parent / 'fixtures/openrouter'
MODEL = 'anthropic/claude-opus-5'


def pi_request():
    return json.loads((FIXTURES / 'pi_0844_initial_request.json').read_text())


def compile(body, **overrides):
    options = dict(model=MODEL, max_output_tokens=16384, reasoning_effort='medium') | overrides
    return compile_openrouter_request(body, **options)


def test_pi_request_is_rebuilt_around_server_choices():
    body = pi_request()
    body['model'] = 'guest-picked-model'
    body['reasoning_effort'] = 'high'
    out = compile(body)
    assert out['model'] == MODEL and out['reasoning'] == {'effort': 'medium'}
    assert out['max_tokens'] == 16384 and 'max_completion_tokens' not in out
    assert out['stream_options'] == {'include_usage': True} and 'store' not in out
    assert 'provider' not in out and 'plugins' not in out


@pytest.mark.parametrize('change', [
    {'provider': {'order': ['anywhere']}}, {'plugins': [{'id': 'web'}]}, {'response_format': {'type': 'json_object'}},
    {'store': True}, {'stream': False}, {'max_completion_tokens': 16385},
    {'messages': [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'https://x.test/a.png'}}]}]},
    {'messages': [{'role': 'assistant', 'tool_calls': [{'id': 'a', 'type': 'function',
                                                        'function': {'name': 'not_a_tool', 'arguments': '{}'}}]}]},
])
def test_guest_cannot_reach_past_the_boundary(change):
    with pytest.raises(ValueError):
        compile(pi_request() | change)


def test_tool_round_trip_with_reasoning_replay_is_accepted():
    body = pi_request()
    body['messages'] += [
        {'role': 'assistant', 'content': None,
         'tool_calls': [{'id': 'toolu_1', 'type': 'function', 'function': {'name': 'bash', 'arguments': '{"command":"echo hi"}'}}],
         'reasoning_details': [{'type': 'reasoning.text', 'text': 'thinking', 'signature': 'sig', 'format': 'anthropic-claude-v1', 'index': 0}]},
        {'role': 'tool', 'tool_call_id': 'toolu_1', 'content': 'hi'}]
    assert compile(body)['messages'][-1]['tool_call_id'] == 'toolu_1'


def test_models_outside_the_allowlist_are_refused():
    with pytest.raises(ValueError):
        compile(pi_request(), model='openai/gpt-something')
    with pytest.raises(AccessDenied):
        OpenRouterProfile('openai/gpt-something', 100, 20, 2, 3)


async def _through_service(setup, data, *, requests=None):
    registry, caps, run, token = setup
    stream = Stream(data)
    def respond(request):
        if requests is not None:
            requests.append(request)
        return httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=stream)
    async with httpx.AsyncClient(trust_env=False, transport=httpx.MockTransport(respond)) as client:
        svc = OpenRouterRunService(caps, OpenRouterHTTPTransport('synthetic-key', client=client))
        svc.bind_profile(run.run_id, OpenRouterProfile(MODEL, 2000, 200, 2, 3))
        body = pi_request()
        body['max_completion_tokens'] = 200
        return b''.join([part async for part in svc.stream(token, 'req', body)]), stream


@pytest.mark.asyncio
async def test_live_opus_stream_passes_and_settles_on_its_usage(setup):
    data = (FIXTURES / 'opus5_tool_call_stream.txt').read_bytes()
    requests = []
    published, stream = await _through_service(setup, data, requests=requests)
    assert b'"finish_reason":"tool_calls"' in published and published.rstrip().endswith(b'data: [DONE]')
    assert spend(setup[0]) == (0, 1632 * 2 + 49 * 3)
    sent = requests[0]
    assert str(sent.url) == 'https://openrouter.ai/api/v1/chat/completions'
    assert sent.headers['authorization'] == 'Bearer synthetic-key'
    assert json.loads(sent.content)['model'] == MODEL and stream.closed


@pytest.mark.asyncio
@pytest.mark.parametrize('old,new', [
    (b'"model":"anthropic/claude-opus-5"', b'"model":"anthropic/claude-other"'),   # a fallback model
    (b'"total_tokens":1681', b'"total_tokens":1'),                                # usage that does not add up
    (b'data: [DONE]', b''),                                                       # no terminal frame
    (b'"audio_tokens":0,"video_tokens":0', b'"audio_tokens":5,"video_tokens":0'), # another billing dimension
])
async def test_a_stream_off_contract_charges_the_reservation(setup, old, new):
    data = (FIXTURES / 'opus5_tool_call_stream.txt').read_bytes().replace(old, new)
    with pytest.raises(RuntimeError):
        await _through_service(setup, data)
    assert spend(setup[0]) == (0, 2000 * 2 + 200 * 3)
