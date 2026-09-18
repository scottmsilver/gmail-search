import asyncio
import json
from pathlib import Path

import httpx
import pytest

from gmail_search.gateway.gemini import GeminiProfile, GeminiRunService, compile_gemini_request
from gmail_search.gateway.gemini_http import GeminiHTTPTransport
from gmail_search.gateway.provider import AnthropicRunService, ProviderProfile, ReplayRejected
from gmail_search.gateway.registry import AccessDenied
from test_gateway_provider import setup as setup_fixture, spend

setup = setup_fixture
MODEL = 'gemini-3.8-flash'


def body():
    return {'contents': [{'role': 'user', 'parts': [{'text': 'hello'}]}],
            'generationConfig': {'maxOutputTokens': 10}}


def compile(value):
    return compile_gemini_request(value, max_output_tokens=20, thinking_level='HIGH')


def test_pi_0844_google_request_fixture_is_accepted_without_mutation():
    fixture = Path(__file__).parent.joinpath('fixtures/cli_compat/gemini_pi_initial.json')
    value = json.loads(fixture.read_text())
    original = json.loads(json.dumps(value))
    out = compile_gemini_request(value, max_output_tokens=1024, thinking_level='HIGH')
    assert out['generationConfig']['thinkingConfig'] == {
        'thinkingLevel': 'HIGH', 'includeThoughts': True}
    assert value == original


def test_normalizer_fixed_thinking_and_detached_input():
    value = body()
    value['generationConfig']['thinkingConfig'] = {'thinkingLevel': 'LOW', 'includeThoughts': True}
    out = compile(value)
    assert out['generationConfig']['thinkingConfig'] == {'thinkingLevel': 'HIGH', 'includeThoughts': True}
    assert value['generationConfig']['thinkingConfig']['thinkingLevel'] == 'LOW'


@pytest.mark.parametrize('extra', [
    {'model': 'guest-model'}, {'cachedContent': 'caches/a'}, {'serviceTier': 'priority'},
    {'tools': [{'googleSearch': {}}]}, {'tools': [{'codeExecution': {}}]},
    {'contents': [{'role': 'user', 'parts': [{'fileData': {'fileUri': 'https://example.test'}}]}]},
    {'generationConfig': {'maxOutputTokens': 21}},
    {'generationConfig': {'maxOutputTokens': True}},
    {'generationConfig': {'maxOutputTokens': 10, 'candidateCount': 2}},
])
def test_normalizer_rejects_external_capabilities(extra):
    with pytest.raises(ValueError):
        compile(body() | extra)


def test_local_function_roundtrip_preserves_inert_urls():
    value = body()
    value['tools'] = [{'functionDeclarations': [{'name': 'bash', 'description': 'local',
        'parametersJsonSchema': {'type': 'object', 'properties': {'command': {'type': 'string'}}}}]}]
    value['contents'] += [
        {'role': 'model', 'parts': [{'functionCall': {'name': 'bash', 'args': {'command': 'echo https://example.test'}}, 'thoughtSignature': 'c2ln'}]},
        {'role': 'user', 'parts': [{'functionResponse': {'name': 'bash', 'response': {'output': '42'}}}]}]
    assert compile(value)['contents'] == value['contents']


def frame(*, finish=False, usage=True):
    result = {'modelVersion': MODEL, 'responseId': 'synthetic', 'candidates': [
        {'index': 0, 'content': {'role': 'model', 'parts': [{'text': '42'}]}}]}
    if finish:
        result['candidates'][0]['finishReason'] = 'STOP'
    if usage:
        result['usageMetadata'] = {'promptTokenCount': 7, 'candidatesTokenCount': 3,
                                   'thoughtsTokenCount': 2, 'totalTokenCount': 12}
    return b'data: ' + json.dumps(result).encode() + b'\n\n'


class Stream(httpx.AsyncByteStream):
    def __init__(self, data):
        self.data, self.closed = data, False

    async def __aiter__(self):
        for index in range(0, len(self.data), 13):
            yield self.data[index:index + 13]

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
async def test_http_service_fixed_endpoint_budget_thinking_usage_and_replay(setup):
    registry, caps, run, token = setup
    stream = Stream(frame(finish=True))
    calls = []
    def handler(request):
        calls.append(request)
        assert spend(registry) == (230, 0)
        assert str(request.url) == f'https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:streamGenerateContent?alt=sse'
        assert request.headers['x-goog-api-key'] == 'synthetic-key'
        assert 'authorization' not in request.headers and 'cookie' not in request.headers
        assert json.loads(request.content)['generationConfig']['thinkingConfig']['thinkingLevel'] == 'HIGH'
        return httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=stream)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False,
            headers={'authorization': 'untrusted-default'}, cookies={'bad': 'cookie'}) as client:
        adapter = GeminiHTTPTransport('synthetic-key', model=MODEL, client=client)
        svc = GeminiRunService(caps, adapter)
        svc.bind_profile(run.run_id, GeminiProfile(MODEL, 100, 20, 2, 3))
        assert b''.join([part async for part in svc.stream(token, 'req', body())]) == frame(finish=True)
        assert spend(registry) == (0, 29)  # 7 input + (3 answer + 2 thinking)
        with pytest.raises(ReplayRejected):
            [part async for part in svc.stream(token, 'req', body())]
        assert len(calls) == 1 and stream.closed
        with pytest.raises(AccessDenied):
            AnthropicRunService(caps, adapter).profile(run.run_id)
        with pytest.raises(AccessDenied):
            svc.bind_profile(run.run_id, ProviderProfile('model', 100, 20, 2, 3))


@pytest.mark.asyncio
@pytest.mark.parametrize('data', [frame(finish=False), frame(finish=True, usage=False),
    frame(finish=True).replace(b'"totalTokenCount": 12', b'"totalTokenCount": 1'),
    frame(finish=True).replace(MODEL.encode(), b'wrong-model'),
    frame(finish=True) + frame(finish=True), b'data: {"a":1,"a":2}\n\n'])
async def test_bad_upstream_conservatively_charges_and_closes(setup, data):
    registry, caps, run, token = setup
    stream = Stream(data)
    async with httpx.AsyncClient(trust_env=False, transport=httpx.MockTransport(
            lambda req: httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=stream))) as client:
        svc = GeminiRunService(caps, GeminiHTTPTransport('synthetic', model=MODEL, client=client))
        svc.bind_profile(run.run_id, GeminiProfile(MODEL, 100, 20, 2, 3))
        with pytest.raises(RuntimeError):
            [part async for part in svc.stream(token, 'req', body())]
        assert spend(registry) == (0, 230) and stream.closed


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['cancel', 'revoke', 'timeout'])
async def test_google_http_interruption_closes_socket_and_settles(setup, mode):
    registry, caps, run, token = setup
    started = asyncio.Event()
    class Waiting(Stream):
        async def __aiter__(self):
            started.set()
            yield frame(finish=False)
            await asyncio.Event().wait()
    stream = Waiting(b'')
    async with httpx.AsyncClient(trust_env=False, transport=httpx.MockTransport(
            lambda req: httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=stream))) as client:
        svc = GeminiRunService(caps, GeminiHTTPTransport('synthetic', model=MODEL, client=client))
        svc.bind_profile(run.run_id, GeminiProfile(MODEL, 100, 20, 2, 3,
                                                   timeout_seconds=.1 if mode == 'timeout' else 10))
        async def collect():
            return [part async for part in svc.stream(token, 'req', body())]
        task = asyncio.create_task(collect())
        await asyncio.wait_for(started.wait(), 2)
        if mode == 'cancel':
            task.cancel()
        elif mode == 'revoke':
            caps.revoke(token)
        with pytest.raises({'cancel': asyncio.CancelledError, 'revoke': AccessDenied, 'timeout': TimeoutError}[mode]):
            await asyncio.wait_for(task, 2)
        assert stream.closed and spend(registry) == (0, 230)


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['redirect', 'compressed', 'frame', 'body', 'service'])
async def test_google_response_transport_limits(setup, mode):
    registry, caps, run, token = setup
    stream = Stream(frame(finish=True))
    calls = []
    def handler(request):
        calls.append(request)
        headers = {'content-type': 'text/event-stream'}
        if mode == 'redirect':
            headers['location'] = 'https://untrusted.example.test/'
        if mode == 'compressed':
            headers['content-encoding'] = 'gzip'
        return httpx.Response(302 if mode == 'redirect' else 200, headers=headers, stream=stream)
    async with httpx.AsyncClient(trust_env=False, transport=httpx.MockTransport(handler)) as client:
        limits = {'max_frame_bytes': 128, 'max_body_bytes': 128 if mode == 'body' else 1024} if mode in ('body', 'frame') else {}
        svc = GeminiRunService(caps, GeminiHTTPTransport('synthetic', model=MODEL, client=client, **limits))
        svc.bind_profile(run.run_id, GeminiProfile(MODEL, 100, 20, 2, 3,
                        max_response_bytes=2 if mode == 'service' else 1024))
        with pytest.raises((RuntimeError, AccessDenied)):
            [part async for part in svc.stream(token, 'req', body())]
        assert stream.closed and len(calls) == 1 and spend(registry) == (0, 230)


def test_google_profile_persisted_and_cross_family_rebinding_denied(setup):
    _, caps, run, _ = setup
    service = GeminiRunService(caps, None)
    profile = GeminiProfile(MODEL, 100, 20, 2, 3)
    service.bind_profile(run.run_id, profile)
    assert GeminiRunService(caps, None).profile(run.run_id) == profile
    with pytest.raises(AccessDenied):
        service.bind_profile(run.run_id, GeminiProfile(MODEL, 100, 20, 2, 3, thinking_level='LOW'))
    with pytest.raises(AccessDenied):
        AnthropicRunService(caps, None).bind_profile(run.run_id, ProviderProfile(MODEL, 100, 20, 2, 3))


@pytest.mark.asyncio
async def test_revoked_google_run_never_opens_transport(setup):
    registry, caps, run, token = setup
    def handler(request):
        pytest.fail('Revoked request opened HTTP transport')
    async with httpx.AsyncClient(trust_env=False, transport=httpx.MockTransport(handler)) as client:
        svc = GeminiRunService(caps, GeminiHTTPTransport('synthetic', model=MODEL, client=client))
        svc.bind_profile(run.run_id, GeminiProfile(MODEL, 100, 20, 2, 3))
        caps.revoke(token)
        with pytest.raises(AccessDenied):
            [part async for part in svc.stream(token, 'req', body())]
        assert spend(registry) == (0, 0)


def live_frame(usage, *, finish=False, thought=False):
    """Shapes Gemini 3.8 Flash actually streamed on 2026-09-18."""
    part = {'text': 'thinking', 'thought': True} if thought else {'text': '42'}
    result = {'modelVersion': MODEL, 'responseId': 'live', 'candidates': [
        {'index': 0, 'content': {'role': 'model', 'parts': [part]}}], 'usageMetadata': usage}
    if finish:
        result['candidates'][0]['finishReason'] = 'STOP'
    return b'data: ' + json.dumps(result).encode() + b'\n\n'


LIVE_STREAM = (
    # Thought-only first frame: no candidatesTokenCount; the tier is reported.
    live_frame({'promptTokenCount': 7, 'totalTokenCount': 7, 'serviceTier': 'standard'}, thought=True)
    # Terminal frame: a larger prompt count, and implicit cache usage inside it.
    + live_frame({'promptTokenCount': 9, 'candidatesTokenCount': 3, 'thoughtsTokenCount': 2,
                  'totalTokenCount': 14, 'cachedContentTokenCount': 5,
                  'cacheTokensDetails': [{'modality': 'TEXT', 'tokenCount': 5}],
                  'serviceTier': 'standard'}, finish=True))


async def _stream_through_service(setup, data):
    registry, caps, run, token = setup
    stream = Stream(data)
    async with httpx.AsyncClient(trust_env=False, transport=httpx.MockTransport(
            lambda req: httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=stream))) as client:
        svc = GeminiRunService(caps, GeminiHTTPTransport('synthetic', model=MODEL, client=client))
        svc.bind_profile(run.run_id, GeminiProfile(MODEL, 100, 20, 2, 3))
        return b''.join([part async for part in svc.stream(token, 'req', body())])


@pytest.mark.asyncio
async def test_live_stream_shapes_are_accepted_and_settle_on_the_terminal_usage(setup):
    """Every one of these refused a real Pi run until relaxed. Cached tokens are
    charged at the full input rate: over-counting, never under."""
    assert await _stream_through_service(setup, LIVE_STREAM) == LIVE_STREAM
    assert spend(setup[0]) == (0, 9 * 2 + 5 * 3)


@pytest.mark.asyncio
@pytest.mark.parametrize('old,new', [
    (b'"serviceTier": "standard"', b'"serviceTier": "priority"'),
    (b'"promptTokenCount": 9, "candidatesTokenCount": 3, "thoughtsTokenCount": 2, "totalTokenCount": 14',
     b'"promptTokenCount": 6, "candidatesTokenCount": 3, "thoughtsTokenCount": 2, "totalTokenCount": 11'),
    (b'"cachedContentTokenCount": 5', b'"cachedContentTokenCount": 10'),
])
async def test_live_stream_limits_still_hold(setup, old, new):
    """Another tier, a falling prompt count, or more cached than prompted: refused."""
    data = LIVE_STREAM.replace(old, new, 1)
    assert data != LIVE_STREAM
    with pytest.raises(RuntimeError):
        await _stream_through_service(setup, data)
