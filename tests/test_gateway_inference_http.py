"""Worker inference HTTP boundary: auth, fixed routing, and streaming cleanup."""
import asyncio
import json
from pathlib import Path
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, HTTPException

from gmail_search.gateway.gemini import GeminiProfile, GeminiRunService, MODEL
from gmail_search.gateway.gemini_http import GeminiHTTPTransport
from gmail_search.gateway.inference_http import add_inference_routes
from gmail_search.gateway.provider import AnthropicRunService, ProviderProfile
from gmail_search.gateway.provider_http import AnthropicHTTPTransport
from gmail_search.gateway.registry import AccessDenied, Registry
from gmail_search.gateway.capabilities import Capabilities


class RecordingService:
    def __init__(self):
        self.authorized = 0
        self.calls = []

    async def _authorize(self, token):
        self.authorized += 1
        if token != 'capability':
            raise AccessDenied()
        return SimpleNamespace(owner_id='alice')

    async def stream(self, token, request_key, body):
        if set(body) != {'accepted'}:
            raise ValueError('request shape is not accepted by the fixed provider service')
        self.calls.append((token, request_key, body))
        yield b'data: synthetic\n\n'


def app(*, anthropic=None, gemini=None):
    result = FastAPI()
    def token_from_request(request):
        value = request.headers.get('authorization')
        if value is None or not value.startswith('Bearer '):
            raise HTTPException(401, 'Run capability required')
        return value[7:]
    add_inference_routes(result, anthropic=anthropic, gemini=gemini,
                         token_from_request=token_from_request)
    return result


def provider_event(kind, **data):
    return ('event: ' + kind + '\ndata: ' + json.dumps(dict(type=kind, **data)) + '\n\n').encode()


def anthropic_sse():
    return b''.join((
        provider_event('message_start', message={'id': 'msg_synthetic', 'type': 'message',
                       'role': 'assistant', 'model': 'claude-sonnet-4-6', 'content': [],
                       'stop_reason': None, 'stop_sequence': None,
                       'usage': {'input_tokens': 7, 'output_tokens': 0}}),
        provider_event('content_block_start', index=0, content_block={'type': 'text', 'text': ''}),
        provider_event('content_block_delta', index=0, delta={'type': 'text_delta', 'text': '42'}),
        provider_event('content_block_stop', index=0),
        provider_event('message_delta', delta={'stop_reason': 'end_turn', 'stop_sequence': None},
                       usage={'output_tokens': 3}),
        provider_event('message_stop'),
    ))


def gemini_sse():
    value = {'modelVersion': MODEL, 'responseId': 'synthetic', 'candidates': [
        {'index': 0, 'content': {'role': 'model', 'parts': [{'text': '42'}]}, 'finishReason': 'STOP'}],
        'usageMetadata': {'promptTokenCount': 7, 'candidatesTokenCount': 3,
                          'thoughtsTokenCount': 2, 'totalTokenCount': 12}}
    return b'data: ' + json.dumps(value).encode() + b'\n\n'


class Wire(httpx.AsyncByteStream):
    def __init__(self, data):
        self.data = data

    async def __aiter__(self):
        for index in range(0, len(self.data), 17):
            yield self.data[index:index + 17]

    async def aclose(self):
        pass


def inference_state(tmp_path, *, key):
    tmp_path.mkdir(mode=0o700)
    registry = Registry(tmp_path / 'registry.sqlite', is_active=lambda owner: owner == 'alice')
    budget = registry.create_budget('alice', 1_000_000)
    run = registry.start_run('alice', 'conversation', request_key=key, budget_id=budget)
    caps = Capabilities(registry)
    return registry, caps, run, caps.issue(run.run_id, audience='inference', operations={'generate'}).secret


@pytest.mark.asyncio
async def test_qualified_cli_fixtures_stream_through_fixed_anthropic_and_gemini_routes(tmp_path):
    anthropic_registry, anthropic_caps, anthropic_run, anthropic_token = inference_state(tmp_path / 'anthropic', key='a')
    gemini_registry, gemini_caps, gemini_run, gemini_token = inference_state(tmp_path / 'gemini', key='g')
    calls = []

    def anthropic_handler(request):
        calls.append(str(request.url))
        assert request.headers['x-api-key'] == 'synthetic-anthropic'
        assert 'authorization' not in request.headers
        return httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=Wire(anthropic_sse()))

    def gemini_handler(request):
        calls.append(str(request.url))
        assert request.headers['x-goog-api-key'] == 'synthetic-gemini'
        assert 'authorization' not in request.headers
        return httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=Wire(gemini_sse()))

    async with httpx.AsyncClient(transport=httpx.MockTransport(anthropic_handler), trust_env=False) as upstream_a, \
            httpx.AsyncClient(transport=httpx.MockTransport(gemini_handler), trust_env=False) as upstream_g:
        anthropic = AnthropicRunService(anthropic_caps, AnthropicHTTPTransport('synthetic-anthropic', client=upstream_a))
        anthropic.bind_profile(anthropic_run.run_id, ProviderProfile(
            'claude-sonnet-4-6', 100, 40_000, 2, 3, client_profile='claude-2.1.272', effort='high'))
        gemini = GeminiRunService(gemini_caps, GeminiHTTPTransport('synthetic-gemini', model=MODEL, client=upstream_g))
        gemini.bind_profile(gemini_run.run_id, GeminiProfile(MODEL, 100, 2_000, 2, 3))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app(anthropic=anthropic, gemini=gemini)),
                                     base_url='http://gateway') as client:
            claude_body = Path(__file__).parent.joinpath('fixtures/cli_compat/claude_initial.json').read_bytes()
            claude = await client.post('/v1/messages', headers={
                'authorization': 'Bearer ' + anthropic_token, 'content-type': 'application/json'}, content=claude_body)
            pi_body = Path(__file__).parent.joinpath('fixtures/cli_compat/gemini_pi_initial.json').read_bytes()
            google = await client.post('/v1beta/models/gemini-3.8-flash:streamGenerateContent?alt=sse', headers={
                'authorization': 'Bearer ' + gemini_token, 'content-type': 'application/json'}, content=pi_body)
    assert claude.status_code == google.status_code == 200
    assert claude.content == anthropic_sse() and google.content == gemini_sse()
    assert claude.headers['x-gateway-request-key'].startswith('gw-')
    assert google.headers['x-gateway-request-key'].startswith('gw-')
    assert calls == ['https://api.anthropic.com/v1/messages',
                     'https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:streamGenerateContent?alt=sse']
    # These are synthetic local registry figures; MockTransport made no provider call.
    assert anthropic_registry.path.exists() and gemini_registry.path.exists()


@pytest.mark.asyncio
async def test_stable_request_key_is_optional_and_replay_is_not_retried():
    class ReplayService(RecordingService):
        def __init__(self):
            super().__init__()
            self.claims = set()

        async def stream(self, token, request_key, body):
            if request_key in self.claims:
                from gmail_search.gateway.provider import ReplayRejected
                raise ReplayRejected()
            self.claims.add(request_key)
            async for chunk in super().stream(token, request_key, body):
                yield chunk

    service = ReplayService()
    headers = {'authorization': 'Bearer capability', 'content-type': 'application/json'}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app(anthropic=service)),
                                 base_url='http://gateway') as client:
        first = await client.post('/v1/messages', headers={**headers, 'x-gateway-request-key': 'trusted-1'},
                                  json={'accepted': True})
        repeated = await client.post('/v1/messages', headers={**headers, 'x-gateway-request-key': 'trusted-1'},
                                     json={'accepted': True})
        generated_a = await client.post('/v1/messages', headers=headers, json={'accepted': True})
        generated_b = await client.post('/v1/messages', headers=headers, json={'accepted': True})
    assert first.status_code == generated_a.status_code == generated_b.status_code == 200
    assert repeated.status_code == 409
    assert generated_a.headers['x-gateway-request-key'] != generated_b.headers['x-gateway-request-key']


@pytest.mark.asyncio
@pytest.mark.parametrize('publish_first', [False, True], ids=['before-first-byte', 'after-first-byte'])
async def test_client_cancellation_closes_real_provider_iterator_and_settles_reservation(tmp_path, publish_first):
    registry, caps, run, token = inference_state(tmp_path / 'cancel', key='cancel')

    class WaitingTransport:
        status_code = 200

        def __init__(self):
            self.started = asyncio.Event()
            self.closed = asyncio.Event()

        @asynccontextmanager
        async def stream(self, **kwargs):
            self.started.set()
            try:
                yield self
            finally:
                self.closed.set()

        async def __aiter__(self):
            if publish_first:
                yield b'data: initial\n\n'
            await asyncio.Event().wait()

    transport = WaitingTransport()
    service = AnthropicRunService(caps, transport)
    service.bind_profile(run.run_id, ProviderProfile('server-model', 100, 20, 2, 3))
    body = {'model': 'server-model', 'messages': [{'role': 'user', 'content': 'hello'}],
            'max_tokens': 10, 'stream': True}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app(anthropic=service)),
                                 base_url='http://gateway') as client:
        task = asyncio.create_task(client.post('/v1/messages', headers={
            'authorization': 'Bearer ' + token, 'content-type': 'application/json'}, json=body))
        await asyncio.wait_for(transport.started.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    await asyncio.wait_for(transport.closed.wait(), 2)
    with registry._transaction() as db:
        budget = db.execute('SELECT reserved, spent FROM budgets').fetchone()
    assert tuple(budget) == (0, 230)


@pytest.mark.asyncio
async def test_authentication_precedes_streaming_body_consumption():
    service = RecordingService()

    class UnreadableBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            pytest.fail('route read a body before authenticating its run capability')

        async def aclose(self):
            pass

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app(anthropic=service)),
                                 base_url='http://gateway') as client:
        response = await client.post('/v1/messages', content=UnreadableBody())
    assert response.status_code == 401
    assert service.authorized == 0


@pytest.mark.asyncio
async def test_fixed_routes_reject_owner_model_paths_and_duplicate_json_keys():
    service = RecordingService()
    headers = {'authorization': 'Bearer capability', 'content-type': 'application/json'}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app(anthropic=service)),
                                 base_url='http://gateway') as client:
        assert (await client.post('/v1/messages/other', headers=headers, content=b'{}')).status_code == 404
        assert (await client.post('/v1/messages?model=other', headers=headers, content=b'{}')).status_code == 400
        assert (await client.post('/v1/messages', headers=headers,
                                  content=b'{"owner_id":"other","owner_id":"other"}')).status_code == 400
        assert (await client.post('/v1/messages', headers=headers,
                                  content=b'{"model":"other"}')).status_code == 400
    assert not service.calls


@pytest.mark.asyncio
async def test_actual_service_rejects_client_selected_owner_or_model_before_upstream(tmp_path):
    _, caps, run, token = inference_state(tmp_path / 'fixed', key='fixed')
    calls = []

    def handler(request):
        calls.append(request)
        pytest.fail('an invalid client-selected owner or model reached the upstream transport')

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False) as upstream:
        service = AnthropicRunService(caps, AnthropicHTTPTransport('synthetic', client=upstream))
        service.bind_profile(run.run_id, ProviderProfile('server-model', 100, 20, 2, 3))
        headers = {'authorization': 'Bearer ' + token, 'content-type': 'application/json'}
        body = {'model': 'server-model', 'messages': [{'role': 'user', 'content': 'hello'}],
                'max_tokens': 10, 'stream': True}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app(anthropic=service)),
                                     base_url='http://gateway') as client:
            owner = await client.post('/v1/messages', headers=headers, json=body | {'owner_id': 'bob'})
            model = await client.post('/v1/messages', headers=headers, json=body | {'model': 'guest-model'})
    assert owner.status_code == model.status_code == 400
    assert not calls


@pytest.mark.asyncio
async def test_google_route_accepts_only_its_exact_sse_query():
    service = RecordingService()
    headers = {'authorization': 'Bearer capability', 'content-type': 'application/json'}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app(gemini=service)),
                                 base_url='http://gateway') as client:
        accepted = await client.post('/v1beta/models/gemini-3.8-flash:streamGenerateContent?alt=sse',
                                     headers=headers, json={'accepted': True})
        wrong = await client.post('/v1beta/models/gemini-3.8-flash:streamGenerateContent?alt=json',
                                  headers=headers, json={'accepted': True})
        extra = await client.post('/v1beta/models/gemini-3.8-flash:streamGenerateContent?alt=sse&model=other',
                                  headers=headers, json={'accepted': True})
    assert accepted.status_code == 200
    assert wrong.status_code == extra.status_code == 400


@pytest.mark.asyncio
async def test_process_and_owner_admission_reject_before_body_and_release_after_cancel():
    class BlockingService:
        def __init__(self):
            self.owners = {'a1': 'alice', 'a2': 'alice', 'b1': 'bob', 'b2': 'bob', 'c': 'carol'}
            self.active = set()
            self.ready = asyncio.Event()
            self.closed = asyncio.Event()

        async def _authorize(self, token):
            return SimpleNamespace(owner_id=self.owners[token])

        async def stream(self, token, request_key, body):
            assert body == {'accepted': True}
            self.active.add(request_key)
            if len(self.active) == 4:
                self.ready.set()
            try:
                yield b'data: held\n\n'
                await asyncio.Event().wait()
            finally:
                self.active.discard(request_key)
                self.closed.set()

    class UnreadableBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            pytest.fail('admission rejection read the request body')

        async def aclose(self):
            pass

    service = BlockingService()
    headers = lambda token: {'authorization': 'Bearer ' + token, 'content-type': 'application/json'}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app(anthropic=service, gemini=service)),
                                 base_url='http://gateway') as client:
        held = [asyncio.create_task(client.post('/v1/messages', headers=headers(token), json={'accepted': True}))
                for token in ('a1', 'a2')]
        await asyncio.sleep(0)
        owner_denied = await client.post('/v1/messages', headers=headers('a1'), content=UnreadableBody())
        held += [asyncio.create_task(client.post('/v1beta/models/gemini-3.8-flash:streamGenerateContent?alt=sse',
                                                 headers=headers(token), json={'accepted': True}))
                 for token in ('b1', 'b2')]
        await asyncio.wait_for(service.ready.wait(), 2)
        global_denied = await client.post('/v1/messages', headers=headers('c'), content=UnreadableBody())
        held[0].cancel()
        with pytest.raises(asyncio.CancelledError):
            await held[0]
        await asyncio.wait_for(service.closed.wait(), 2)
        released = asyncio.create_task(client.post('/v1/messages', headers=headers('c'), json={'accepted': True}))
        for _ in range(20):
            if len(service.active) == 4:
                break
            await asyncio.sleep(0)
        assert len(service.active) == 4
        released.cancel()
        with pytest.raises(asyncio.CancelledError):
            await released
        for task in held[1:]:
            task.cancel()
        await asyncio.gather(*held[1:], return_exceptions=True)
    assert owner_denied.status_code == global_denied.status_code == 429


@pytest.mark.asyncio
async def test_gateway_factory_mounts_optional_inference_with_shared_capability_auth():
    from gmail_search.gateway.http import create_gateway_app
    service = RecordingService()
    gateway = create_gateway_app(None, anthropic=service, gemini=service)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway), base_url='http://gateway') as client:
        for path in ('/v1/messages', '/v1beta/models/gemini-3.8-flash:streamGenerateContent?alt=sse'):
            denied = await client.post(path, json={'accepted': True})
            assert denied.status_code == 401
            result = await client.post(path, headers={'Authorization': 'Bearer capability'}, json={'accepted': True})
            assert result.status_code == 200
            assert result.content == b'data: synthetic\n\n'
            assert result.headers['cache-control'] == 'private, no-store'
    assert len(service.calls) == 2
