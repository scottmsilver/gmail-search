"""Deterministic cancellation regressions; synthetic transport and local budgets only."""
import asyncio
from contextlib import asynccontextmanager
import json
import threading

import pytest
from fastapi import FastAPI

from gmail_search.gateway import inference_http
from gmail_search.gateway.capabilities import Capabilities
from gmail_search.gateway.provider import AnthropicRunService, ProviderProfile
from gmail_search.gateway.registry import Registry


class GatedTransport:
    status_code = 200

    def __init__(self):
        self.started = asyncio.Event()
        self.exiting = asyncio.Event()
        self.finish_close = asyncio.Event()
        self.finish_close.set()
        self.closed = False
        self.close_interrupted = False

    @asynccontextmanager
    async def stream(self, **kwargs):
        self.started.set()
        try:
            yield self
        finally:
            self.exiting.set()
            try:
                await self.finish_close.wait()
                self.closed = True
            except asyncio.CancelledError:
                self.close_interrupted = True
                raise

    async def __aiter__(self):
        yield b'data: synthetic\n\n'
        await asyncio.Event().wait()


def balance(registry):
    with registry._transaction() as db:
        return tuple(db.execute('SELECT reserved,spent FROM budgets').fetchone())


@pytest.fixture
def state(tmp_path, monkeypatch):
    tmp_path.chmod(0o700)
    registry = Registry(tmp_path / 'registry.sqlite', is_active=lambda owner: owner == 'alice')
    budget = registry.create_budget('alice', 100_000)
    run = registry.start_run('alice', 'conversation', request_key='start', budget_id=budget)
    caps = Capabilities(registry)
    token = caps.issue(run.run_id, audience='inference', operations=['generate']).secret
    transport = GatedTransport()
    service = AnthropicRunService(caps, transport)
    service.bind_profile(run.run_id, ProviderProfile('server-model', 100, 20, 2, 3))
    releases = []
    original_release = inference_http._Admission.release

    async def release(admission, owner):
        releases.append((transport.closed, balance(registry)))
        await original_release(admission, owner)

    monkeypatch.setattr(inference_http._Admission, 'release', release)
    app = FastAPI()
    inference_http.add_inference_routes(app, anthropic=service, token_from_request=lambda request: token)
    body = {'model': 'server-model', 'messages': [{'role': 'user', 'content': 'hello'}],
            'max_tokens': 10, 'stream': True}
    return app, service, registry, run, token, transport, releases, body


def connection(body):
    delivered = False

    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {'type': 'http.request', 'body': json.dumps(body).encode(), 'more_body': False}
        await asyncio.Event().wait()

    scope = {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.4'},
             'http_version': '1.1', 'method': 'POST', 'path': '/v1/messages',
             'raw_path': b'/v1/messages', 'root_path': '', 'scheme': 'http',
             'query_string': b'', 'headers': [(b'content-type', b'application/json')],
             'server': ('test', 80), 'client': ('test', 123)}
    return scope, receive


@pytest.mark.asyncio
async def test_response_header_send_failure_closes_and_settles_before_admission_release(state):
    app, _, registry, _, token, transport, releases, body = state
    scope, receive = connection(body)
    sent = []

    async def send(message):
        sent.append(message)
        if message['type'] == 'http.response.start':
            raise OSError('synthetic disconnected socket')

    with pytest.raises(Exception):
        await app(scope, receive, send)
    assert transport.closed
    assert releases == [(True, (0, 230))]
    assert token not in repr(sent)


@pytest.mark.asyncio
async def test_cancel_during_prefetch_watcher_handoff_closes_and_settles(state, monkeypatch):
    app, _, _, _, _, transport, releases, body = state
    scope, receive = connection(body)
    watcher_exiting, finish_watcher = asyncio.Event(), asyncio.Event()

    async def disconnect(request):
        try:
            await asyncio.Event().wait()
        finally:
            watcher_exiting.set()
            await finish_watcher.wait()

    monkeypatch.setattr(inference_http, '_disconnect', disconnect)
    sent = []

    async def send(message):
        sent.append(message)

    task = asyncio.create_task(app(scope, receive, send))
    await asyncio.wait_for(watcher_exiting.wait(), 2)
    task.cancel()
    finish_watcher.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert transport.closed
    assert releases == [(True, (0, 230))]
    assert not sent


@pytest.mark.asyncio
async def test_repeated_http_cancel_drains_transport_and_settlement_before_release(state):
    app, _, _, _, token, transport, releases, body = state
    transport.finish_close.clear()
    scope, receive = connection(body)
    first_sent = asyncio.Event()
    sent = []

    async def send(message):
        sent.append(message)
        if message['type'] == 'http.response.body':
            first_sent.set()

    task = asyncio.create_task(app(scope, receive, send))
    await asyncio.wait_for(first_sent.wait(), 2)
    task.cancel()
    await asyncio.wait_for(transport.exiting.wait(), 2)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    transport.finish_close.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not transport.close_interrupted
    assert transport.closed
    assert releases == [(True, (0, 230))]
    assert token not in repr(sent)


@pytest.mark.asyncio
async def test_cancelled_reserve_waits_for_thread_commit_and_settles_created_claim(state, monkeypatch):
    _, service, registry, run, token, transport, _, body = state
    entered = asyncio.Event()
    allow_commit = threading.Event()
    loop = asyncio.get_running_loop()
    reserve = registry.reserve

    def gated_reserve(*args):
        loop.call_soon_threadsafe(entered.set)
        assert allow_commit.wait(2)
        return reserve(*args)

    monkeypatch.setattr(registry, 'reserve', gated_reserve)
    iterator = service.stream(token, 'cancelled-reserve', body)
    task = asyncio.create_task(anext(iterator))
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    allow_commit.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    # Reading after the task completes must already observe the final ledger.
    assert balance(registry) == (0, 230)
    assert not transport.started.is_set()
    with registry._transaction() as db:
        assert db.execute('SELECT charged FROM reservations WHERE run_id=? AND request_key=?',
                          (run.run_id, 'cancelled-reserve')).fetchone()['charged'] == 230


@pytest.mark.asyncio
async def test_cancel_after_prefetch_completed_before_waiter_resumes_closes_result(state, monkeypatch):
    app, _, _, _, _, transport, releases, body = state
    scope, receive = connection(body)
    original_open = inference_http._open

    async def open_and_cancel(*args):
        result = await original_open(*args)
        # This callback runs before asyncio.wait's completion callback resumes
        # the request, although the prefetch task already owns a completed result.
        asyncio.get_running_loop().call_soon(task.cancel)
        return result

    monkeypatch.setattr(inference_http, '_open', open_and_cancel)

    async def send(message):
        pytest.fail('cancelled prefetch must not publish headers')

    task = asyncio.create_task(app(scope, receive, send))
    with pytest.raises(asyncio.CancelledError):
        await task
    assert transport.closed
    assert releases == [(True, (0, 230))]


@pytest.mark.asyncio
async def test_response_construction_failure_closes_prefetched_result(state, monkeypatch):
    app, _, _, _, token, transport, releases, body = state
    scope, receive = connection(body)
    sent = []

    def fail_response(*args, **kwargs):
        raise RuntimeError('synthetic credential must stay private: ' + token)

    monkeypatch.setattr(inference_http, '_InferenceResponse', fail_response)

    async def send(message):
        sent.append(message)

    await app(scope, receive, send)
    assert sent[0]['status'] == 503
    assert token not in repr(sent)
    assert transport.closed
    assert releases == [(True, (0, 230))]


@pytest.mark.asyncio
async def test_repeated_direct_provider_cancel_does_not_interrupt_transport_close(state):
    _, service, registry, _, token, transport, _, body = state
    transport.finish_close.clear()
    iterator = service.stream(token, 'direct-cancel', body)
    await anext(iterator)
    task = asyncio.create_task(anext(iterator))
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.wait_for(transport.exiting.wait(), 2)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    transport.finish_close.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert transport.closed and not transport.close_interrupted
    assert balance(registry) == (0, 230)


@pytest.mark.asyncio
async def test_repeated_cancel_waits_for_settlement_ack_before_releasing_admission(state, monkeypatch):
    app, _, registry, _, _, transport, releases, body = state
    scope, receive = connection(body)
    first_sent, settling = asyncio.Event(), asyncio.Event()
    allow_settlement = threading.Event()
    loop = asyncio.get_running_loop()
    settle = registry.settle

    def gated_settle(*args):
        loop.call_soon_threadsafe(settling.set)
        assert allow_settlement.wait(2)
        return settle(*args)

    monkeypatch.setattr(registry, 'settle', gated_settle)

    async def send(message):
        if message['type'] == 'http.response.body':
            first_sent.set()

    task = asyncio.create_task(app(scope, receive, send))
    await asyncio.wait_for(first_sent.wait(), 2)
    task.cancel()
    await asyncio.wait_for(settling.wait(), 2)
    try:
        assert transport.closed
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not releases and not task.done()
        assert balance(registry) == (230, 0)
    finally:
        allow_settlement.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert releases == [(True, (0, 230))]


@pytest.mark.asyncio
@pytest.mark.parametrize('existing_charge', [None, 23])
async def test_cancelled_replay_does_not_settle_or_overwrite_original_claim(state, monkeypatch, existing_charge):
    _, service, registry, run, token, transport, _, body = state
    registry.reserve(run.run_id, 'original', 230)
    if existing_charge is not None:
        registry.settle(run.run_id, 'original', existing_charge)
    expected = balance(registry)
    entered = asyncio.Event()
    allow_return = threading.Event()
    loop = asyncio.get_running_loop()
    reserve = registry.reserve

    def gated_reserve(*args):
        claim = reserve(*args)
        loop.call_soon_threadsafe(entered.set)
        assert allow_return.wait(2)
        return claim

    monkeypatch.setattr(registry, 'reserve', gated_reserve)
    iterator = service.stream(token, 'original', body)
    task = asyncio.create_task(anext(iterator))
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    allow_return.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert balance(registry) == expected
    assert not transport.started.is_set()
