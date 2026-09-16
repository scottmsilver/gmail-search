import asyncio
from contextlib import asynccontextmanager

import pytest

from gmail_search.gateway.capabilities import Capabilities
from gmail_search.gateway.registry import AccessDenied, Registry
from gmail_search.gateway.provider import AnthropicRunService, ProviderProfile, ProviderUsage, ReplayRejected


@pytest.fixture
def setup(tmp_path):
    tmp_path.chmod(0o700)
    registry = Registry(tmp_path / 'registry.sqlite', is_active=lambda owner: owner == 'alice')
    budget = registry.create_budget('alice', 100_000)
    run = registry.start_run('alice', 'conversation', request_key='start', budget_id=budget)
    caps = Capabilities(registry)
    token = caps.issue(run.run_id, audience='inference', operations=['generate']).secret
    return registry, caps, run, token


class Transport:
    def __init__(self, *, usage=ProviderUsage(7, 3), wait=False, status=200):
        self.usage, self.wait, self.status_code = usage, wait, status
        self.calls, self.closed = [], False
        self.started = asyncio.Event()

    @asynccontextmanager
    async def stream(self, **kwargs):
        self.calls.append(kwargs)
        self.started.set()
        try:
            yield self
        finally:
            self.closed = True

    async def __aiter__(self):
        yield b'data: synthetic\n\n'
        if self.wait:
            await asyncio.Event().wait()


def service(setup, transport, **profile):
    registry, caps, run, token = setup
    result = AnthropicRunService(caps, transport)
    result.bind_profile(run.run_id, ProviderProfile(model='server-model', input_token_limit=100,
                        output_token_limit=20, input_units_per_token=2, output_units_per_token=3, **profile))
    return result


def body(**extra):
    return dict(model='server-model', messages=[{'role':'user','content':'hello'}], max_tokens=10, stream=True, **extra)


async def collect(svc, token, request_key='request', request=None):
    return [chunk async for chunk in svc.stream(token, request_key, body() if request is None else request)]


def spend(registry):
    with registry._transaction() as db:
        row = db.execute('SELECT reserved,spent FROM budgets').fetchone()
        return tuple(row)


@pytest.mark.asyncio
async def test_fixed_endpoint_reserve_before_send_trusted_usage_and_no_replay(setup):
    registry, caps, run, token = setup
    transport = Transport()
    original = transport.stream
    @asynccontextmanager
    async def check(**kwargs):
        assert spend(registry) == (230, 0)
        async with original(**kwargs) as stream:
            yield stream
    transport.stream = check
    svc = service(setup, transport)
    assert await collect(svc, token) == [b'data: synthetic\n\n']
    assert transport.calls[0]['url'] == 'https://api.anthropic.com/v1/messages'
    assert transport.calls[0]['follow_redirects'] is False
    assert transport.calls[0]['body']['model'] == 'server-model'
    assert spend(registry) == (0, 23)
    with pytest.raises(ReplayRejected):
        await collect(svc, token)
    assert len(transport.calls) == 1 and spend(registry) == (0, 23)


@pytest.mark.asyncio
async def test_wrong_model_and_unbound_run_do_not_reserve_or_send(setup):
    registry, caps, run, token = setup
    transport = Transport()
    svc = AnthropicRunService(caps, transport)
    with pytest.raises(AccessDenied):
        await collect(svc, token)
    svc = service(setup, transport)
    bad = body(); bad['model'] = 'guest-model'
    with pytest.raises(ValueError):
        await collect(svc, token, request=bad)
    assert not transport.calls and spend(registry) == (0, 0)


@pytest.mark.asyncio
@pytest.mark.parametrize('usage', [None, ProviderUsage(-1, 3), ProviderUsage(1000, 3), ProviderUsage(1, True)])
async def test_unknown_or_invalid_usage_charged_conservatively(setup, usage):
    transport = Transport(usage=usage)
    svc = service(setup, transport)
    await collect(svc, setup[3])
    assert spend(setup[0]) == (0, 230)


@pytest.mark.asyncio
async def test_revocation_closes_waiting_upstream_and_discards_results(setup):
    transport = Transport(wait=True)
    svc = service(setup, transport)
    task = asyncio.create_task(collect(svc, setup[3]))
    await transport.started.wait()
    setup[1].revoke(setup[3])
    with pytest.raises(AccessDenied):
        await asyncio.wait_for(task, 2)
    assert transport.closed and spend(setup[0]) == (0, 230)


@pytest.mark.asyncio
async def test_cancellation_closes_upstream_and_keeps_charge(setup):
    transport = Transport(wait=True)
    svc = service(setup, transport)
    task = asyncio.create_task(collect(svc, setup[3]))
    await transport.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert transport.closed and spend(setup[0]) == (0, 230)


@pytest.mark.asyncio
async def test_deadline_closes_upstream(setup):
    transport = Transport(wait=True)
    svc = service(setup, transport, timeout_seconds=.05)
    with pytest.raises(TimeoutError):
        await collect(svc, setup[3])
    assert transport.closed and spend(setup[0]) == (0, 230)


@pytest.mark.asyncio
async def test_redirect_refused_without_emitting_bytes(setup):
    transport = Transport(status=302)
    svc = service(setup, transport)
    with pytest.raises(AccessDenied):
        await collect(svc, setup[3])
    assert transport.closed and spend(setup[0]) == (0, 230)


def test_profile_binding_is_durable_and_immutable(setup):
    transport = Transport()
    svc = service(setup, transport)
    with pytest.raises(AccessDenied):
        svc.bind_profile(setup[2].run_id, ProviderProfile('other',100,20,2,3))
    reopened = AnthropicRunService(setup[1], transport)
    assert reopened.profile(setup[2].run_id).model == 'server-model'


@pytest.mark.asyncio
async def test_concurrent_replay_does_not_send_twice(setup):
    transport = Transport(wait=True)
    svc = service(setup, transport)
    task = asyncio.create_task(collect(svc, setup[3]))
    await transport.started.wait()
    with pytest.raises(ReplayRejected):
        await collect(svc, setup[3])
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert len(transport.calls) == 1 and spend(setup[0]) == (0, 230)


@pytest.mark.asyncio
async def test_closed_consumer_cancels_upstream(setup):
    transport = Transport(wait=True)
    svc = service(setup, transport)
    stream = svc.stream(setup[3], 'request', body())
    assert await anext(stream) == b'data: synthetic\n\n'
    await stream.aclose()
    assert transport.closed and spend(setup[0]) == (0, 230)


@pytest.mark.asyncio
async def test_suspended_consumer_does_not_prevent_revocation_shutdown(setup):
    transport = Transport(wait=True)
    svc = service(setup, transport)
    stream = svc.stream(setup[3], 'request', body())
    await anext(stream)
    setup[1].revoke(setup[3])
    async with asyncio.timeout(2):
        while not transport.closed:
            await asyncio.sleep(.01)
    with pytest.raises(AccessDenied):
        await anext(stream)
    assert spend(setup[0]) == (0, 230)


@pytest.mark.asyncio
async def test_response_size_limit_aborts_and_charges_reservation(setup):
    svc = service(setup, Transport(), max_response_bytes=2)
    with pytest.raises(AccessDenied):
        await collect(svc, setup[3])
    assert spend(setup[0]) == (0, 230)


@pytest.mark.asyncio
async def test_revoked_capability_prevents_reservation(setup):
    transport = Transport()
    svc = service(setup, transport)
    setup[1].revoke(setup[3])
    with pytest.raises(AccessDenied):
        await collect(svc, setup[3])
    assert not transport.calls and spend(setup[0]) == (0, 0)


@pytest.mark.asyncio
async def test_exhausted_shared_budget_prevents_provider_call(setup):
    transport = Transport()
    svc = service(setup, transport)
    setup[0].reserve(setup[2].run_id, 'other-call', 99_999)
    with pytest.raises(AccessDenied):
        await collect(svc, setup[3])
    assert not transport.calls and spend(setup[0]) == (99_999, 0)


@pytest.mark.asyncio
async def test_transport_error_is_sanitized_and_charges_reservation(setup):
    transport = Transport()
    @asynccontextmanager
    async def fail(**kwargs):
        raise RuntimeError('provider secret and private request')
        yield
    transport.stream = fail
    svc = service(setup, transport)
    with pytest.raises(RuntimeError) as error:
        await collect(svc, setup[3])
    assert 'secret' not in str(error.value) and 'private' not in str(error.value)
    assert spend(setup[0]) == (0, 230)


@pytest.mark.asyncio
async def test_server_bound_cli_profile_compiles_hints_without_forwarding(setup):
    transport=Transport()
    svc=service(setup,transport,client_profile='claude-2.1.272')
    request=body(metadata={'user_id':'synthetic-device'},output_config={'effort':'high'})
    await collect(svc,setup[3],request=request)
    assert 'metadata' not in transport.calls[0]['body']
    assert 'output_config' not in transport.calls[0]['body']
