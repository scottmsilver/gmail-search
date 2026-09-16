"""Worker event append derives replay ownership from its capability."""
import pytest
from fastapi.testclient import TestClient
from gmail_search.gateway.capabilities import Capabilities
from gmail_search.gateway.events import Events
from gmail_search.gateway.http import create_gateway_app
from gmail_search.gateway.registry import Registry, AccessDenied


def test_event_route_binds_owner_and_rejects_identity_override_or_revocation(tmp_path):
    registry = Registry(tmp_path/'registry', is_active=lambda owner: owner in {'alice', 'bob'})
    caps = Capabilities(registry)
    events = Events(caps)
    run = registry.start_run('alice', 'conversation', request_key='request')
    token = caps.issue(run.run_id, audience='events', operations={'append'}).secret
    headers = {'Authorization': 'Bearer '+token}
    with TestClient(create_gateway_app(None, events=events)) as client:
        assert client.post('/v1/events', json={'type':'text'}).status_code == 401
        result = client.post('/v1/events', headers=headers, json={'type':'text', 'text':'synthetic'})
        assert result.status_code == 200 and result.json() == {'seq':1}
        assert result.headers['cache-control'] == 'private, no-store'
        for event in ({'type':'text', 'owner_id':'bob'}, {'type':'text', 'run_id':run.run_id}):
            assert client.post('/v1/events', headers=headers, json=event).status_code == 403
        assert client.post('/v1/events?owner=bob', headers=headers, json={'type':'text'}).status_code == 400
        assert client.get('/v1/events', headers=headers).status_code == 405
        caps.revoke(token)
        assert client.post('/v1/events', headers=headers, content=b'invalid-json').status_code == 403
    assert events.read('alice','conversation',run.run_id) == [{'seq':1,'event':{'type':'text','text':'synthetic'}}]
    with pytest.raises(AccessDenied):
        events.read('bob','conversation',run.run_id)


def test_event_route_bounds_body_and_wrong_audience(tmp_path):
    registry = Registry(tmp_path/'registry', is_active=lambda owner: owner == 'alice')
    caps = Capabilities(registry)
    events = Events(caps)
    run = registry.start_run('alice', 'conversation', request_key='request')
    token = caps.issue(run.run_id, audience='events', operations={'append'}).secret
    other = caps.issue(run.run_id, audience='sql', operations={'query'}).secret
    with TestClient(create_gateway_app(None, events=events)) as client:
        assert client.post('/v1/events', headers={'Authorization':'Bearer '+other}, json={'type':'text'}).status_code == 403
        result = client.post('/v1/events', headers={'Authorization':'Bearer '+token}, json={'type':'text', 'text':'x'*32768})
        assert result.status_code == 413
    assert events.read('alice','conversation',run.run_id) == []


@pytest.mark.asyncio
async def test_event_write_drain_survives_repeated_request_cancellation():
    import asyncio
    import threading
    from gmail_search.gateway.event_http import _settled_call
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    def operation():
        entered.set()
        assert release.wait(3)
        finished.set()
        return 1
    task = asyncio.create_task(_settled_call(operation))
    try:
        async with asyncio.timeout(2):
            while not entered.is_set():
                await asyncio.sleep(.001)
        task.cancel()
        await asyncio.sleep(.01)
        task.cancel()
        await asyncio.sleep(.01)
        assert not task.done()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert finished.is_set()
