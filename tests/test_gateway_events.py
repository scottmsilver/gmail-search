"""Bounded durable event replay for authenticated run owners."""
import pytest
from gmail_search.gateway.capabilities import Capabilities
from gmail_search.gateway.registry import Registry, AccessDenied
from gmail_search.gateway.events import Events


@pytest.fixture
def state(tmp_path):
    active = {'alice','bob'}
    registry = Registry(tmp_path/'registry', is_active=lambda owner: owner in active)
    caps = Capabilities(registry)
    run = registry.start_run('alice','conversation',request_key='request')
    token = caps.issue(run.run_id, audience='events',operations={'append'}).secret
    return registry, caps, run, token, active


def test_replay_survives_finish_restart_and_owner_check(state):
    registry, caps, run, token, _ = state
    events = Events(caps)
    assert events.append(token, {'type':'text','text':'one'}) == 1
    assert events.append(token, {'type':'text','text':'two'}) == 2
    registry.finish(run.run_id,status='completed')
    replay = Events(caps).read('alice','conversation',run.run_id,after=1)
    assert replay == [{'seq':2,'event':{'type':'text','text':'two'}}]
    for owner,conversation in [('bob','conversation'),('alice','other')]:
        with pytest.raises(AccessDenied):
            events.read(owner,conversation,run.run_id)
    with pytest.raises(AccessDenied):
        events.append(token, {'type':'text','text':'late'})


def test_event_limits_and_revocation(state):
    registry, caps, run, token, active = state
    events = Events(caps,max_events=2,max_event_bytes=64,max_run_bytes=64)
    events.append(token, {'type':'text','text':'x'})
    with pytest.raises(AccessDenied):
        events.append(token, {'type':'text','text':'x'*100})
    events.append(token, {'type':'text','text':'y'})
    with pytest.raises(AccessDenied):
        events.append(token, {'type':'text','text':'z'})
    active.remove('alice')
    with pytest.raises(AccessDenied):
        events.read('alice','conversation',run.run_id)


@pytest.mark.parametrize('event', [{'type':'text','text':float('nan')},{'type':'text','text':b'bytes'}, {'owner_id':'bob'}, {'type':'text','text':'\ud800'}])
def test_invalid_events_fail_closed(state,event):
    _,caps,_,token,_=state
    with pytest.raises(AccessDenied):
        Events(caps).append(token,event)


def test_stale_battle_cannot_append_after_new_writer(state):
    registry,caps,_,_,_=state
    branch = registry.start_run('bob','conversation',request_key='branch',writer=False)
    secret = caps.issue(branch.run_id,audience='events',operations={'append'}).secret
    registry.start_run('bob','conversation',request_key='writer')
    with pytest.raises(AccessDenied):
        Events(caps).append(secret, {'type':'text','text':'stale'})


def test_owner_quota_across_runs_and_terminal_cleanup(state):
    registry,caps,first,secret,_=state
    events=Events(caps,max_owner_bytes=40)
    events.append(secret,{'type':'text','text':'first'})
    registry.finish(first.run_id,status='completed')
    second=registry.start_run('alice','conversation',request_key='second')
    next_secret=caps.issue(second.run_id,audience='events',operations={'append'}).secret
    with pytest.raises(AccessDenied):
        events.append(next_secret,{'type':'text','text':'second'})
    with pytest.raises(AccessDenied):
        events.purge_terminal('bob',first.run_id)
    with pytest.raises(AccessDenied):
        events.purge_terminal('alice',second.run_id)
    events.purge_terminal('alice',first.run_id)
    assert events.append(next_secret,{'type':'text','text':'second'})==1
    assert events.read('alice','conversation',first.run_id)==[]
