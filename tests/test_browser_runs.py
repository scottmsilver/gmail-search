import asyncio
import logging
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from gmail_search.gateway import inference_http
from gmail_search.gateway.browser_runs import BrowserRuns
from gmail_search.gateway.capabilities import Capabilities
from gmail_search.gateway.events import Events
from gmail_search.gateway.http import _token
from gmail_search.gateway.inference_http import add_inference_routes
from gmail_search.gateway.registry import AccessDenied, Registry
from gmail_search.gateway.worker import WorkerController, WorkerLimits


class Backend:
    namespace = 'browser-fixture'

    def __init__(self, events, capabilities):
        self.events,self.capabilities = events,capabilities
        self.handles = set()
        self.complete = True
        self.stop_fails = False
        self.prompts = {}
        self.runtimes = {}

    def launch(self, handle, lease, limits):
        self.handles.add(handle)
        token = self.capabilities.issue(lease.run_id,audience='events',operations=['append']).secret
        self.events.append(token,{'type':'tool_start','name':'query_emails_batch','args':{}})
        if self.complete:
            self.events.append(token,{'type':'text','text':'Answer for '+self.prompts[lease.run_id]})
            self.events.append(token,{'type':'status','state':'runner_completed'})

    def renew(self, handle, lease):
        assert handle in self.handles

    def stop(self, handle):
        if self.stop_fails:
            raise RuntimeError('synthetic stop failure')
        self.handles.discard(handle)

    def inventory(self):
        return set(self.handles)


@pytest.fixture
def bundle(tmp_path):
    tmp_path.chmod(0o700)
    registry = Registry(tmp_path/'registry.sqlite',is_active=lambda owner:owner in ('alice','bob'))
    caps = Capabilities(registry)
    events = Events(caps)
    backend = Backend(events,caps)
    workers = WorkerController(registry,backend,limits=WorkerLimits(wall_seconds=1800))
    budgets = {owner:registry.create_budget(owner,1000) for owner in ('alice','bob')}
    saved = []
    def prepare(lease,prompt,runtime):
        backend.prompts[lease.run_id] = prompt
        backend.runtimes[lease.run_id] = runtime
    def persist(*args):
        saved.append(args)
        return True
    runs = BrowserRuns(workers,events,prepare_input=prepare,persist_answer=persist,
        budget_for=budgets.__getitem__,poll_seconds=.01)
    return SimpleNamespace(runs=runs,backend=backend,registry=registry,saved=saved)


async def wait_state(bundle, run_id, wanted):
    async with asyncio.timeout(3):
        while True:
            state = bundle.runs.snapshot('alice','conversation',run_id)
            if state['state'] == wanted:
                return state
            await asyncio.sleep(.01)


@pytest.mark.asyncio
async def test_question_to_worker_events_saved_answer_and_stop_ack(bundle):
    run = await bundle.runs.start('alice','conversation','my arbitrary question')
    try:
        state = await wait_state(bundle,run,'completed')
        assert state['answer'] == 'Answer for my arbitrary question'
        assert bundle.saved == [('alice','conversation',run,state['answer'])]
        assert [row['event']['type'] for row in state['events']] == ['tool_start','text','status']
        assert bundle.backend.handles == set()
        with pytest.raises(AccessDenied):
            bundle.runs.snapshot('bob','conversation',run)
        with pytest.raises(AccessDenied):
            await bundle.runs.cancel('bob','conversation',run)
    finally:
        await bundle.runs.close()


@pytest.mark.asyncio
async def test_cancel_revokes_run_and_reaps_worker(bundle):
    bundle.backend.complete = False
    run = await bundle.runs.start('alice','conversation','question')
    try:
        await wait_state(bundle,run,'running')
        await bundle.runs.cancel('alice','conversation',run)
        assert bundle.runs.snapshot('alice','conversation',run)['state'] == 'cancelled'
        assert not bundle.backend.handles
        with pytest.raises(AccessDenied):
            bundle.registry.heartbeat(run)
    finally:
        await bundle.runs.close()


@pytest.mark.asyncio
async def test_failed_stop_is_stopping_until_cleanup_retry(bundle):
    bundle.backend.stop_fails = True
    run = await bundle.runs.start('alice','conversation','question')
    try:
        state = await wait_state(bundle,run,'stopping')
        assert state['answer'] == '' and bundle.saved == []
        assert bundle.backend.handles
        bundle.backend.stop_fails = False
        await bundle.runs.cancel('alice','conversation',run)
        assert not bundle.backend.handles
    finally:
        bundle.backend.stop_fails = False
        await bundle.runs.close()


@pytest.mark.asyncio
async def test_immediate_stop_before_drive_starts_is_terminal(bundle):
    run = await bundle.runs.start('alice','conversation','question')
    task = bundle.runs._tasks[run]
    task.cancel()
    await bundle.runs.cancel('alice','conversation',run)
    assert bundle.runs.snapshot('alice','conversation',run)['state'] == 'cancelled'
    assert not bundle.backend.handles


@pytest.mark.asyncio
async def test_unicode_question_limit_precedes_any_worker_or_run(bundle):
    with pytest.raises(ValueError):
        await bundle.runs.start('alice','conversation','é'*8193)
    assert not bundle.backend.handles and not bundle.backend.prompts
    with bundle.registry._transaction() as db:
        assert db.execute('SELECT count(*) FROM runs').fetchone()[0] == 0


@pytest.mark.asyncio
async def test_cancel_completed_preserves_persisted_answer(bundle):
    run=await bundle.runs.start('alice','conversation','question')
    before=await wait_state(bundle,run,'completed')
    await asyncio.sleep(.02)
    await bundle.runs.cancel('alice','conversation',run)
    assert bundle.runs.snapshot('alice','conversation',run)==before
    assert len(bundle.saved)==1
    await bundle.runs.close()


@pytest.mark.asyncio
async def test_preparation_failure_is_failed_not_user_cancelled(bundle):
    def fail(*args):raise ValueError('fixture failure')
    bundle.runs.prepare_input=fail
    run=await bundle.runs.start('alice','conversation','question')
    await wait_state(bundle,run,'failed')
    assert not bundle.backend.handles and not bundle.saved
    await bundle.runs.close()


@pytest.mark.asyncio
async def test_close_drains_inflight_admission(bundle):
    import threading
    entered=threading.Event();release=threading.Event()
    original=bundle.runs._open
    def gated(*args):
        entered.set();release.wait(3);return original(*args)
    bundle.runs._open=gated
    start=asyncio.create_task(bundle.runs.start('alice','conversation','question'))
    await asyncio.to_thread(entered.wait,1)
    close=asyncio.create_task(bundle.runs.close())
    await asyncio.sleep(.03)
    try:assert not close.done(), 'close must retain admission ownership'
    finally:release.set()
    await asyncio.gather(start,return_exceptions=True)
    await close
    with bundle.registry._transaction() as db:
        assert db.execute("SELECT count(*) FROM runs WHERE status='active'").fetchone()[0]==0
        assert db.execute("SELECT count(*) FROM browser_runs WHERE state='starting'").fetchone()[0]==0
    assert not bundle.backend.handles


@pytest.mark.asyncio
async def test_repeated_close_cancellation_drains_stop_ack(bundle):
    import threading
    bundle.backend.complete=False
    run=await bundle.runs.start('alice','conversation','question')
    await wait_state(bundle,run,'running')
    entered=threading.Event();release=threading.Event();original=bundle.backend.stop
    def gated(handle):
        entered.set();release.wait(3);original(handle)
    bundle.backend.stop=gated
    close=asyncio.create_task(bundle.runs.close())
    await asyncio.to_thread(entered.wait,1)
    close.cancel();close.cancel()
    second=asyncio.create_task(bundle.runs.close())
    await asyncio.sleep(.03)
    try:
        assert not close.done() and not second.done()
        assert bundle.backend.handles
    finally:release.set()
    await asyncio.gather(close,second,return_exceptions=True)
    assert not bundle.backend.handles
    assert bundle.runs.snapshot('alice','conversation',run)['state']=='cancelled'


@pytest.mark.asyncio
async def test_snapshot_rechecks_owner_after_event_read(bundle):
    run=await bundle.runs.start('alice','conversation','question')
    await wait_state(bundle,run,'completed')
    original=bundle.runs.events.read
    def revoked(*args,**kwargs):
        result=original(*args,**kwargs)
        bundle.registry.is_active=lambda owner:False
        return result
    bundle.runs.events.read=revoked
    with pytest.raises(AccessDenied):bundle.runs.snapshot('alice','conversation',run)
    await bundle.runs.close()


@pytest.mark.asyncio
async def test_publication_rechecks_owner_after_external_persistence(bundle):
    def persist(*args):
        bundle.saved.append(args)
        bundle.registry.is_active=lambda owner:False
        return True
    bundle.runs.persist_answer=persist
    run=await bundle.runs.start('alice','conversation','question')
    async with asyncio.timeout(3):
        while run in bundle.runs._tasks:await asyncio.sleep(.01)
    with bundle.registry._transaction() as db:
        row=db.execute('SELECT state,answer FROM browser_runs WHERE run_id=?',(run,)).fetchone()
    assert tuple(row)==('failed','')
    assert not bundle.backend.handles
    await bundle.runs.close()


@pytest.mark.asyncio
async def test_cancel_during_external_persistence_prevents_publication(bundle):
    import threading
    entered=threading.Event();release=threading.Event()
    def persist(*args):
        entered.set();release.wait(3);bundle.saved.append(args);return True
    bundle.runs.persist_answer=persist
    run=await bundle.runs.start('alice','conversation','question')
    await asyncio.to_thread(entered.wait,1)
    cancel=asyncio.create_task(bundle.runs.cancel('alice','conversation',run))
    await asyncio.sleep(.03)
    try:assert not cancel.done()
    finally:release.set()
    await cancel
    assert bundle.runs.snapshot('alice','conversation',run)['state']=='cancelled'
    assert bundle.runs.snapshot('alice','conversation',run)['answer']==''
    await bundle.runs.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('change',['deadline','new_fence'])
async def test_persistence_drift_never_publishes_stale_answer(bundle,change):
    extra=[]
    def persist(*args):
        bundle.saved.append(args)
        if change=='deadline':
            now=bundle.registry.clock();bundle.registry.clock=lambda:now+61
        else:
            # Also proves persistence is outside the SQLite write transaction.
            extra.append(bundle.registry.start_run('alice','conversation',request_key='new-writer'))
        return True
    bundle.runs.persist_answer=persist
    run=await bundle.runs.start('alice','conversation','question')
    state=await wait_state(bundle,run,'failed')
    assert state['answer']==''
    await bundle.runs.close()
    for lease in extra:bundle.registry.cancel(lease.run_id)


@pytest.mark.asyncio
async def test_cancelled_start_drains_admission_and_stop(bundle):
    bundle.backend.complete=False
    import threading
    entered=threading.Event();release=threading.Event();original=bundle.runs._open
    def gated(*args):
        entered.set();release.wait(3);return original(*args)
    bundle.runs._open=gated
    start=asyncio.create_task(bundle.runs.start('alice','conversation','question'))
    await asyncio.to_thread(entered.wait,1);start.cancel();start.cancel()
    await asyncio.sleep(.02)
    try:assert not start.done()
    finally:release.set()
    with pytest.raises(asyncio.CancelledError):await start
    assert not bundle.backend.handles
    with bundle.registry._transaction() as db:
        assert db.execute("SELECT count(*) FROM runs WHERE status='active'").fetchone()[0]==0
        assert db.execute("SELECT state FROM browser_runs").fetchone()[0]=='cancelled'
    await bundle.runs.close()


@pytest.mark.asyncio
async def test_close_stop_failure_retains_stopping_and_reports_failed_ack(bundle):
    bundle.backend.complete=False
    run=await bundle.runs.start('alice','conversation','question')
    await wait_state(bundle,run,'running')
    bundle.backend.stop_fails=True
    with pytest.raises(AccessDenied):await bundle.runs.close()
    assert bundle.runs.snapshot('alice','conversation',run)['state']=='stopping'
    assert bundle.backend.handles
    bundle.backend.stop_fails=False
    await bundle.runs.cancel('alice','conversation',run)
    assert not bundle.backend.handles


@pytest.mark.asyncio
async def test_failed_run_metadata_distinguishes_failure(bundle):
    def failed(*args):raise RuntimeError('fixture')
    bundle.runs.prepare_input=failed
    run=await bundle.runs.start('alice','conversation','question')
    await wait_state(bundle,run,'failed')
    with bundle.registry._transaction() as db:
        assert db.execute('SELECT status FROM runs WHERE run_id=?',(run,)).fetchone()[0]=='failed'
    await bundle.runs.close()


@pytest.mark.asyncio
async def test_unpublished_admission_cleanup_works_after_owner_disabled(bundle):
    bundle.backend.complete=False
    run=await bundle.runs.start('alice','conversation','question')
    await wait_state(bundle,run,'running')
    bundle.registry.is_active=lambda owner:False
    with pytest.raises(AccessDenied):await bundle.runs.cancel('alice','conversation',run)
    assert hasattr(bundle.runs,'abandon_unpublished')
    await bundle.runs.abandon_unpublished(run)
    assert not bundle.backend.handles
    with bundle.registry._transaction() as db:
        # The drive may observe revocation before compensation requests Stop.
        assert db.execute('SELECT state FROM browser_runs WHERE run_id=?',(run,)).fetchone()[0] in ('failed','cancelled')
        assert db.execute('SELECT state FROM workers WHERE run_id=?',(run,)).fetchone()[0]=='stopped'
    with pytest.raises(AccessDenied):await bundle.runs.abandon_unpublished('unowned-id')
    await bundle.runs.close()


@pytest.mark.asyncio
async def test_close_multiple_workers_serializes_backend_stop_ack(bundle):
    import threading
    bundle.backend.complete=False
    first=await bundle.runs.start('alice','conversation','first')
    await wait_state(bundle,first,'running')
    second=await bundle.runs.start('bob','other','second')
    async with asyncio.timeout(3):
        while bundle.runs.snapshot('bob','other',second)['state']!='running':await asyncio.sleep(.01)
    entered=threading.Event();release=threading.Event();original=bundle.backend.stop
    def gated(handle):
        if not entered.is_set():entered.set();release.wait(3)
        original(handle)
    bundle.backend.stop=gated
    close=asyncio.create_task(bundle.runs.close())
    await asyncio.to_thread(entered.wait,1)
    await asyncio.sleep(.05);release.set()
    await close
    assert not bundle.backend.handles
    assert bundle.runs.snapshot('alice','conversation',first)['state']=='cancelled'
    assert bundle.runs.snapshot('bob','other',second)['state']=='cancelled'


@pytest.mark.asyncio
async def test_snapshot_does_not_pair_old_event_page_with_new_completion(bundle,monkeypatch):
    bundle.backend.complete = False
    run = await bundle.runs.start('alice','conversation','question')
    await wait_state(bundle,run,'running')
    original = bundle.runs.events.read
    def complete_after_read(*args,**kwargs):
        result = original(*args,**kwargs)
        bundle.runs._state(run,'completed','answer committed after event read')
        return result
    monkeypatch.setattr(bundle.runs.events,'read',complete_after_read)
    try:
        result = bundle.runs.snapshot('alice','conversation',run)
        assert result['state'] == 'running'
        assert result['answer'] == ''
    finally:
        monkeypatch.setattr(bundle.runs.events,'read',original)
        # Restore the simulated interleaving before normal controller cleanup.
        with bundle.registry._transaction() as db:
            db.execute("UPDATE browser_runs SET state='running',answer='' WHERE run_id=?",(run,))
        await bundle.runs.close()


@pytest.mark.asyncio
async def test_startup_recovery_stops_unowned_previous_process_run(bundle):
    bundle.backend.complete=False
    lease=bundle.runs._open('alice','conversation')
    bundle.runs.workers.start(lease.run_id)
    assert bundle.backend.handles
    await bundle.runs.recover()
    assert not bundle.backend.handles
    assert bundle.runs.snapshot('alice','conversation',lease.run_id)['state']=='cancelled'
    await bundle.runs.close()


@pytest.mark.asyncio
async def test_startup_recovery_failure_retains_stopping_state(bundle):
    bundle.backend.complete=False
    lease=bundle.runs._open('alice','conversation')
    bundle.runs.workers.start(lease.run_id)
    bundle.backend.stop_fails=True
    with pytest.raises(AccessDenied):
        await bundle.runs.recover()
    assert bundle.backend.handles
    assert bundle.runs.snapshot('alice','conversation',lease.run_id)['state']=='stopping'
    bundle.backend.stop_fails=False
    await bundle.runs.recover()
    assert not bundle.backend.handles
    await bundle.runs.close()


def test_browser_deadline_leaves_margin_inside_worker_wall_limit(bundle):
    from dataclasses import replace
    bundle.registry.clock=lambda:1000.0
    bundle.runs.workers.limits=replace(bundle.runs.workers.limits,wall_seconds=180)
    lease=bundle.runs._open('alice','conversation')
    assert lease.deadline==1175.0
    bundle.registry.cancel(lease.run_id)


# A refused model call ends its run promptly, with the reason shown and logged (#21).
OVER_BUDGET = 10**6  # The fixture budget is 1000 tokens.


class ModelService:
    """Authorizes like the real run services, then reserves `units` of budget."""
    def __init__(self, capabilities, units=1):
        self.capabilities, self.units = capabilities, units

    async def _authorize(self, token):
        return await asyncio.to_thread(self.capabilities.authorize, token, audience='inference', operation='generate')

    async def stream(self, token, request_key, body):
        lease = await self._authorize(token)
        await asyncio.to_thread(self.capabilities.registry.reserve, lease.run_id, request_key, self.units)
        yield b'data: synthetic\n\n'


def capabilities_of(bundle):
    return bundle.runs.events.capabilities


async def running_run_and_token(bundle):
    bundle.backend.complete = False
    run = await bundle.runs.start('alice', 'conversation', 'synthetic question')
    await wait_state(bundle, run, 'running')
    token = capabilities_of(bundle).issue(run, audience='inference', operations=['generate'], ttl=600).secret
    return run, token


async def call_model(bundle, token, *, units=1, body=b'{"model":"synthetic"}', on_refused=None):
    app = FastAPI()
    add_inference_routes(app, anthropic=ModelService(capabilities_of(bundle), units), token_from_request=_token,
                         on_refused=on_refused or bundle.runs.refuse_capability)
    headers = {'authorization': 'Bearer ' + token, 'content-type': 'application/json'}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://gateway') as client:
        return await client.post('/v1/messages', headers=headers, content=body)


def exhaust_budget(bundle, token, monkeypatch):
    return {'units': OVER_BUDGET}


def revoke_token(bundle, token, monkeypatch):
    capabilities_of(bundle).revoke(token)
    return {}


def fill_admission(bundle, token, monkeypatch):
    monkeypatch.setattr(inference_http, 'INFERENCE_OWNER_STREAMS', 0)
    return {}


@pytest.mark.asyncio
@pytest.mark.parametrize('refuse,status,reason', [
    (exhaust_budget, 402, 'The run stopped: Token budget exhausted'),
    (revoke_token, 403, 'The run stopped: its access to the model service was revoked'),
    (fill_admission, 429, 'The run stopped: the model service is at capacity'),
])
async def test_refused_call_fails_the_run_with_its_reason(bundle, monkeypatch, caplog, refuse, status, reason):
    caplog.set_level(logging.WARNING)
    run, token = await running_run_and_token(bundle)
    try:
        response = await call_model(bundle, token, **refuse(bundle, token, monkeypatch))
        assert response.status_code == status
        state = await wait_state(bundle, run, 'failed')
        assert state['reason'].startswith(reason)
        assert not bundle.backend.handles
        assert f'inference refused /v1/messages {status}' in caplog.text
        assert f'run {run[:8]} ended, model call refused: {reason}' in caplog.text
    finally:
        await bundle.runs.close()


@pytest.mark.asyncio
async def test_invalid_request_is_not_a_refusal(bundle):
    run, token = await running_run_and_token(bundle)
    try:
        response = await call_model(bundle, token, body=b'{"a":1,"a":2}')
        assert response.status_code == 400
        await asyncio.sleep(.1)
        state = bundle.runs.snapshot('alice', 'conversation', run)
        assert state['state'] == 'running' and state['reason'] is None
    finally:
        await bundle.runs.close()


@pytest.mark.asyncio
async def test_refusal_after_completion_leaves_the_answer(bundle):
    run, token = await running_run_and_token(bundle)
    events = capabilities_of(bundle).issue(run, audience='events', operations=['append']).secret
    bundle.runs.events.append(events, {'type': 'text', 'text': 'Synthetic answer'})
    bundle.runs.events.append(events, {'type': 'status', 'state': 'runner_completed'})
    await wait_state(bundle, run, 'completed')
    response = await call_model(bundle, token)  # A straggling call after the run's access ended.
    assert response.status_code == 403
    state = bundle.runs.snapshot('alice', 'conversation', run)
    assert state['state'] == 'completed' and state['reason'] is None
    await bundle.runs.close()


@pytest.mark.asyncio
async def test_unknown_or_malformed_token_refusal_is_ignored(bundle):
    for token in ('0' * 64, 'not-a-capability'):
        await bundle.runs.refuse_capability(token, 'The run stopped: synthetic')
    assert bundle.runs._refusals == {}


@pytest.mark.asyncio
async def test_a_failing_refusal_hook_never_changes_the_response(bundle, caplog):
    async def broken(token, reason):
        raise RuntimeError('synthetic hook failure')
    run, token = await running_run_and_token(bundle)
    try:
        response = await call_model(bundle, token, units=OVER_BUDGET, on_refused=broken)
        assert response.status_code == 402
        assert 'inference refusal not reported: RuntimeError' in caplog.text
    finally:
        await bundle.runs.close()
