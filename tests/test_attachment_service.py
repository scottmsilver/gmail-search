"""Capability-bound attachment jobs must stop and reap before returning."""
import asyncio
import json
import threading

import pytest

from gmail_search.gateway.attachment_sandbox import AttachmentDenied, AttachmentInput
from gmail_search.gateway.attachment_service import RunAttachmentService
from gmail_search.gateway.capabilities import Capabilities
from gmail_search.gateway.registry import AccessDenied, Registry


RAW = json.dumps({'text': 'synthetic', 'pages': [], 'truncated': False}).encode()


@pytest.mark.asyncio
async def test_cancelled_cleanup_task_does_not_busy_loop():
    cleanup = asyncio.create_task(asyncio.sleep(0))
    cleanup.cancel()
    await asyncio.gather(cleanup, return_exceptions=True)
    with pytest.raises(asyncio.CancelledError):
        await RunAttachmentService._drain(cleanup)


@pytest.mark.asyncio
async def test_admission_precedes_loading_and_is_released_after_cleanup(tmp_path):
    from gmail_search.gateway.attachment_sandbox import AttachmentDenied
    registry = Registry(tmp_path / 'capacity.sqlite', is_active=lambda owner: True)
    capabilities = Capabilities(registry)
    tokens = []
    for owner in ('alice', 'bob'):
        run = registry.start_run(owner, 'conversation', request_key=owner, writer=False)
        tokens.append(capabilities.issue(run.run_id, audience='attachment', operations={'parse'}).secret)
    entered = asyncio.Event()
    loads = []
    async def load(owner, aid):
        loads.append((owner, aid))
        entered.set()
        await asyncio.Event().wait()
    api = RunAttachmentService(capabilities, Backend(), load=load, max_jobs=1, max_owner_jobs=1)
    task = asyncio.create_task(api.parse(tokens[0], 1))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        for token in tokens:
            with pytest.raises(AttachmentDenied):
                await api.parse(token, 2)
        assert loads == [('alice', 1)]
        assert api.active_jobs == {'alice': 1}
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert api.active_jobs == {}


@pytest.mark.asyncio
async def test_failed_teardown_does_not_release_controller_admission(setup):
    class FailedStopBackend:
        def start(self, data, mime, options):
            class FailedJob:
                def wait(self):
                    return RAW
                def stop(self):
                    raise RuntimeError('Synthetic teardown unavailable')
            return FailedJob()
    async def load(owner, aid):
        return AttachmentInput(owner, aid, 'application/pdf', b'synthetic')
    api = service(setup, FailedStopBackend(), load)
    with pytest.raises(RuntimeError, match='teardown'):
        await api.parse(setup[2], 1)
    assert api.active_jobs == {'alice': 1}
    with pytest.raises(AttachmentDenied):
        await api.parse(setup[2], 2)


@pytest.mark.asyncio
async def test_remote_backend_receives_only_derived_context_and_host_authorizer(setup):
    from gmail_search.gateway.attachment_remote import SSHAttachmentBackend
    from test_attachment_remote_jobs import Transport
    transport = Transport()
    backend = SSHAttachmentBackend(setup[0], transport=transport, poll_interval=.01)
    async def load(owner, aid):
        return AttachmentInput(owner, aid, 'application/pdf', b'synthetic')
    api = service(setup, backend, load)
    result = await api.parse(setup[2], 12)
    assert result.text == 'synthetic'
    start = transport.calls[0]
    assert start['context']['owner_id'] == 'alice' and start['context']['attachment_id'] == 12
    assert setup[2] not in json.dumps(transport.calls)
    assert transport.calls[-1]['op'] == 'stop'
    assert api.active_jobs == {}


class Job:
    def __init__(self, backend):
        self.backend = backend
        self.waiting = threading.Event()
        self.finished = threading.Event()
        self.stop_entered = threading.Event()
        self.allow_teardown = threading.Event()
        self.allow_teardown.set()

    def wait(self):
        self.waiting.set()
        if not self.finished.wait(3):
            raise TimeoutError('synthetic job did not stop')
        return RAW

    def stop(self):
        self.stop_entered.set()
        self.finished.set()
        if not self.allow_teardown.wait(3):
            raise TimeoutError('synthetic teardown did not complete')
        self.backend.running.remove(self)


class Backend:
    def __init__(self):
        self.jobs = []
        self.running = set()

    def start(self, data, mime_type, options):
        if self.running:
            raise RuntimeError('capacity')
        job = Job(self)
        self.jobs.append(job)
        self.running.add(job)
        return job


@pytest.fixture
def setup(tmp_path):
    tmp_path.chmod(0o700)
    registry = Registry(tmp_path/'attachment.sqlite', is_active=lambda owner: owner == 'alice')
    run = registry.start_run('alice', 'conversation', request_key='attachment', writer=False)
    capabilities = Capabilities(registry)
    token = capabilities.issue(run.run_id, audience='attachment', operations={'parse'}).secret
    return registry, capabilities, token


def service(setup, backend, loader):
    return RunAttachmentService(setup[1], backend, load=loader, watch_interval=.01)


@pytest.mark.asyncio
async def test_revocation_stops_and_reaps_running_job_before_suppressing_result(setup):
    backend = Backend()

    async def load(owner, attachment):
        assert (owner, attachment) == ('alice', 12)
        return AttachmentInput(owner, attachment, 'application/pdf', b'%PDF-synthetic')

    api = service(setup, backend, load)
    task = asyncio.create_task(api.parse(setup[2], 12))
    while not backend.jobs or not backend.jobs[0].waiting.is_set():
        await asyncio.sleep(.001)
    setup[1].revoke(setup[2])
    while not backend.jobs[0].stop_entered.is_set():
        await asyncio.sleep(.001)
    with pytest.raises(AccessDenied):
        await asyncio.wait_for(task, 1)
    assert not backend.running


@pytest.mark.asyncio
async def test_repeated_cancellation_during_stop_still_drains_teardown(setup):
    backend = Backend()

    async def load(owner, attachment):
        return AttachmentInput(owner, attachment, 'application/pdf', b'%PDF-synthetic')

    api = service(setup, backend, load)
    task = asyncio.create_task(api.parse(setup[2], 12))
    while not backend.jobs or not backend.jobs[0].waiting.is_set():
        await asyncio.sleep(.001)
    job = backend.jobs[0]
    job.allow_teardown.clear()
    task.cancel()
    while not job.stop_entered.is_set():
        await asyncio.sleep(.001)
    task.cancel()
    await asyncio.sleep(.01)
    assert job in backend.running
    job.allow_teardown.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 1)
    assert not backend.running


@pytest.mark.asyncio
async def test_revocation_during_successful_job_teardown_suppresses_result(setup):
    backend = Backend()

    async def load(owner, attachment):
        return AttachmentInput(owner, attachment, 'application/pdf', b'%PDF-synthetic')

    api = service(setup, backend, load)
    task = asyncio.create_task(api.parse(setup[2], 12))
    while not backend.jobs or not backend.jobs[0].waiting.is_set():
        await asyncio.sleep(.001)
    job = backend.jobs[0]
    job.allow_teardown.clear()
    job.finished.set()
    while not job.stop_entered.is_set():
        await asyncio.sleep(.001)
    setup[1].revoke(setup[2])
    job.allow_teardown.set()
    with pytest.raises(AccessDenied):
        await asyncio.wait_for(task, 1)
    assert not backend.running


@pytest.mark.asyncio
async def test_caller_cancellation_waits_for_teardown_before_capacity_is_released(setup):
    backend = Backend()

    async def load(owner, attachment):
        return AttachmentInput(owner, attachment, 'application/pdf', b'%PDF-synthetic')

    api = service(setup, backend, load)
    task = asyncio.create_task(api.parse(setup[2], 12))
    while not backend.jobs or not backend.jobs[0].waiting.is_set():
        await asyncio.sleep(.001)
    job = backend.jobs[0]
    job.allow_teardown.clear()
    task.cancel()
    while not job.stop_entered.is_set():
        await asyncio.sleep(.001)
    assert job in backend.running
    with pytest.raises(AttachmentDenied):
        await api.parse(setup[2], 13)
    job.allow_teardown.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 1)
    assert not backend.running


@pytest.mark.asyncio
async def test_revocation_during_async_load_prevents_backend_launch(setup):
    backend = Backend()
    entered = asyncio.Event()

    async def load(owner, attachment):
        entered.set()
        await asyncio.Event().wait()

    api = service(setup, backend, load)
    task = asyncio.create_task(api.parse(setup[2], 12))
    await entered.wait()
    setup[1].revoke(setup[2])
    with pytest.raises(AccessDenied):
        await asyncio.wait_for(task, 1)
    assert not backend.jobs


@pytest.mark.asyncio
async def test_malformed_request_is_denied_before_loader_or_backend_and_owner_is_not_a_parameter(setup):
    backend = Backend()
    loaded = False

    async def load(owner, attachment):
        nonlocal loaded
        loaded = True
        return AttachmentInput(owner, attachment, 'application/pdf', b'x')

    api = service(setup, backend, load)
    with pytest.raises(AttachmentDenied):
        await api.parse(setup[2], 0)
    with pytest.raises(TypeError):
        await api.parse(setup[2], 12, owner_id='bob')
    assert not loaded and not backend.jobs
