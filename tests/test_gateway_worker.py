"""Controller lifecycle tests with a synthetic backend; never launches a VM."""
import pytest

from gmail_search.gateway.registry import AccessDenied, Registry
from gmail_search.gateway.capabilities import Capabilities
from gmail_search.gateway.worker import WorkerController, WorkerLimits


class Backend:
    namespace = 'synthetic-worker'

    def __init__(self):
        self.guests = {}
        self.stopped = []
        self.launches = 0
        self.fail_stop = False
        self.crash_launch = False

    def launch(self, handle, lease, limits):
        self.launches += 1
        self.guests[handle] = (lease, limits)
        if self.crash_launch:
            raise KeyboardInterrupt('synthetic controller crash after launch')

    def renew(self, handle, lease):
        old, limits = self.guests[handle]
        self.guests[handle] = (lease, limits)

    def stop(self, handle):
        if self.fail_stop:
            raise RuntimeError('synthetic stop failure')
        self.guests.pop(handle, None)
        self.stopped.append(handle)

    def inventory(self):
        return set(self.guests)


@pytest.fixture
def worker(tmp_path):
    now = [100.0]
    active = {'alice', 'bob'}
    registry = Registry(tmp_path/'runs.sqlite', is_active=lambda owner: owner in active, clock=lambda: now[0])
    backend = Backend()
    controller = WorkerController(registry, backend, max_workers=2, max_owner_workers=1)
    return controller, registry, backend, now, active


def run(registry, owner='alice', conversation='conversation'):
    return registry.start_run(owner, conversation, request_key='request')


def test_launch_is_idempotent_and_binds_fixed_limits(worker):
    controller, registry, backend, _, _ = worker
    lease = run(registry)
    handle = controller.start(lease.run_id)
    assert controller.start(lease.run_id) == handle
    assert backend.launches == 1
    assert len(handle) == 32
    assert backend.guests[handle] == (lease, WorkerLimits())
    with pytest.raises(TypeError):
        controller.start(lease.run_id, image='/guest/chosen')


def test_worker_quotas_bound_distinct_runs(worker):
    controller, registry, backend, _, _ = worker
    controller.start(run(registry).run_id)
    with pytest.raises(AccessDenied):
        controller.start(run(registry, conversation='second').run_id)
    controller.start(run(registry, owner='bob').run_id)
    assert len(backend.guests) == 2


@pytest.mark.parametrize('change', ['cancel', 'expiry', 'revoke', 'fence', 'worker_loss'])
def test_restart_reconciles_invalid_and_missing_workers(worker, change):
    controller, registry, backend, now, active = worker
    lease = run(registry)
    handle = controller.start(lease.run_id)
    caps = Capabilities(registry)
    secret = caps.issue(lease.run_id, audience='mail', operations={'search'}).secret
    if change == 'cancel':
        registry.cancel(lease.run_id)
    elif change == 'expiry':
        now[0] += 61
    elif change == 'revoke':
        active.remove('alice')
    elif change == 'fence':
        registry.finish(lease.run_id, status='failed')
        registry.start_run('alice', 'conversation', request_key='new')
    else:
        backend.guests.pop(handle)
    restarted = WorkerController(registry, backend, max_workers=2, max_owner_workers=1)
    restarted.reconcile()
    assert handle not in backend.guests
    with pytest.raises(AccessDenied):
        caps.authorize(secret, audience='mail', operation='search')
    assert restarted.reconcile() == 0


def test_crash_after_launch_before_binding_completion_is_torn_down(worker):
    controller, registry, backend, _, _ = worker
    lease = run(registry)
    backend.crash_launch = True
    with pytest.raises(KeyboardInterrupt):
        controller.start(lease.run_id)
    assert len(backend.guests) == 1
    backend.crash_launch = False
    restarted = WorkerController(registry, backend, max_workers=2, max_owner_workers=1)
    assert restarted.reconcile() == 1
    assert not backend.guests
    with pytest.raises(AccessDenied):
        restarted.start(lease.run_id)


def test_failed_teardown_revokes_first_and_retries_on_restart(worker):
    controller, registry, backend, _, _ = worker
    lease = run(registry)
    handle = controller.start(lease.run_id)
    caps = Capabilities(registry)
    secret = caps.issue(lease.run_id, audience='mail', operations={'search'}).secret
    backend.fail_stop = True
    with pytest.raises(AccessDenied):
        controller.cancel(lease.run_id)
    assert handle in backend.guests
    with pytest.raises(AccessDenied):
        caps.authorize(secret, audience='mail', operation='search')
    backend.fail_stop = False
    restarted = WorkerController(registry, backend, max_workers=2, max_owner_workers=1)
    assert restarted.reconcile() == 1
    assert handle not in backend.guests


def test_orphan_backend_handle_is_removed_after_controller_restart(worker):
    controller, registry, backend, _, _ = worker
    orphan = 'f' * 32
    backend.guests[orphan] = None
    assert controller.reconcile() == 1
    assert orphan in backend.stopped


def test_heartbeat_requires_live_guest_and_renews_watchdog(worker):
    controller, registry, backend, now, _ = worker
    lease = run(registry)
    handle = controller.start(lease.run_id)
    now[0] += 10
    refreshed = controller.heartbeat(lease.run_id)
    assert backend.guests[handle][0].lease_expires == refreshed.lease_expires == 170
    backend.guests.pop(handle)
    with pytest.raises(AccessDenied):
        controller.heartbeat(lease.run_id)


def test_limits_are_positive_and_bounded():
    for changes in ({'vcpus': 0}, {'memory_mib': True}, {'pids': 10**9}, {'disk_bytes': -1}):
        with pytest.raises(ValueError):
            WorkerLimits(**changes)


def test_cancellation_revokes_even_while_launch_holds_controller_lock(worker):
    controller, registry, backend, _, _ = worker
    lease = run(registry)
    original_launch = backend.launch
    def racing_launch(handle, launch_lease, limits):
        original_launch(handle, launch_lease, limits)
        # Teardown may be busy, but revocation must happen immediately.
        with pytest.raises(AccessDenied):
            controller.cancel(lease.run_id)
    backend.launch = racing_launch
    with pytest.raises(AccessDenied):
        controller.start(lease.run_id)
    assert not backend.guests


def test_cleanup_continues_past_one_failed_guest(worker):
    controller, registry, backend, _, _ = worker
    first = run(registry)
    second = run(registry, owner='bob')
    handle1 = controller.start(first.run_id)
    handle2 = controller.start(second.run_id)
    registry.cancel(first.run_id)
    registry.cancel(second.run_id)
    original_stop = backend.stop
    def stop(handle):
        if handle == handle1:
            raise RuntimeError('synthetic failure')
        original_stop(handle)
    backend.stop = stop
    with pytest.raises(AccessDenied):
        controller.reconcile()
    assert handle1 in backend.guests
    assert handle2 not in backend.guests
