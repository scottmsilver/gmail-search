"""Real parser transport sockets with a synthetic VM lifecycle double."""
import importlib.util
from pathlib import Path
import socket
import struct
import sys
import tempfile
import threading
import time

import pytest


def load(name):
    path = Path(__file__).parents[1] / 'deploy/public/worker' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def backend(tmp_path, monkeypatch):
    load('firecracker_backend')
    module = load('attachment_backend')
    jail_temp = tempfile.TemporaryDirectory(prefix='att-jails-')
    jails = Path(jail_temp.name)

    class VM:
        def __init__(self):
            self.entered = threading.Event()
            self.allow_launch = threading.Event()
            self.allow_launch.set()
            self.stopped = threading.Event()
            self.fail_stop = False
            self.stop_calls = 0
            self.orphans = set()
            self.renewals = []

        def inventory(self):
            return self.orphans

        def launch(self, handle, *_):
            self.entered.set()
            assert self.allow_launch.wait(3), 'test launch was never released'
            if self.stopped.is_set():
                raise RuntimeError('cancelled launch')
            (jails / handle / 'root').mkdir(parents=True)

        def stop(self, handle):
            self.stop_calls += 1
            self.stopped.set()
            self.allow_launch.set()
            if self.fail_stop:
                raise RuntimeError('teardown acknowledgement unavailable')
            self.orphans.discard(handle)

        def renew(self, handle, lease):
            self.renewals.append((handle, lease))

    vm = VM()
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'JAILS', jails)
    monkeypatch.setattr(module, 'boundary', lambda: None)
    monkeypatch.setattr(module, 'SyntheticAttachmentBackend', lambda: vm)
    monkeypatch.setattr(module.os, 'chown', lambda *_: None)
    controller = module.AttachmentFirecrackerBackend()
    jobs = []

    def start():
        job = controller.start(b'synthetic', 'application/pdf', {'dpi': 100, 'pages': []})
        jobs.append(job)
        return job

    try:
        yield controller, vm, start, jails
    finally:
        vm.fail_stop = False
        vm.allow_launch.set()
        for job in jobs:
            job.stop()
        jail_temp.cleanup()


def peer_for(job, root):
    peer = socket.socket(socket.AF_UNIX)
    peer.settimeout(2)
    until = time.monotonic() + 2
    while True:
        try:
            peer.connect(str(root / job.handle / 'root/gateway.vsock_8002'))
            break
        except (FileNotFoundError, ConnectionRefusedError):
            if time.monotonic() >= until:
                peer.close()
                raise
            time.sleep(.005)
    size = struct.unpack('!I', exact(peer, 4))[0]
    exact(peer, size + len(b'synthetic'))
    return peer


def exact(peer, size):
    data = b''
    while len(data) < size:
        chunk = peer.recv(size - len(data))
        assert chunk, 'unexpected transport EOF'
        data += chunk
    return data


def test_start_returns_before_launch_and_stop_interrupts_launch(backend):
    _, vm, start, _ = backend
    vm.allow_launch.clear()
    before = time.monotonic()
    job = start()
    assert time.monotonic() - before < .5
    assert vm.entered.wait(1)
    job.stop()
    assert not job.thread.is_alive()
    job.stop()
    assert vm.stop_calls == 1
    with pytest.raises(Exception):
        job.wait()


@pytest.mark.parametrize('connected', [False, True])
def test_stop_interrupts_listener_and_partial_response(backend, connected):
    _, vm, start, root = backend
    job = start()
    peer = peer_for(job, root) if connected else None
    try:
        if peer:
            peer.sendall(b'\x00\x00')  # partial length frame keeps recv blocked
        before = time.monotonic()
        job.stop()
        assert time.monotonic() - before < 1
        assert not job.thread.is_alive()
        assert vm.stop_calls == 1
    finally:
        if peer:
            peer.close()


def test_stop_failure_retains_capacity_until_successful_retry(backend):
    controller, vm, start, root = backend
    job = start()
    with peer_for(job, root):
        vm.fail_stop = True
        with pytest.raises(RuntimeError, match='acknowledgement'):
            job.stop()
        assert controller.pending_jobs() == (job,)
        with pytest.raises(BlockingIOError):
            start()
        vm.fail_stop = False
        job.stop()
        job.stop()
        assert controller.pending_jobs() == ()
        assert vm.stop_calls == 2
        another = start()
        another.stop()


def test_wait_returns_bounded_response_but_holds_capacity_until_stop(backend):
    _, vm, start, root = backend
    job = start()
    with peer_for(job, root) as peer:
        peer.sendall(struct.pack('!I', 2) + b'{}')
        assert job.wait() == b'{}'
        with pytest.raises(BlockingIOError):
            start()
        job.stop()
        assert vm.stop_calls == 1


def test_old_worker_inventory_blocks_new_capacity_after_controller_restart(backend):
    _, vm, start, _ = backend
    vm.orphans.add('a' * 32)
    with pytest.raises(RuntimeError, match='reconciliation'):
        start()
    assert not vm.entered.is_set()


def test_short_lease_renewal_is_monotonic_and_stops_after_cancel(backend):
    _, vm, start, root = backend
    job = start()
    with peer_for(job, root):
        assert job.renew(1) is True
        lease = vm.renewals[-1][1]
        assert 0 < lease.lease_expires - time.time() <= 3
        assert lease.deadline == job.deadline
        with pytest.raises(ValueError):
            job.renew(1)
        with pytest.raises(ValueError):
            job.renew(True)
        job.stop()
        with pytest.raises(RuntimeError):
            job.renew(2)


def test_pending_launch_does_not_extend_lease(backend):
    _, vm, start, _ = backend
    vm.allow_launch.clear()
    job = start()
    assert vm.entered.wait(1)
    assert job.renew(1) is False
    assert vm.renewals == []
    job.stop()


def test_reconcile_clears_orphans_before_admission(backend):
    controller, vm, _, _ = backend
    vm.orphans.add('a' * 32)
    controller.reconcile()
    assert vm.inventory() == set()
