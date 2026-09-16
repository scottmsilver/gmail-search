"""Durable worker lifecycle using a fake parser; no VM or private data."""
import importlib.util
from pathlib import Path
import sys
import time

import pytest

from gmail_search.gateway import attachment_rpc as rpc
from test_attachment_rpc import request


def load():
    sys.modules['attachment_rpc'] = rpc
    path = Path(__file__).parents[1] / 'deploy/public/worker/attachment_manager.py'
    spec = importlib.util.spec_from_file_location('attachment_manager', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Job:
    result = error = None
    def __init__(self):
        self.stops = 0
        self.renewals = []
        self.fail_stop = False
    def stop(self):
        self.stops += 1
        if self.fail_stop:
            raise RuntimeError('stop acknowledgement unavailable')
    def renew(self, seq):
        self.renewals.append(seq)
        return True


class Backend:
    def __init__(self):
        self.jobs = []
        self.reconciled = 0
        self.fail_reconcile = False
    def reconcile(self):
        self.reconciled += 1
        if self.fail_reconcile:
            raise RuntimeError('orphan still alive')
        for job in self.jobs:
            job.stop()
    def start(self, data, mime_type, options, **kwargs):
        assert data == b'x'
        assert kwargs['lease_seconds'] == 3
        job = Job()
        self.jobs.append(job)
        return job


@pytest.fixture
def setup(tmp_path):
    module = load()
    backend = Backend()
    manager = module.Manager(tmp_path / 'private', backend, controller_uid=1234)
    yield manager, backend, module
    manager.close()


def start():
    h = request()
    h['context']['deadline'] = time.time() + 40
    h['context_sha256'] = rpc.context_digest(h['context'])
    return h


def action(h, op, seq=1):
    result = {key: h[key] for key in rpc.COMMON}
    result.update(op=op, payload_size=0)
    if op == 'poll':
        result['renew_seq'] = seq
    return result


def call(manager, header, payload=b'', uid=1234):
    return manager.handle(header, payload, peer_uid=uid)[0]


def test_same_job_retry_lost_ack_and_changed_binding(setup):
    manager, backend, _ = setup
    h = start()
    assert call(manager, h, b'x')['status'] == 'running'
    assert call(manager, h, b'x')['status'] == 'running'
    assert len(backend.jobs) == 1
    h['options']['dpi'] = 120
    assert call(manager, h, b'x')['code'] == 'binding_mismatch'


def test_stop_before_start_is_durable(setup):
    manager, backend, module = setup
    h = start()
    assert call(manager, action(h, 'stop'))['status'] == 'stopped'
    directory = manager.state_dir
    manager.close()
    other = module.Manager(directory, backend, controller_uid=1234)
    try:
        assert call(other, h, b'x')['code'] == 'stopped'
        assert backend.jobs == []
    finally:
        other.close()


def test_controller_and_context_binding(setup):
    manager, backend, _ = setup
    h = start()
    assert call(manager, h, b'x', uid=999)['code'] == 'denied'
    assert backend.jobs == []
    call(manager, h, b'x')
    poll = action(h, 'poll')
    poll['context_sha256'] = '0' * 64
    assert call(manager, poll)['code'] == 'binding_mismatch'
    assert backend.jobs[0].renewals == []


def test_poll_sequence_replay_does_not_renew(setup):
    manager, backend, _ = setup
    h = start()
    call(manager, h, b'x')
    assert call(manager, action(h, 'poll', 5))['status'] == 'running'
    assert call(manager, action(h, 'poll', 5))['code'] == 'renewal_replay'
    assert call(manager, action(h, 'poll', 4))['code'] == 'renewal_replay'
    assert backend.jobs[0].renewals == [5]


def test_result_keeps_capacity_until_stop_ack(setup):
    manager, backend, _ = setup
    h = start()
    call(manager, h, b'x')
    backend.jobs[0].result = b'{"text":"synthetic"}'
    response, data = manager.handle(action(h, 'poll'), b'', peer_uid=1234)
    assert response['status'] == 'done' and data == backend.jobs[0].result
    assert call(manager, start(), b'x')['code'] == 'busy'
    backend.jobs[0].fail_stop = True
    assert call(manager, action(h, 'stop'))['code'] == 'stop_pending'
    assert call(manager, start(), b'x')['code'] == 'busy'
    backend.jobs[0].fail_stop = False
    assert call(manager, action(h, 'stop'))['status'] == 'stopped'
    assert call(manager, start(), b'x')['status'] == 'running'


def test_restart_reconciles_orphans_before_admission(setup):
    manager, backend, module = setup
    h = start()
    call(manager, h, b'x')
    directory = manager.state_dir
    manager.close()
    backend.fail_reconcile = True
    with pytest.raises(RuntimeError, match='orphan'):
        module.Manager(directory, backend, controller_uid=1234)
    backend.fail_reconcile = False
    other = module.Manager(directory, backend, controller_uid=1234)
    try:
        assert backend.jobs[0].stops >= 1
        assert call(other, h, b'x')['code'] == 'stopped'
        assert call(other, start(), b'x')['status'] == 'running'
    finally:
        other.close()


def test_expired_lease_stops_without_poll(setup):
    manager, backend, _ = setup
    h = start()
    call(manager, h, b'x')
    manager.tick(now=time.monotonic() + 4)
    assert backend.jobs[0].stops == 1
    assert call(manager, action(h, 'poll'))['code'] == 'stopped'


def test_metadata_private_and_contains_no_payload(setup):
    manager, _, _ = setup
    h = start()
    call(manager, h, b'x')
    assert manager.state_dir.stat().st_mode & 0o777 == 0o700
    assert (manager.state_dir / 'jobs.sqlite').stat().st_mode & 0o777 == 0o600
    assert b'alice' not in (manager.state_dir / 'jobs.sqlite').read_bytes()


def test_server_request_response_and_rejects_trailing_bytes(setup, tmp_path):
    import socket
    import threading
    manager, backend, module = setup
    path = str(tmp_path / 'rpc.sock')
    server = module.Server(manager, path, controller_uid=__import__('os').getuid())
    # Server credentials must match manager's trusted identity, not request data.
    manager.controller_uid = __import__('os').getuid()
    done = threading.Event()
    thread = threading.Thread(target=server.serve, args=(done,))
    thread.start()
    try:
        for suffix in (b'trailing', b''):
            with socket.socket(socket.AF_UNIX) as peer:
                peer.settimeout(2)
                peer.connect(path)
                h = start()
                peer.sendall(rpc.encode_frame(h, b'x') + suffix)
                peer.shutdown(socket.SHUT_WR)
                if suffix:
                    try:
                        assert peer.recv(1) == b''
                    except ConnectionResetError:
                        pass  # Kernel may reset a socket closed with unread junk.
                else:
                    response, payload = rpc.read_frame(peer.recv, response=True)
                    assert response['status'] == 'running'
                    assert response['request_id'] == h['request_id'] and payload == b''
        assert len(backend.jobs) == 1
    finally:
        done.set()
        thread.join(3)
        server.close()


def test_frontend_validates_and_proxies_fixed_socket(setup, tmp_path):
    import os
    import threading
    manager, _, module = setup
    path = str(tmp_path / 'rpc.sock')
    manager.controller_uid = os.getuid()
    server = module.Server(manager, path, controller_uid=os.getuid())
    done = threading.Event()
    thread = threading.Thread(target=server.serve, args=(done,))
    thread.start()
    spec = importlib.util.spec_from_file_location('attachment_rpc_frontend',
        Path(__file__).parents[1] / 'deploy/public/worker/attachment_rpc_frontend.py')
    frontend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frontend)
    source_r, source_w = os.pipe()
    sink_r, sink_w = os.pipe()
    h = start()
    os.write(source_w, rpc.encode_frame(h, b'x'))
    os.close(source_w)
    try:
        frontend.relay(source_r, sink_w, socket_path=path)
        os.close(sink_w)
        sink_w = None
        with os.fdopen(sink_r, 'rb') as stream:
            response, data = rpc.read_frame(stream.read, response=True)
            assert response['status'] == 'running' and data == b''
    finally:
        os.close(source_r)
        if sink_w is not None:
            os.close(sink_w)
        done.set()
        thread.join(3)
        server.close()


def test_idle_connection_deadline_and_peer_identity(setup, tmp_path, monkeypatch):
    import os
    import socket
    import threading
    manager, backend, module = setup
    monkeypatch.setattr(module, 'READ_SECONDS', .03)
    path = str(tmp_path / 'rpc.sock')
    server = module.Server(manager, path, controller_uid=os.getuid())
    done = threading.Event()
    thread = threading.Thread(target=server.serve, args=(done,))
    thread.start()
    try:
        with socket.socket(socket.AF_UNIX) as peer:
            peer.settimeout(1)
            peer.connect(path)
            peer.sendall(b'\x00')
            assert peer.recv(1) == b''
        server.controller_uid = os.getuid() + 10000
        with socket.socket(socket.AF_UNIX) as peer:
            peer.settimeout(1)
            peer.connect(path)
            assert peer.recv(1) == b''
        assert backend.jobs == []
    finally:
        done.set()
        thread.join(3)
        server.close()


def test_frontend_output_timeout_cannot_block_on_full_pipe(setup, tmp_path, monkeypatch):
    import os
    import threading
    manager, backend, module = setup
    path = str(tmp_path / 'rpc.sock')
    manager.controller_uid = os.getuid()
    backend_start = backend.start
    def start_large(*args, **kwargs):
        job = backend_start(*args, **kwargs)
        job.result = b'z' * 100000
        return job
    backend.start = start_large
    server = module.Server(manager, path, controller_uid=os.getuid())
    done = threading.Event()
    server_thread = threading.Thread(target=server.serve, args=(done,))
    server_thread.start()
    spec = importlib.util.spec_from_file_location('attachment_rpc_frontend',
        Path(__file__).parents[1] / 'deploy/public/worker/attachment_rpc_frontend.py')
    frontend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frontend)
    monkeypatch.setattr(frontend, 'READ_SECONDS', .05)
    source_r, source_w = os.pipe()
    sink_r, sink_w = os.pipe()
    __import__('fcntl').fcntl(sink_w, __import__('fcntl').F_SETPIPE_SZ, 4096)
    os.write(source_w, rpc.encode_frame(start(), b'x'))
    os.close(source_w)
    errors = []
    def run():
        try:
            frontend.relay(source_r, sink_w, socket_path=path)
        except Exception as error:
            errors.append(error)
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        thread.join(.5)
        assert not thread.is_alive(), 'frontend blocked past absolute output deadline'
        assert len(errors) == 1 and isinstance(errors[0], TimeoutError)
    finally:
        os.close(sink_r)  # Unblock a regressed blocking writer before test cleanup.
        thread.join(1)
        os.close(sink_w)
        os.close(source_r)
        done.set()
        server_thread.join(3)
        server.close()


def test_metadata_limit_never_evicts_stop_tombstones(setup, monkeypatch):
    manager, backend, module = setup
    monkeypatch.setattr(module, 'MAX_RECORDS', 1)
    stopped = start()
    assert call(manager, action(stopped, 'stop'))['status'] == 'stopped'
    assert call(manager, action(start(), 'stop'))['code'] == 'metadata_full'
    assert call(manager, start(), b'x')['code'] == 'metadata_full'
    assert call(manager, stopped, b'x')['code'] == 'stopped'
    assert backend.jobs == []


def test_no_concurrent_manager_or_unsafe_state_directory(setup, tmp_path):
    manager, backend, module = setup
    with pytest.raises(BlockingIOError):
        module.Manager(manager.state_dir, backend, controller_uid=1234)
    unsafe = tmp_path / 'unsafe'
    unsafe.mkdir()
    # chmod, not mkdir(mode=...): mkdir's mode is masked by the ambient umask,
    # so under `umask 0077` this directory came out 0700 — safe — and the test
    # failed for the environment rather than for the code.
    unsafe.chmod(0o755)
    with pytest.raises(RuntimeError, match='Unsafe'):
        module.Manager(unsafe, backend, controller_uid=1234)
    symlink = tmp_path / 'alias'
    symlink.symlink_to(manager.state_dir)
    with pytest.raises(RuntimeError, match='Unsafe'):
        module.Manager(symlink, backend, controller_uid=1234)


def test_pending_launch_cannot_extend_lease(setup):
    manager, backend, _ = setup
    h = start()
    call(manager, h, b'x')
    backend.jobs[0].renew = lambda seq: False
    initial_expiry = manager.jobs[h['job_id']]['lease']
    assert call(manager, action(h, 'poll'))['status'] == 'running'
    assert manager.jobs[h['job_id']]['lease'] == initial_expiry
    manager.tick(now=initial_expiry + .01)
    assert backend.jobs[0].stops == 1


def test_persisting_binding_does_not_extend_context_deadline(setup):
    manager, backend, _ = setup
    h = start()
    original_start = backend.start
    deadlines = []
    def record_deadline(*args, **kwargs):
        deadlines.append(kwargs['deadline'])
        return original_start(*args, **kwargs)
    backend.start = record_deadline
    assert call(manager, h, b'x')['status'] == 'running'
    assert deadlines[0] <= h['context']['deadline']
