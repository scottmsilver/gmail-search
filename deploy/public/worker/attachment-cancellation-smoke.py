#!/usr/bin/env python3
"""Actual nested parser VM cancellation; synthetic input, no external services."""
import json
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, '/opt/gmail-worker')
import attachment_backend
from firecracker_backend import SyntheticAttachmentBackend, ROOT, process_start, boundary

boundary()
actual = SyntheticAttachmentBackend()
assert not actual.inventory(), 'Synthetic worker must start empty'
running = threading.Event()
release = threading.Event()


class PausedLaunch:
    """Hold only the controller after the actual VMM is running, before input."""
    def inventory(self):
        return actual.inventory()

    def launch(self, handle, lease, limits):
        actual.launch(handle, lease, limits)
        running.set()
        if not release.wait(20):
            raise TimeoutError('Synthetic launch pause timed out')

    def stop(self, handle):
        try:
            actual.stop(handle)
        finally:
            release.set()


attachment_backend.SyntheticAttachmentBackend = PausedLaunch
controller = attachment_backend.AttachmentFirecrackerBackend()
data = (Path('/home/worker/attachment-fixtures') / 'hello.pdf').read_bytes()
begin = time.monotonic()
job = controller.start(data, 'application/pdf', {'dpi': 100, 'pages': []})
start_seconds = time.monotonic() - begin
assert start_seconds < 1, 'start blocked on VM launch'
try:
    assert running.wait(18), 'actual parser VMM failed to reach running'
    state = json.loads((ROOT / 'runs' / job.handle / 'state.json').read_text())
    assert job.handle in actual.inventory()
    before = time.monotonic()
    job.stop()
    stop_seconds = time.monotonic() - before
    job.stop()  # Acknowledged teardown is idempotent.
    assert not job.thread.is_alive()
    assert not controller.pending_jobs()
    assert job.handle not in actual.inventory()
    assert process_start(state['pid']) is None, 'VMM process survived cancellation'
    try:
        job.wait()
    except Exception:
        pass
    else:
        raise AssertionError('Cancelled job returned output')
    print(json.dumps({'status': 'passed', 'actual_vm_cancelled_during_launch': True,
                      'start_seconds': start_seconds, 'stop_seconds': stop_seconds,
                      'worker_joined': True, 'vmm_reaped': True,
                      'evidence': state['evidence']}, indent=2))
finally:
    release.set()
    job.stop()
