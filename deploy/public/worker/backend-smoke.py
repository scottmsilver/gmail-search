#!/usr/bin/env python3
"""Run only in the synthetic outer worker; no application data or credentials."""
from dataclasses import dataclass
import json
import os
import sys
import time
from types import SimpleNamespace
import uuid

sys.path.insert(0, '/opt/gmail-worker')
from firecracker_backend import FirecrackerBackend, ROOT, CGROUP, JAILS


@dataclass
class Limits:
    vcpus: int = 1
    memory_mib: int = 256
    pids: int = 64
    disk_bytes: int = 1024**3
    output_bytes: int = 1024**2
    wall_seconds: int = 30


backend = FirecrackerBackend()
handle = uuid.uuid4().hex
lease = SimpleNamespace(run_id=uuid.uuid4().hex, lease_expires=time.time()+12, deadline=time.time()+25)
pid = os.fork()
if pid == 0:
    try:
        backend.launch(handle, lease, Limits())
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(1)
    os._exit(0)  # The launching controller vanishes; watchdog must survive it.
_, status = os.waitpid(pid, 0)
assert os.waitstatus_to_exitcode(status) == 0
state = json.loads((ROOT/'runs'/handle/'state.json').read_text())
assert state['status'] == 'running', state
print('VMM enforcement evidence:', json.dumps(state['evidence']), flush=True)
assert state['evidence']['Seccomp:'].endswith('2')
assert state['evidence']['Uid:'].split()[1:] == ['65534']*4
assert state['evidence']['memory.max'] == str(256*1024**2)
assert state['evidence']['pids.max'] == '64'
assert state['evidence']['cpu.max'] == '100000 100000'
end = time.monotonic()+20
while time.monotonic() < end and handle in backend.inventory():
    time.sleep(.1)
assert handle not in backend.inventory(), 'Watchdog did not stop orphaned worker'
text = (ROOT/'runs'/handle/'serial.log').read_text(errors='replace')
assert 'GMAIL_BACKEND_SYNTHETIC_HELLO' in text.splitlines(), text[-2000:]
assert 'eth0:' not in text and 'ens3:' not in text
assert not (CGROUP/handle).exists()
assert not (JAILS/handle).exists()
print('PASS: actual guest shell; readonly root/no NIC; cgroup CPU/memory/pids; jailer UID/seccomp; controller death followed by lease teardown', flush=True)

handle = uuid.uuid4().hex
lease = SimpleNamespace(run_id=uuid.uuid4().hex, lease_expires=time.time()+10, deadline=time.time()+25)
backend.launch(handle, lease, Limits())
lease.lease_expires = time.time()+15
backend.renew(handle, lease)
backend.stop(handle)
assert handle not in backend.inventory()
print('PASS: renewed lease then explicit cancellation tears down VMM cgroup/jail/netns', flush=True)

# Restart cleanup also handles loss of the independent supervisor itself.
handle = uuid.uuid4().hex
lease = SimpleNamespace(run_id=uuid.uuid4().hex, lease_expires=time.time()+10, deadline=time.time()+25)
backend.launch(handle, lease, Limits())
process = json.loads((ROOT/'runs'/handle/'supervisor.json').read_text())
os.kill(process['pid'], 9)
time.sleep(.2)
restarted = FirecrackerBackend()
assert handle in restarted.inventory()
restarted.stop(handle)
assert handle not in restarted.inventory()
assert not (CGROUP/handle).exists()
print('PASS: supervisor crash detected by restarted backend; full VMM cgroup removed', flush=True)
