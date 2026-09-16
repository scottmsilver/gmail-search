#!/usr/bin/env python3
"""Trusted outer controller: actual inner-VM CLIs via fixed synthetic gateways."""
from dataclasses import dataclass
import json
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

sys.path.insert(0, '/opt/gmail-worker')
from firecracker_backend import SyntheticAgentBackend, ROOT, JAILS


@dataclass
class Limits:
    vcpus: int = 1
    memory_mib: int = 1024
    pids: int = 128
    disk_bytes: int = 1024**3
    output_bytes: int = 2*1024**2
    wall_seconds: int = 180


backend = SyntheticAgentBackend()
handle = uuid.uuid4().hex
lease = SimpleNamespace(run_id=uuid.uuid4().hex, lease_expires=time.time()+175, deadline=time.time()+175)
relays = []
artifact_root = ROOT/'synthetic-artifacts'/handle
try:
    backend.launch(handle, lease, Limits())
    state = json.loads((ROOT/'runs'/handle/'state.json').read_text())
    print('INNER_VM_HANDLE ' + handle, flush=True)
    print('INNER_VM_ENFORCEMENT ' + json.dumps(state['evidence']), flush=True)
    for vsock, upstream in ((8000,18081), (8001,18082)):
        log = (ROOT/'runs'/handle/f'relay-{vsock}.log').open('wb')
        relays.append(subprocess.Popen(['/usr/bin/python3', '/opt/gmail-worker/vsock_http_relay.py',
            '--socket', str(JAILS/handle/'root'/f'gateway.vsock_{vsock}'), '--upstream-port', str(upstream),
            '--synthetic-artifacts', str(artifact_root)], stdout=log, stderr=subprocess.STDOUT))
        log.close()
    deadline = time.monotonic()+155
    serial = ROOT/'runs'/handle/'serial.log'
    while time.monotonic() < deadline:
        content = serial.read_text(errors='replace') if serial.exists() else ''
        if 'INNER_AGENT_SMOKE_PASS' in content.splitlines():
            break
        if 'Traceback (most recent call last)' in content or handle not in backend.inventory():
            raise RuntimeError(content[-10000:])
        time.sleep(.2)
    else:
        raise RuntimeError(content[-10000:])
    print('\n'.join(line for line in content.splitlines() if line.startswith('INNER_')), flush=True)
    uploaded = [path for path in artifact_root.iterdir() if path.is_file() and path.name != '.lock']
    assert len(uploaded) == 2, [p.name for p in uploaded]
    assert all(path.read_bytes() == b'42' for path in uploaded)
    print('PASS: actual Pi and native Claude inside jailed no-NIC inner VM ran Bash/Python; two bounded artifact byte uploads contain42', flush=True)
finally:
    for process in relays:
        process.terminate()
    for process in relays:
        process.wait(timeout=5)
    backend.stop(handle)
