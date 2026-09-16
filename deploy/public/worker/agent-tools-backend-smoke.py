#!/usr/bin/env python3
"""Trusted synthetic outer controller; config arrives only on bounded stdin."""
from dataclasses import dataclass
import json
import os
import select
import socket
import struct
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid
sys.path.insert(0,'/opt/gmail-worker')
from firecracker_backend import SyntheticAgentToolsBackend,ROOT,JAILS,boundary
from guest_run_bootstrap import read_config


@dataclass
class Limits:
    vcpus:int=1
    memory_mib:int=1024
    pids:int=128
    disk_bytes:int=1024**3
    output_bytes:int=2*1024**2
    wall_seconds:int=130


def _cleanup(backend, handle, relay):
    try:
        if relay is not None:
            try:
                relay.terminate()
            except ProcessLookupError:
                pass
            try:
                relay.wait(timeout=5)
            except subprocess.TimeoutExpired:
                relay.kill()
                relay.wait(timeout=5)
    finally:
        # Relay failure must never skip stopping/reaping the actual VM.
        backend.stop(handle)
        if backend.inventory():
            raise RuntimeError('Worker inventory not empty after stop')
        print('INNER_VM_STOP_ACK', flush=True)


def main():
    boundary()
    end=time.monotonic()+5
    def read(size):
        remaining=end-time.monotonic()
        if remaining<=0 or not select.select([0],[],[],remaining)[0]:raise TimeoutError('Bootstrap input deadline')
        return os.read(0,size)
    config=read_config(read)
    raw=json.dumps(config,separators=(',',':')).encode()
    assert len(raw)<=4096
    backend=SyntheticAgentToolsBackend()
    if backend.inventory():raise RuntimeError('Existing worker jobs require reconciliation')
    handle=uuid.uuid4().hex
    lease=SimpleNamespace(run_id=uuid.uuid4().hex,lease_expires=time.time()+125,deadline=time.time()+125)
    relay=None
    try:
        backend.launch(handle,lease,Limits())
        state=json.loads((ROOT/'runs'/handle/'state.json').read_text())
        print('INNER_VM_ENFORCEMENT '+json.dumps(state['evidence']),flush=True)
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as server:
            path=JAILS/handle/'root/gateway.vsock_8002'
            server.bind(str(path));os.chown(path,65534,65534);os.chmod(path,0o600)
            server.listen(1);server.settimeout(10)
            log=(ROOT/'runs'/handle/'tools-relay.log').open('wb')
            relay=subprocess.Popen(['/usr/bin/python3','/opt/gmail-worker/vsock_http_relay.py',
                '--socket',str(JAILS/handle/'root/gateway.vsock_8000'),'--upstream-port','18081'],
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            log.close()
            peer,_=server.accept()
            with peer:
                peer.settimeout(5)
                peer.sendall(struct.pack('!I',len(raw))+raw)
                peer.shutdown(socket.SHUT_WR)
            del raw,config
        # Bootstrap listener is closed permanently before the CLI workflow.
        deadline=time.monotonic()+105
        serial=ROOT/'runs'/handle/'serial.log'
        while time.monotonic()<deadline:
            content=serial.read_text(errors='replace') if serial.exists() else ''
            if 'INNER_MAIL_TOOLS_PASS' in content.splitlines():break
            if 'INNER_MAIL_TOOLS_FAILED' in content.splitlines() or handle not in backend.inventory():
                raise RuntimeError('Synthetic guest workflow failed')
            time.sleep(.2)
        else:raise TimeoutError('Synthetic guest workflow deadline')
        print('\n'.join(line for line in content.splitlines() if line.startswith('INNER_MAIL_TOOLS')),flush=True)
    finally:
        _cleanup(backend,handle,relay)


if __name__=='__main__':main()
