#!/usr/bin/env python3
"""Explicit enrolled production VM profile; never an environment guard bypass."""
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import sys

if __name__ == '__main__':
    sys.path.insert(0, '/opt/gmail-worker')
import firecracker_backend as backend

CONFIG = Path('/etc/gmail-worker/production.json')
HOSTNAME = 'gmail-execution-worker'
MODULE = Path('/opt/gmail-worker/production_worker_profile.py')
ASSETS = backend.PINS | {'agent-full.squashfs': backend.AGENT_FULL_PIN}


def validate_enrollment(data, *, hostname, machine_id):
    if (type(data) is not dict or set(data) != {'version', 'purpose', 'hostname', 'machine_id'}
            or type(data['version']) is not int or data['version'] != 1
            or data['purpose'] != 'gmail-full-agent-production'
            or data['hostname'] != HOSTNAME or hostname != HOSTNAME
            or type(data['machine_id']) is not str
            or not re.fullmatch('[a-f0-9]{32}', data['machine_id'])
            or data['machine_id'] == '0' * 32 or data['machine_id'] != machine_id):
        raise RuntimeError('Worker is not the enrolled dedicated production VM')


def secure_directory(path):
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
        raise RuntimeError('Unsafe production worker directory')


def open_owned(path, *, owner=0):
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        raise RuntimeError('Unsafe production worker file') from exc
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != owner or info.st_mode & 0o022:
        os.close(fd)
        raise RuntimeError('Unsafe production worker file')
    return os.fdopen(fd, 'rb')


def verify_asset(path, expected, *, owner=0):
    with open_owned(path, owner=owner) as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
            raise RuntimeError('Pinned production worker asset mismatch')


def boundary():
    if os.geteuid() != 0 or socket.gethostname() != HOSTNAME:
        raise RuntimeError('Production worker requires root on the enrolled dedicated VM')
    for directory in (Path('/etc'), CONFIG.parent, Path('/opt'), MODULE.parent,
                      Path('/var/lib'), backend.ROOT, backend.ROOT/'images'):
        secure_directory(directory)
    with open_owned(CONFIG) as stream:
        raw = stream.read(4097)
    if len(raw) > 4096:
        raise RuntimeError('Oversized production enrollment')
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError('Duplicate enrollment field')
            result[key] = value
        return result
    validate_enrollment(json.loads(raw, object_pairs_hook=unique_pairs),
                        hostname=socket.gethostname(), machine_id=Path('/etc/machine-id').read_text().strip())
    if not Path('/sys/fs/cgroup/cgroup.controllers').is_file():
        raise RuntimeError('Production worker requires cgroup v2')
    if not stat.S_ISCHR(Path('/dev/kvm').stat().st_mode):
        raise RuntimeError('Production worker requires KVM')
    for program in MODULE.parent.glob('*.py'):
        with open_owned(program):
            pass


def prepare_images():
    """Verify preinstalled assets; production never copies from a home directory."""
    boundary()
    for name, pin in ASSETS.items():
        verify_asset(backend.ROOT/'images'/name, pin)
    for name in ('firecracker', 'jailer'):
        verify_asset(Path('/usr/local/bin')/name, ASSETS[name])
    Path('/sys/fs/cgroup/cgroup.subtree_control').write_text('+cpu +memory +pids')
    backend.CGROUP.mkdir(exist_ok=True)
    (backend.CGROUP/'cgroup.subtree_control').write_text('+cpu +memory +pids')


class ProductionFullAgentBackend(backend.FirecrackerBackend):
    namespace = 'production-firecracker'
    profile = 'agent_full'
    supervisor_module = MODULE

    def check_boundary(self):
        boundary()


def main():
    if sys.argv[1:] == ['--check']:
        prepare_images()
        print('Production worker enrollment and pinned assets verified')
    elif sys.argv[1:] == ['--reap']:
        worker = ProductionFullAgentBackend()
        for handle in worker.inventory():
            worker.stop(handle)
    elif len(sys.argv) == 3 and sys.argv[1] == '_supervise':
        boundary()
        handle = backend.checked_handle(sys.argv[2])
        lease = backend.read_json(backend.ROOT/'runs'/handle/'lease.json')
        if lease.get('profile') != 'agent_full':
            raise RuntimeError('Production supervisor only accepts the fixed full-agent profile')
        backend.supervise(handle, boundary_check=boundary, image_prepare=prepare_images, serial_output=False)
    elif len(sys.argv) == 1:
        prepare_images()
        from full_agent_manager import main as manager_main
        manager_main(ProductionFullAgentBackend)
    else:
        raise SystemExit('Only fixed production manager commands are accepted')


if __name__ == '__main__':
    main()
