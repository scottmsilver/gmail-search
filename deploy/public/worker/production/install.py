#!/usr/bin/env python3
"""Install reviewed production files locally inside an enrolled dedicated VM.

No downloads, user creation, key generation, service start or SSH reload.
Arguments are trusted administrator inputs, never RPC/browser inputs.
"""
import hashlib
import json
import os
from pathlib import Path
import pwd
import shutil
import socket
import stat
import subprocess
import sys


MODULES = ('firecracker_backend.py','production_worker_profile.py','full_agent_manager.py',
           'full_agent_rpc_server.py','full_agent_rpc_frontend.py','guest_agent_bootstrap.py',
           'guest_tool_config.py','guest_run_bootstrap.py','vsock_http_relay.py')

def main():
    if len(sys.argv) != 4 or os.geteuid() != 0 or socket.gethostname() != 'gmail-execution-worker':
        raise SystemExit('Usage on dedicated VM as root: install.py REPO ASSETS ENROLLMENT')
    repo, assets, enrollment = map(Path, sys.argv[1:])
    source = repo/'deploy/public/worker'
    sys.path.insert(0, str(source))
    from production_worker_profile import ASSETS, validate_enrollment
    validate_enrollment(json.loads(enrollment.read_text()), hostname=socket.gethostname(),
                        machine_id=Path('/etc/machine-id').read_text().strip())
    # Accounts/keys are provisioned independently and are never copied from a host home.
    account_ids = set()
    group_ids = set()
    for account in ('gmail-full-agent-rpc', 'gmail-gateway-tunnel'):
        entry = pwd.getpwnam(account)
        if (entry.pw_uid in (0,65534) or entry.pw_gid in (0,65534)
                or entry.pw_uid in account_ids or entry.pw_gid in group_ids):
            raise RuntimeError('Distinct dedicated unprivileged accounts required')
        account_ids.add(entry.pw_uid); group_ids.add(entry.pw_gid)
    for name, digest in ASSETS.items():
        with (assets/name).open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
                raise RuntimeError('Source artifact pin mismatch: ' + name)
    def directory(path, mode):
        path.mkdir(mode=mode, parents=True, exist_ok=True)
        info = path.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
            raise RuntimeError('Unsafe installation directory')
        path.chmod(mode)
    def install(src, dest, mode):
        if dest.is_symlink():
            raise RuntimeError('Refusing symlink install target')
        temp = dest.with_name(dest.name + '.install-new')
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
        with os.fdopen(fd, 'wb') as output, src.open('rb') as input:
            shutil.copyfileobj(input, output)
        temp.chmod(mode); os.replace(temp, dest)
    target = Path('/opt/gmail-worker')
    for path, mode in ((target,0o755),(Path('/etc/gmail-worker'),0o755),
                       (Path('/var/lib/gmail-worker'),0o700),(Path('/var/lib/gmail-worker/images'),0o700)):
        directory(path,mode)
    for name in MODULES:
        install(source/name,target/name,0o644)
    install(repo/'src/gmail_search/gateway/full_agent_rpc.py',target/'full_agent_rpc.py',0o644)
    for name in ASSETS:
        install(assets/name,Path('/var/lib/gmail-worker/images')/name,0o444)
    for name in ('firecracker','jailer'):
        install(assets/name,Path('/usr/local/bin')/name,0o755)
    install(enrollment,Path('/etc/gmail-worker/production.json'),0o600)
    for unit in ('gmail-full-agent-manager.service','gmail-worker-clock-sync.service'):
        install(source/'production'/unit,Path('/etc/systemd/system')/unit,0o644)
    # The worker cannot reach NTP; lease deadlines depend on this clock.
    install(source/'production/phc_clock_sync.py',target/'phc_clock_sync.py',0o644)
    # SSH policy is staged for explicit effective-policy validation before reload.
    install(source/'production/sshd-worker.conf',Path('/etc/gmail-worker/sshd-worker.conf.pending'),0o644)
    subprocess.run(['/usr/bin/python3','-I',str(target/'production_worker_profile.py'),'--check'],check=True)
    print('Installed; services remain stopped. Validate SSH policy and run synthetic qualification before admission.')


if __name__ == '__main__':
    main()
