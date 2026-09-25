#!/usr/bin/env python3
"""Synthetic-only nested Firecracker backend and independent lease supervisor.

Run as root ONLY inside the clean synthetic-execution-worker outer VM. No
production host launcher, guest runtime, or capability relay is provided.
The controller passes leases/limits; image paths and executable flags are fixed.
"""
from contextlib import contextmanager
import ctypes
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import select
import shutil
import socket
import stat
import subprocess
import sys
import time

ROOT = Path('/var/lib/gmail-worker')
JAILS = Path('/srv/jailer/firecracker')
CGROUP = Path('/sys/fs/cgroup/gmail-worker')
MODULE = Path('/opt/gmail-worker/firecracker_backend.py')
ATTACHMENT_PIN = '89bf4da702506dadacc5cd08b9f4d737f27abb592eb35b7a768cc22d590f9068'
# Separately built full-agent image; historical pins remain unchanged.
AGENT_FULL_PIN = 'b71e81e8e481113280a48661bc157741aa3961bf9d708179ca68fc5a65dd9d39'
AGENT_PI_MCP_PIN = '478461755f2f344572bb0784685205fb173795bb5e3acae6fe523fb73dc62bbc'
AGENT_MCP_PIN = '54303c8873abc96c27ea8cc99930e4713016a5d1b091376ed858a472f58ba7ba'
AGENT_TOOLS_PIN = '4b981cd6a2e1eb2f28acad2d02ac6c365b4ef48b03345eb098578b9a1dd71402'
RUNTIME_PIN = '0546f3aad681b012a1b44b90ac6f18188a6e25419220806e6a7a6ed4245bb32a'
PINS = {
    'firecracker': '34237ad1a6fcec150a85786488a64acc3b8f419057136113a9858c8bea68bf88',
    'jailer': 'a8683b88775f4d95802fb6f4db8ee5fba22aca31e0e9cfa99560608c560cdfef',
    'vmlinux': 'e41c7048bd2475e7e788153823fcb9166a7e0b78c4c443bd6446d015fa735f53',
    'rootfs.squashfs': 'f4fbb71a581c2f4cd204900ceaf280b71b031c58479250cb0430e0b29774ef5c',
}


def checked_handle(handle):
    if type(handle) is not str or not re.fullmatch('[a-f0-9]{32}', handle):
        raise ValueError('Invalid trusted worker handle')
    return handle


def boundary():
    if os.geteuid() != 0 or socket.gethostname() != 'synthetic-execution-worker':
        raise RuntimeError('Backend is restricted to root in the clean synthetic outer worker')
    if not Path('/sys/fs/cgroup/cgroup.controllers').is_file():
        raise RuntimeError('cgroup v2 required')


def atomic_json(path, data):
    temporary = path.with_name(path.name + '.new')
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(data, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read_json(path):
    return json.loads(path.read_text())


def process_start(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except FileNotFoundError:
        return None


def supervisor_alive(path):
    if not (path/'supervisor.json').exists():
        return False
    process = read_json(path/'supervisor.json')
    return process_start(process['pid']) == process['start']


def fixed_config(limits, *, profile='shell'):
    if profile not in ('shell', 'agent', 'attachment', 'agent_tools', 'agent_mcp', 'agent_pi_mcp', 'agent_full'):
        raise ValueError('Unsupported fixed worker profile')
    if not 128 <= limits['memory_mib'] <= 16384 or not 1 <= limits['vcpus'] <= 8:
        raise ValueError('Unsupported fixed machine limits')
    config = {
        'boot-source': {'kernel_image_path': '/vmlinux',
                        'boot_args': 'console=ttyS0 reboot=k panic=1 pci=off root=/dev/vda rootfstype=squashfs ro init=/bin/sh'},
        'drives': [{'drive_id': 'rootfs', 'path_on_host': '/rootfs.squashfs', 'is_root_device': True, 'is_read_only': True}],
        'machine-config': {'vcpu_count': limits['vcpus'], 'mem_size_mib': limits['memory_mib'] - 64},
        'vsock': {'guest_cid': 3, 'uds_path': '/gateway.vsock'},
    }
    if profile in ('agent', 'attachment', 'agent_tools', 'agent_mcp', 'agent_pi_mcp', 'agent_full'):
        runtime = {'attachment': '/attachment.squashfs', 'agent': '/runtime.squashfs',
                   'agent_tools': '/agent-tools.squashfs', 'agent_mcp': '/agent-mcp.squashfs',
                   'agent_pi_mcp': '/agent-pi-mcp.squashfs', 'agent_full': '/agent-full.squashfs'}[profile]
        config['drives'].append({'drive_id': 'runtime', 'path_on_host': runtime,
                                 'is_root_device': False, 'is_read_only': True})
    return config



def prepare_jail(jail, config):
    """Create the jail root and the VMM's config so the jailed VMM can read them.

    The VMM runs as uid 65534 once the jailer drops privileges, and the manager
    unit runs under UMask=0077: mkdir(mode=0o755) lands as 0700 and write_text()
    as 0600, both root-only, and Firecracker panicked reading its own config
    ("Unable to open or read from the configuration file: Permission denied").
    mkdir and write_text only request a mode; chmod sets it. The config holds
    paths and machine limits, no secrets.
    """
    jail.mkdir(parents=True, mode=0o755)
    jail.chmod(0o755)
    path = jail/'config.json'
    path.write_text(json.dumps(config))
    path.chmod(0o644)
    return path


def vmm_exit_report(child, output):
    """Why the VMM died, or None while it is alive.

    A VMM that dies early becomes a zombie of this subreaper. Its /proc entry
    persists, still named `firecracker`, reporting Seccomp 0 -- so a readiness
    loop that only reads /proc spent its whole window polling a corpse and
    blamed seccomp. Reap it and report its exit status and its own log instead.
    """
    try:
        pid, status = os.waitpid(child, os.WNOHANG)
    except ChildProcessError:
        return None
    if pid == 0:
        return None
    os.set_blocking(output.fileno(), False)
    log = (output.read() or b'').decode(errors='replace')[-2000:].strip()
    return f'VMM exited before readiness (exit code {os.waitstatus_to_exitcode(status)}): {log}'


def finalize_stopped_state(path):
    """Mark a torn-down run stopped without erasing why it failed.

    `supervise()` records {'status': 'failed', 'error': ...}; teardown used to
    write {'status': 'stopped'} over it unconditionally, so a failed guest was
    indistinguishable from a clean exit. Production discards guest serial output
    by design, which left no record of the failure at all.
    """
    state = read_json(path/'state.json') if (path/'state.json').exists() else {}
    if state.get('status') == 'failed':
        return
    atomic_json(path/'state.json', {'status': 'stopped'})


class FirecrackerBackend:
    namespace = 'synthetic-firecracker'
    profile = 'shell'
    supervisor_module = MODULE

    def check_boundary(self):
        boundary()

    def __init__(self):
        self.check_boundary()
        ROOT.mkdir(mode=0o700, exist_ok=True)
        info = ROOT.lstat()
        if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700 or info.st_uid != 0:
            raise RuntimeError('Unsafe backend state directory')
        (ROOT/'runs').mkdir(mode=0o700, exist_ok=True)

    @contextmanager
    def lock(self, handle):
        path = ROOT/'runs'/checked_handle(handle)
        path.mkdir(mode=0o700, exist_ok=True)
        fd = os.open(path/'lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield path
        finally:
            os.close(fd)

    def launch(self, handle, lease, limits):
        with self.lock(handle) as path:
            if (path/'state.json').exists():
                old = read_json(path/'lease.json')
                if (read_json(path/'state.json')['status'] == 'running'
                        and old['run_id'] == lease.run_id and old['deadline'] == lease.deadline
                        and old.get('profile', 'shell') == self.profile):
                    return
                raise RuntimeError('Partially used handle cannot launch again')
            conf = asdict(limits)
            fixed_config(conf, profile=self.profile)
            now = time.time()
            if not now < lease.lease_expires <= lease.deadline or lease.deadline - now > limits.wall_seconds:
                raise ValueError('Invalid online lease')
            data = {'run_id': lease.run_id, 'deadline': lease.deadline, 'profile': self.profile,
                    'lease_expires': lease.lease_expires, 'limits': conf}
            atomic_json(path/'lease.json', data)
            atomic_json(path/'state.json', {'status': 'launching'})
            # The supervisor is outside the VMM cgroup and survives controller exit.
            subprocess.Popen(['/usr/bin/python3', str(self.supervisor_module), '_supervise', handle],
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True, close_fds=True, env={'PATH': '/usr/sbin:/usr/bin:/sbin:/bin'})
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            state = read_json(path/'state.json')
            if state['status'] == 'running':
                return
            if state['status'] in ('stopped', 'failed'):
                raise RuntimeError('Synthetic VMM launch failed: ' + state.get('error', ''))
            time.sleep(.05)
        self.stop(handle)
        raise TimeoutError('Synthetic VMM launch timed out')

    def renew(self, handle, lease):
        with self.lock(handle) as path:
            old = read_json(path/'lease.json')
            if (old['run_id'] != lease.run_id or old['deadline'] != lease.deadline
                    or not time.time() < lease.lease_expires <= old['deadline']
                    or read_json(path/'state.json')['status'] != 'running'):
                raise RuntimeError('Lease renewal refused')
            old['lease_expires'] = max(old['lease_expires'], lease.lease_expires)
            atomic_json(path/'lease.json', old)

    def inventory(self):
        handles = set()
        for path in (ROOT/'runs').iterdir():
            checked_handle(path.name)
            if (path/'state.json').exists() and read_json(path/'state.json')['status'] not in ('stopped', 'failed'):
                handles.add(path.name)
        if CGROUP.exists():
            handles.update(checked_handle(p.name) for p in CGROUP.iterdir() if p.is_dir())
        return handles

    def stop(self, handle):
        checked_handle(handle)
        with self.lock(handle) as path:
            atomic_json(path/'stop.json', {})
        kill_cgroup(handle)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            state = read_json(path/'state.json') if (path/'state.json').exists() else {}
            if state.get('status') in ('stopped', 'failed') or not supervisor_alive(path):
                break
            time.sleep(.05)
        if state.get('status') not in ('stopped', 'failed') and supervisor_alive(path):
            raise RuntimeError('Supervisor has not completed teardown; retain binding and retry')
        cleanup(handle)
        if (path/'supervisor.json').exists():
            try:
                os.waitpid(read_json(path/'supervisor.json')['pid'], os.WNOHANG)
            except ChildProcessError:
                pass  # Original controller died; outer init adopts/reaps it.
        with self.lock(handle) as path:
            finalize_stopped_state(path)


class SyntheticAgentBackend(FirecrackerBackend):
    # Trusted fixed profile; no image/config argument reaches start().
    # Both fixed profiles share this physical worker's inventory namespace.
    namespace = 'synthetic-firecracker'
    profile = 'agent'


def kill_cgroup(handle):
    path = CGROUP/checked_handle(handle)
    if path.exists():
        (path/'cgroup.kill').write_text('1')


def cleanup(handle):
    kill_cgroup(handle)
    path = CGROUP/handle
    deadline = time.monotonic() + 5
    while path.exists() and time.monotonic() < deadline:
        if 'populated 0' in (path/'cgroup.events').read_text():
            path.rmdir()
            break
        time.sleep(.05)
    if path.exists():
        raise RuntimeError('VMM cgroup remains populated')
    subprocess.run(['/usr/sbin/ip', 'netns', 'del', 'gms-' + handle], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
    jail = JAILS/handle
    if jail.exists():
        shutil.rmtree(jail)


def prepare_images():
    boundary()
    images = ROOT/'images'
    images.mkdir(mode=0o700, exist_ok=True)
    for name, expected in PINS.items():
        source = Path('/home/worker/microvm')/name
        target = images/name
        if not target.exists():
            shutil.copyfile(source, target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise RuntimeError('Pinned image/binary mismatch')
        target.chmod(0o500 if name in ('firecracker', 'jailer') else 0o444)
    # Jailer uses the executable basename as its directory component.
    for name in ('firecracker', 'jailer'):
        installed = Path('/usr/local/bin')/name
        if not installed.exists() or hashlib.sha256(installed.read_bytes()).hexdigest() != PINS[name]:
            shutil.copyfile(images/name, installed)
            installed.chmod(0o755)
    Path('/sys/fs/cgroup/cgroup.subtree_control').write_text('+cpu +memory +pids')
    CGROUP.mkdir(exist_ok=True)
    (CGROUP/'cgroup.subtree_control').write_text('+cpu +memory +pids')


class SyntheticFullAgentBackend(FirecrackerBackend):
    # Full v3 tools and user-prompt RPC runner; historical profiles unchanged.
    profile = 'agent_full'


class SyntheticAgentPiMCPBackend(FirecrackerBackend):
    # Pi adapter qualification has its own immutable image and fixed startup.
    profile = 'agent_pi_mcp'


class SyntheticAgentMCPBackend(FirecrackerBackend):
    # Native Claude typed MCP qualification; no changes to older image pins.
    profile = 'agent_mcp'


class SyntheticAgentToolsBackend(FirecrackerBackend):
    # Separate immutable qualification image; existing agent profile is unchanged.
    profile = 'agent_tools'


class SyntheticAttachmentBackend(FirecrackerBackend):
    profile = 'attachment'


def supervise(handle, *, boundary_check=boundary, image_prepare=prepare_images, serial_output=True):
    boundary_check()
    checked_handle(handle)
    path = ROOT/'runs'/handle
    atomic_json(path/'supervisor.json', {'pid': os.getpid(), 'start': process_start(os.getpid())})
    lease = read_json(path/'lease.json')
    limits = lease['limits']
    profile = lease.get('profile', 'shell')
    hard_end = time.monotonic() + min(lease['deadline'] - time.time(), limits['wall_seconds'])
    child = None
    proc = None
    error = ''
    try:
        image_prepare()
        if (path/'stop.json').exists():
            raise RuntimeError('Cancelled before launch')
        jail = JAILS/handle/'root'
        # Before any image is linked in: the jail must exist, with modes the VMM can read.
        prepare_jail(jail, fixed_config(limits, profile=profile))
        names = ['vmlinux', 'rootfs.squashfs']
        if profile in ('agent', 'attachment', 'agent_tools', 'agent_mcp', 'agent_pi_mcp', 'agent_full'):
            filename, pin = {'attachment': ('attachment.squashfs', ATTACHMENT_PIN),
                             'agent': ('runtime.squashfs', RUNTIME_PIN),
                             'agent_tools': ('agent-tools.squashfs', AGENT_TOOLS_PIN),
                             'agent_mcp': ('agent-mcp.squashfs', AGENT_MCP_PIN),
                             'agent_pi_mcp': ('agent-pi-mcp.squashfs', AGENT_PI_MCP_PIN),
                             'agent_full': ('agent-full.squashfs', AGENT_FULL_PIN)}[profile]
            runtime = ROOT/'images'/filename
            if hashlib.sha256(runtime.read_bytes()).hexdigest() != pin:
                raise RuntimeError('Pinned synthetic runtime image mismatch')
            names.append(filename)
        for name in names:
            if (ROOT/'images'/name).stat().st_size > limits['disk_bytes']:
                raise RuntimeError('Fixed image exceeds disk limit')
            os.link(ROOT/'images'/name, jail/name)
        subprocess.run(['/usr/sbin/ip', 'netns', 'add', 'gms-' + handle], check=True, timeout=5)
        if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
            raise RuntimeError('subreaper unavailable')
        args = ['/usr/local/bin/jailer', '--id', handle, '--exec-file', '/usr/local/bin/firecracker',
                '--uid', '65534', '--gid', '65534', '--new-pid-ns',
                '--netns', '/run/netns/gms-' + handle, '--cgroup-version', '2', '--parent-cgroup', 'gmail-worker',
                '--cgroup', f"memory.max={limits['memory_mib'] * 1024**2}", '--cgroup', 'memory.swap.max=0',
                '--cgroup', f"pids.max={limits['pids']}", '--cgroup', f"cpu.max={limits['vcpus'] * 100000} 100000",
                '--resource-limit', 'no-file=128', '--resource-limit', f"fsize={limits['output_bytes']}",
                '--', '--no-api', '--config-file', '/config.json']
        proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if proc.wait(timeout=5) != 0:
            raise RuntimeError('jailer launch failed: ' + proc.stdout.read(2048).decode(errors='replace'))
        child = int((jail/'firecracker.pid').read_text())
        # /proc diagnostics describe the real VMM, not the exited jailer parent.
        ready_by = time.monotonic() + 5
        while True:
            died = vmm_exit_report(child, proc.stdout)
            if died:
                raise RuntimeError(died)
            status = Path(f'/proc/{child}/status').read_text()
            if re.search(r'^Seccomp:\s+2$', status, re.MULTILINE):
                break
            if time.monotonic() >= ready_by:
                # Say what was observed. "readiness check failed" alone cannot
                # distinguish a VMM that never applies its filter from one that
                # died, and the production worker keeps no console output.
                seen = next((line for line in status.splitlines() if line.startswith('Seccomp')), 'Seccomp: absent')
                name = next((line for line in status.splitlines() if line.startswith('Name:')), 'Name: ?')
                raise RuntimeError(f'VMM seccomp readiness check failed (pid={child} {name.strip()} '
                                   f'{seen.strip()}; waited {time.monotonic() - (ready_by - 5):.1f}s)')
            time.sleep(.02)
        evidence = {key: line for key in ('Uid:', 'Seccomp:', 'NSpid:') for line in status.splitlines() if line.startswith(key)}
        evidence.update({name: (CGROUP/handle/name).read_text().strip() for name in ('memory.max', 'memory.swap.max', 'pids.max', 'cpu.max')})
        atomic_json(path/'state.json', {'status': 'running', 'pid': child, 'evidence': evidence})
        os.set_blocking(proc.stdout.fileno(), False)
        written, sent = 0, False
        boot_at = time.monotonic()
        with (path/'serial.log' if serial_output else Path('/dev/null')).open('wb') as log:
            while True:
                pid, exit_status = os.waitpid(child, os.WNOHANG)
                if pid:
                    child = None
                    if os.waitstatus_to_exitcode(exit_status) != 0:
                        error = 'VMM exited unsuccessfully'
                    break
                current = read_json(path/'lease.json')
                if ((path/'stop.json').exists() or time.time() >= min(current['lease_expires'], current['deadline'])
                        or time.monotonic() >= hard_end):
                    kill_cgroup(handle)
                    os.waitpid(child, 0)
                    child = None
                    break
                readable, _, _ = select.select([proc.stdout], [], [], .05)
                if readable:
                    chunk = os.read(proc.stdout.fileno(), 65536)
                    if written + len(chunk) > limits['output_bytes']:
                        raise RuntimeError('Serial output quota exceeded')
                    log.write(chunk)
                    log.flush()
                    written += len(chunk)
                if not sent and time.monotonic() - boot_at >= 3:
                    if profile == 'attachment':
                        command = (b'mount -t proc proc /proc; mount -t sysfs sysfs /sys; '
                                   b'mount -t tmpfs -o size=128m,mode=1777,nodev,nosuid tmpfs /tmp; '
                                   b'mkdir /tmp/parser; mount -t squashfs -o ro /dev/vdb /tmp/parser; '
                                   b'/usr/bin/python3 -I /tmp/parser/guest-attachment-parser.py\n')
                    elif profile in ('agent', 'agent_tools', 'agent_mcp', 'agent_pi_mcp', 'agent_full'):
                        command = (b'mount -t proc proc /proc; mount -t sysfs sysfs /sys; '
                                   b'mount -t tmpfs -o size=256m,mode=1777,nodev,nosuid tmpfs /tmp; '
                                   b'mkdir /tmp/runtime; mount -t squashfs -o ro /dev/vdb /tmp/runtime; '
                                   b'/usr/bin/ip link set lo up; /usr/bin/python3 -I /tmp/runtime/' +
                                   {'agent_tools': b'guest-mail-tools-smoke.py\n', 'agent_mcp': b'guest-mail-mcp-smoke.py\n',
                                    'agent_pi_mcp': b'guest-pi-mail-mcp-smoke.py\n',
                                    'agent_full': b'guest_agent.py\n',
                                    'agent': b'guest-agent-smoke.py\n'}[profile])
                    else:
                        command = b'mount -t proc proc /proc; mount -t sysfs sysfs /sys; echo GMAIL_BACKEND_SYNTHETIC_HELLO; cat /proc/net/dev; cat /proc/mounts\n'
                    proc.stdin.write(command)
                    proc.stdin.flush()
                    sent = True
    except BaseException as exc:
        error = str(exc)[:2000]
    finally:
        if child is not None:
            kill_cgroup(handle)
            os.waitpid(child, 0)
        if proc is not None:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            if proc.stdin:
                proc.stdin.close()
            if proc.stdout:
                proc.stdout.close()
        try:
            cleanup(handle)
        except Exception as exc:
            error = str(exc)[:2000]
        atomic_json(path/'state.json', {'status': 'failed' if error else 'stopped', 'error': error})


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] != '_supervise':
        raise SystemExit('Only the trusted backend may start a supervisor')
    supervise(sys.argv[2])
