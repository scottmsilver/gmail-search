"""scripts/move-worker-disk.sh [--dry-run] [--source DIR] move|rollback|status

Moves the production worker VM's disk and boot files to `worker.vm_dir` in
deploy.json and runs the VM from a systemd user unit (#27). The owner picks the
window; see docs/worker-vm-move.md.

move      stop the VM gracefully, copy and verify its directory, render boot.sh
          and the unit for the new home, start it and health-check it. Any
          failure after the stop restarts the VM from the untouched source.
rollback  stop the unit and boot the source directory again (recorded in
          <vm_dir>/.moved-from). The window only: see the runbook.
status    where the VM runs from, and the unit's state. Read-only.

--dry-run prints what move would do and runs only read-only checks.
The source directory is never modified or deleted.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
import time
from typing import Callable

from . import lock
from .config import DEFAULT_PATH, DeployConfig, load_config
from .host import Host
from .runner import CommandFailed, Runner

VM_NAME = 'gmail-production-worker'
UNIT = 'gmail-production-worker.service'
LEGACY_UNIT = 'gmail-production-worker-legacy'
QEMU = '/usr/bin/qemu-system-x86_64'
SMP, MEMORY_MB = 4, 4096
UNIT_DIR = Path('~/.config/systemd/user')
MARKER = '.moved-from'
STOP_SECONDS = 180
HEALTH_SECONDS = 300
# Everything rendered into boot.sh and the unit must be a plain absolute path,
# or a name that cannot start like an option.
SAFE_PATH = re.compile(r'^/[A-Za-z0-9._/@:+-]*$')
SAFE_NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._:-]*$')


class MoveRefused(RuntimeError):
    pass


def say(message: str) -> None:
    print(message, flush=True)


# ── what the VM runs ─────────────────────────────────────────────────
def _plain(value, pattern: re.Pattern) -> str:
    text = str(value)
    if not pattern.match(text):
        raise MoveRefused(f'refusing to render {text!r}: only plain paths and names')
    return text


def _safe_path(value) -> str:
    return _plain(value, SAFE_PATH)


def _safe_name(value) -> str:
    return _plain(value, SAFE_NAME)


def qemu_argv(vm_dir: Path, host: str, port: int) -> list[str]:
    """The worker's qemu command line, as the hand-written boot.sh had it."""
    d = _safe_path(vm_dir)
    return [QEMU, '-name', VM_NAME, '-nodefaults', '-no-user-config',
            '-machine', 'q35,accel=kvm', '-cpu', 'host', '-smp', str(SMP), '-m', str(MEMORY_MB),
            '-drive', f'file={d}/worker.qcow2,if=virtio,format=qcow2',
            '-drive', f'file={d}/seed.iso,if=virtio,format=raw,readonly=on',
            '-netdev', f'user,id=worker,restrict=on,hostfwd=tcp:{_safe_name(host)}:{int(port)}-:22',
            '-device', 'virtio-net-pci,netdev=worker',
            '-rtc', 'base=utc,clock=host', '-display', 'none', '-serial', f'file:{d}/serial.log', '-monitor', 'none']


def render_boot_script(vm_dir: Path, host: str, port: int) -> str:
    argv = qemu_argv(vm_dir, host, port)
    return '\n'.join([
        '#!/usr/bin/env bash', '# Rendered by scripts/move-worker-disk.sh; see docs/worker-vm-move.md.',
        'set -euo pipefail', 'umask 077', f'cd {_safe_path(vm_dir)}', 'sha512sum --check --status base.sha512',
        'exec ' + ' '.join(argv), ''])


def _stop_command(config: DeployConfig) -> str:
    w = config.worker
    ssh = ['/usr/bin/ssh', '-F', '/dev/null', '-i', _safe_path(w.key),
           '-o', f'UserKnownHostsFile={_safe_path(w.known_hosts)}',
           '-o', 'StrictHostKeyChecking=yes', '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes',
           '-o', 'ConnectTimeout=10', '-p', str(int(w.port)), f'{_safe_name(w.user)}@{_safe_name(w.host)}',
           'sudo', '-n', 'systemctl', 'poweroff']
    # $$MAINPID reaches sh as $MAINPID. systemd sends SIGTERM once ExecStop
    # returns, so it waits for the guest to power off first.
    return f"/bin/sh -c '{' '.join(ssh)} || true; while kill -0 $$MAINPID 2>/dev/null; do sleep 1; done'"


def render_unit(vm_dir: Path, config: DeployConfig) -> str:
    return '\n'.join([
        '[Unit]', 'Description=Gmail Search production worker VM (qemu)',
        '# Rendered by scripts/move-worker-disk.sh from deploy.json worker.vm_dir; see docs/worker-vm-move.md.', '',
        '[Service]', 'Type=simple', f'ExecStart=/bin/bash {_safe_path(vm_dir)}/boot.sh',
        f'ExecStop={_stop_command(config)}', f'TimeoutStopSec={STOP_SECONDS}',
        'Restart=on-failure', 'RestartSec=10', 'UMask=0077', '',
        '[Install]', 'WantedBy=default.target', ''])


# ── finding the running VM ───────────────────────────────────────────
def _read_argv(cmdline: Path) -> list[str] | None:
    try:
        raw = cmdline.read_bytes()
    except OSError:
        return None
    return [part.decode(errors='replace') for part in raw.split(b'\0') if part]


def _is_our_qemu(entry: Path) -> bool:
    """Our uid, running the qemu binary (which may have been upgraded under it)."""
    try:
        return entry.stat().st_uid == os.getuid() and os.readlink(entry / 'exe') in (QEMU, QEMU + ' (deleted)')
    except OSError:
        return False


def find_vm(proc_root: Path = Path('/proc')) -> tuple[int, list[str]] | None:
    """The pid and argv of our qemu named VM_NAME, matched exactly, not by pattern."""
    found = []
    for entry in proc_root.iterdir():
        if entry.name.isdigit():
            argv = _read_argv(entry / 'cmdline')
            if argv and argv[0] == QEMU and argv[1:3] == ['-name', VM_NAME] and _is_our_qemu(entry):
                found.append((int(entry.name), argv))
    if len(found) > 1:
        raise MoveRefused(f'{len(found)} qemu processes named {VM_NAME}: pids {[pid for pid, _ in found]}')
    return found[0] if found else None


def disk_dir(argv: list[str]) -> Path:
    for arg in argv:
        match = re.match(r'^file=(.+)/worker\.qcow2,', arg)
        if match:
            return Path(match.group(1))
    raise MoveRefused('the running VM has no worker.qcow2 drive')


# ── copying and verifying ────────────────────────────────────────────
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def tree_entries(root: Path) -> dict[str, os.stat_result]:
    entries = {}
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            path = Path(dirpath, name)
            entries[str(path.relative_to(root))] = path.lstat()
    return entries


def allocated_bytes(root: Path) -> int:
    return sum(st.st_blocks * 512 for st in tree_entries(root).values() if stat.S_ISREG(st.st_mode))


def _entry_problem(rel: str, src: Path, dst: Path, a: os.stat_result, b: os.stat_result) -> str | None:
    if stat.S_IFMT(a.st_mode) != stat.S_IFMT(b.st_mode) or stat.S_IMODE(a.st_mode) != stat.S_IMODE(b.st_mode):
        return f'{rel}: type or mode differs'
    if stat.S_ISLNK(a.st_mode):
        return None if os.readlink(src / rel) == os.readlink(dst / rel) else f'{rel}: link target differs'
    if not stat.S_ISREG(a.st_mode):
        return None
    if a.st_size != b.st_size:
        return f'{rel}: size {b.st_size} != {a.st_size}'
    if b.st_blocks > a.st_blocks * 1.01 + 2048:
        return f'{rel}: copy is not sparse ({b.st_blocks * 512} bytes allocated, source {a.st_blocks * 512})'
    if sha256(src / rel) != sha256(dst / rel):
        return f'{rel}: sha256 differs'
    return None


def verify_copy(src: Path, dst: Path, ignore: frozenset[str] = frozenset()) -> list[str]:
    """Every entry in both trees, with the same type, mode, size and sha256."""
    a, b = tree_entries(src), tree_entries(dst)
    for rel in ignore:
        a.pop(rel, None)
        b.pop(rel, None)
    problems = [f'{rel}: missing in copy' for rel in sorted(a.keys() - b.keys())]
    problems += [f'{rel}: not in source' for rel in sorted(b.keys() - a.keys())]
    for rel in sorted(a.keys() & b.keys()):
        problem = _entry_problem(rel, src, dst, a[rel], b[rel])
        if problem:
            problems.append(problem)
    if stat.S_IMODE(src.stat().st_mode) != stat.S_IMODE(dst.stat().st_mode):
        problems.append('.: directory mode differs')
    return problems


def tree_fingerprint(root: Path) -> str:
    """Changes when any entry's type, mode, size or mtime does."""
    rows = sorted((rel, st.st_mode, st.st_size, st.st_mtime_ns) for rel, st in tree_entries(root).items())
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def write_replacing(path: Path, text: str, mode: int) -> None:
    """Write a new file and rename it over `path`, so a symlink there is
    replaced, never followed."""
    tmp = path.with_name(f'.{path.name}.tmp-{time.time_ns()}')
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
    with os.fdopen(fd, 'w') as handle:
        handle.write(text)
    os.replace(tmp, path)


def refuse_untrusted(path: Path) -> None:
    """Only what this user owns and nobody else can write is copied, booted or believed."""
    st = path.lstat()
    if stat.S_ISLNK(st.st_mode) or st.st_uid != os.getuid() or st.st_mode & 0o022:
        raise MoveRefused(f'{path} is a symlink, not ours, or group/world-writable')


def read_marker(vm_dir: Path) -> dict | None:
    marker = vm_dir / MARKER
    return json.loads(marker.read_text()) if marker.exists() else None


def copy_is_current(vm_dir: Path, source: Path, running) -> bool:
    """A verified copy the source has not moved past: the VM is not running
    from the source, and the source disk is unchanged since the copy."""
    marker = read_marker(vm_dir)
    if marker is None or (running and disk_dir(running[1]) == source):
        return False
    return tree_fingerprint(source) == marker.get('sourceFingerprint')


# ── the machine, behind one object so tests can fake it ──────────────
@dataclass
class Mover:
    host: Host
    vm_dir: Path
    land_lock: Path
    unit_dir: Path = field(default_factory=lambda: UNIT_DIR.expanduser())
    proc_root: Path = Path('/proc')
    free_bytes: Callable[[Path], int] = lambda path: shutil.disk_usage(path).free
    sleep: Callable[[float], None] = time.sleep
    stop_seconds: int = STOP_SECONDS
    health_seconds: int = HEALTH_SECONDS

    @property
    def worker(self):
        return self.host.config.worker

    def argv_for(self, vm_dir: Path) -> list[str]:
        return qemu_argv(vm_dir, self.worker.host, self.worker.port)

    def unit_path(self) -> Path:
        return self.unit_dir / UNIT

    def systemctl(self, *args: str, check: bool = True) -> int:
        return self.host.runner.run(['systemctl', '--user', *args], check=check)

    # ── read-only checks ─────────────────────────────────────────────
    def refuse_active_runs(self) -> None:
        active = self.host.active_runs(self.host.config.registry)
        if active:
            raise MoveRefused(f'{active} run(s) active; move when idle')

    def check_live_argv(self, argv: list[str], source: Path) -> None:
        """The rendered argv must be exactly the running one, directory aside."""
        expected = self.argv_for(source)
        if argv != expected:
            differ = sorted(set(argv) ^ set(expected))
            raise MoveRefused(f'the running VM argv is not the one this tool renders; differs in {differ}')

    def refuse_outside_references(self, source: Path) -> None:
        """The directory must be self-contained, or the moved VM would still
        read the old one: no symlinks, and every image in the qcow2 backing
        chain names its backing file within the directory and has no external
        data file."""
        links = [rel for rel, st in tree_entries(source).items() if stat.S_ISLNK(st.st_mode)]
        if links:
            raise MoveRefused(f'{source} has symlinks, so a copy would not be self-contained: {sorted(links)[:10]}')
        chain = json.loads(self.host.runner.capture(
            ['qemu-img', 'info', '-U', '--backing-chain', '--output=json', '--', source / 'worker.qcow2']))
        for image in chain:
            outside = [image.get('backing-filename', ''),
                       image.get('format-specific', {}).get('data', {}).get('data-file', '')]
            if any('/' in name for name in outside) or outside[1]:
                raise MoveRefused(f"{image.get('filename')} refers to {outside}; "
                                  'only a backing file in the same directory can move')

    def check_space(self, source: Path) -> None:
        need, parent = allocated_bytes(source), self.vm_dir.parent
        free = self.free_bytes(parent if parent.exists() else Path.home())
        if free < need * 1.1:
            raise MoveRefused(f'{parent}: {free} bytes free, the copy needs {need} (+10%)')

    def vm_alive(self, pid: int, argv: list[str]) -> bool:
        """That pid still runs that argv: a reused pid does not count."""
        return _read_argv(self.proc_root / str(pid) / 'cmdline') == argv

    def refuse_second_vm(self) -> None:
        running = find_vm(self.proc_root)
        if running:
            raise RuntimeError(f'a worker VM still runs (pid {running[0]}); not starting a second on the same disk port')

    def source_for(self, running, given: Path | None) -> Path:
        marker = read_marker(self.vm_dir)
        if marker:
            refuse_untrusted(self.vm_dir / MARKER)
            return Path(marker['source'])
        if running:
            return disk_dir(running[1])
        if given:
            return given
        raise MoveRefused('the VM is not running and no --source was given')

    # ── steps that change the machine ────────────────────────────────
    def stop_vm(self, pid: int, argv: list[str]) -> None:
        say(f'stopping the worker VM (pid {pid}) with a guest poweroff')
        self.host.worker_ssh('sudo -n systemctl poweroff', check=False)
        waited = 0
        while self.vm_alive(pid, argv):
            if waited >= self.stop_seconds:
                raise RuntimeError(f'qemu pid {pid} still running {waited}s after poweroff; not killing it. '
                                   'Nothing was copied or changed; check the VM before re-running')
            self.sleep(2)
            waited += 2

    def copy(self, source: Path) -> None:
        """Into a fresh directory of our own; it is removed unless it verifies."""
        self.vm_dir.parent.mkdir(parents=True, exist_ok=True)
        partial = Path(tempfile.mkdtemp(prefix=self.vm_dir.name + '.partial-', dir=self.vm_dir.parent))
        try:
            say(f'copying {source} -> {partial}')
            self.host.runner.run(['cp', '-a', '--sparse=always', '--reflink=auto', '--', f'{source}/.', partial])
            problems = verify_copy(source, partial)
            if problems:
                raise RuntimeError('copy did not verify: ' + '; '.join(problems[:10]))
            marker = {'source': str(source), 'movedAt': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                      'files': len(tree_entries(source)), 'allocatedBytes': allocated_bytes(source),
                      'sourceFingerprint': tree_fingerprint(source)}
            (partial / MARKER).write_text(json.dumps(marker, indent=2) + '\n')
            self.replace_with(partial)
        finally:
            if partial.exists():
                shutil.rmtree(partial)
        say(f"verified {marker['files']} entries (size, mode, sha256); now at {self.vm_dir}")

    def replace_with(self, partial: Path) -> None:
        """Rename the verified copy into place. An older copy (ours: the
        preconditions refuse a vm_dir without a marker) is renamed to a fresh
        name first, and only that name is removed."""
        stale = self.vm_dir.with_name(f'{self.vm_dir.name}.stale-{time.time_ns()}')
        if self.vm_dir.exists():
            os.rename(self.vm_dir, stale)
        os.rename(partial, self.vm_dir)
        if stale.exists():
            shutil.rmtree(stale)

    def reverify(self, source: Path) -> None:
        """A copy kept from an earlier attempt is checked again before it boots;
        only the files this tool writes into it are left out."""
        say(f're-verifying the copy at {self.vm_dir} against {source}')
        problems = verify_copy(source, self.vm_dir, ignore=frozenset({MARKER, 'boot.sh'}))
        if problems:
            raise RuntimeError(f'{self.vm_dir} no longer matches {source} (' + '; '.join(problems[:10]) +
                               '). If the VM ran from it, start the unit; otherwise move it aside and re-run')

    def install_unit(self) -> None:
        write_replacing(self.vm_dir / 'boot.sh', render_boot_script(self.vm_dir, self.worker.host, self.worker.port),
                        0o700)
        self.unit_dir.mkdir(parents=True, exist_ok=True)
        write_replacing(self.unit_path(), render_unit(self.vm_dir, self.host.config), 0o644)
        self.systemctl('daemon-reload')
        self.systemctl('enable', UNIT)

    def wait_worker(self, unit: str) -> None:
        """The unit active, the pinned SSH host key answering, the guest manager active."""
        manager, waited = self.host.config.services['manager'], 0
        while True:
            if self.host.unit_active(unit) and self.host.worker_ssh(
                    f'sudo -n systemctl is-active --quiet {manager}', check=False) == 0:
                say(f'healthy: {unit} active, worker SSH and {manager} answer')
                return
            if waited >= self.health_seconds:
                raise RuntimeError(f'worker not healthy {waited}s after starting {unit}')
            self.sleep(5)
            waited += 5

    def start_source(self, source: Path) -> None:
        self.refuse_second_vm()
        say(f'starting the VM from {source} as {LEGACY_UNIT}')
        self.host.runner.run(['systemd-run', '--user', f'--unit={LEGACY_UNIT}', '--collect',
                              '/bin/bash', source / 'boot.sh'])
        self.wait_worker(LEGACY_UNIT + '.service')

    # ── commands ─────────────────────────────────────────────────────
    def status(self) -> None:
        running = find_vm(self.proc_root)
        where = f'pid {running[0]} from {disk_dir(running[1])}' if running else 'not running'
        say(f'worker VM: {where}')
        say(f"{UNIT}: {'active' if self.host.unit_active(UNIT) else 'not active'}"
            f"{'' if self.unit_path().exists() else ' (not installed)'}")
        marker = read_marker(self.vm_dir)
        say(f'{self.vm_dir}: ' + (f"moved from {marker['source']} at {marker['movedAt']}" if marker
                                    else 'exists, no move marker' if self.vm_dir.exists() else 'absent'))

    def already_moved(self, running) -> bool:
        return bool(running) and disk_dir(running[1]) == self.vm_dir.resolve() and self.host.unit_active(UNIT)

    def move(self, given: Path | None, dry_run: bool) -> None:
        running = find_vm(self.proc_root)
        if self.already_moved(running):
            say(f'already moved: {UNIT} runs the VM from {self.vm_dir}; nothing to do')
            return
        source = self.source_for(running, given)
        self.check_preconditions(source, running)
        if dry_run:
            self.print_plan(source, running)
            return
        with lock.held(self.land_lock, 'move-worker-disk'):
            self.refuse_active_runs()
            self.cut_over(source, running)

    def check_preconditions(self, source: Path, running) -> None:
        src, dst = source.resolve(), self.vm_dir.resolve()
        if src.is_relative_to(dst) or dst.is_relative_to(src):
            raise MoveRefused(f'source {source} and worker.vm_dir {self.vm_dir} must not contain each other')
        if not (source / 'worker.qcow2').is_file() or not (source / 'boot.sh').is_file():
            raise MoveRefused(f'{source} has no worker.qcow2 and boot.sh')
        for path in (source, source / 'boot.sh', *([self.vm_dir] if self.vm_dir.exists() else [])):
            refuse_untrusted(path)
        self.refuse_outside_references(source)
        if self.vm_dir.exists() and read_marker(self.vm_dir) is None:
            raise MoveRefused(f'{self.vm_dir} exists but was not made by this tool; move it aside first')
        if running:
            self.check_live_argv(running[1], source)
        if not copy_is_current(self.vm_dir, source, running):
            self.check_space(source)
        self.refuse_active_runs()

    def print_plan(self, source: Path, running) -> None:
        copied = copy_is_current(self.vm_dir, source, running)
        steps = [f'take {self.land_lock} and re-check that no run is active',
                 f'stop the VM (pid {running[0]}) with a guest poweroff, wait up to {self.stop_seconds}s'
                 if running else 'the VM is not running: nothing to stop',
                 f'{self.vm_dir} already holds a verified copy: skip the copy' if copied else
                 f'copy {source} ({allocated_bytes(source)} bytes allocated) to a new {self.vm_dir.name}.partial-* '
                 'directory, verify size, mode, sparseness and sha256 of every file, rename into place',
                 f'write {self.vm_dir}/boot.sh and {self.unit_path()}, daemon-reload, enable {UNIT}',
                 f'start {UNIT}, wait up to {self.health_seconds}s for SSH and the guest manager',
                 f'on any failure after the stop: stop {UNIT}, boot {source} again as {LEGACY_UNIT}']
        say(f'dry run: source {source}, target {self.vm_dir}; read-only checks passed')
        say('argv check: rendered command line equals the running one with only the directory changed'
            if running else 'argv check: skipped, the VM is not running')
        for number, step in enumerate(steps, 1):
            say(f'  {number}. {step}')
        say('--- boot.sh ---\n' + render_boot_script(self.vm_dir, self.worker.host, self.worker.port))
        say(f'--- {self.unit_path()} ---\n' + render_unit(self.vm_dir, self.host.config))

    def cut_over(self, source: Path, running) -> None:
        current = copy_is_current(self.vm_dir, source, running)
        if running:
            self.stop_vm(*running)
        try:
            if current:
                self.reverify(source)
            else:
                self.copy(source)
            self.install_unit()
            self.systemctl('start', UNIT)
            self.wait_worker(UNIT)
        except Exception as error:
            restored = self.restore_source(source)
            raise RuntimeError(f'move failed ({error}); the VM is back on {source}: '
                               f'{"yes" if restored else "NO, see the runbook"}') from error
        say(f'moved: the worker VM runs from {self.vm_dir} under {UNIT}; {source} is untouched')

    def restore_source(self, source: Path) -> bool:
        try:
            self.systemctl('disable', '--now', UNIT, check=False)
            self.start_source(source)
            return True
        except Exception as error:  # report, do not mask the original failure
            say(f'restoring the source failed: {error}')
            return False

    def rollback(self, dry_run: bool) -> None:
        if read_marker(self.vm_dir) is None:
            raise MoveRefused(f'{self.vm_dir} has no {MARKER}: nothing to roll back to')
        source = self.source_for(None, None)
        for path in (source, source / 'boot.sh'):
            refuse_untrusted(path)
        if dry_run:
            say(f'dry run: would stop and disable {UNIT}, then boot {source} as {LEGACY_UNIT}')
            return
        with lock.held(self.land_lock, 'move-worker-disk rollback'):
            self.refuse_active_runs()
            self.systemctl('disable', '--now', UNIT)
            try:
                self.start_source(source)
            except Exception as error:
                raise RuntimeError(f'rollback failed ({error}); the moved VM is stopped. Start it again with: '
                                   f'systemctl --user enable --now {UNIT}') from error
        say(f'rolled back: the VM runs from {source}; {self.vm_dir} is kept')


# ── entry point ──────────────────────────────────────────────────────
def parse_args(argv):
    p = argparse.ArgumentParser(prog='scripts/move-worker-disk.sh', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=['move', 'rollback', 'status'])
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--source', type=Path, help='the current VM directory, when the VM is not running')
    p.add_argument('--config', type=Path, default=DEFAULT_PATH)
    return p.parse_args(argv)


def mover_for(config: DeployConfig, runner: Runner) -> Mover:
    if config.worker.vm_dir is None:
        raise MoveRefused("deploy.json has no worker.vm_dir; add it (see docs/worker-vm-move.md)")
    common = runner.capture(['git', 'rev-parse', '--path-format=absolute', '--git-common-dir']).strip()
    return Mover(Host(config, runner), config.worker.vm_dir, Path(common).parent / '.runtime/issue-loop/land.lock')


def run(argv) -> int:
    args = parse_args(argv)
    runner = Runner()
    mover = mover_for(load_config(args.config), runner)
    if args.command == 'status':
        mover.status()
    elif args.command == 'rollback':
        mover.rollback(args.dry_run)
    else:
        mover.move(args.source.resolve() if args.source else None, args.dry_run)
    return 0


def main(argv=None) -> int:
    try:
        return run(sys.argv[1:] if argv is None else argv)
    except (RuntimeError, ValueError, CommandFailed, OSError) as error:
        print(f'move-worker-disk FAILED: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
