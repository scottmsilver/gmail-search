"""The worker-disk move (#27) against throwaway directories, a fake /proc and a
fake systemctl/ssh runner: nothing live is read or touched."""
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from gmail_search.deploy import worker_disk
from gmail_search.deploy.config import load_config
from gmail_search.deploy.host import Host
from gmail_search.deploy.worker_disk import Mover, MoveRefused

from test_deploy_activate import config as deploy_config
from test_deploy_config_image import write_config

# The running worker's argv as observed on 2026-09-25, with its directory
# replaced by a synthetic one. The rendered argv must equal it exactly.
OLD = '/srv/synthetic/production-worker'
LIVE_ARGV = [
    '/usr/bin/qemu-system-x86_64', '-name', 'gmail-production-worker', '-nodefaults', '-no-user-config',
    '-machine', 'q35,accel=kvm', '-cpu', 'host', '-smp', '4', '-m', '4096',
    '-drive', f'file={OLD}/worker.qcow2,if=virtio,format=qcow2',
    '-drive', f'file={OLD}/seed.iso,if=virtio,format=raw,readonly=on',
    '-netdev', 'user,id=worker,restrict=on,hostfwd=tcp:127.0.0.1:22093-:22',
    '-device', 'virtio-net-pci,netdev=worker',
    '-rtc', 'base=utc,clock=host', '-display', 'none', '-serial', f'file:{OLD}/serial.log', '-monitor', 'none']


class FakeMachine:
    """systemctl, ssh and systemd-run as state changes on a fake /proc; cp runs for real."""

    def __init__(self, proc: Path, *, healthy=True):
        self.proc, self.healthy, self.calls, self.active = proc, healthy, [], set()
        self.stop_works = True
        self.chain = [{'filename': 'worker.qcow2', 'backing-filename': 'base.qcow2'}, {'filename': 'base.qcow2'}]

    def spawn(self, pid: int, argv, exe=None):
        (self.proc / str(pid)).mkdir(parents=True)
        (self.proc / str(pid) / 'cmdline').write_bytes(b'\0'.join(a.encode() for a in argv) + b'\0')
        (self.proc / str(pid) / 'exe').symlink_to(exe or argv[0])

    def kill(self, pid: int):
        for item in (self.proc / str(pid)).iterdir():
            item.unlink()
        (self.proc / str(pid)).rmdir()

    def alive(self, pid: int) -> bool:
        return (self.proc / str(pid)).exists()

    def pids(self):
        return [int(p.name) for p in self.proc.iterdir()]

    def boot(self, unit: str, vm_dir: str):
        self.spawn(max(self.pids(), default=100) + 1, worker_disk.qemu_argv(Path(vm_dir), '127.0.0.1', 22093))
        self.active.add(unit)

    def run(self, args, check=True, **kw):
        args = [str(a) for a in args]
        self.calls.append(args)
        if args[0] == 'cp':
            return subprocess.run(args, check=check).returncode
        if args[0] == 'ssh' and args[-1].endswith('poweroff'):
            for pid in self.pids():
                self.kill(pid)
            self.active.clear()
            return 255  # the connection drops as the guest powers off
        if args[0] == 'ssh':
            return 0 if self.healthy and self.pids() else 1
        if args[:3] == ['systemctl', '--user', 'start']:
            self.boot(args[3], self.unit_vm_dir())
        if args[:2] == ['systemctl', '--user'] and '--now' in args and self.stop_works:
            for pid in self.pids():
                self.kill(pid)
            self.active.discard(args[-1])
        if args[0] == 'systemd-run':
            self.boot(args[2].split('=', 1)[1] + '.service', str(Path(args[-1]).parent))
        return 0

    def unit_vm_dir(self) -> str:
        unit = (self.unit_dir / worker_disk.UNIT).read_text()
        return next(line for line in unit.splitlines() if line.startswith('ExecStart=')).split()[-1][:-len('/boot.sh')]

    def capture(self, args, **kw):
        self.calls.append([str(a) for a in args])
        if args[0] == 'qemu-img':
            return json.dumps(self.chain)
        return 'active\n' if args[-1] in self.active else 'inactive\n'


def sparse_file(path: Path, size: int, data: bytes = b'qcow2-synthetic'):
    with open(path, 'wb') as handle:
        handle.write(data)
        handle.truncate(size)


def make_source(root: Path) -> Path:
    source = root / 'old-worktree/data/production-worker'
    (source / 'keys').mkdir(parents=True)
    source.chmod(0o700)
    sparse_file(source / 'worker.qcow2', 64 << 20)
    (source / 'base.qcow2').write_bytes(b'base')
    (source / 'base.qcow2').chmod(0o400)
    (source / 'seed.iso').write_bytes(b'seed')
    (source / 'boot.sh').write_text('#!/usr/bin/env bash\nexit 0\n')
    (source / 'keys/admin').write_text('synthetic key')
    (source / 'keys/admin').chmod(0o600)
    return source


def snapshot(*roots: Path) -> dict:
    """Every path, mode and content hash under the roots (the lock's parent included)."""
    seen = {}
    for root in roots:
        if root.exists():
            for dirpath, dirnames, filenames in os.walk(root):
                for name in dirnames + filenames:
                    path = Path(dirpath, name)
                    body = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else 'dir'
                    seen[str(path)] = (path.lstat().st_mode, body)
    return seen


@pytest.fixture
def world(tmp_path):
    source = make_source(tmp_path)
    proc = tmp_path / 'proc'
    proc.mkdir()
    machine = FakeMachine(proc)
    machine.unit_dir = tmp_path / 'units'
    machine.spawn(4242, [a.replace(OLD, str(source)) for a in LIVE_ARGV])
    runs = {'active': 0}
    host = Host(deploy_config(tmp_path), machine, sleep=lambda s: None, active_runs=lambda path: runs['active'])
    mover = Mover(host, tmp_path / 'share/gmail-search/worker-vm', tmp_path / 'runtime/land.lock',
                  unit_dir=machine.unit_dir, proc_root=proc, sleep=lambda s: None,
                  stop_seconds=10, health_seconds=10)
    return mover, machine, source, runs, tmp_path


def leftovers(mover):
    return sorted(p.name for p in mover.vm_dir.parent.iterdir() if p.name != mover.vm_dir.name)


def changed_calls(machine):
    return [c for c in machine.calls if c[:3] != ['systemctl', '--user', 'is-active'] and c[0] != 'qemu-img']


# ── rendering ────────────────────────────────────────────────────────
def test_rendered_argv_is_the_live_one_with_the_directory_swapped():
    assert worker_disk.qemu_argv(Path(OLD), '127.0.0.1', 22093) == LIVE_ARGV


def test_boot_script_is_valid_bash_and_execs_the_argv(tmp_path):
    script = worker_disk.render_boot_script(Path('/srv/new/worker-vm'), '127.0.0.1', 22093)
    (tmp_path / 'boot.sh').write_text(script)
    subprocess.run(['bash', '-n', tmp_path / 'boot.sh'], check=True)
    assert 'cd /srv/new/worker-vm\n' in script and 'sha512sum --check --status base.sha512' in script
    assert 'exec ' + ' '.join(worker_disk.qemu_argv(Path('/srv/new/worker-vm'), '127.0.0.1', 22093)) in script


def test_unit_starts_boot_script_and_stops_by_guest_poweroff(tmp_path):
    unit = worker_disk.render_unit(Path('/srv/new/worker-vm'), deploy_config(tmp_path))
    assert 'ExecStart=/bin/bash /srv/new/worker-vm/boot.sh' in unit
    stop = next(line for line in unit.splitlines() if line.startswith('ExecStop='))
    assert f'-i {tmp_path}/key' in stop and '-p 22093 worker-admin@127.0.0.1 sudo -n systemctl poweroff' in stop
    assert 'kill -0 $$MAINPID' in stop and 'WantedBy=default.target' in unit


@pytest.mark.parametrize('bad', ['/srv/with space', "/srv/q'uote", '/srv/%h', '/srv/$HOME', 'relative/dir', '-oX'])
def test_rendering_refuses_paths_that_are_not_plain(bad):
    with pytest.raises(MoveRefused, match='only plain paths'):
        worker_disk.render_boot_script(Path(bad), '127.0.0.1', 22093)


@pytest.mark.parametrize('field', ['user', 'host'])
def test_an_ssh_destination_that_reads_as_an_option_is_refused(tmp_path, field):
    config = deploy_config(tmp_path)
    worker = dataclasses.replace(config.worker, **{field: '-oProxyCommand=touch'})
    with pytest.raises(MoveRefused, match='only plain paths'):
        worker_disk.render_unit(Path('/srv/new/worker-vm'), dataclasses.replace(config, worker=worker))


# ── finding the VM ───────────────────────────────────────────────────
def test_find_vm_matches_the_name_exactly(world):
    mover, machine, source, _, _ = world
    machine.spawn(5000, ['/usr/bin/qemu-system-x86_64', '-name', 'gmail-synthetic-worker'])
    machine.spawn(5001, ['/bin/bash', '-c', 'echo -name gmail-production-worker'])
    pid, argv = worker_disk.find_vm(mover.proc_root)
    assert pid == 4242 and worker_disk.disk_dir(argv) == source


@pytest.mark.parametrize('exe', ['/tmp/not-qemu', worker_disk.QEMU + ' (deleted)'])
def test_find_vm_checks_the_executable(world, exe):
    mover, machine, _, _, _ = world
    machine.kill(4242)
    machine.spawn(4242, LIVE_ARGV, exe=exe)
    found = worker_disk.find_vm(mover.proc_root)
    assert (found is not None) == exe.endswith('(deleted)')  # an upgraded qemu still counts


def test_a_reused_pid_is_not_the_vm(world):
    mover, machine, _, _, _ = world
    argv = worker_disk.find_vm(mover.proc_root)[1]
    assert mover.vm_alive(4242, argv)
    machine.kill(4242)
    machine.spawn(4242, ['/bin/sleep', '100'])
    assert not mover.vm_alive(4242, argv)


def test_two_vms_with_the_name_refuse(world):
    mover, machine, _, _, _ = world
    machine.spawn(5000, LIVE_ARGV)
    with pytest.raises(MoveRefused, match='2 qemu processes'):
        worker_disk.find_vm(mover.proc_root)


# ── verification ─────────────────────────────────────────────────────
def copy_of(source: Path, dst: Path) -> Path:
    subprocess.run(['cp', '-a', '--sparse=always', source, dst], check=True)
    return dst


def test_a_faithful_sparse_copy_verifies(tmp_path):
    source = make_source(tmp_path)
    dst = copy_of(source, tmp_path / 'copy')
    assert worker_disk.verify_copy(source, dst) == []
    assert (dst / 'worker.qcow2').stat().st_blocks * 512 < 1 << 20


def test_same_size_different_bytes_is_caught(tmp_path):
    source = make_source(tmp_path)
    dst = copy_of(source, tmp_path / 'copy')
    with open(dst / 'worker.qcow2', 'r+b') as handle:
        handle.write(b'X')
    assert worker_disk.verify_copy(source, dst) == ['worker.qcow2: sha256 differs']


def test_mode_missing_extra_and_size_are_caught(tmp_path):
    source = make_source(tmp_path)
    dst = copy_of(source, tmp_path / 'copy')
    (dst / 'keys/admin').chmod(0o644)
    (dst / 'seed.iso').unlink()
    (dst / 'stray').write_text('x')
    (dst / 'boot.sh').write_text('longer than before, same name\n')
    assert worker_disk.verify_copy(source, dst) == [
        'seed.iso: missing in copy', 'stray: not in source', 'boot.sh: size 30 != 27', 'keys/admin: type or mode differs']


def test_a_copy_that_lost_sparseness_is_caught(tmp_path):
    source = make_source(tmp_path)
    dst = copy_of(source, tmp_path / 'copy')
    (dst / 'worker.qcow2').write_bytes((source / 'worker.qcow2').read_bytes())  # fully allocated
    assert worker_disk.verify_copy(source, dst) == [
        f'worker.qcow2: copy is not sparse ({(dst / "worker.qcow2").stat().st_blocks * 512} bytes allocated, '
        f'source {(source / "worker.qcow2").stat().st_blocks * 512})']


# ── dry run and refusals ─────────────────────────────────────────────
def test_dry_run_writes_nothing_and_runs_no_command(world, capsys):
    mover, machine, source, _, root = world
    before = snapshot(root)
    mover.move(None, dry_run=True)
    assert snapshot(root) == before
    assert changed_calls(machine) == []
    out = capsys.readouterr().out
    assert f'source {source}' in out and 'rendered command line equals the running one' in out
    assert f'ExecStart=/bin/bash {mover.vm_dir}/boot.sh' in out


def test_active_runs_refuse_before_anything_changes(world):
    mover, machine, _, runs, root = world
    runs['active'] = 2
    before = snapshot(root)
    with pytest.raises(MoveRefused, match='2 run'):
        mover.move(None, dry_run=False)
    assert snapshot(root) == before and changed_calls(machine) == [] and machine.alive(4242)


def test_a_live_argv_this_tool_would_not_render_refuses(world):
    mover, machine, source, _, _ = world
    machine.kill(4242)
    machine.spawn(4242, [a.replace(OLD, str(source)) for a in LIVE_ARGV] + ['-snapshot'])
    with pytest.raises(MoveRefused, match="differs in \\['-snapshot'\\]"):
        mover.move(None, dry_run=True)


def test_a_target_this_tool_did_not_make_refuses(world):
    mover, machine, _, _, _ = world
    mover.vm_dir.mkdir(parents=True)
    with pytest.raises(MoveRefused, match='not made by this tool'):
        mover.move(None, dry_run=True)
    assert changed_calls(machine) == []


@pytest.mark.parametrize('where', ['target inside source', 'source inside target'])
def test_nested_source_and_target_refuse(world, where):
    mover, machine, source, _, _ = world
    mover.vm_dir = source / 'worker-vm' if where == 'target inside source' else source.parent
    if where == 'source inside target':
        (source.parent / '.moved-from').write_text(json.dumps({'source': str(source)}))
    with pytest.raises(MoveRefused, match='must not contain each other'):
        mover.move(None, dry_run=False)
    assert changed_calls(machine) == [] and machine.alive(4242)


def test_a_group_writable_source_refuses(world):
    mover, machine, source, _, _ = world
    source.chmod(0o770)
    with pytest.raises(MoveRefused, match='group/world-writable'):
        mover.move(None, dry_run=True)


@pytest.mark.parametrize('image', [
    {'filename': 'worker.qcow2', 'backing-filename': '/elsewhere/base.qcow2'},
    {'filename': 'worker.qcow2', 'backing-filename': '../base.qcow2'},
    {'filename': 'base.qcow2', 'backing-filename': '/elsewhere/root.qcow2'},
    {'filename': 'base.qcow2', 'format-specific': {'type': 'qcow2', 'data': {'data-file': 'base.raw'}}},
])
def test_a_disk_chain_reaching_outside_the_directory_refuses(world, image):
    mover, machine, _, _, _ = world
    machine.chain = [machine.chain[0], image] if image['filename'] == 'base.qcow2' else [image]
    with pytest.raises(MoveRefused, match='only a backing file in the same directory'):
        mover.move(None, dry_run=True)


def test_a_symlink_in_the_source_refuses(world):
    mover, _, source, _, _ = world
    (source / 'pinned').symlink_to('/elsewhere/assets')
    with pytest.raises(MoveRefused, match="symlinks.*\\['pinned'\\]"):
        mover.move(None, dry_run=True)


def test_too_little_space_refuses(world):
    mover, _, _, _, _ = world
    mover.free_bytes = lambda path: 1
    with pytest.raises(MoveRefused, match='bytes free'):
        mover.move(None, dry_run=True)


def test_a_held_landing_lock_refuses(world):
    mover, machine, _, _, _ = world
    mover.land_lock.mkdir(parents=True)
    (mover.land_lock / 'owner').write_text('land #1\n')  # no pid: a live holder
    with pytest.raises(RuntimeError, match='landing lock held'):
        mover.move(None, dry_run=False)
    assert changed_calls(machine) == [] and machine.alive(4242)


# ── the move ─────────────────────────────────────────────────────────
def test_move_copies_verifies_switches_and_leaves_the_source_alone(world):
    mover, machine, source, _, _ = world
    before = snapshot(source)
    mover.move(None, dry_run=False)
    assert snapshot(source) == before
    marker = json.loads((mover.vm_dir / '.moved-from').read_text())
    assert marker['source'] == str(source) and marker['files'] == 6
    assert (mover.vm_dir / 'keys/admin').read_text() == 'synthetic key' and (mover.vm_dir / 'keys/admin').stat().st_mode & 0o777 == 0o600
    assert f'cd {mover.vm_dir}\n' in (mover.vm_dir / 'boot.sh').read_text()
    running = worker_disk.find_vm(mover.proc_root)
    assert worker_disk.disk_dir(running[1]) == mover.vm_dir and not machine.alive(4242)
    steps = [c[:4] for c in changed_calls(machine) if c[0] != 'ssh' or 'poweroff' in c[-1]]
    assert [s[0] if s[0] != 'systemctl' else ' '.join(s[2:4]) for s in steps] == [
        'ssh', 'cp', 'daemon-reload', 'enable gmail-production-worker.service', 'start gmail-production-worker.service']
    assert not mover.land_lock.exists() and leftovers(mover) == []


def test_move_is_idempotent(world, capsys):
    mover, machine, _, _, _ = world
    mover.move(None, dry_run=False)
    machine.calls.clear()
    mover.move(None, dry_run=False)
    assert 'already moved' in capsys.readouterr().out and changed_calls(machine) == []


def test_resume_after_the_stop_uses_the_verified_copy(world):
    mover, machine, source, _, _ = world
    mover.host.worker_ssh('sudo -n systemctl poweroff', check=False)  # stopped, then the tool died
    mover.copy(source)
    machine.calls.clear()
    mover.move(None, dry_run=False)
    assert [c[0] for c in changed_calls(machine)].count('cp') == 0
    assert worker_disk.disk_dir(worker_disk.find_vm(mover.proc_root)[1]) == mover.vm_dir


def test_a_kept_copy_that_no_longer_matches_is_not_booted(world):
    mover, machine, source, _, _ = world
    machine.kill(4242)
    mover.copy(source)
    with open(mover.vm_dir / 'worker.qcow2', 'r+b') as handle:
        handle.write(b'damaged')
    with pytest.raises(RuntimeError, match='no longer matches.*worker.qcow2: sha256 differs'):
        mover.move(source, dry_run=False)
    assert not any(c[:3] == ['systemctl', '--user', 'start'] for c in machine.calls)
    assert worker_disk.disk_dir(worker_disk.find_vm(mover.proc_root)[1]) == source


def test_a_boot_sh_symlink_in_a_kept_copy_is_replaced_not_followed(world):
    mover, machine, source, _, _ = world
    machine.kill(4242)
    mover.copy(source)
    disk = hashlib.sha256((mover.vm_dir / 'worker.qcow2').read_bytes()).hexdigest()
    (mover.vm_dir / 'boot.sh').unlink()
    (mover.vm_dir / 'boot.sh').symlink_to(mover.vm_dir / 'worker.qcow2')
    mover.move(source, dry_run=False)
    assert hashlib.sha256((mover.vm_dir / 'worker.qcow2').read_bytes()).hexdigest() == disk
    boot = mover.vm_dir / 'boot.sh'
    assert not boot.is_symlink() and boot.stat().st_mode & 0o777 == 0o700 and 'exec ' in boot.read_text()


def test_rollback_refuses_an_untrusted_source(world):
    mover, machine, source, _, _ = world
    mover.move(None, dry_run=False)
    source.chmod(0o770)
    with pytest.raises(MoveRefused, match='group/world-writable'):
        mover.rollback(dry_run=False)
    assert not any(c[0] == 'systemd-run' for c in machine.calls)


def test_a_copy_the_running_source_moved_past_is_made_again(world):
    mover, machine, source, _, _ = world
    mover.copy(source)
    (mover.vm_dir / 'serial.log').write_text('stale copy was booted')
    mover.move(None, dry_run=False)
    assert [c[0] for c in machine.calls].count('cp') == 2
    assert not (mover.vm_dir / 'serial.log').exists() and leftovers(mover) == []


def test_a_changed_source_file_other_than_the_disk_is_copied_again(world):
    mover, machine, source, _, _ = world
    machine.kill(4242)
    mover.copy(source)
    (source / 'seed.iso').write_bytes(b'reseeded')
    mover.move(source, dry_run=False)
    assert [c[0] for c in machine.calls].count('cp') == 2
    assert (mover.vm_dir / 'seed.iso').read_bytes() == b'reseeded'


def test_directories_the_tool_did_not_make_are_left_alone(world):
    mover, _, _, _, _ = world
    mover.vm_dir.parent.mkdir(parents=True)
    for name in ('worker-vm.partial', 'worker-vm.partial-old', 'worker-vm.stale'):
        (mover.vm_dir.parent / name).mkdir()
        (mover.vm_dir.parent / name / 'keep').write_text('x')
    mover.move(None, dry_run=False)
    assert leftovers(mover) == ['worker-vm.partial', 'worker-vm.partial-old', 'worker-vm.stale']


def test_restore_never_starts_a_second_vm(world):
    mover, machine, source, _, _ = world
    machine.healthy, machine.stop_works = False, False
    with pytest.raises(RuntimeError, match='back on .*: NO'):
        mover.move(None, dry_run=False)
    assert not any(c[0] == 'systemd-run' for c in machine.calls) and len(machine.pids()) == 1


def test_unhealthy_after_start_boots_the_source_again(world):
    mover, machine, source, _, _ = world
    machine.healthy = False
    with pytest.raises(RuntimeError, match=f'the VM is back on {source}: NO'):
        mover.move(None, dry_run=False)  # the source is unhealthy too in this fake
    assert ['systemctl', '--user', 'disable', '--now', worker_disk.UNIT] in machine.calls
    legacy = [c for c in machine.calls if c[0] == 'systemd-run']
    assert legacy == [['systemd-run', '--user', '--unit=gmail-production-worker-legacy', '--collect',
                       '/bin/bash', str(source / 'boot.sh')]]
    assert worker_disk.disk_dir(worker_disk.find_vm(mover.proc_root)[1]) == source


def test_a_bad_copy_never_replaces_anything_and_the_source_boots_again(world, monkeypatch):
    mover, machine, source, _, _ = world
    monkeypatch.setattr(worker_disk, 'verify_copy', lambda a, b: ['worker.qcow2: sha256 differs'])
    with pytest.raises(RuntimeError, match=f'copy did not verify.*back on {source}: yes'):
        mover.move(None, dry_run=False)
    assert not mover.vm_dir.exists() and leftovers(mover) == []
    assert worker_disk.disk_dir(worker_disk.find_vm(mover.proc_root)[1]) == source


def test_rollback_boots_the_recorded_source_and_keeps_the_copy(world):
    mover, machine, source, _, _ = world
    mover.move(None, dry_run=False)
    mover.rollback(dry_run=False)
    assert worker_disk.disk_dir(worker_disk.find_vm(mover.proc_root)[1]) == source
    assert (mover.vm_dir / 'worker.qcow2').exists()


def test_rollback_dry_run_and_without_a_marker(world, capsys):
    mover, machine, _, _, _ = world
    with pytest.raises(MoveRefused, match='nothing to roll back'):
        mover.rollback(dry_run=True)
    mover.move(None, dry_run=False)
    machine.calls.clear()
    mover.rollback(dry_run=True)
    assert 'would stop and disable' in capsys.readouterr().out and changed_calls(machine) == []


# ── config ───────────────────────────────────────────────────────────
def test_vm_dir_is_read_and_expanded(tmp_path):
    config = load_config(write_config(tmp_path))
    assert config.worker.vm_dir == Path('~/.local/share/gmail-search/worker-vm').expanduser()


def test_a_config_without_vm_dir_still_loads_and_the_mover_names_the_key(tmp_path):
    path = write_config(tmp_path)
    data = json.loads(path.read_text())
    del data['worker']['vm_dir']
    path.write_text(json.dumps(data))
    config = load_config(path)
    assert config.worker.vm_dir is None
    with pytest.raises(MoveRefused, match='worker.vm_dir'):
        worker_disk.mover_for(config, runner=None)
