"""Validate fixed synthetic configuration without launching a VMM or using root."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('firecracker_backend', Path(__file__).parents[1]/'deploy/public/worker/firecracker_backend.py')
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)


def test_backend_refuses_development_host():
    with pytest.raises(RuntimeError, match='clean synthetic'):
        backend.FirecrackerBackend()


def test_fixed_machine_has_readonly_root_and_no_network_device():
    config = backend.fixed_config({'memory_mib': 256, 'vcpus': 1})
    assert 'network-interfaces' not in config
    assert config['vsock']['guest_cid'] == 3
    assert config['drives'][0]['is_read_only']
    assert config['boot-source']['kernel_image_path'] == '/vmlinux'


@pytest.mark.parametrize('value', ['../host', '/proc/1', 'a'*31, 'A'*32, 'x;rm', None])
def test_handles_cannot_select_paths(value):
    with pytest.raises(ValueError):
        backend.checked_handle(value)


def test_supervisor_liveness_does_not_treat_zombie_as_running():
    import os
    import time
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    try:
        deadline = time.monotonic() + 2
        while backend.process_start(pid) is not None and time.monotonic() < deadline:
            time.sleep(.01)
        assert backend.process_start(pid) is None
    finally:
        os.waitpid(pid, 0)


def test_agent_profile_adds_only_immutable_runtime_drive():
    config = backend.fixed_config({'memory_mib': 1024, 'vcpus': 1}, profile='agent')
    assert len(config['drives']) == 2
    assert all(drive['is_read_only'] for drive in config['drives'])
    assert config['drives'][1]['path_on_host'] == '/runtime.squashfs'
    assert 'network-interfaces' not in config
    with pytest.raises(ValueError):
        backend.fixed_config({'memory_mib': 1024, 'vcpus': 1}, profile='/guest/chosen')


def test_attachment_profile_has_only_fixed_readonly_parser_runtime():
    config = backend.fixed_config({'memory_mib': 768, 'vcpus': 1}, profile='attachment')
    assert config['drives'][1]['path_on_host'] == '/attachment.squashfs'
    assert len(config['drives']) == 2
    assert all(drive['is_read_only'] for drive in config['drives'])
    assert 'network-interfaces' not in config
    assert backend.SyntheticAttachmentBackend.profile == 'attachment'


def test_agent_tools_qualification_has_separate_pinned_readonly_profile():
    import re
    config=backend.fixed_config({'memory_mib':1024,'vcpus':1},profile='agent_tools')
    assert len(config['drives'])==2
    assert config['drives'][1]['path_on_host']=='/agent-tools.squashfs'
    assert all(drive['is_read_only'] for drive in config['drives'])
    assert 'network-interfaces' not in config
    assert backend.SyntheticAgentToolsBackend.profile=='agent_tools'
    assert re.fullmatch('[a-f0-9]{64}',backend.AGENT_TOOLS_PIN)
    assert backend.AGENT_TOOLS_PIN != backend.RUNTIME_PIN


def test_agent_mcp_qualification_keeps_old_pins_and_separate_readonly_drive():
    config=backend.fixed_config({'memory_mib':1024,'vcpus':1},profile='agent_mcp')
    assert len(config['drives'])==2 and config['drives'][1]['path_on_host']=='/agent-mcp.squashfs'
    assert all(drive['is_read_only'] for drive in config['drives'])
    assert 'network-interfaces' not in config
    assert backend.SyntheticAgentMCPBackend.profile=='agent_mcp'
    assert backend.AGENT_MCP_PIN not in (backend.RUNTIME_PIN,backend.AGENT_TOOLS_PIN,backend.ATTACHMENT_PIN)
    assert backend.AGENT_TOOLS_PIN=='4b981cd6a2e1eb2f28acad2d02ac6c365b4ef48b03345eb098578b9a1dd71402'


def test_agent_pi_mcp_has_its_own_fixed_readonly_profile_and_preserves_claude_pin():
    config=backend.fixed_config({'memory_mib':1024,'vcpus':1},profile='agent_pi_mcp')
    assert len(config['drives'])==2 and config['drives'][1]['path_on_host']=='/agent-pi-mcp.squashfs'
    assert all(drive['is_read_only'] for drive in config['drives'])
    assert 'network-interfaces' not in config
    assert backend.SyntheticAgentPiMCPBackend.profile=='agent_pi_mcp'
    assert backend.AGENT_MCP_PIN=='54303c8873abc96c27ea8cc99930e4713016a5d1b091376ed858a472f58ba7ba'
    assert backend.AGENT_PI_MCP_PIN not in (backend.RUNTIME_PIN,backend.ATTACHMENT_PIN,backend.AGENT_TOOLS_PIN,backend.AGENT_MCP_PIN)


def test_stop_does_not_erase_a_recorded_failure(tmp_path, monkeypatch):
    """Teardown must not destroy why the run failed.

    `supervise()` records its verdict as {'status': 'failed', 'error': ...}.
    `stop()` then wrote {'status': 'stopped'} over it unconditionally, so every
    failed guest looked identical to a clean exit -- which is why a guest that
    exited before using any capability could not be diagnosed at all: the
    production worker discards serial output by design, and the one remaining
    record was being overwritten during teardown.
    """
    path = tmp_path / 'run'
    path.mkdir()
    backend.atomic_json(path / 'state.json', {'status': 'failed', 'error': 'jailer launch failed: boom'})
    backend.finalize_stopped_state(path)
    kept = backend.read_json(path / 'state.json')
    assert kept['status'] == 'failed'
    assert 'boom' in kept['error']


def test_stop_still_marks_a_clean_run_stopped(tmp_path):
    path = tmp_path / 'run'
    path.mkdir()
    backend.atomic_json(path / 'state.json', {'status': 'running', 'pid': 1})
    backend.finalize_stopped_state(path)
    assert backend.read_json(path / 'state.json') == {'status': 'stopped'}


@pytest.mark.parametrize('cls_name', [n for n in dir(backend)
                                      if isinstance(getattr(backend, n), type)
                                      and issubclass(getattr(backend, n), backend.FirecrackerBackend)])
def test_every_backend_can_stop(cls_name):
    """Teardown is a class method, not a module function.

    The change that added finalize_stopped_state() put it at column 0 directly
    above `def stop`, which closed the class: stop() became a nested function
    and every backend silently lost it. The worker could then neither cancel nor
    reap a guest, and the manager crashed at startup trying to reap one. The
    helper's own tests passed throughout, because they never touched stop().
    """
    assert callable(getattr(getattr(backend, cls_name), 'stop', None)), f'{cls_name} has no stop()'


# ── a VMM that cannot read its own config ────────────────────────────────────

def test_the_jail_is_readable_by_the_vmm_under_a_restrictive_umask(tmp_path):
    """The VMM runs as uid 65534 once the jailer drops privileges, and the
    manager unit runs under UMask=0077. `mkdir(mode=0o755)` therefore landed as
    0700 and `write_text()` as 0600 -- both root-only -- and every guest died at
    once: "Firecracker panicked ... Unable to open or read from the
    configuration file: Permission denied". Proven by running the same launch
    under umask 022 (boots, seccomp filter in one second) and 077 (panics).
    """
    import os
    import stat

    previous = os.umask(0o077)
    try:
        config = backend.prepare_jail(tmp_path / 'jail', {'machine-config': {'vcpu_count': 1}})
    finally:
        os.umask(previous)
    assert stat.S_IMODE((tmp_path / 'jail').stat().st_mode) == 0o755
    assert stat.S_IMODE(config.stat().st_mode) == 0o644
    assert '"vcpu_count": 1' in config.read_text()


def test_a_vmm_that_dies_early_is_reported_as_dead_not_as_unfiltered():
    """An early-dying VMM is a zombie of this subreaper. Its /proc entry persists,
    still named `firecracker`, with Seccomp 0 -- so the readiness loop spent its
    whole window reading a corpse and blamed seccomp. It must say the VMM died,
    with its exit status and its own last words."""
    import subprocess
    import sys
    import time

    child = subprocess.Popen([sys.executable, '-c', "print('Firecracker panicked: nope'); raise SystemExit(3)"],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    report = None
    for _ in range(100):
        report = backend.vmm_exit_report(child.pid, child.stdout)
        if report:
            break
        time.sleep(0.02)
    assert report and 'exit code 3' in report and 'Firecracker panicked: nope' in report, report


def test_a_live_vmm_is_not_reported_as_dead():
    import subprocess

    child = subprocess.Popen(['sleep', '5'], stdout=subprocess.PIPE)
    try:
        assert backend.vmm_exit_report(child.pid, child.stdout) is None
    finally:
        child.kill(); child.wait()
