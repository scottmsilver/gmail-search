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
