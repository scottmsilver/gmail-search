"""Production enrollment rejects synthetic/foreign hosts and mutable assets."""
import importlib
from pathlib import Path
import sys

import pytest

WORKER = Path(__file__).parents[1] / 'deploy/public/worker'
sys.path.insert(0, str(WORKER))


def module():
    return importlib.import_module('production_worker_profile')


def enrollment():
    return dict(version=1, purpose='gmail-full-agent-production',
                hostname='gmail-execution-worker', machine_id='a' * 32)


def test_closed_enrollment_accepts_only_exact_production_identity():
    p = module()
    p.validate_enrollment(enrollment(), hostname='gmail-execution-worker', machine_id='a' * 32)
    for host in ('synthetic-execution-worker', 'sukkot', 'localhost'):
        with pytest.raises(RuntimeError):
            p.validate_enrollment(enrollment(), hostname=host, machine_id='a' * 32)
    with pytest.raises(RuntimeError):
        p.validate_enrollment(enrollment(), hostname='gmail-execution-worker', machine_id='b' * 32)


@pytest.mark.parametrize('changed', [dict(version=True), dict(version=2), dict(machine_id='0'*32),
    dict(hostname='synthetic-execution-worker'), dict(image='/tmp/evil'), dict(purpose='test')])
def test_enrollment_cannot_select_paths_images_or_relax_host_binding(changed):
    data = enrollment() | changed
    with pytest.raises(RuntimeError):
        module().validate_enrollment(data, hostname=data['hostname'], machine_id=data['machine_id'])


def test_asset_verification_rejects_symlinks_writable_files_and_changed_bytes(tmp_path):
    import hashlib
    import os
    p = module()
    asset = tmp_path/'image'; asset.write_bytes(b'qualified'); asset.chmod(0o444)
    pin = hashlib.sha256(b'qualified').hexdigest()
    p.verify_asset(asset, pin, owner=os.geteuid())
    link = tmp_path/'link'; link.symlink_to(asset)
    with pytest.raises(RuntimeError): p.verify_asset(link, pin, owner=os.geteuid())
    asset.chmod(0o666)
    with pytest.raises(RuntimeError): p.verify_asset(asset, pin, owner=os.geteuid())
    asset.chmod(0o444)
    with pytest.raises(RuntimeError): p.verify_asset(asset, '0'*64, owner=os.geteuid())
    with pytest.raises(RuntimeError): p.verify_asset(asset, pin, owner=os.geteuid()+1)


def test_profile_keeps_full_image_and_qualified_isolation():
    p = module()
    import firecracker_backend as backend
    assert p.ProductionFullAgentBackend.profile == 'agent_full'
    assert p.ProductionFullAgentBackend.supervisor_module == Path('/opt/gmail-worker/production_worker_profile.py')
    assert backend.AGENT_FULL_PIN == 'ebed416c98c2dcd169508668c414929898779e25e3334f07d853a254329afaa3'
    config = backend.fixed_config(dict(memory_mib=1024,vcpus=1),profile='agent_full')
    assert 'network-interfaces' not in config
    assert all(d['is_read_only'] for d in config['drives'])
    with pytest.raises(RuntimeError, match='clean synthetic'):
        backend.SyntheticFullAgentBackend()
    with pytest.raises(RuntimeError): p.ProductionFullAgentBackend()


def test_installed_module_allowlist_imports_without_source_checkout(tmp_path):
    import shutil
    import subprocess
    import importlib.util
    spec = importlib.util.spec_from_file_location('worker_install',WORKER/'production/install.py')
    installer = importlib.util.module_from_spec(spec); spec.loader.exec_module(installer)
    for name in installer.MODULES:
        shutil.copyfile(WORKER/name,tmp_path/name)
    shutil.copyfile(WORKER.parents[2]/'src/gmail_search/gateway/full_agent_rpc.py',tmp_path/'full_agent_rpc.py')
    result = subprocess.run([sys.executable,'-I','-c',
        'import sys; sys.path.insert(0,sys.argv[1]); import production_worker_profile, full_agent_manager, full_agent_rpc_server, full_agent_rpc_frontend',str(tmp_path)],capture_output=True,text=True)
    assert result.returncode == 0, result.stderr


def test_production_supervisor_discards_guest_console_content(monkeypatch):
    p = module()
    observed = {}
    monkeypatch.setattr(p, 'boundary', lambda: None)
    monkeypatch.setattr(p.backend, 'read_json', lambda _: {'profile':'agent_full'})
    monkeypatch.setattr(p.backend, 'supervise', lambda handle, **kwargs: observed.update(kwargs))
    monkeypatch.setattr(sys, 'argv', ['production_worker_profile.py','_supervise','a'*32])
    p.main()
    assert observed['serial_output'] is False


@pytest.mark.parametrize('account,expected', [
    ('gmail-full-agent-rpc', {'disableforwarding': 'yes', 'maxsessions': '1'}),
    ('gmail-gateway-tunnel', {'allowtcpforwarding': 'remote', 'maxsessions': '0',
                            'permitlisten': '127.0.0.1:18081', 'permitopen': 'none'}),
])
def test_production_ssh_policy_parses_with_real_sshd(account, expected, tmp_path):
    import shutil
    import subprocess
    sshd = shutil.which('sshd')
    if sshd is None:
        pytest.skip('OpenSSH server is required for effective-policy validation')
    host_key = tmp_path/'host-key'
    subprocess.run(['ssh-keygen', '-q', '-t', 'ed25519', '-N', '', '-f', str(host_key)],
                   check=True)
    result = subprocess.run([
        sshd, '-T', '-h', str(host_key),
        '-f', str(WORKER/'production/sshd-worker.conf'), '-C',
        f'user={account},host=gmail-execution-worker,addr=10.0.2.2',
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    effective = dict(line.split(' ', 1) for line in result.stdout.splitlines())
    for option, value in (expected | {'permituserenvironment': 'no',
                                    'passwordauthentication': 'no'}).items():
        assert effective[option] == value


def test_ssh_policy_parses_when_included_inside_an_open_match_block(tmp_path):
    """The drop-in must survive being included at non-global scope.

    `Include` does not close an enclosing `Match`, so a fragment inherits
    whatever scope was open where it was included. Any earlier-sorting file in
    `sshd_config.d/` that leaves a `Match` open therefore decides whether this
    one parses. A global-only directive here is rejected in that scope and
    takes the *whole* server config down with it, which is what stopped worker
    enrolment on 2026-09-15:

        sshd -t failed PermitUserEnvironment inside Match

    The test above parses the fragment standalone and cannot see this; only
    inclusion under an open Match reproduces it.
    """
    import shutil
    import subprocess

    sshd = shutil.which('sshd')
    if sshd is None:
        pytest.skip('OpenSSH server is required for effective-policy validation')
    host_key = tmp_path / 'host-key'
    subprocess.run(['ssh-keygen', '-q', '-t', 'ed25519', '-N', '', '-f', str(host_key)], check=True)
    main = tmp_path / 'sshd_config'
    main.write_text(
        f'HostKey {host_key}\n'
        'Match User someone-else\n'
        '    PermitTTY no\n'
        f'Include {WORKER / "production/sshd-worker.conf"}\n'
    )
    result = subprocess.run([sshd, '-t', '-f', str(main)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
