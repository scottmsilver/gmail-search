"""Exact public runtime input checks; no VM, dependency installs, or root needed."""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


SOURCE = Path(__file__).parents[1] / 'deploy/public/worker/verify_pi_mcp_runtime_inputs.py'


def verifier():
    assert SOURCE.exists(), 'Runtime builder must verify actual executable input bytes'
    spec = importlib.util.spec_from_file_location('verify_pi_mcp_runtime_inputs', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def inputs(tmp_path):
    runtime, packages = tmp_path / 'runtime', tmp_path / 'packages'
    for path, content in {
        runtime / 'bin/node': b'node',
        runtime / 'lib/pi/dist/cli.js': b'qualified CLI',
        packages / 'node_modules/adapter/index.ts': b'qualified adapter',
        packages / 'package.json': b'{}',
        packages / 'package-lock.json': b'{"lockfileVersion":3}',
    }.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (runtime / 'bin/node').chmod(0o755)
    return runtime, packages


@pytest.mark.parametrize('target', ['cli', 'adapter'])
def test_changed_executable_bytes_rejected_with_unchanged_package_metadata(inputs, target):
    module = verifier()
    runtime, packages = inputs
    expected = module.fingerprints(runtime, packages, trusted=False)
    path = runtime / 'lib/pi/dist/cli.js' if target == 'cli' else packages / 'node_modules/adapter/index.ts'
    path.write_bytes(b'changed executable')
    with pytest.raises(ValueError, match='fingerprint'):
        module.verify(runtime, packages, expected, trusted=False)


@pytest.mark.parametrize('mutation', ['extra', 'missing', 'executable'])
def test_tree_membership_and_executable_modes_are_verified(inputs, mutation):
    module = verifier()
    runtime, packages = inputs
    expected = module.fingerprints(runtime, packages, trusted=False)
    if mutation == 'extra':
        (packages / 'node_modules/extra.js').write_text('injected')
    elif mutation == 'missing':
        (runtime / 'lib/pi/dist/cli.js').unlink()
    else:
        (runtime / 'bin/node').chmod(0o644)
    with pytest.raises(ValueError, match='fingerprint'):
        module.verify(runtime, packages, expected, trusted=False)


@pytest.mark.parametrize('target', ['../../outside', '/etc/passwd', '../escape/secret'])
def test_symlink_escape_rejected_before_reading_target(inputs, target):
    module = verifier()
    runtime, packages = inputs
    (packages / 'node_modules/link').symlink_to(target)
    with pytest.raises(ValueError, match='symlink'):
        module.fingerprints(runtime, packages, trusted=False)


def test_internal_symlink_is_covered_and_snapshot_has_same_content(inputs, tmp_path):
    module = verifier()
    runtime, packages = inputs
    (packages / 'node_modules/link').symlink_to('adapter/index.ts')
    expected = module.fingerprints(runtime, packages, trusted=False)
    snapshot = tmp_path / 'snapshot'
    module.snapshot(runtime, packages, snapshot)
    module.verify(snapshot, snapshot / 'pi-pkgs', expected, trusted=False)
    assert os.readlink(snapshot / 'pi-pkgs/node_modules/link') == 'adapter/index.ts'
    (snapshot / 'lib/pi/dist/cli.js').write_text('changed during copy')
    with pytest.raises(ValueError, match='fingerprint'):
        module.verify(snapshot, snapshot / 'pi-pkgs', expected, trusted=False)


@pytest.mark.parametrize('uid,mode', [(1000, 0o100644), (0, 0o100664), (0, 0o100666), (0, 0o104755)])
def test_untrusted_owner_write_permissions_and_special_bits_rejected(uid, mode):
    module = verifier()
    with pytest.raises(ValueError, match='trusted'):
        module.trusted_metadata(SimpleNamespace(st_uid=uid, st_mode=mode), 'fixture')


def test_root_owned_private_and_readonly_metadata_accepted():
    module = verifier()
    for mode in (0o100644, 0o100755, 0o40700, 0o40755, 0o120777):
        module.trusted_metadata(SimpleNamespace(st_uid=0, st_mode=mode), 'fixture')


def test_special_files_rejected_without_opening(inputs):
    module = verifier()
    runtime, packages = inputs
    os.mkfifo(packages / 'node_modules/pipe')
    with pytest.raises(ValueError, match='file type'):
        module.fingerprints(runtime, packages, trusted=False)


def test_default_verification_rejects_user_owned_inputs(inputs):
    module = verifier()
    if os.getuid() == 0:
        pytest.skip('Requires ordinary unprivileged test user')
    with pytest.raises(ValueError, match='trusted'):
        module.fingerprints(*inputs)


def test_symlink_chain_escape_rejected(inputs, tmp_path):
    module = verifier()
    runtime, packages = inputs
    (tmp_path / 'external').write_text('outside')
    (packages / 'node_modules/adapter/link').symlink_to('../../../external')
    (packages / 'node_modules/link').symlink_to('adapter/link')
    with pytest.raises(ValueError, match='symlink'):
        module.fingerprints(runtime, packages, trusted=False)


def test_symlink_input_root_rejected(inputs):
    module = verifier()
    runtime, packages = inputs
    (runtime / 'alias').symlink_to('lib', target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        module.tree_fingerprint(runtime / 'alias', trusted=False)
