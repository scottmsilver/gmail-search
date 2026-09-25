"""Activation, rollback, postcheck and dry runs against a throwaway install
tree, with fake systemctl/ssh runners: nothing live is touched."""
import json
from pathlib import Path

import pytest

from gmail_search.deploy import phases
from gmail_search.deploy.__main__ import phases_for
from gmail_search.deploy.config import DeployConfig, WorkerAccess
from gmail_search.deploy.host import Host
from gmail_search.deploy.runner import CommandFailed


class FakeRunner:
    def __init__(self, fail_when=None):
        self.calls, self.fail_when = [], fail_when or (lambda args: False)

    def run(self, args, **kw):
        args = [str(a) for a in args]
        self.calls.append(args)
        if self.fail_when(args):
            raise CommandFailed(args, 1)
        return 0

    def capture(self, args, **kw):
        self.calls.append([str(a) for a in args])
        return 'active\n'


def config(root: Path) -> DeployConfig:
    return DeployConfig(
        install_root=root, registry=root / 'registry.sqlite', public_web_env=root / 'web.env',
        public_host='gms.example', web_node_modules=root / 'nm',
        web_local_url='http://127.0.0.1:3001', health_host='127.0.0.1', health_ports=(8091, 3001),
        image_inputs_cache=root / 'inputs',
        services={'controller': 'api.service', 'web': 'web.service', 'manager': 'mgr.service',
                  'clock_sync': 'clock.service'},
        worker=WorkerAccess(host='127.0.0.1', port=22093, user='worker-admin', key=root / 'key',
                            known_hosts=root / 'kh', opt_dir='/opt/gmail-worker',
                            image_path='/var/lib/gmail-worker/images/agent-full.squashfs'))


@pytest.fixture
def install(tmp_path):
    releases = tmp_path / 'public-releases'
    for name in ('old', 'new'):
        (releases / name).mkdir(parents=True)
    for link in ('invited-current', 'public-current'):
        (tmp_path / link).symlink_to(releases / 'old')
    state = phases.State(tmp_path / 'state/state.json')
    state.set(release='new', target='f' * 40, kinds=['controller'], releaseDir=str(releases / 'new'),
              runningDir=str(releases / 'old'))
    return tmp_path, state


def host_for(root, runner, *, healthy=True, http=None):
    return Host(config(root), runner, port_open=lambda h, p: healthy, sleep=lambda s: None, health_seconds=10,
                http_status=http or (lambda url, host: 404 if url.endswith('bad!') else 200),
                active_runs=lambda path: 0)


def current(root):
    return {Path(p).name: (root / p).resolve().name for p in ('invited-current', 'public-current')}


def test_activate_swaps_both_links_and_restarts(install):
    root, state = install
    runner = FakeRunner()
    phases.activate_phase(host_for(root, runner), state)
    assert current(root) == {'invited-current': 'new', 'public-current': 'new'}
    assert ['systemctl', '--user', 'restart', 'api.service', 'web.service'] in runner.calls
    assert not any(call[0] == 'ssh' for call in runner.calls)


def test_unhealthy_after_swap_rolls_back_and_says_so(install):
    root, state = install
    ports = {'up': False}

    def port_open(host, port):
        return ports['up']

    runner = FakeRunner()
    host = host_for(root, runner)
    host.port_open = port_open
    original = host.wait_healthy
    calls = {'n': 0}

    def wait():
        calls['n'] += 1
        ports['up'] = calls['n'] > 1  # the new release never comes up; the old one does
        return original()

    host.wait_healthy = wait
    with pytest.raises(RuntimeError, match='rollback succeeded'):
        phases.activate_phase(host, state)
    assert current(root) == {'invited-current': 'old', 'public-current': 'old'}


def test_worker_failure_restores_the_worker_and_the_links(install):
    root, state = install
    state.set(kinds=['worker'], workerFiles=['full_agent_manager.py'], image=None)
    (state.path.parent / 'worker').mkdir()
    (state.path.parent / 'worker/full_agent_manager.py').write_text('x')
    runner = FakeRunner(fail_when=lambda args: args[0] == 'ssh' and 'install -o root' in args[-1])
    with pytest.raises(RuntimeError, match='rollback succeeded'):
        phases.activate_phase(host_for(root, runner), state)
    restore = [c for c in runner.calls if c[0] == 'ssh' and '.prev' in c[-1] and 'mv -f' in c[-1]]
    assert restore, 'worker files were not restored'
    assert current(root)['invited-current'] == 'old'


def test_worker_install_script_quotes_and_keeps_a_previous_copy(install):
    root, _ = install
    script = phases.worker_install_script(config(root), ['full_agent_manager.py'], True, '/tmp/gms-deploy-x')
    assert script.startswith('sudo -n bash -c ')
    for needle in ('agent-full.squashfs.prev', 'full_agent_manager.py.prev', 'systemctl restart mgr.service'):
        assert needle in script


def test_failed_postcheck_leaves_the_release_up_and_prints_rollback(install):
    root, state = install
    state.set(activatedAt='t')
    host = host_for(root, FakeRunner(), http=lambda url, host: 500)
    with pytest.raises(RuntimeError, match='--phase rollback --release new'):
        phases.postcheck_phase(host, root, state)
    record = json.loads((root / '.runtime/issue-loop/deploys.jsonl').read_text())
    assert record['result'] == 'activated-postcheck-failed'


def test_postcheck_passes_and_records(install):
    root, state = install
    state.set(activatedAt='t')
    phases.postcheck_phase(host_for(root, FakeRunner()), root, state)
    record = json.loads((root / '.runtime/issue-loop/deploys.jsonl').read_text())
    assert record['result'] == 'deployed' and record['http']['/c/bad!'] == 404


def test_dry_run_never_activates():
    assert phases_for('all', True) == ['plan', 'package', 'qualify', 'preflight']
    with pytest.raises(SystemExit):
        phases_for('activate', True)


def test_dry_run_packages_outside_the_live_releases(install):
    root, _ = install
    host = host_for(root, FakeRunner())
    assert phases.release_dir(host, root / 'main', 'x', True) == root / 'main/.runtime/deploy/dry-run/x'
    assert phases.release_dir(host, root / 'main', 'x', False) == root / 'public-releases/x'


def test_preflight_refuses_active_runs(install):
    root, state = install
    state.set(worktree=str(root / 'no-tree'), qualifiedAt='t')
    host = host_for(root, FakeRunner())
    host.active_runs = lambda path: 2
    with pytest.raises(RuntimeError, match='2 run'):
        phases.preflight_phase(host, root, state)
