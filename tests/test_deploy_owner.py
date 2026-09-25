"""The owner track (#52): plan, package, activate, rollback and postcheck for
serve, mcp, the owner web and supervise, and the one-time cutover of their
unit files. A throwaway repo, install tree and unit directory, with a fake
systemd: nothing live is touched."""
from datetime import date
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from gmail_search.deploy import checkout_checks, owner_cutover, phases, plan
from gmail_search.deploy.config import DeployConfig, OwnerTrack, WorkerAccess
from gmail_search.deploy.host import Host
from gmail_search.deploy.owner_units import OWNER_UNITS, WATCHDOG_TIMER, Machine
from gmail_search.deploy.runner import Runner, without_git_env

REPO_ROOT = Path(__file__).resolve().parents[1]
PORTS = {'gmail-search-serve.service': 8090, 'gmail-search-mcp.service': 7878, 'gmail-search-web.service': 3000}
HEALTHY = {8090: 200, 7878: 401, 3000: 200}
SCHEMA = 'CREATE TABLE t();\n'


def blob_of(text: str) -> str:
    data = text.encode()
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def checks_source(reviewed: str) -> str:
    return f"REVIEWED_SCHEMA_BLOB = '{reviewed}'\n"


def test_main_declares_its_own_schema_reviewed():
    """Every owner daemon runs pg_schema.sql on the live database at boot. A
    change to it lands only together with REVIEWED_SCHEMA_BLOB, i.e. with a
    stated review against the live catalog."""
    schema = (REPO_ROOT / checkout_checks.SCHEMA_PATH).read_bytes()
    blob = hashlib.sha1(b'blob %d\0' % len(schema) + schema).hexdigest()
    assert blob == checkout_checks.REVIEWED_SCHEMA_BLOB, (
        f'{checkout_checks.SCHEMA_PATH} changed: review what it runs at boot against the live catalog, '
        f'then set REVIEWED_SCHEMA_BLOB to {blob}')
    assert plan.schema_reviewed(blob, (REPO_ROOT / phases.CHECKS_PATH).read_text())


def test_the_runner_drops_what_picks_a_repository_and_keeps_transport():
    env = {'GIT_DIR': '/r/.git', 'GIT_INDEX_FILE': '/r/.git/index', 'GIT_WORK_TREE': '/r', 'GIT_CONFIG_COUNT': '1',
           'GIT_CONFIG_KEY_0': 'core.bare', 'GIT_CONFIG_VALUE_0': 'true', 'GIT_SSH_COMMAND': 'ssh -i k', 'PATH': '/bin'}
    assert without_git_env(env) == {'GIT_SSH_COMMAND': 'ssh -i k', 'PATH': '/bin'}


# ── plan: pure ───────────────────────────────────────────────────────
@pytest.mark.parametrize('paths, owner', [
    (['src/gmail_search/agents/mcp_tools_server.py'], True),
    (['web/app/page.tsx'], True),
    (['templates/index.html'], True),
    (['scripts/serve_watchdog.sh'], True),
    (['docs/x.md', 'tests/test_x.py', 'scripts/other.sh', 'web/scripts/test-x.mjs'], False),
    (['deploy/public/worker/full_agent_manager.py'], False),
])
def test_what_the_owner_daemons_run(paths, owner):
    assert plan.classify_owner(paths) is owner


def test_owner_paths_are_the_owner_tracks_business_only_when_it_exists():
    assert plan.classify(['templates/index.html'], []).manual == ['templates/index.html']
    batch = plan.classify(['templates/index.html'], [], owner_track=True)
    assert batch.manual == [] and batch.kinds == set()


def test_schema_reviewed_needs_the_commits_own_declaration():
    blob = blob_of(SCHEMA)
    assert plan.schema_reviewed(blob, 'x = 1\n' + checks_source(blob))
    assert not plan.schema_reviewed(blob, checks_source('0' * 40))
    assert not plan.schema_reviewed(blob, '')


# ── a repo, an install tree, a fake systemd ──────────────────────────
def git(cwd, *args) -> str:
    return subprocess.run(['git', '-C', str(cwd), *args], check=True, capture_output=True, text=True).stdout.strip()


def commit(repo: Path, files: dict) -> str:
    for rel, text in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)
    git(repo, 'add', '-A')
    git(repo, '-c', 'user.name=t', '-c', 'user.email=t@example.invalid', 'commit', '-qm', 'c')
    return git(repo, 'rev-parse', 'HEAD')


class Systemd:
    """Owner units with pids, ports and answers. `breaks[unit] = n`: the next
    n restarts of that unit come back answering 502."""

    def __init__(self, unit_dir: Path | None = None, answers=None, breaks=None):
        self.pid = {u: 100 + i for i, u in enumerate(OWNER_UNITS)}
        self.answers, self.breaks = {**HEALTHY, **(answers or {})}, dict(breaks or {})
        self.down, self.events, self.next_pid, self.unit_dir = set(), [], 1000, unit_dir

    def __call__(self, args):
        verb = args[2]
        if verb == 'is-active':
            return 'active\n'
        if verb in ('stop', 'start', 'restart', 'daemon-reload'):
            self.events.append(' '.join(args[2:]))
            if verb == 'restart':
                unit = args[-1]
                self.pid[unit], self.next_pid = self.next_pid, self.next_pid + 10
                (self.down.add if self.breaks.get(unit, 0) > 0 else self.down.discard)(unit)
                self.breaks[unit] = max(0, self.breaks.get(unit, 0) - 1)
            return ''
        prop, unit = args[4], args[-1]
        if prop == 'ExecStart':
            return self.exec_start(unit)
        return {'MainPID': str(self.pid.get(unit, 0)), 'ActiveState': 'active',
                'Environment': 'PATH=/node/bin:/usr/bin'}.get(prop, '')

    def exec_start(self, unit):
        files = [self.unit_dir / unit, *sorted((self.unit_dir / f'{unit}.d').glob('*.conf'))]
        lines = [ln for f in files if f.exists() for ln in f.read_text().splitlines() if ln.startswith('ExecStart=')]
        return lines[-1].split('=', 1)[1].replace('%h', str(Path.home())) if lines else ''

    def unit_of(self, pid):
        return next((u for u, p in self.pid.items() if p == pid), None)

    def listening(self, pid):
        unit = self.unit_of(pid)
        return [('127.0.0.1', PORTS[unit])] if unit in PORTS else []

    def process_tree(self, pid):
        return [pid, pid + 1] if self.unit_of(pid) == 'gmail-search-supervise.service' else [pid]

    def http_status(self, method, url):
        port = int(url.split(':')[2].split('/')[0])
        unit = next(u for u, p in PORTS.items() if p == port)
        return 502 if unit in self.down else self.answers[port]

    def restarts(self):
        return [e.split()[-1] for e in self.events if e.startswith('restart') and e.split()[-1] in OWNER_UNITS]


class FakeRunner(Runner):
    """git and tar run for real; systemctl, uv and next are faked and recorded."""

    def __init__(self, systemd, fail_when=lambda args: False):
        self.systemd, self.fail_when, self.calls = systemd, fail_when, []

    def _fake(self, args):
        args = [str(a) for a in args]
        if args[0] in ('git', 'tar'):
            return None
        self.calls.append(args)
        if self.fail_when(args):
            from gmail_search.deploy.runner import CommandFailed
            raise CommandFailed(args, 1)
        return self.systemd(args) if args[0] == 'systemctl' else ''

    def run(self, args, **kw):
        out = self._fake(args)
        return super().run(args, **kw) if out is None else 0

    def capture(self, args, **kw):
        out = self._fake(args)
        return super().capture(args, **kw) if out is None else out


def config(root: Path) -> DeployConfig:
    return DeployConfig(
        install_root=root, registry=root / 'registry.sqlite', public_web_env=root / 'web.env',
        public_host='gms.example', web_node_modules=root / 'nm', web_local_url='http://127.0.0.1:3001',
        health_host='127.0.0.1', health_ports=(8091,), image_inputs_cache=root / 'inputs',
        services={'controller': 'api.service', 'web': 'web.service', 'manager': 'mgr.service',
                  'clock_sync': 'clock.service'},
        worker=WorkerAccess(host='127.0.0.1', port=22, user='w', key=root / 'k', known_hosts=root / 'kh',
                            opt_dir='/opt/w', image_path='/var/w.squashfs'),
        owner=OwnerTrack(checkout=root / 'checkout', web_env_local=root / 'checkout/web/.env.local'))


def host_for(root: Path, systemd: Systemd, **runner_kw) -> Host:
    runner = FakeRunner(systemd, **runner_kw)
    m = Machine(root / 'checkout', runner, health_host='127.0.0.1', listening=systemd.listening,
                process_tree=systemd.process_tree, http_status=systemd.http_status, sleep=lambda s: None)
    return Host(config(root), runner, port_open=lambda h, p: True, sleep=lambda s: None, health_seconds=10,
                http_status=lambda url, host: 404 if url.endswith('bad!') else 200, active_runs=lambda p: 0,
                owner_units=m)


@pytest.fixture
def repo(tmp_path):
    """The checkout: base commit (what everything runs) and a target after it."""
    main = tmp_path / 'checkout'
    main.mkdir()
    git(main, 'init', '-q', '-b', 'main')
    base = commit(main, {
        'deploy/public/worker/prepare-full-agent-runtime.sh': 'for name in guest_agent.py; do\n',
        checkout_checks.SCHEMA_PATH: SCHEMA, phases.CHECKS_PATH: checks_source(blob_of(SCHEMA)),
        'src/gmail_search/agents/mcp_tools_server.py': 'v1\n', 'templates/index.html': '<p>\n',
        'web/app/page.tsx': 'v1\n', 'scripts/serve_watchdog.sh': '#!/bin/sh\n', 'uv.lock': 'lock\n'})
    return main, base


@pytest.fixture
def install(tmp_path, repo):
    """Both tracks running the base commit."""
    main, base = repo
    for track, link_names in (('public-releases', ('invited-current', 'public-current')),
                              ('owner-releases', ('owner-current',))):
        running = tmp_path / track / 'old'
        running.mkdir(parents=True)
        (running / 'QUALIFIED.json').write_text(json.dumps({'commit': base}))
        for name in link_names:
            (tmp_path / name).symlink_to(running)
    return tmp_path, main, base


def plan_for(root, main, systemd=None):
    return phases.plan_phase(host_for(root, systemd or Systemd()), main, 'HEAD', 'loop', date(2026, 9, 25),
                             fetch=False)


# ── plan: against a repo ─────────────────────────────────────────────
def test_before_the_cutover_an_mcp_change_ships_the_controller_only(install):
    root, main, _ = install
    (root / 'owner-current').unlink()
    commit(main, {'src/gmail_search/agents/mcp_tools_server.py': 'v2\n'})
    planned = plan_for(root, main)
    assert planned['action'] == 'ship' and planned['kinds'] == ['controller'] and 'ownerRunningDir' not in planned


def test_an_mcp_change_ships_the_owner_track(install):
    root, main, _ = install
    commit(main, {'src/gmail_search/agents/mcp_tools_server.py': 'v2\n'})
    planned = plan_for(root, main)
    assert planned['action'] == 'ship' and planned['kinds'] == ['controller', 'owner']
    assert planned['ownerRunningDir'] == str(root / 'owner-releases/old')


def test_a_template_change_ships_the_owner_track_alone(install):
    root, main, _ = install
    commit(main, {'templates/index.html': '<p>new\n'})
    planned = plan_for(root, main)
    assert planned['action'] == 'ship' and planned['kinds'] == ['owner']


def test_a_docs_change_leaves_the_owner_track_alone(install):
    root, main, _ = install
    commit(main, {'docs/x.md': 'x\n'})
    assert plan_for(root, main)['action'] == 'skip'


def test_an_unreviewed_schema_refuses_the_whole_deploy(install):
    root, main, _ = install
    commit(main, {checkout_checks.SCHEMA_PATH: SCHEMA + 'CREATE INDEX i ON t();\n'})
    planned = plan_for(root, main)
    assert planned['action'].startswith('refuse:') and 'REVIEWED_SCHEMA_BLOB' in planned['action']


def test_a_schema_change_that_carries_its_review_ships(install):
    root, main, _ = install
    new = SCHEMA + 'CREATE INDEX i ON t();\n'
    commit(main, {checkout_checks.SCHEMA_PATH: new, phases.CHECKS_PATH: checks_source(blob_of(new))})
    assert plan_for(root, main)['kinds'] == ['controller', 'owner']


def test_an_owner_release_the_target_does_not_contain_refuses(install):
    root, main, base = install
    git(main, 'switch', '-q', '-c', 'side')
    side = commit(main, {'src/gmail_search/x.py': 'side\n'})
    git(main, 'switch', '-q', 'main')
    commit(main, {'src/gmail_search/agents/mcp_tools_server.py': 'v2\n'})
    (root / 'owner-releases/old/QUALIFIED.json').write_text(json.dumps({'commit': side}))
    assert plan_for(root, main)['action'] == 'refuse:target does not contain the owner release commit'


def test_an_owner_track_behind_a_current_invited_stack_catches_up(install):
    root, main, _ = install
    target = commit(main, {'src/gmail_search/agents/mcp_tools_server.py': 'v2\n'})
    (root / 'public-releases/old/QUALIFIED.json').write_text(json.dumps({'commit': target}))
    planned = plan_for(root, main)
    assert planned['action'] == 'ship' and planned['kinds'] == ['owner']


def test_no_deploy_while_the_cutover_is_under_way(install):
    root, main, _ = install
    commit(main, {'src/gmail_search/agents/mcp_tools_server.py': 'v2\n'})
    records = main / phases.CUTOVER_DIR
    records.mkdir(parents=True)
    (records / 'state.json').write_text(json.dumps({'done': ['point owner-current']}))
    assert 'cutover has started and not finished' in plan_for(root, main)['action']
    (records / 'state.json').write_text(json.dumps({'finished': 't'}))
    assert plan_for(root, main)['action'] == 'ship'
    (records / 'rollback.json').write_text('{}')
    assert plan_for(root, main)['action'].startswith('refuse:')


# ── package ──────────────────────────────────────────────────────────
def test_an_owner_release_is_the_tree_its_own_venv_and_the_owner_web(install):
    root, main, base = install
    host = host_for(root, Systemd())
    out = root / 'owner-releases/new'
    phases.package_owner(host, main, out, base, root / 'logs')
    assert (out / 'templates/index.html').read_text() == '<p>\n'
    assert (out / 'scripts/serve_watchdog.sh').exists() and out.stat().st_mode & 0o077 == 0
    assert (out / 'web/node_modules').readlink() == root / 'nm'
    assert (out / 'web/.env.local').readlink() == root / 'checkout/web/.env.local'
    assert not (root / 'logs/owner-tree.tar').exists()
    uv, build = [c for c in host.runner.calls if c[0] != 'systemctl']
    assert uv == ['env', '-u', 'VIRTUAL_ENV', '-u', 'UV_PROJECT_ENVIRONMENT', 'uv', 'sync', '--locked', '--extra', 'dev']
    assert build == ['env', 'PATH=/node/bin:/usr/bin', str(out / 'web/node_modules/.bin/next'), 'build']


def test_dry_run_packages_the_owner_release_outside_the_live_ones(install):
    root, main, _ = install
    host = host_for(root, Systemd())
    assert phases.owner_release_dir(host, main, 'x', True) == main / '.runtime/deploy/dry-run/owner/x'
    assert phases.owner_release_dir(host, main, 'x', False) == root / 'owner-releases/x'


def test_preflight_refuses_an_owner_release_that_moved_since_plan(install):
    root, main, _ = install
    state = phases.State(root / 'state/state.json')
    (root / 'owner-releases/other').mkdir()
    state.set(worktree=str(root / 'no-tree'), releaseDir=str(root / 'public-releases/old'),
              runningDir=str(root / 'public-releases/old'), qualifiedAt='t', target='f' * 40, release='new',
              kinds=['owner'], ownerReleaseDir=str(root / 'owner-releases/new'),
              ownerRunningDir=str(root / 'owner-releases/other'))
    with pytest.raises(RuntimeError, match='owner release changed since plan'):
        phases.preflight_phase(host_for(root, Systemd()), main, state)


# ── activate / rollback / postcheck ──────────────────────────────────
@pytest.fixture
def activating(install):
    root, main, _ = install
    for name in ('new',):
        (root / 'public-releases' / name).mkdir()
        (root / 'owner-releases' / name).mkdir()
    state = phases.State(root / 'state/state.json')
    state.set(release='new', target='f' * 40, kinds=['controller', 'owner'],
              releaseDir=str(root / 'public-releases/new'), runningDir=str(root / 'public-releases/old'),
              ownerReleaseDir=str(root / 'owner-releases/new'), ownerRunningDir=str(root / 'owner-releases/old'))
    return root, state


def owner_current(root):
    return (root / 'owner-current').resolve().name


def test_activate_swaps_owner_current_and_restarts_in_order_with_the_watchdog_held(activating):
    root, state = activating
    systemd = Systemd()
    phases.activate_phase(host_for(root, systemd), state)
    assert owner_current(root) == 'new' and (root / 'invited-current').resolve().name == 'new'
    assert systemd.restarts() == list(OWNER_UNITS)
    owner_events = [e for e in systemd.events if 'api.service' not in e]
    assert owner_events[0] == f'stop {WATCHDOG_TIMER}' and owner_events[-1] == f'start {WATCHDOG_TIMER}'


def test_an_owner_only_release_leaves_the_invited_stack_running(activating):
    root, state = activating
    state.set(kinds=['owner'])
    systemd = Systemd()
    host = host_for(root, systemd)
    phases.activate_phase(host, state)
    assert owner_current(root) == 'new' and (root / 'invited-current').resolve().name == 'old'
    assert not any('api.service' in e for e in systemd.events)


def test_a_deploy_planned_before_the_cutover_does_not_activate_after_it(activating):
    root, state = activating
    state.data.pop('ownerReleaseDir'), state.data.pop('ownerRunningDir')
    state.set(kinds=['controller'])
    with pytest.raises(RuntimeError, match='owner release changed since plan'):
        phases.activate_phase(host_for(root, Systemd()), state)
    assert (root / 'invited-current').resolve().name == 'old'


def test_an_owner_refusal_is_not_hidden_behind_a_current_invited_stack(install):
    root, main, _ = install
    target = commit(main, {checkout_checks.SCHEMA_PATH: SCHEMA + '-- unreviewed\n'})
    (root / 'public-releases/old/QUALIFIED.json').write_text(json.dumps({'commit': target}))
    assert 'REVIEWED_SCHEMA_BLOB' in plan_for(root, main)['action']


def test_the_watchdog_is_held_from_before_the_swap_and_left_off_if_it_was_off(activating):
    root, state = activating
    systemd = Systemd()
    seen = {}
    real = systemd.__call__

    def watch(args):
        if args[2] == 'stop' and args[-1] == WATCHDOG_TIMER:
            seen['link at stop'] = owner_current(root)
        return 'inactive' if args[-1] == WATCHDOG_TIMER and args[2] == 'show' else real(args)

    host = host_for(root, systemd)
    host.runner.systemd = watch
    phases.activate_phase(host, state)
    assert seen == {'link at stop': 'old'}
    assert [e for e in systemd.events if WATCHDOG_TIMER in e] == [f'stop {WATCHDOG_TIMER}']


def test_unhealthy_owner_units_refuse_before_anything_changes(activating):
    root, state = activating
    systemd = Systemd(answers={7878: 502})
    host = host_for(root, systemd)
    with pytest.raises(RuntimeError, match='owner units not healthy before the deploy'):
        phases.activate_phase(host, state)
    assert owner_current(root) == 'old' and (root / 'invited-current').resolve().name == 'old'
    assert systemd.events == [] and host.runner.calls == [c for c in host.runner.calls if c[2] == 'show']


def test_an_mcp_that_does_not_come_back_rolls_both_tracks_back(activating):
    root, state = activating
    systemd = Systemd(breaks={'gmail-search-mcp.service': 1})
    with pytest.raises(RuntimeError, match='mcp.service not healthy.*rollback succeeded'):
        phases.activate_phase(host_for(root, systemd), state)
    assert owner_current(root) == 'old' and (root / 'invited-current').resolve().name == 'old'
    assert systemd.restarts() == ['gmail-search-serve.service', 'gmail-search-mcp.service', *OWNER_UNITS]
    timer = [e for e in systemd.events if WATCHDOG_TIMER in e]
    assert timer == [f'stop {WATCHDOG_TIMER}', f'start {WATCHDOG_TIMER}'] * 2


def test_a_failed_invited_activation_never_touches_the_owner_units(activating):
    root, state = activating
    systemd = Systemd()
    host = host_for(root, systemd, fail_when=lambda args: args[2:3] == ['restart'] and 'api.service' in args)
    with pytest.raises(RuntimeError, match='activate failed'):
        phases.activate_phase(host, state)
    assert owner_current(root) == 'old' and systemd.restarts() == []


def test_rollback_phase_puts_back_both_tracks(activating):
    root, state = activating
    systemd = Systemd()
    host = host_for(root, systemd)
    phases.activate_phase(host, state)
    phases.rollback_phase(host, state)
    assert owner_current(root) == 'old' and (root / 'invited-current').resolve().name == 'old'
    assert systemd.restarts() == [*OWNER_UNITS, *OWNER_UNITS]


def test_postcheck_needs_the_mcp_401_and_records_the_owner_units(activating):
    root, state = activating
    systemd = Systemd()
    host = host_for(root, systemd)
    phases.activate_phase(host, state)
    phases.postcheck_phase(host, root, state)
    record = json.loads((root / '.runtime/issue-loop/deploys.jsonl').read_text())
    assert record['result'] == 'deployed'
    assert record['owner']['gmail-search-mcp.service baseline'] == "pid 1020, {'7878': 401}"
    systemd.answers[7878] = 502
    with pytest.raises(RuntimeError, match='POSTCHECK FAILED'):
        phases.postcheck_phase(host, root, state)


# ── the one-time cutover ─────────────────────────────────────────────
UNITS = {
    'gmail-search-serve.service': '[Service]\nWorkingDirectory=%h/{c}\nExecStart=%h/{c}/.venv/bin/gmail-search serve '
                                  '--data-dir %h/{c}/data\nStandardOutput=append:%h/{c}/data/serve.log\n',
    'gmail-search-mcp.service': '[Service]\nWorkingDirectory=%h/{c}\n'
                                'ExecStart=%h/{c}/.venv/bin/python -m gmail_search.agents.mcp_tools_server\n',
    'gmail-search-web.service': '[Service]\nWorkingDirectory=%h/{c}/web\n'
                                'ExecStart=%h/{c}/web/node_modules/.bin/next start -p 3000\n',
    'gmail-search-web.service.d/loopback.conf': '[Service]\nExecStart=\n'
                                                'ExecStart=%h/{c}/web/node_modules/.bin/next start --port 3000\n',
    'gmail-search-supervise.service': '[Service]\nWorkingDirectory=%h/{c}\nEnvironment=PATH=%h/{c}/.venv/bin:/usr/bin\n'
                                      'ExecStart=%h/{c}/.venv/bin/gmail-search supervise --data-dir %h/{c}/data\n',
    'gmail-search-serve-watchdog.service': '[Service]\nType=oneshot\nExecStart=%h/{c}/scripts/serve_watchdog.sh\n',
}


@pytest.fixture
def cutover(tmp_path, repo, monkeypatch):
    """Units written against a checkout under a fake home; invited-current at
    the base commit; no owner track yet."""
    main, base = repo
    monkeypatch.setenv('HOME', str(tmp_path))
    unit_dir = tmp_path / 'units'
    for rel, text in UNITS.items():
        (unit_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        (unit_dir / rel).write_text(text.format(c='checkout'))
    running = tmp_path / 'public-releases/live'
    running.mkdir(parents=True)
    (running / 'QUALIFIED.json').write_text(json.dumps({'commit': base, 'release': 'live'}))
    (tmp_path / 'invited-current').symlink_to(running)
    systemd = Systemd(unit_dir=unit_dir)
    host = host_for(tmp_path, systemd)
    args = owner_cutover.parse_args(['switch', '--unit-dir', str(unit_dir)])
    state = phases.State(main / '.runtime/owner-cutover/state.json')
    return tmp_path, host, systemd, args, state


def snapshot(unit_dir: Path) -> dict:
    return {str(p.relative_to(unit_dir)): p.read_text() for p in sorted(unit_dir.rglob('*')) if p.is_file()}


def test_rewrite_moves_code_and_keeps_state_in_the_checkout(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path))
    c, o = tmp_path / 'checkout', tmp_path / 'owner-current'
    new = owner_cutover.rewrite_unit(UNITS['gmail-search-supervise.service'].format(c='checkout'), c, o)
    assert 'Environment=PATH=%h/owner-current/.venv/bin:/usr/bin' in new
    assert 'ExecStart=%h/owner-current/.venv/bin/gmail-search supervise --data-dir %h/checkout/data' in new
    assert 'WorkingDirectory=%h/checkout\n' in new and owner_cutover.leftovers(new, c) == []
    web = owner_cutover.rewrite_unit(UNITS['gmail-search-web.service'].format(c='checkout'), c, o)
    assert 'WorkingDirectory=%h/owner-current/web\n' in web
    absolute = owner_cutover.rewrite_unit(f'ExecStart={c}/.venv/bin/python -m x\n', c, o)
    assert absolute == f'ExecStart={o}/.venv/bin/python -m x\n'


def test_a_unit_naming_the_checkout_some_other_way_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path))
    c = tmp_path / 'checkout'
    text = 'WorkingDirectory=%h/checkout\nExecStart=%h/checkout/tools/run.sh\nEnvironment=X=%h/checkout\n'
    assert owner_cutover.leftovers(text, c) == ['ExecStart=%h/checkout/tools/run.sh', 'Environment=X=%h/checkout']
    assert owner_cutover.leftovers('ExecStart=%h/checkout-other/bin/x\n', c) == []


def test_cutover_check_is_read_only_and_passes(cutover):
    root, host, systemd, args, state = cutover
    before = snapshot(args.unit_dir)
    assert owner_cutover.check(host.owner_units, host, state, args) == 0
    assert snapshot(args.unit_dir) == before and systemd.events == [] and not state.path.exists()


def test_cutover_dry_run_prints_and_changes_nothing(cutover, capsys):
    root, host, systemd, args, state = cutover
    host.owner_units.dry_run = True
    before = snapshot(args.unit_dir)
    assert owner_cutover.switch(host.owner_units, host, state, args) == 0
    out = capsys.readouterr().out
    assert snapshot(args.unit_dir) == before and systemd.events == [] and not state.path.exists()
    assert not (root / 'owner-current').exists() and not (root / 'owner-releases').exists()
    assert 'would rewrite' in out and 'would run: systemctl --user daemon-reload' in out
    assert [ln.split()[-1] for ln in out.splitlines() if 'systemctl --user restart' in ln] == list(OWNER_UNITS)


def test_cutover_switches_verifies_and_rolls_back_byte_identical(cutover):
    root, host, systemd, args, state = cutover
    before = snapshot(args.unit_dir)
    assert owner_cutover.switch(host.owner_units, host, state, args) == 0
    assert (root / 'owner-current').resolve() == root / 'owner-releases/live'
    assert json.loads((root / 'owner-releases/live/QUALIFIED.json').read_text())['release'] == 'live'
    after = snapshot(args.unit_dir)
    assert '%h/owner-current/.venv/bin/python' in after['gmail-search-mcp.service']
    assert '%h/owner-current/scripts/serve_watchdog.sh' in after['gmail-search-serve-watchdog.service']
    assert systemd.events[:2] == [f'stop {WATCHDOG_TIMER}', 'daemon-reload'] and systemd.restarts() == list(OWNER_UNITS)
    assert systemd.events[-1] == f'start {WATCHDOG_TIMER}'
    assert not (root / '.wt').exists() or not any((root / '.wt').iterdir())
    assert owner_cutover.switch(host.owner_units, host, state, args) == 0  # finished: nothing again
    assert systemd.restarts() == list(OWNER_UNITS)
    assert owner_cutover.rollback(host.owner_units, host, state, args) == 0
    assert snapshot(args.unit_dir) == before and not (root / 'owner-current').exists()
    assert systemd.restarts() == [*OWNER_UNITS, *OWNER_UNITS] and not state.path.exists()


def test_cutover_refuses_an_unhealthy_unit_or_an_existing_track(cutover):
    root, host, systemd, args, state = cutover
    systemd.answers[8090] = 503
    assert owner_cutover.switch(host.owner_units, host, state, args) == 2
    systemd.answers[8090] = 200
    (root / 'owner-current').symlink_to(root)
    assert owner_cutover.switch(host.owner_units, host, state, args) == 2
    assert systemd.events == [] and not state.path.exists()


def test_a_cutover_that_stopped_after_the_rewrite_resumes_at_the_restart(cutover):
    root, host, systemd, args, state = cutover
    host.runner.fail_when = lambda a: a[2:3] == ['daemon-reload']
    with pytest.raises(Exception, match='daemon-reload'):
        owner_cutover.switch(host.owner_units, host, state, args)
    assert state.data['done'][-1] == 'rewrite the unit files'
    host.runner.fail_when = lambda a: False
    assert owner_cutover.switch(host.owner_units, host, state, args) == 0
    assert systemd.restarts() == list(OWNER_UNITS)


def test_rollback_without_a_cutover_refuses(cutover):
    root, host, systemd, args, state = cutover
    with pytest.raises(RuntimeError, match='nothing to roll back'):
        owner_cutover.rollback(host.owner_units, host, state, args)
