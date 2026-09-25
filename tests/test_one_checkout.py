"""scripts/one-checkout.sh (#57) against a throwaway origin, production clone
and main worktree, with systemctl, uv, next and gh faked: git is real, nothing
live is touched."""
import json
from pathlib import Path
import subprocess

import pytest

from gmail_search.deploy import checkout_checks as checks
from gmail_search.deploy import one_checkout as oc
from gmail_search.deploy.phases import State
from gmail_search.deploy.runner import CommandFailed, Runner

OLD = 'prod/old-branch'
DIRTY = 'tests/test_bm25_deleted_statistics.py'
PORTS = {'gmail-search-serve.service': 8090, 'gmail-search-mcp.service': 7878, 'gmail-search-web.service': 3000}
ANSWERS = {8090: 200, 7878: 401, 3000: 200}


def _deps(names) -> str:
    return '[' + ', '.join(f'{{ name = "{n}" }}' for n in names) + ']'


def lock(**versions) -> str:
    """A uv.lock: `requests` is a runtime dependency (via `urllib3`), `pytest` dev-only."""
    v = {'requests': '2.0', 'urllib3': '1.0', 'pytest': '8.0', **versions}
    pkgs = [('gmail-search', '0.1.0', ['requests']), ('requests', v['requests'], ['urllib3']),
            ('urllib3', v['urllib3'], []), ('pytest', v['pytest'], [])]
    if 'xdist' in v:
        pkgs.append(('pytest-xdist', v['xdist'], []))
    out = [f'[[package]]\nname = "{n}"\nversion = "{ver}"\ndependencies = {_deps(deps)}\n' for n, ver, deps in pkgs]
    out[0] += f'[package.optional-dependencies]\ndev = {_deps(["pytest"])}\n'
    return '\n'.join(out)


# ── the pure checks ──────────────────────────────────────────────────
def test_a_dev_only_lock_change_passes_and_names_the_packages():
    finding = checks.check_lock(lock(), lock(xdist='3.8', pytest='8.1'))
    assert finding.ok and '+pytest-xdist 3.8' in finding.detail and 'pytest 8.0 -> 8.1' in finding.detail


def test_a_transitive_runtime_change_is_refused():
    finding = checks.check_lock(lock(), lock(urllib3='2.0'))
    assert not finding.ok and 'urllib3 1.0 -> 2.0' in finding.detail


def test_only_the_four_expected_paths_may_be_dirty():
    assert checks.check_dirty(set(checks.EXPECTED_DIRTY)).ok
    assert checks.check_dirty(set()).ok
    refused = checks.check_dirty({DIRTY, 'src/gmail_search/cli.py'})
    assert not refused.ok and 'src/gmail_search/cli.py' in refused.detail


def test_each_unmerged_commit_needs_its_own_acknowledgement():
    unmerged = ['453e1f7' + 'a' * 33 + ' fix(ci): one', 'f2fc858' + 'b' * 33 + ' fix(crawl): two']
    assert checks.check_unmerged(unmerged, {'453e1f7', 'f2fc858b'}).ok
    refused = checks.check_unmerged(unmerged, {'453e1f7'})
    assert not refused.ok and 'f2fc858' in refused.detail and '453e1f7' not in refused.detail


def test_an_acknowledgement_matching_two_commits_is_refused():
    unmerged = ['1234567a' + 'a' * 32 + ' one', '1234567b' + 'b' * 32 + ' two']
    refused = checks.check_unmerged(unmerged, {'1234567'})
    assert not refused.ok and 'more than one' in refused.detail


def test_a_change_to_a_package_pulled_in_by_a_runtime_extra_is_refused():
    """`psycopg[binary]`: psycopg-binary is runtime even though it is an optional dependency."""
    def with_binary(version):
        return lock().replace('dependencies = [{ name = "requests" }]',
                              'dependencies = [{ name = "requests" }, { name = "psycopg", extra = ["binary"] }]', 1) + (
            f'\n[[package]]\nname = "psycopg"\nversion = "3.2"\ndependencies = []\n'
            f'[package.optional-dependencies]\nbinary = [{{ name = "psycopg-binary" }}]\n'
            f'\n[[package]]\nname = "psycopg-binary"\nversion = "{version}"\ndependencies = []\n')
    finding = checks.check_lock(with_binary('3.2'), with_binary('3.3'))
    assert not finding.ok and 'psycopg-binary 3.2 -> 3.3' in finding.detail


def test_an_unreviewed_schema_blob_is_refused_with_the_flag_to_pass():
    assert checks.check_schema(checks.REVIEWED_SCHEMA_BLOB).ok
    refused = checks.check_schema('b' * 40)
    assert not refused.ok and f'--schema-reviewed {"b" * 40}' in refused.detail


def test_a_web_lockfile_change_is_refused():
    same = {p: 'x' for p in checks.WEB_LOCKFILES}
    assert checks.check_web_lockfiles(same, dict(same)).ok
    assert not checks.check_web_lockfiles(same, {**same, 'web/package-lock.json': 'y'}).ok


@pytest.mark.parametrize('state,labels,ok', [('CLOSED', [], True), ('OPEN', ['loop:merged'], True),
                                             ('OPEN', ['loop:working'], False)])
def test_an_issue_counts_as_done_when_closed_or_landed(state, labels, ok):
    assert checks.check_issue(55, state, labels).ok is ok


def test_allow_unmerged_refuses_short_shas():
    with pytest.raises(SystemExit):
        oc.parse_args(['check', '--allow-unmerged', 'abc'])


# ── a throwaway production checkout ─────────────────────────────────
def git(cwd, *args) -> str:
    return subprocess.run(['git', '-C', str(cwd), *args], check=True, capture_output=True, text=True).stdout.strip()


def write(root: Path, files: dict) -> None:
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)


def commit(root: Path, files: dict, message: str) -> None:
    write(root, files)
    git(root, 'add', '-A')
    git(root, '-c', 'user.name=t', '-c', 'user.email=t@example.invalid', 'commit', '-qm', message)


@pytest.fixture
def repo(tmp_path):
    """origin/main one commit past the old branch; production on the old
    branch with one expected edit; a second worktree holding main."""
    origin, seed, prod = tmp_path / 'origin.git', tmp_path / 'seed', tmp_path / 'prod'
    git(tmp_path, 'init', '-q', '--bare', '-b', 'main', str(origin))
    git(tmp_path, 'clone', '-q', str(origin), str(seed))
    commit(seed, {'uv.lock': lock(), 'web/package.json': '{}', 'web/package-lock.json': '{}',
                  checks.SCHEMA_PATH: 'CREATE TABLE t();\n', DIRTY: 'old\n', '.gitignore': 'web/.next/\n.runtime/\n'},
           'base')
    git(seed, 'push', '-q', 'origin', f'main:{OLD}', 'main')
    commit(seed, {'uv.lock': lock(xdist='3.8'), checks.SCHEMA_PATH: 'CREATE TABLE t();\nCREATE INDEX i ON t();\n',
                  'NEW_ON_MAIN': 'x\n'}, 'main moves on')
    git(seed, 'push', '-q', 'origin', 'main')
    git(tmp_path, 'clone', '-q', str(origin), str(prod))
    git(prod, 'switch', '-q', OLD)
    git(prod, 'worktree', 'add', '-q', str(tmp_path / 'main-wt'), 'main')
    git(prod, 'fetch', '-q', 'origin')
    (prod / DIRTY).write_text('edited by another session\n')
    write(prod, {'web/.next/BUILD_ID': 'old-build'})
    return prod


class Units:
    """Fake systemd: units with pids, ports and answers."""

    def __init__(self, answers=None):
        self.pid = {u: 100 + i for i, u in enumerate((*oc.OWNER_UNITS, *oc.SHARING_UNITS))}
        self.answers = {**ANSWERS, **(answers or {})}
        self.events, self.next_pid = [], 1000

    def systemctl(self, args):
        verb, unit = args[2], args[-1]
        if verb in ('stop', 'start'):
            self.events.append(f'{verb} {unit}')
            if unit in self.pid:
                self.next_pid += 10
                self.pid[unit] = self.next_pid if verb == 'start' else 0
            return ''
        return {'MainPID': str(self.pid.get(unit, 0)), 'Environment': 'PATH=/node/bin:/usr/bin',
                'ActiveState': 'active' if self.pid.get(unit) else 'inactive'}.get(args[4], '')

    def unit_of(self, pid):
        return next((u for u, p in self.pid.items() if p and p == pid), None)

    def listening(self, pid):
        unit = self.unit_of(pid)
        return [('127.0.0.1', PORTS[unit])] if unit in PORTS else []

    def process_tree(self, pid):
        return [pid, pid + 1] if self.unit_of(pid) == 'gmail-search-supervise.service' else [pid]

    def http_status(self, method, url):
        return self.answers[int(url.split(':')[2].split('/')[0])]


class FakeRunner(Runner):
    """git runs for real; systemctl, gh and the build commands are faked."""

    def __init__(self, units, fail_when=lambda args: False):
        self.units, self.calls, self.fail_when = units, [], fail_when

    def _fake(self, args):
        args = [str(a) for a in args]
        if self.fail_when(args):
            raise CommandFailed(args, 1)
        if args[0] == 'git':
            return None
        self.calls.append(args)
        if args[0] == 'systemctl':
            return self.units.systemctl(args)
        if args[0] == 'gh':
            return json.dumps({'state': 'CLOSED', 'labels': []})
        if args[0] in ('cp', 'rm'):
            subprocess.run(args, check=True)
        return ''

    def run(self, args, **kw):
        out = self._fake(args)
        return super().run(args, **kw) if out is None else 0

    def capture(self, args, **kw):
        out = self._fake(args)
        return super().capture(args, **kw) if out is None else out


def machine(prod, units, *, dry_run=False, fail_when=lambda args: False):
    return oc.Machine(prod, FakeRunner(units, fail_when), dry_run=dry_run, health_host='127.0.0.1',
                      listening=units.listening, process_tree=units.process_tree, http_status=units.http_status,
                      sleep=lambda s: None)


def args_for(prod, command='switch', *extra):
    blob = git(prod, 'rev-parse', f'origin/main:{checks.SCHEMA_PATH}')
    return oc.parse_args([command, '--after-issue', '1', '--schema-reviewed', blob, *extra])


def state_of(prod):
    return State(prod / '.runtime/one-checkout/state.json')


def snapshot(prod):
    return git(prod, 'rev-parse', '--abbrev-ref', 'HEAD'), (prod / DIRTY).read_text()


def changes(runner):
    return [c for c in runner.calls if c[0] != 'gh' and c[2:3] != ['show']]


# ── switch ───────────────────────────────────────────────────────────
def test_dry_run_prints_every_step_and_changes_nothing(repo, capsys):
    units = Units()
    m = machine(repo, units, dry_run=True)
    assert oc.switch(m, state_of(repo), args_for(repo)) == 0
    out = capsys.readouterr().out
    assert snapshot(repo) == (OLD, 'edited by another session\n')
    assert git(repo.parent / 'main-wt', 'rev-parse', '--abbrev-ref', 'HEAD') == 'main'
    assert not (repo / '.runtime').exists() and changes(m.runner) == [] and units.events == []
    starts = [line.split()[-1] for line in out.splitlines() if 'would run: systemctl --user start' in line]
    assert starts == [*oc.OWNER_UNITS, oc.WATCHDOG_TIMER]
    assert 'switch --detach' in out and 'uv sync --locked --extra dev' in out and 'next build' in out


def test_switch_moves_to_main_saves_the_patch_and_restarts_in_order(repo):
    units = Units()
    m = machine(repo, units)
    shared_before = {u: units.pid[u] for u in oc.SHARING_UNITS}
    assert oc.switch(m, state_of(repo), args_for(repo)) == 0
    assert git(repo, 'rev-parse', 'HEAD') == git(repo, 'rev-parse', 'origin/main')
    assert git(repo, 'rev-parse', '--abbrev-ref', 'HEAD') == 'main' and git(repo, 'status', '--porcelain') == ''
    assert git(repo.parent / 'main-wt', 'rev-parse', '--abbrev-ref', 'HEAD') == 'HEAD'
    data = state_of(repo).data
    patch = Path(data['patch'])
    assert 'edited by another session' in patch.read_text() and patch.stat().st_mode & 0o777 == 0o600
    assert data['old_branch'] == OLD and data['finished']
    stops = [e.split()[1] for e in units.events if e.startswith('stop')]
    starts = [e.split()[1] for e in units.events if e.startswith('start')]
    assert stops == [oc.WATCHDOG_TIMER, *reversed(oc.OWNER_UNITS)]
    assert starts == [*oc.OWNER_UNITS, oc.WATCHDOG_TIMER]
    assert {u: units.pid[u] for u in oc.SHARING_UNITS} == shared_before
    assert ['env', 'PATH=/node/bin:/usr/bin', str(repo / 'web/node_modules/.bin/next'), 'build'] in m.runner.calls


def test_a_finished_switch_is_not_repeated(repo):
    units = Units()
    oc.switch(machine(repo, units), state_of(repo), args_for(repo))
    again = machine(repo, units)
    assert oc.switch(again, state_of(repo), args_for(repo)) == 0
    assert units.events.count(f'stop {oc.OWNER_UNITS[0]}') == 1


def test_an_unexpected_edit_stops_the_switch_before_anything_changes(repo):
    (repo / 'uv.lock').write_text(lock() + '\n# edited\n')
    units = Units()
    assert oc.switch(machine(repo, units), state_of(repo), args_for(repo)) == 2
    assert units.events == [] and not state_of(repo).path.exists()


def test_an_unhealthy_baseline_stops_the_switch(repo):
    units = Units(answers={8090: 503})
    assert oc.switch(machine(repo, units), state_of(repo), args_for(repo)) == 2
    assert units.events == []


def test_a_unit_that_does_not_come_back_like_before_fails_by_name(repo):
    units = Units()
    m = machine(repo, units)
    real_start = units.systemctl

    def mcp_breaks(args):
        out = real_start(args)
        if args[2] == 'start' and args[-1] == 'gmail-search-mcp.service':
            units.answers[7878] = 502
        return out
    units.systemctl = mcp_breaks
    with pytest.raises(RuntimeError, match='gmail-search-mcp.service not healthy'):
        oc.switch(m, state_of(repo), args_for(repo))
    assert 'start owner units' not in state_of(repo).data['done']


def test_a_switch_that_stopped_part_way_resumes_after_the_last_done_step(repo):
    units = Units()
    broken = machine(repo, units, fail_when=lambda args: args[-1] == 'build')
    with pytest.raises(CommandFailed):
        oc.switch(broken, state_of(repo), args_for(repo))
    assert state_of(repo).data['done'][-1] == 'uv sync'
    resumed = machine(repo, units)
    assert oc.switch(resumed, state_of(repo), args_for(repo)) == 0
    assert [c for c in resumed.runner.calls if c[:2] == ['git', 'fetch']] == []
    assert not any('sync' in c for c in resumed.runner.calls)
    assert state_of(repo).data['finished']


# ── rollback ─────────────────────────────────────────────────────────
def test_rollback_restores_the_branch_the_edits_the_build_and_main_worktree(repo):
    units = Units()
    oc.switch(machine(repo, units), state_of(repo), args_for(repo))
    (repo / 'web/.next/BUILD_ID').write_text('new-build')
    back = machine(repo, units)
    assert oc.rollback(back, state_of(repo), args_for(repo, 'rollback')) == 0
    assert snapshot(repo) == (OLD, 'edited by another session\n')
    assert (repo / 'web/.next/BUILD_ID').read_text() == 'old-build'
    assert git(repo.parent / 'main-wt', 'rev-parse', '--abbrev-ref', 'HEAD') == 'main'
    assert not state_of(repo).path.exists()
    assert len(list((repo / '.runtime/one-checkout').glob('state-rolled-back-*.json'))) == 1
    starts = [e.split()[1] for e in units.events if e.startswith('start')]
    assert starts[-5:] == [*oc.OWNER_UNITS, oc.WATCHDOG_TIMER]


def test_rollback_without_a_switch_refuses(repo):
    with pytest.raises(RuntimeError, match='nothing to roll back'):
        oc.rollback(machine(repo, Units()), state_of(repo), args_for(repo, 'rollback'))


def test_rollback_dry_run_changes_nothing(repo):
    units = Units()
    oc.switch(machine(repo, units), state_of(repo), args_for(repo))
    events = list(units.events)
    oc.rollback(machine(repo, units, dry_run=True), state_of(repo), args_for(repo, 'rollback'))
    assert units.events == events and git(repo, 'rev-parse', '--abbrev-ref', 'HEAD') == 'main'
    assert state_of(repo).path.exists() and not state_of(repo).path.with_name('rollback.json').exists()


def test_rollback_after_the_loop_worktree_was_removed_skips_reattaching_it(repo):
    units = Units()
    oc.switch(machine(repo, units), state_of(repo), args_for(repo))
    git(repo, 'worktree', 'remove', str(repo.parent / 'main-wt'))
    assert oc.rollback(machine(repo, units), state_of(repo), args_for(repo, 'rollback')) == 0
    assert snapshot(repo) == (OLD, 'edited by another session\n')


def test_an_edit_after_the_patch_was_saved_stops_the_switch_before_any_discard(repo):
    units = Units()
    real = units.systemctl

    def someone_edits_during_the_stop(args):
        if args[2] == 'stop' and args[-1] == oc.OWNER_UNITS[0]:
            (repo / DIRTY).write_text('edited again, after the patch\n')
        return real(args)
    units.systemctl = someone_edits_during_the_stop
    with pytest.raises(RuntimeError, match='changed after the patch was saved'):
        oc.switch(machine(repo, units), state_of(repo), args_for(repo))
    assert snapshot(repo) == (OLD, 'edited again, after the patch\n')


def test_rollback_refuses_to_start_units_when_the_patch_cannot_go_back(repo):
    units = Units()
    oc.switch(machine(repo, units), state_of(repo), args_for(repo))
    (repo / DIRTY).write_text('an edit made on main after the switch\n')
    with pytest.raises(RuntimeError, match='neither applies nor is applied'):
        oc.rollback(machine(repo, units), state_of(repo), args_for(repo, 'rollback'))
    assert not [e for e in units.events[-5:] if e.startswith('start')]


def test_a_probed_unit_with_no_listening_port_is_not_a_healthy_baseline(repo):
    units = Units()
    silent = machine(repo, units)
    silent.listening = lambda pid: [] if units.unit_of(pid) == 'gmail-search-mcp.service' else units.listening(pid)
    assert oc.switch(silent, state_of(repo), args_for(repo)) == 2
    assert units.events == []


def test_a_clean_checkout_switches_with_an_empty_patch_and_rolls_back(repo):
    git(repo, 'checkout', '--', DIRTY)
    units = Units()
    assert oc.switch(machine(repo, units), state_of(repo), args_for(repo)) == 0
    assert Path(state_of(repo).data['patch']).read_text() == ''
    assert oc.rollback(machine(repo, units), state_of(repo), args_for(repo, 'rollback')) == 0
    assert snapshot(repo) == (OLD, 'old\n')


def test_a_crash_between_the_discard_and_the_branch_switch_resumes(repo):
    units = Units()
    crash = machine(repo, units, fail_when=lambda args: args[-2:] == ['switch', 'main'])
    with pytest.raises(CommandFailed):
        oc.switch(crash, state_of(repo), args_for(repo))
    assert git(repo, 'status', '--porcelain') == ''
    assert oc.switch(machine(repo, units), state_of(repo), args_for(repo)) == 0
    assert git(repo, 'rev-parse', '--abbrev-ref', 'HEAD') == 'main'


def test_a_crash_just_after_the_discard_is_not_mistaken_for_a_new_edit(repo):
    units = Units()
    m = machine(repo, units)
    state = state_of(repo)
    oc.new_state(m, state, git(repo, 'rev-parse', 'origin/main'), oc.baseline(m))
    oc.save_patch(m, state)
    git(repo, 'checkout', '--', DIRTY)  # the discard ran, its step was never recorded
    assert oc._unchanged_since_saved(m, state) == []


def test_nested_extras_are_followed_into_the_runtime_closure():
    base = lock().replace('dependencies = [{ name = "requests" }]',
                          'dependencies = [{ name = "requests" }, { name = "parent", extra = ["feat"] }]', 1)

    def with_leaf(version):
        return base + (
            '\n[[package]]\nname = "parent"\nversion = "1"\ndependencies = []\n'
            '[package.optional-dependencies]\nfeat = [{ name = "middle", extra = ["nested"] }]\n'
            '\n[[package]]\nname = "middle"\nversion = "1"\ndependencies = []\n'
            '[package.optional-dependencies]\nnested = [{ name = "leaf" }]\n'
            f'\n[[package]]\nname = "leaf"\nversion = "{version}"\ndependencies = []\n')
    finding = checks.check_lock(with_leaf('1'), with_leaf('2'))
    assert not finding.ok and 'leaf 1 -> 2' in finding.detail


def test_a_dev_package_becoming_runtime_at_the_same_version_is_refused():
    now_runtime = lock().replace('dependencies = [{ name = "requests" }]',
                                 'dependencies = [{ name = "requests" }, { name = "pytest" }]', 1)
    finding = checks.check_lock(lock(), now_runtime)
    assert not finding.ok and 'pytest now runtime' in finding.detail
    back = checks.check_lock(now_runtime, lock())
    assert not back.ok and 'pytest no longer runtime' in back.detail


def test_a_runtime_registry_change_at_the_same_version_is_refused():
    moved = lock().replace('name = "urllib3"\nversion = "1.0"\n',
                           'name = "urllib3"\nversion = "1.0"\nsource = { registry = "https://mirror.invalid/simple" }\n')
    finding = checks.check_lock(lock(), moved)
    assert not finding.ok and 'urllib3 1.0 (source changed)' in finding.detail


def test_rollback_with_the_saved_patch_missing_stops_before_starting_units(repo):
    units = Units()
    oc.switch(machine(repo, units), state_of(repo), args_for(repo))
    Path(state_of(repo).data['patch']).unlink()
    with pytest.raises(RuntimeError, match='is missing'):
        oc.rollback(machine(repo, units), state_of(repo), args_for(repo, 'rollback'))
    assert not [e for e in units.events[-5:] if e.startswith('start')]


def test_rollback_reattaches_a_worktree_detached_by_a_step_that_never_recorded(repo):
    units = Units()
    stop = machine(repo, units, fail_when=lambda args: args[-2:] == ['--', DIRTY])
    with pytest.raises(CommandFailed):
        oc.switch(stop, state_of(repo), args_for(repo))
    data = state_of(repo).data
    state_of(repo).set(done=[d for d in data['done'] if d != 'release main'])
    assert oc.rollback(machine(repo, units), state_of(repo), args_for(repo, 'rollback')) == 0
    assert git(repo.parent / 'main-wt', 'rev-parse', '--abbrev-ref', 'HEAD') == 'main'


def test_edits_saved_and_discarded_earlier_are_what_rollback_restores(repo, tmp_path):
    earlier = tmp_path / 'stray.patch'
    earlier.write_text(git(repo, 'diff', 'HEAD', '--binary') + '\n')
    git(repo, 'checkout', '--', DIRTY)
    units = Units()
    args = args_for(repo, 'switch', '--saved-patch', str(earlier))
    assert oc.switch(machine(repo, units), state_of(repo), args) == 0
    assert git(repo, 'rev-parse', '--abbrev-ref', 'HEAD') == 'main' and git(repo, 'status', '--porcelain') == ''
    assert oc.rollback(machine(repo, units), state_of(repo), args_for(repo, 'rollback')) == 0
    assert snapshot(repo) == (OLD, 'edited by another session\n')


def test_a_saved_patch_with_edits_still_in_the_tree_is_refused(repo, tmp_path):
    earlier = tmp_path / 'stray.patch'
    earlier.write_text(git(repo, 'diff', 'HEAD', '--binary') + '\n')
    units = Units()
    with pytest.raises(RuntimeError, match='still has uncommitted edits'):
        oc.switch(machine(repo, units), state_of(repo), args_for(repo, 'switch', '--saved-patch', str(earlier)))
    assert units.events == []
