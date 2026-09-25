"""scripts/one-checkout.sh [--dry-run] <check|switch|rollback> [options]

Moves the production checkout (the one holding .git, where serve, mcp,
supervise and the owner's web run) from its old branch onto `main` in one
window, and back (#57). docs/one-checkout-runbook.md is the procedure; this is
its executable half.

  check     read-only preconditions; exit 0 only when switch may run
  switch    save the patch, stop the owner units, discard the four edits, move
            to main, uv sync, rebuild web, start the units and health-check them
  rollback  the reverse, from the state the switch recorded

--dry-run prints every command switch or rollback would run and runs none.
Run it from a worktree of main, never from the production checkout itself.
Each step is recorded in <checkout>/.runtime/one-checkout/state.json, so a
switch or rollback that stopped part way re-runs from where it stopped.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Callable

from . import checkout_checks as checks
from .config import DEFAULT_PATH, load_config
from .host import _active_runs, _port_open
from .phases import State, now
from .runner import CommandFailed, Runner

say = lambda message: print(f'==> {message}', flush=True)  # noqa: E731

# The units that run the checkout's own code. Started backend first, so mcp
# and web never come up against a serve still on the old code, and supervise
# last: its children (watch, update, reindex, summarize, crawl) write
# constantly and every one of them runs init_db. Stopped in the reverse.
OWNER_UNITS = ('gmail-search-serve.service', 'gmail-search-mcp.service',
               'gmail-search-web.service', 'gmail-search-supervise.service')
WATCHDOG_TIMER = 'gmail-search-serve-watchdog.timer'
# Share the checkout's .venv or web/node_modules but run a release's code; the
# switch must leave them running (same MainPID before and after).
SHARING_UNITS = ('gmail-search-invited-api.service', 'gmail-search-public-web.service')
# Each unit's own readiness request and the answers that mean healthy (mcp:
# up, with its OAuth gate on); the port is read from the running unit.
PROBES = {'gmail-search-serve.service': ('GET', '/healthz?ready=1', {200}),
          'gmail-search-mcp.service': ('POST', '/mcp', {401}),
          'gmail-search-web.service': ('GET', '/', {200})}
HEALTH_SECONDS = {'gmail-search-serve.service': 300}
DEFAULT_HEALTH_SECONDS = 120
TARGET_BRANCH = 'main'


# ── the machine, behind one object so tests fake it ──────────────────
def _listening(pid: int) -> list[tuple[str, int]]:
    out = subprocess.run(['ss', '-ltnpH'], capture_output=True, text=True).stdout
    found = []
    for line in out.splitlines():
        if f'pid={pid},' in line:
            addr = line.split()[3]
            host, _, port = addr.rpartition(':')
            found.append((host.strip('[]'), int(port)))
    return found


def _process_tree(pid: int) -> list[int]:
    out = subprocess.run(['ps', '-o', 'pid=', '--ppid', str(pid)], capture_output=True, text=True).stdout
    kids = [int(p) for p in out.split()]
    return [pid] + [q for k in kids for q in _process_tree(k)]


def _http_status(method: str, url: str) -> int:
    import http.client
    from urllib.parse import urlsplit

    parts = urlsplit(url)
    conn = http.client.HTTPConnection(parts.hostname, parts.port, timeout=30)
    try:
        conn.request(method, parts.path + (f'?{parts.query}' if parts.query else ''))
        return conn.getresponse().status
    except OSError:
        return 0
    finally:
        conn.close()


@dataclass
class Machine:
    checkout: Path
    runner: Runner = field(default_factory=Runner)
    dry_run: bool = False
    health_host: str = ''
    registry: Path | None = None
    listening: Callable[[int], list] = _listening
    process_tree: Callable[[int], list] = _process_tree
    http_status: Callable[[str, str], int] = _http_status
    port_open: Callable[[str, int], bool] = _port_open
    active_runs: Callable[[Path], int] = _active_runs
    sleep: Callable[[float], None] = time.sleep

    def git(self, *args, check=True) -> str:
        # --no-optional-locks: a read (status) must not rewrite the checkout's index.
        return self.runner.capture(['git', '--no-optional-locks', '-C', self.checkout, *args], check=check).strip()

    def act(self, args, **kw) -> None:
        """A command that changes something: printed always, run unless dry."""
        where = f" (in {kw['cwd']})" if kw.get('cwd') else ''
        say(f"{'would run' if self.dry_run else 'run'}: {shlex.join(map(str, args))}{where}")
        if not self.dry_run:
            self.runner.run(args, **kw)

    def unit(self, unit: str, prop: str) -> str:
        return self.runner.capture(['systemctl', '--user', 'show', '-p', prop, '--value', unit],
                                   check=False).strip()

    def main_pid(self, unit: str) -> int:
        return int(self.unit(unit, 'MainPID') or 0)

    def unit_tree(self, unit: str) -> list[int]:
        pid = self.main_pid(unit)
        return self.process_tree(pid) if pid else []

    def unit_ports(self, unit: str) -> list[tuple[str, int]]:
        return sorted({a for p in self.unit_tree(unit) for a in self.listening(p)})


# ── check ────────────────────────────────────────────────────────────
def _blob(m: Machine, ref: str, path: str) -> str:
    return m.git('rev-parse', f'{ref}:{path}', check=False)


def _show(m: Machine, ref: str, path: str) -> str:
    return m.runner.capture(['git', '--no-optional-locks', '-C', m.checkout, 'show', f'{ref}:{path}'])


def _dirty(m: Machine) -> set[str]:
    return set(m.git('diff', '--name-only', 'HEAD').splitlines())


def _untracked_in_the_way(m: Machine, target: str) -> checks.Finding:
    untracked = set(m.git('ls-files', '--others', '--exclude-standard').splitlines())
    arriving = set(m.git('diff', '--name-only', 'HEAD', target).splitlines())
    clash = sorted(untracked & arriving)
    return checks.Finding('untracked files', not clash,
                          'would be overwritten: ' + ', '.join(clash) if clash else 'none in the way')


def _issue(m: Machine, number: int) -> checks.Finding:
    raw = m.runner.capture(['gh', 'issue', 'view', str(number), '--json', 'state,labels'],
                           cwd=m.checkout, check=False)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return checks.Finding(f'#{number}', False, 'gh could not read the issue')
    return checks.check_issue(number, data['state'], [lbl['name'] for lbl in data['labels']])


def _local_main(m: Machine, target: str) -> checks.Finding:
    if not m.git('rev-parse', '--verify', '--quiet', f'refs/heads/{TARGET_BRANCH}', check=False):
        return checks.Finding('local main', True, 'absent; switch creates it at the target')
    ok = m.runner.run(['git', '-C', m.checkout, 'merge-base', '--is-ancestor', TARGET_BRANCH, target],
                      check=False) == 0
    return checks.Finding('local main', ok, 'fast-forwards to the target' if ok
                          else 'has commits the target lacks; resolve by hand')


def _quiet(m: Machine) -> list[checks.Finding]:
    lock = m.checkout / '.runtime/issue-loop/land.lock'
    held = lock.exists()
    runs = m.active_runs(m.registry) if m.registry else 0
    return [checks.Finding('landing lock', not held, f'held: {(lock / "owner").read_text().strip()}'
                           if held and (lock / 'owner').exists() else 'held' if held else 'free'),
            checks.Finding('active user runs', runs == 0, f'{runs} active')]


def _cwd_and_parent(proc: Path) -> tuple[Path, int]:
    ppid = int((proc / 'stat').read_text().rsplit(')', 1)[1].split()[1])
    return Path(os.readlink(proc / 'cwd')), ppid


def _other_sessions(m: Machine) -> list[str]:
    """Outermost processes working in the checkout that no owner unit started
    (another session, a shell). Reported, not refused: the owner's own shell
    is one."""
    units = {p for u in (*OWNER_UNITS, *SHARING_UNITS) for p in m.unit_tree(u)}
    inside = {}
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            cwd, ppid = _cwd_and_parent(proc)
        except (OSError, ValueError, IndexError):
            continue
        if cwd == m.checkout and int(proc.name) not in units | {os.getpid()}:
            inside[int(proc.name)] = ppid
    return [f'{pid} {_comm(pid)}' for pid, ppid in sorted(inside.items()) if ppid not in inside]


def _comm(pid: int) -> str:
    try:
        return Path(f'/proc/{pid}/comm').read_text().strip()
    except OSError:
        return '(exited)'


def preconditions(m: Machine, target: str, args) -> list[checks.Finding]:
    old_lock, new_lock = _show(m, 'HEAD', 'uv.lock'), _show(m, target, 'uv.lock')
    unmerged = m.git('log', '--format=%H %s', f'{target}..HEAD').splitlines()
    return [
        *(_issue(m, n) for n in args.after_issue),
        checks.check_dirty(_dirty(m)),
        _untracked_in_the_way(m, target),
        checks.check_unmerged(unmerged, set(args.allow_unmerged)),
        _local_main(m, target),
        checks.check_lock(old_lock, new_lock),
        checks.check_web_lockfiles({p: _blob(m, 'HEAD', p) for p in checks.WEB_LOCKFILES},
                                   {p: _blob(m, target, p) for p in checks.WEB_LOCKFILES}),
        checks.check_schema(_blob(m, target, checks.SCHEMA_PATH), args.schema_reviewed),
        *_quiet(m),
    ]


def report(findings: list[checks.Finding]) -> bool:
    for finding in findings:
        print(finding.line(), flush=True)
    return all(f.ok for f in findings)


# ── health ───────────────────────────────────────────────────────────
def _url(m: Machine, host: str, port: int, path: str) -> str:
    host = m.health_host if host in ('0.0.0.0', '*', '::', '') else host
    return f'http://{host}:{port}{path}'


def probe_unit(m: Machine, unit: str) -> dict:
    """What the unit answers now: its ports and each one's status for the
    unit's own readiness request."""
    method, path, _ = PROBES.get(unit, (None, None, None))
    ports = m.unit_ports(unit)
    return {'pid': m.main_pid(unit), 'ports': ports, 'children': len(m.unit_tree(unit)) - 1,
            'status': {str(port): m.http_status(method, _url(m, host, port, path))
                       for host, port in ports} if method else {}}


def baseline(m: Machine) -> dict:
    return {u: probe_unit(m, u) for u in (*OWNER_UNITS, *SHARING_UNITS)}


def wait_like_before(m: Machine, unit: str, before: dict) -> str:
    """Poll until the unit is active, on the same ports, and each answers the
    status it answered before the switch (plus children for supervise)."""
    deadline = HEALTH_SECONDS.get(unit, DEFAULT_HEALTH_SECONDS)
    for _ in range(deadline // 5):
        if m.unit(unit, 'ActiveState') == 'active':
            now_ = probe_unit(m, unit)
            same = {str(p): s for p, s in now_['status'].items()} == before['status']
            kids = now_['children'] > 0 if unit == 'gmail-search-supervise.service' else True
            if same and [list(p) for p in now_['ports']] == [list(p) for p in before['ports']] and kids:
                return f"{unit} healthy: pid {now_['pid']}, {now_['status'] or 'no probe'}"
        m.sleep(5)
    raise RuntimeError(f'{unit} not healthy after {deadline}s (before: {before}); '
                       f'journalctl --user -u {unit} -n 50')


def _baseline_ok(unit: str, b: dict) -> bool:
    if not b['pid']:
        return False
    if unit in PROBES:
        return bool(b['status']) and all(s in PROBES[unit][2] for s in b['status'].values())
    return unit != 'gmail-search-supervise.service' or b.get('children', 0) > 0


def healthy_baseline(before: dict) -> list[checks.Finding]:
    """The switch judges health as "answers like before", so before must be
    good: every unit running, each probed one listening and answering healthy,
    supervise with children."""
    return [checks.Finding(f'{u} baseline', _baseline_ok(u, b),
                           f"pid {b['pid']}, {b['status'] or 'no probe'}"
                           + (f", {b.get('children', 0)} children" if u == 'gmail-search-supervise.service' else ''))
            for u, b in before.items()]


def sharing_pids(m: Machine) -> dict[str, int]:
    return {u: m.main_pid(u) for u in SHARING_UNITS}


def sharing_units_untouched(m: Machine, before: dict[str, int]) -> None:
    for unit, pid in sharing_pids(m).items():
        if pid != before[unit]:
            raise RuntimeError(f'{unit} restarted meanwhile (pid {before[unit]} -> {pid})')
        say(f'{unit} untouched: pid {pid}')


# ── the steps ────────────────────────────────────────────────────────
def _step(state: State, name: str, do: Callable[[], None], m: Machine) -> None:
    done = state.data.get('done', [])
    if name in done:
        say(f'{name}: already done')
        return
    say(name)
    do()
    if not m.dry_run:
        state.set(done=[*done, name])


def stop_units(m: Machine) -> None:
    m.act(['systemctl', '--user', 'stop', WATCHDOG_TIMER])
    for unit in reversed(OWNER_UNITS):
        m.act(['systemctl', '--user', 'stop', unit])


def start_units(m: Machine, before: dict) -> None:
    for unit in OWNER_UNITS:
        m.act(['systemctl', '--user', 'start', unit])
        if not m.dry_run:
            say(wait_like_before(m, unit, before[unit]))
    m.act(['systemctl', '--user', 'start', WATCHDOG_TIMER])


def save_patch(m: Machine, state: State) -> None:
    """The edits rollback puts back: the tree's own, or, when someone already
    saved and discarded them (--saved-patch), a copy of that patch, proven to
    apply to the clean tree."""
    patch, earlier = Path(state.data['patch']), state.data.get('saved_patch')
    say(f"{'would save' if m.dry_run else 'save'} "
        f"{'a copy of ' + earlier if earlier else 'the uncommitted diff'} to {patch}")
    if m.dry_run:
        return
    diff = m.runner.capture(['git', '-C', m.checkout, 'diff', 'HEAD', '--binary'])
    if earlier and diff:
        raise RuntimeError(f'--saved-patch {earlier} given, but the tree still has uncommitted edits')
    patch.parent.mkdir(parents=True, exist_ok=True)
    patch.write_text(Path(earlier).read_text() if earlier else diff)
    patch.chmod(0o600)
    if patch.stat().st_size:  # git refuses to check an empty patch; a clean tree has nothing to prove
        m.runner.run(['git', '-C', m.checkout, 'apply', '--check', *([] if earlier else ['--reverse']), patch])


def backup_web_build(m: Machine, state: State) -> None:
    # A copy interrupted part way is removed first, or `cp -a` would nest the
    # retry inside it.
    m.act(['rm', '-rf', state.data['web_backup']])
    m.act(['cp', '-a', m.checkout / 'web/.next', state.data['web_backup']])


def release_main_elsewhere(m: Machine, state: State) -> None:
    """`main` can be checked out in one worktree only: detach whichever other
    worktree holds it (the loop's gmail-search-main), recorded for rollback."""
    holder = _worktree_on(m, f'refs/heads/{TARGET_BRANCH}')
    if holder and holder != m.checkout:
        if not m.dry_run:
            state.set(detached=str(holder))  # before the act: a crash between still re-attaches
        m.act(['git', '-C', holder, 'switch', '--detach'])


def _worktree_on(m: Machine, ref: str) -> Path | None:
    path = None
    for line in m.git('worktree', 'list', '--porcelain').splitlines():
        if line.startswith('worktree '):
            path = Path(line[len('worktree '):])
        elif line == f'branch {ref}':
            return path
    return None


def _unchanged_since_saved(m: Machine, state: State) -> list[str]:
    """The dirty paths, once proven to be exactly what the saved patch holds:
    anything edited after the patch was taken stops the switch before it
    discards anything."""
    dirty = sorted(_dirty(m))
    if m.dry_run:
        return dirty
    finding = checks.check_dirty(set(dirty))
    patch = Path(state.data['patch'])
    now_diff = m.runner.capture(['git', '--no-optional-locks', '-C', m.checkout, 'diff', 'HEAD', '--binary'])
    if not now_diff and (not patch.stat().st_size or m.runner.run(
            ['git', '-C', m.checkout, 'apply', '--check', patch], check=False) == 0):
        return []  # already discarded: a retry after a crash just past the discard
    if not finding.ok or now_diff != patch.read_text():
        raise RuntimeError('the checkout changed after the patch was saved; nothing discarded. '
                           f"Compare `git -C {m.checkout} diff HEAD` with {state.data['patch']}")
    return dirty


def discard_edits(m: Machine, state: State) -> None:
    dirty = _unchanged_since_saved(m, state)
    if dirty:
        m.act(['git', '-C', m.checkout, 'checkout', 'HEAD', '--', *dirty])


def move_to_target(m: Machine, state: State) -> None:
    """Each command is safe to repeat, so a retry after a crash part way
    through lands in the same place."""
    target = state.data['target']
    if m.git('rev-parse', '--verify', '--quiet', f'refs/heads/{TARGET_BRANCH}', check=False):
        m.act(['git', '-C', m.checkout, 'switch', TARGET_BRANCH])
        m.act(['git', '-C', m.checkout, 'merge', '--ff-only', target])
    else:
        m.act(['git', '-C', m.checkout, 'switch', '-c', TARGET_BRANCH, '--track', 'origin/main'])
        m.act(['git', '-C', m.checkout, 'merge', '--ff-only', target])


def uv_sync(m: Machine) -> None:
    # Without -u, the caller's own venv (VIRTUAL_ENV from `uv run`) could be
    # the one synced instead of the checkout's.
    m.act(['env', '-u', 'VIRTUAL_ENV', '-u', 'UV_PROJECT_ENVIRONMENT', 'uv', 'sync', '--locked', '--extra', 'dev'],
          cwd=m.checkout)


def _unit_path(m: Machine, unit: str) -> str:
    """PATH as the unit runs with it, so the build uses the same node."""
    env = m.unit(unit, 'Environment')
    found = re.search(r'(?:^|\s)PATH=(\S+)', env)
    return found.group(1) if found else os.environ.get('PATH', '')


def build_web(m: Machine) -> None:
    path = _unit_path(m, 'gmail-search-web.service')
    m.act(['env', f'PATH={path}', m.checkout / 'web/node_modules/.bin/next', 'build'], cwd=m.checkout / 'web')


def restore_web_build(m: Machine, state: State) -> None:
    backup = Path(state.data['web_backup'])
    if not backup.exists() and not m.dry_run:
        raise RuntimeError(f'no web build backup at {backup}')
    m.act(['rm', '-rf', m.checkout / 'web/.next'])
    m.act(['cp', '-a', backup, m.checkout / 'web/.next'])


def move_back(m: Machine, state: State) -> None:
    d = state.data
    m.act(['git', '-C', m.checkout, 'switch', d['old_branch']])
    if m.git('rev-parse', 'HEAD') != d['old_head'] and not m.dry_run:
        raise RuntimeError(f"{d['old_branch']} moved since the switch; restore by hand")
    restore_patch(m, Path(d['patch']))
    # Gone once the post-switch cleanup removed the loop's main worktree.
    if d.get('detached') and Path(d['detached']).exists():
        m.act(['git', '-C', d['detached'], 'switch', TARGET_BRANCH])


def restore_patch(m: Machine, patch: Path) -> None:
    """Re-apply the saved edits (the policy scoping among them) before any unit
    starts; already applied is fine, anything else stops the rollback."""
    if m.dry_run:
        m.act(['git', '-C', m.checkout, 'apply', patch])
        return
    if not patch.exists():
        raise RuntimeError(f'{patch} is missing: the saved edits cannot be restored. Resolve by hand.')
    if not patch.stat().st_size:
        say('no saved edits to restore')
        return
    applies = ['git', '-C', m.checkout, 'apply', '--check']
    if m.runner.run([*applies, '--reverse', patch], check=False) == 0:
        say(f'{patch.name} is already applied')
    elif m.runner.run([*applies, patch], check=False) == 0:
        m.act(['git', '-C', m.checkout, 'apply', patch])
    else:
        raise RuntimeError(f'{patch} neither applies nor is applied: the tree has other edits. '
                           'Resolve by hand before starting any unit.')


# ── commands ─────────────────────────────────────────────────────────
def new_state(m: Machine, state: State, target: str, before: dict, saved_patch: str | None = None) -> None:
    stamp = now().replace(':', '')
    base = state.path.parent
    state.set(started=now(), old_branch=m.git('rev-parse', '--abbrev-ref', 'HEAD'),
              old_head=m.git('rev-parse', 'HEAD'), target=target,
              patch=str(base / f'uncommitted-{stamp}.patch'), web_backup=str(base / f'web-next-{stamp}'),
              baseline=before, done=[], saved_patch=saved_patch)


def switch(m: Machine, state: State, args) -> int:
    if state.data.get('finished'):
        say(f"already switched at {state.data['finished']} ({state.path}); nothing to do")
        return 0
    if state.data.get('done'):
        before, target = state.data['baseline'], state.data['target']
        say(f"resuming the switch to {target[:12]}; done: {', '.join(state.data['done'])}")
    else:
        m.act(['git', '-C', m.checkout, 'fetch', 'origin'])
        target = m.git('rev-parse', args.target)
        before = baseline(m)
        if not report([*preconditions(m, target, args), *healthy_baseline(before)]):
            return 2
        if m.dry_run:
            # A stand-in that is never written: a dry run records nothing.
            state = State(state.path.with_name('dry-run.json'))
            state.data = {'patch': '<patch>', 'web_backup': '<web backup>', 'target': target,
                          'saved_patch': args.saved_patch and str(args.saved_patch)}
        else:
            new_state(m, state, target, before, args.saved_patch and str(args.saved_patch.resolve()))
    shared = sharing_pids(m)
    for other in _other_sessions(m):
        say(f'note: another process works in the checkout: {other}')
    _step(state, 'save patch', lambda: save_patch(m, state), m)
    _step(state, 'back up web build', lambda: backup_web_build(m, state), m)
    _step(state, 'stop owner units', lambda: stop_units(m), m)
    _step(state, 'release main', lambda: release_main_elsewhere(m, state), m)
    _step(state, 'discard the saved edits', lambda: discard_edits(m, state), m)
    _step(state, 'check out main', lambda: move_to_target(m, state), m)
    _step(state, 'uv sync', lambda: uv_sync(m), m)
    _step(state, 'build web', lambda: build_web(m), m)
    _step(state, 'start owner units', lambda: start_units(m, before), m)
    if not m.dry_run:
        sharing_units_untouched(m, shared)
        state.set(finished=now())
    say(f"switched: {m.checkout} on {TARGET_BRANCH} at {target[:12]}" if not m.dry_run else 'dry run: nothing changed')
    return 0


def rollback(m: Machine, state: State, args) -> int:
    done = state.data.get('done', [])
    if 'stop owner units' not in done:
        raise RuntimeError(f'nothing to roll back: {state.path} records no change to the checkout or units')
    before, shared = state.data['baseline'], sharing_pids(m)
    back = State(state.path.with_name('rollback.json'))
    say(f"rolling back to {state.data['old_branch']} at {state.data['old_head'][:12]}")
    _step(back, 'stop owner units', lambda: stop_units(m), m)
    if {'release main', 'discard the saved edits', 'check out main'} & set(done) or state.data.get('detached'):
        _step(back, 'check out the old branch', lambda: move_back(m, state), m)
    _step(back, 'uv sync', lambda: uv_sync(m), m)
    if 'build web' in done:
        _step(back, 'restore web build', lambda: restore_web_build(m, state), m)
    _step(back, 'start owner units', lambda: start_units(m, before), m)
    if not m.dry_run:
        sharing_units_untouched(m, shared)
        archive(state, back)
    return 0


def archive(state: State, back: State) -> None:
    """A finished rollback leaves no state behind, so the next switch starts
    fresh; both records are kept beside it."""
    stamp = now().replace(':', '')
    for record in (state, back):
        record.path.rename(record.path.with_name(f'{record.path.stem}-rolled-back-{stamp}.json'))
    say(f'rolled back; records kept in {state.path.parent}')


def check(m: Machine, state: State, args) -> int:
    target = m.git('rev-parse', args.target)
    say(f"{m.checkout}: {m.git('rev-parse', '--abbrev-ref', 'HEAD')} at {m.git('rev-parse', '--short', 'HEAD')}, "
        f'target {args.target} at {target[:12]}')
    ok = report(preconditions(m, target, args))
    for other in _other_sessions(m):
        print(f'note another process works in the checkout: {other}')
    return 0 if ok else 2


def parse_args(argv):
    p = argparse.ArgumentParser(prog='scripts/one-checkout.sh', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=['check', 'switch', 'rollback'])
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--checkout', type=Path, help='the production checkout (default: the one holding .git)')
    p.add_argument('--target', default='origin/main')
    p.add_argument('--after-issue', type=int, action='append', default=None,
                   help='an issue that must be closed or landed first (default: 27 and 55)')
    p.add_argument('--allow-unmerged', type=lambda s: [x for x in s.split(',') if x], default=[],
                   help='comma-separated shas (7+ chars) on the old branch that may be left behind')
    p.add_argument('--schema-reviewed', default=checks.REVIEWED_SCHEMA_BLOB,
                   help="the target pg_schema.sql blob someone has checked against the live catalog")
    p.add_argument('--saved-patch', type=Path,
                   help='the edits were already saved to this patch and discarded: rollback re-applies it')
    p.add_argument('--config', type=Path, default=DEFAULT_PATH)
    args = p.parse_args(argv)
    args.after_issue = args.after_issue or [27, 55]
    if any(len(s) < 7 for s in args.allow_unmerged):
        p.error('--allow-unmerged takes shas of at least 7 characters')
    return args


def production_checkout(runner: Runner) -> Path:
    common = runner.capture(['git', 'rev-parse', '--path-format=absolute', '--git-common-dir']).strip()
    return Path(common).parent


def _drop_inherited_git_env() -> None:
    """Run from a git hook, GIT_DIR and friends would aim every git command
    here at the hook's repository instead of the checkout named by -C."""
    for name in [n for n in os.environ if n.startswith('GIT_')]:
        del os.environ[name]


def run(argv) -> int:
    _drop_inherited_git_env()
    args = parse_args(argv)
    runner = Runner()
    checkout = (args.checkout or production_checkout(runner)).resolve()
    here = Path(runner.capture(['git', 'rev-parse', '--show-toplevel']).strip()).resolve()
    if here == checkout and args.command != 'check':
        raise RuntimeError(f'run this from a worktree of main, not from {checkout} itself')
    config = load_config(args.config)
    m = Machine(checkout, runner, dry_run=args.dry_run, health_host=config.health_host, registry=config.registry)
    state = State(checkout / '.runtime/one-checkout/state.json')
    return {'check': check, 'switch': switch, 'rollback': rollback}[args.command](m, state, args)


def main(argv=None) -> int:
    try:
        return run(sys.argv[1:] if argv is None else argv)
    except (RuntimeError, ValueError, CommandFailed, OSError) as error:
        print(f'one-checkout FAILED: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
