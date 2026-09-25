"""scripts/owner-cutover.sh [--dry-run] <check|switch|rollback> [options]

Moves the owner daemons (serve, mcp, the owner web, supervise, and the serve
watchdog) off the checkout's code and onto the deployer's owner track, once
(#52). After it, every deploy that changes their code ships it to
`owner-current` and restarts them; docs/one-checkout-runbook.md ("Owner
track cutover") is the procedure.

  check     read-only preconditions; exit 0 only when switch may run
  switch    build the first owner release at invited-current's commit, save
            the unit files, point owner-current at the release, rewrite the
            units to run from it, daemon-reload, restart in order, verify
  rollback  put the saved unit files back, restart in order, remove owner-current

The units keep the checkout as their working directory (data/, config.yaml,
the claudebox and pi mounts are state there); only ExecStart, the venv on
PATH, the web's directory and the watchdog script move to owner-current.
--dry-run prints what switch or rollback would do and changes nothing. Each
step is recorded in <checkout>/.runtime/owner-cutover/state.json, so a run
that stopped part way re-runs from where it stopped.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import sys

from . import checkout_checks as checks
from . import lock, plan
from .config import DEFAULT_PATH, DeployConfig, load_config
from .host import Host
from .one_checkout import _step, archive, production_checkout, report
from .owner_units import OWNER_UNITS, Machine, baseline, healthy_baseline, restart_each, say, watchdog_held
from .phases import CHECKS_PATH, CUTOVER_DIR, State, now, package_owner, swap_link
from .runner import CommandFailed, Runner

UNIT_DIR = Path('~/.config/systemd/user')
WATCHDOG_SERVICE = 'gmail-search-serve-watchdog.service'
# What moves to the release, by the path after the checkout: the venv (ExecStart
# and supervise's PATH), the web's next binary, and the watchdog script.
MOVED = ('/.venv/', '/web/node_modules/', '/scripts/serve_watchdog.sh')
# What stays in the checkout: the working directory itself and data/.
KEPT = re.compile(r'^(/data(/\S*)?)?$')


# ── the unit files ───────────────────────────────────────────────────
def spellings(old: Path, new: Path) -> list[tuple[str, str]]:
    """(old, new) as a unit file may spell them: %h-relative (the units' own
    style) when both are under home, and absolute."""
    home = Path.home()
    relative = old.is_relative_to(home) and new.is_relative_to(home)
    return ([(f'%h/{old.relative_to(home)}', f'%h/{new.relative_to(home)}')] if relative else []) + [(str(old), str(new))]


def rewrite_unit(text: str, checkout: Path, owner: Path) -> str:
    for old, new in spellings(checkout, owner):
        for tail in MOVED:
            text = text.replace(old + tail, new + tail)
        text = re.sub(rf'^WorkingDirectory={re.escape(old)}/web$', f'WorkingDirectory={new}/web', text, flags=re.M)
    return text


def leftovers(text: str, checkout: Path) -> list[str]:
    """Directive lines still naming the checkout for anything but its state:
    a unit shaped in a way this rewrite does not know."""
    bad = []
    for line in text.splitlines():
        if line.lstrip().startswith(('#', ';')):
            continue
        for old, _ in spellings(checkout, checkout):
            for found in re.finditer(re.escape(old) + r'(?=/|\s|$)(\S*)', line):
                if not KEPT.match(found.group(1)) or (not found.group(1) and not line.startswith('WorkingDirectory=')):
                    bad.append(line.strip())
    return bad


def unit_files(unit_dir: Path) -> list[Path]:
    """Every file the cutover rewrites: each owner unit, its drop-ins, and the
    watchdog's service."""
    files = []
    for unit in (*OWNER_UNITS, WATCHDOG_SERVICE):
        files.append(unit_dir / unit)
        files.extend(sorted((unit_dir / f'{unit}.d').glob('*.conf')))
    return files


def check_units(unit_dir: Path, checkout: Path, owner: Path) -> list[checks.Finding]:
    found = []
    for path in unit_files(unit_dir):
        name = str(path.relative_to(unit_dir))
        if not path.exists():
            found.append(checks.Finding(name, False, 'missing'))
            continue
        text = path.read_text()
        new = rewrite_unit(text, checkout, owner)
        bad = leftovers(new, checkout)
        found.append(checks.Finding(name, new != text and not bad,
                                    'still names the checkout: ' + ' | '.join(bad) if bad
                                    else 'nothing to move' if new == text else 'rewrites cleanly'))
    return found


# ── check ────────────────────────────────────────────────────────────
def invited_commit(config: DeployConfig) -> tuple[str, Path]:
    running = config.current_links[0].resolve()
    commit = plan.running_commit(running)
    if not commit:
        raise RuntimeError(f'{running} records no commit')
    return commit, running


def check_schema(m: Machine, commit: str) -> checks.Finding:
    blob = m.git('rev-parse', f'{commit}:{checks.SCHEMA_PATH}')
    source = m.git('show', f'{commit}:{CHECKS_PATH}', check=False)
    ok = plan.schema_reviewed(blob, source)
    return checks.Finding('startup schema', ok, f'{checks.SCHEMA_PATH} {blob[:12]} is '
                          + ('the blob that commit declares reviewed' if ok else 'not declared reviewed at that commit'))


def preconditions(m: Machine, config: DeployConfig, unit_dir: Path) -> list[checks.Finding]:
    commit, running = invited_commit(config)
    link = config.owner_current
    return [checks.Finding('owner-current', not link.is_symlink(),
                           f'already points at {link.resolve()}' if link.is_symlink() else 'absent'),
            checks.Finding('first release', True, f'{running.name} at {commit[:12]} (invited-current)'),
            check_schema(m, commit),
            *check_units(unit_dir, config.owner.checkout, link)]


# ── the steps ────────────────────────────────────────────────────────
def build_release(m: Machine, host: Host, state: State) -> None:
    d = state.data
    out, tree = Path(d['release_dir']), Path(d['worktree'])
    if m.dry_run:
        say(f"would build {out} from {d['commit'][:12]} (git archive, uv sync --locked, next build)")
        return
    if not tree.exists():
        m.act(['git', '-C', m.checkout, 'worktree', 'add', '--detach', tree, d['commit']])
    package_owner(host, tree, out, d['commit'], state.path.parent / 'logs')
    qualified = json.loads((Path(d['invited_dir']) / 'QUALIFIED.json').read_text())
    (out / 'QUALIFIED.json').write_text(json.dumps({**qualified, 'ownerCutoverAt': now()}, indent=2) + '\n')
    (out / 'RELEASE').write_text(f"{d['commit'][:7]}: {out.name} (owner cutover) {now()}\n")
    m.act(['git', '-C', m.checkout, 'worktree', 'remove', '--force', tree])


def save_units(m: Machine, state: State, unit_dir: Path) -> None:
    saved = state.path.parent / 'units-before'
    for path in unit_files(unit_dir):
        copy = saved / path.relative_to(unit_dir)
        say(f"{'would save' if m.dry_run else 'save'} {path} -> {copy}")
        if not m.dry_run:
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, copy)


def write_units(m: Machine, state: State, config: DeployConfig, unit_dir: Path) -> None:
    """From the saved copies, so a retry never rewrites a rewritten file."""
    saved = state.path.parent / 'units-before'
    for path in unit_files(unit_dir):
        say(f"{'would rewrite' if m.dry_run else 'rewrite'} {path}")
        if not m.dry_run:
            text = (saved / path.relative_to(unit_dir)).read_text()
            _write_aside(path, rewrite_unit(text, config.owner.checkout, config.owner_current))


def restore_units(m: Machine, state: State, unit_dir: Path) -> None:
    saved = state.path.parent / 'units-before'
    for path in unit_files(unit_dir):
        say(f"{'would restore' if m.dry_run else 'restore'} {path}")
        if not m.dry_run:
            _write_aside(path, (saved / path.relative_to(unit_dir)).read_text())


def _write_aside(path: Path, text: str) -> None:
    scratch = path.with_name(path.name + '.cutover-tmp')
    scratch.write_text(text)
    shutil.copymode(path, scratch)
    scratch.replace(path)


def point_owner_current(m: Machine, config: DeployConfig, release: Path) -> None:
    say(f"{'would point' if m.dry_run else 'point'} {config.owner_current} -> {release}")
    if not m.dry_run:
        swap_link(config.owner_current, release)


def remove_owner_current(m: Machine, config: DeployConfig) -> None:
    link = config.owner_current
    if not link.is_symlink():
        return
    if not link.resolve().is_relative_to(config.owner_releases):
        raise RuntimeError(f'{link} points outside {config.owner_releases}; resolve by hand')
    say(f"{'would remove' if m.dry_run else 'remove'} {link}")
    if not m.dry_run:
        link.unlink()


def verify_loaded(m: Machine, config: DeployConfig) -> None:
    """systemd runs the rewritten units: every ExecStart goes through owner-current."""
    for unit in (*OWNER_UNITS, WATCHDOG_SERVICE):
        start = m.unit(unit, 'ExecStart')
        if str(config.owner_current) not in start:
            raise RuntimeError(f'{unit} ExecStart does not go through {config.owner_current}: {start[:200]}')
        say(f'{unit} runs from {config.owner_current}')


def reload_and_restart(m: Machine, before: dict) -> None:
    """The watchdog held from before systemd loads the rewritten units."""
    with watchdog_held(m):
        m.act(['systemctl', '--user', 'daemon-reload'])
        restart_each(m, before)


# ── commands ─────────────────────────────────────────────────────────
def switch(m: Machine, host: Host, state: State, args) -> int:
    config = host.config
    if state.data.get('finished'):
        say(f"already cut over at {state.data['finished']} ({state.path}); nothing to do")
        return 0
    if state.data.get('done'):
        say(f"resuming the cutover; done: {', '.join(state.data['done'])}")
    else:
        before = baseline(m, OWNER_UNITS)
        if not report([*preconditions(m, config, args.unit_dir), *healthy_baseline(before)]):
            return 2
        commit, invited = invited_commit(config)
        fresh = {'started': now(), 'commit': commit, 'invited_dir': str(invited), 'baseline': before, 'done': [],
                 'release_dir': str(config.owner_releases / invited.name),
                 'worktree': str(Path.home() / '.wt' / f'owner-cutover-{invited.name}')}
        if m.dry_run:
            state = State(state.path.with_name('dry-run.json'))  # never written
            state.data = fresh
        else:
            state.set(**fresh)
    d = state.data
    _step(state, 'build the owner release', lambda: build_release(m, host, state), m)
    _step(state, 'save the unit files', lambda: save_units(m, state, args.unit_dir), m)
    _step(state, 'point owner-current', lambda: point_owner_current(m, config, Path(d['release_dir'])), m)
    _step(state, 'rewrite the unit files', lambda: write_units(m, state, config, args.unit_dir), m)
    _step(state, 'restart owner units', lambda: reload_and_restart(m, d['baseline']), m)
    if not m.dry_run:
        verify_loaded(m, config)
        state.set(finished=now())
    say(f"cut over: owner daemons run {Path(d['release_dir']).name}" if not m.dry_run else 'dry run: nothing changed')
    return 0


def rollback(m: Machine, host: Host, state: State, args) -> int:
    done = state.data.get('done', [])
    if 'save the unit files' not in done:
        raise RuntimeError(f'nothing to roll back: {state.path} records no saved unit files')
    back = State(state.path.with_name('rollback.json'))
    say('rolling back the owner track cutover')
    _step(back, 'restore the unit files', lambda: restore_units(m, state, args.unit_dir), m)
    _step(back, 'restart owner units', lambda: reload_and_restart(m, state.data['baseline']), m)
    _step(back, 'remove owner-current', lambda: remove_owner_current(m, host.config), m)
    if not m.dry_run:
        archive(state, back)
    return 0


def check(m: Machine, host: Host, state: State, args) -> int:
    return 0 if report([*preconditions(m, host.config, args.unit_dir),
                        *healthy_baseline(baseline(m, OWNER_UNITS))]) else 2


def parse_args(argv):
    p = argparse.ArgumentParser(prog='scripts/owner-cutover.sh', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=['check', 'switch', 'rollback'])
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--config', type=Path, default=DEFAULT_PATH)
    p.add_argument('--unit-dir', type=Path, default=UNIT_DIR)
    args = p.parse_args(argv)
    args.unit_dir = args.unit_dir.expanduser()
    return args


def run(argv) -> int:
    args = parse_args(argv)
    runner = Runner()
    config = load_config(args.config)
    if config.owner is None:
        raise RuntimeError(f"{args.config}: no 'owner' block (see deploy/deploy.example.json)")
    repo = production_checkout(runner).resolve()
    m = Machine(repo, runner, dry_run=args.dry_run, health_host=config.health_host)
    host = Host(config, runner, owner_units=m)
    state = State(config.owner.checkout / CUTOVER_DIR / 'state.json')
    command = {'check': check, 'switch': switch, 'rollback': rollback}[args.command]
    if args.command == 'check' or args.dry_run:
        return command(m, host, state, args)
    # No landing (and so no deploy's qualify) meanwhile; a deploy's plan and
    # activate refuse while the records say a cutover is under way.
    with lock.held(repo / '.runtime/issue-loop/land.lock', f'owner-cutover {args.command}'):
        return command(m, host, state, args)


def main(argv=None) -> int:
    try:
        return run(sys.argv[1:] if argv is None else argv)
    except (RuntimeError, ValueError, CommandFailed, OSError) as error:
        print(f'owner-cutover FAILED: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
