"""scripts/deploy.sh [--dry-run] [--name <word>] [--target <ref>] [--phase <name>] [--release <name>]

Phases: plan, package, qualify, preflight, activate, postcheck (all by default);
`--phase rollback|clean --release <name>` act on an earlier run. --dry-run runs
plan..preflight into .runtime/deploy/dry-run/ and never swaps or restarts.
Also: --update-pin (rebuild the worker image from this checkout and write
AGENT_FULL_PIN, uncommitted) and --seed-image-inputs <extracted image root>.
"""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys
import tempfile

from . import image, lock, phases
from .config import DEFAULT_PATH, load_config
from .host import Host
from .runner import CommandFailed, Runner

PHASES = ['plan', 'package', 'qualify', 'preflight', 'activate', 'postcheck']
EXTRA = ['rollback', 'clean']


def parse_args(argv):
    p = argparse.ArgumentParser(prog='scripts/deploy.sh', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--name', default='loop')
    p.add_argument('--target', default='origin/main')
    p.add_argument('--phase', default='all', choices=['all', *PHASES, *EXTRA])
    p.add_argument('--release')
    p.add_argument('--config', type=Path, default=DEFAULT_PATH)
    p.add_argument('--update-pin', action='store_true')
    p.add_argument('--seed-image-inputs', type=Path, metavar='ROOT')
    return p.parse_args(argv)


def phases_for(phase: str, dry_run: bool) -> list[str]:
    wanted = PHASES if phase == 'all' else [phase]
    if not dry_run:
        return wanted
    kept = [name for name in wanted if name not in ('activate', 'postcheck', 'rollback')]
    if not kept:
        raise SystemExit(f'--phase {phase} cannot run in --dry-run')
    return kept


def main_checkout(runner: Runner) -> Path:
    common = runner.capture(['git', 'rev-parse', '--path-format=absolute', '--git-common-dir']).strip()
    return Path(common).parent


def state_for(main: Path, release: str) -> phases.State:
    return phases.State(main / '.runtime/deploy' / release / 'state.json')


def under_land_lock(main: Path, release: str, step) -> None:
    """Activate and rollback restart services: never alongside a landing or
    the owner track cutover (scripts/owner-cutover.sh), which hold the same lock."""
    with lock.held(main / '.runtime/issue-loop/land.lock', f'deploy {release}', wait_seconds=1800):
        step()


def run(argv) -> int:
    args = parse_args(argv)
    runner = Runner()
    repo = Path(runner.capture(['git', 'rev-parse', '--show-toplevel']).strip())
    config = load_config(args.config)
    if args.seed_image_inputs:
        print(image.seed_inputs(repo, config.image_inputs_cache, args.seed_image_inputs.resolve()))
        return 0
    if args.update_pin:
        with tempfile.TemporaryDirectory(dir=repo / '.runtime' if (repo / '.runtime').exists() else None) as work:
            _, sha = image.build(repo, config.image_inputs_cache, Path(work), runner, log_dir=Path(work) / 'logs')
        image.write_pin(repo, sha)
        print(f'AGENT_FULL_PIN = {sha} (written, not committed)')
        return 0
    host, main = Host(config, runner), main_checkout(runner)
    wanted = phases_for(args.phase, args.dry_run)
    last = main / '.runtime/deploy/last.json'
    if wanted[0] == 'plan':
        planned = phases.plan_phase(host, main, args.target, args.name, date.today())
        print(f"plan: {planned['action']}  target={planned['target'][:7]} running={str(planned['running'])[:7]} "
              f"kinds={','.join(planned['kinds']) or '-'}")
        if planned['action'] != 'ship':
            return 0 if not planned['action'].startswith('refuse') else 2
        release = planned['release'] + ('-dry' if args.dry_run else '')
        state = state_for(main, release)
        state.set(**{**planned, 'release': release, 'dryRun': args.dry_run})
        last.parent.mkdir(parents=True, exist_ok=True)
        last.write_text(json.dumps({'release': release}) + '\n')
    else:
        release = args.release or json.loads(last.read_text())['release']
        state = state_for(main, release)
    for name in wanted[1:] if wanted[0] == 'plan' else wanted:
        {'package': lambda: phases.package_phase(host, main, state, dry_run=args.dry_run),
         'qualify': lambda: phases.qualify_phase(host, main, state),
         'preflight': lambda: phases.preflight_phase(host, main, state),
         'activate': lambda: under_land_lock(main, release, lambda: phases.activate_phase(host, state)),
         'postcheck': lambda: phases.postcheck_phase(host, main, state),
         'rollback': lambda: under_land_lock(main, release, lambda: phases.rollback_phase(host, state)),
         'clean': lambda: phases.clean_phase(host, main, state)}[name]()
    return 0


def main(argv=None) -> int:
    try:
        return run(sys.argv[1:] if argv is None else argv)
    except (RuntimeError, ValueError, CommandFailed, OSError) as error:
        print(f'deploy FAILED: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
