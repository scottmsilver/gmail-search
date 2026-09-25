"""The six phases (plan, package, qualify, preflight, activate, postcheck) plus
rollback and clean. Each reads and writes the run's State; the deployer's
decisions live in plan.py and image.py."""
from __future__ import annotations

from datetime import datetime, timezone
import glob
import json
import os
from pathlib import Path
import shlex
import shutil

from . import image, lock, notes, plan
from .checkout_checks import SCHEMA_PATH
from .config import read_env_file
from .host import Host
from .owner_units import OWNER_UNITS, WEB_UNIT, Machine, baseline, healthy_baseline, restart_each, watchdog_held

say = lambda message: print(f'==> {message}', flush=True)  # noqa: E731
CHECKS_PATH = 'src/gmail_search/deploy/checkout_checks.py'
# owner_cutover's records, under the owner checkout.
CUTOVER_DIR = Path('.runtime/owner-cutover')
VALID_CONVERSATION = '/c/abc123def456'
INVALID_CONVERSATION = '/c/bad!'


def now() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


class State:
    """This run's record, `.runtime/deploy/<release>/state.json`."""

    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text()) if path.exists() else {}

    def set(self, **patch):
        self.data.update(patch)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Written aside and renamed, so a crash mid-write leaves the old record.
        scratch = self.path.with_name(self.path.name + '.tmp')
        scratch.write_text(json.dumps(self.data, indent=2) + '\n')
        os.replace(scratch, self.path)

    def need(self, *keys):
        missing = [k for k in keys if k not in self.data]
        if missing:
            raise RuntimeError(f'no {", ".join(missing)} in {self.path}: run the earlier phases first')
        return self.data


def git(host: Host, repo: Path, *args, check=True) -> str:
    return host.runner.capture(['git', '-C', repo, *args], check=check).strip()


def is_ancestor(host: Host, repo: Path, older: str, newer: str) -> bool:
    return host.runner.run(['git', '-C', repo, 'merge-base', '--is-ancestor', older, newer], check=False) == 0


# ── plan ─────────────────────────────────────────────────────────────
def running_release(host: Host) -> Path:
    return host.config.current_links[0].resolve()


def plan_phase(host: Host, main: Path, target: str, word: str, today, *, fetch=True) -> dict:
    if fetch:
        host.runner.run(['git', '-C', main, 'fetch', '-q', 'origin'])
    target_sha = git(host, main, 'rev-parse', f'{target}^{{commit}}')
    running_dir = running_release(host)
    running = plan.running_commit(running_dir)
    if running:
        running = git(host, main, 'rev-parse', f'{running}^{{commit}}')
    paths = git(host, main, 'diff', '--name-only', f'{running}..{target_sha}').split() if running else []
    guest = image.parse_guest_files(git(host, main, 'show', f'{target_sha}:{image.PREPARE}'))
    owner = _owner_plan(host, main, target_sha)
    batch = (plan.classify(paths, guest, owner_track=owner_track_on(host)) if running
             else plan.Batch(manual=['<unknown running commit>']))
    if owner.get('ship'):
        batch.kinds.add(plan.OWNER)
    action = plan.decide(running, target_sha, batch=batch,
                         target_contains_running=bool(running) and is_ancestor(host, main, running, target_sha),
                         running_contains_target=bool(running) and is_ancestor(host, main, target_sha, running))
    if action == 'skip' and running == target_sha and owner.get('ship'):
        action = 'ship'  # the invited stack is current, the owner track is not
    if owner.get('refuse') and (action == 'ship' or running == target_sha):
        action = 'refuse:' + owner['refuse']
    taken = {p.name for p in host.config.releases.iterdir()} if host.config.releases.exists() else set()
    return {'action': action, 'target': target_sha, 'running': running, 'runningDir': str(running_dir),
            'kinds': sorted(batch.kinds), 'manual': batch.manual, 'paths': paths,
            'release': plan.release_name(word, today, taken),
            **({'ownerRunningDir': owner['runningDir']} if owner.get('runningDir') else {})}


def owner_track_on(host: Host) -> bool:
    """The owner track ships once the one-time cutover (owner_cutover) made
    owner-current; until then deploys leave the owner daemons alone."""
    return host.config.owner is not None and host.config.owner_current.is_symlink()


def owner_unchanged_since_plan(host: Host, data: dict) -> None:
    """The owner track as plan saw it: no cutover under way, and owner-current
    (or its absence) the same. A deploy planned before a cutover must not
    activate in the middle of one."""
    if host.config.owner is None:
        return
    if cutover_busy(host):
        raise RuntimeError('the owner track cutover has started and not finished; nothing changed')
    now_ = str(host.config.owner_current.resolve()) if owner_track_on(host) else None
    if now_ != data.get('ownerRunningDir'):
        raise RuntimeError('the running owner release changed since plan; plan again')


def cutover_busy(host: Host) -> bool:
    """A cutover switch or rollback that started and has not finished: the
    unit files and owner-current may disagree until it does."""
    records = host.config.owner.checkout / CUTOVER_DIR
    state = records / 'state.json'
    switching = state.exists() and not json.loads(state.read_text()).get('finished')
    return switching or (records / 'rollback.json').exists()


def _owner_plan(host: Host, main: Path, target_sha: str) -> dict:
    """{} (no track), {'refuse': why}, or the running owner release and
    whether the target changes anything the owner daemons run."""
    if not owner_track_on(host):
        return {}
    if cutover_busy(host):
        return {'refuse': 'the owner track cutover has started and not finished; finish or roll it back'}
    running_dir = host.config.owner_current.resolve()
    running = plan.running_commit(running_dir)
    if not running:
        return {'refuse': f'{running_dir} records no commit'}
    running = git(host, main, 'rev-parse', f'{running}^{{commit}}')
    if not is_ancestor(host, main, running, target_sha):
        return {'refuse': 'target does not contain the owner release commit'}
    if not plan.classify_owner(git(host, main, 'diff', '--name-only', f'{running}..{target_sha}').split()):
        return {'runningDir': str(running_dir)}
    schema = git(host, main, 'rev-parse', f'{target_sha}:{SCHEMA_PATH}')
    if not plan.schema_reviewed(schema, git(host, main, 'show', f'{target_sha}:{CHECKS_PATH}', check=False)):
        return {'refuse': f'{SCHEMA_PATH} at the target is not the blob its REVIEWED_SCHEMA_BLOB names; '
                          'serve runs it on the live database at boot, so review it and update that constant'}
    return {'ship': True, 'runningDir': str(running_dir)}


# ── package ──────────────────────────────────────────────────────────
def build_worktree(release: str, dry_run: bool) -> Path:
    return Path('~/.wt').expanduser() / (f'deploy-dry-run-{release}' if dry_run else f'deploy-{release}')


def release_dir(host: Host, main: Path, release: str, dry_run: bool) -> Path:
    return (main / '.runtime/deploy/dry-run' / release) if dry_run else host.config.releases / release


def owner_release_dir(host: Host, main: Path, release: str, dry_run: bool) -> Path:
    return (main / '.runtime/deploy/dry-run/owner' / release) if dry_run else host.config.owner_releases / release


def package_phase(host: Host, main: Path, state: State, *, dry_run: bool) -> None:
    data = state.need('target', 'release', 'kinds', 'runningDir')
    tree = build_worktree(data['release'], dry_run)
    if not tree.exists():
        host.runner.run(['git', '-C', main, 'worktree', 'add', '--detach', tree, data['target']])
    out = release_dir(host, main, data['release'], dry_run)
    logs = state.path.parent / 'logs'
    shutil.rmtree(out, ignore_errors=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(data['runningDir'], out, symlinks=True)
    shutil.rmtree(out / 'api/src', ignore_errors=True)
    shutil.copytree(tree / 'src', out / 'api/src', ignore=shutil.ignore_patterns('__pycache__'))
    (out / 'QUALIFIED.json').unlink(missing_ok=True)
    if plan.WEB in data['kinds']:
        _package_web(host, tree, out, logs)
    _package_notes(host, main, data, out)
    worker = _package_worker(host, tree, state, data['kinds'], logs)
    (out / 'RELEASE').write_text(f"{data['target'][:7]}: {data['release']} {now()}\n")
    _private(out)
    owner = {}
    if plan.OWNER in data['kinds']:
        owner_out = owner_release_dir(host, main, data['release'], dry_run)
        package_owner(host, tree, owner_out, data['target'], logs)
        (owner_out / 'RELEASE').write_text(f"{data['target'][:7]}: {data['release']} {now()}\n")
        owner = {'ownerReleaseDir': str(owner_out)}
    state.set(worktree=str(tree), releaseDir=str(out), **worker, **owner)
    say(f"packaged {data['release']} ({', '.join(data['kinds'])}) at {out}")


def _package_web(host: Host, tree: Path, out: Path, logs: Path) -> None:
    shutil.rmtree(out / 'web', ignore_errors=True)
    shutil.copytree(tree / 'web', out / 'web', ignore=shutil.ignore_patterns('node_modules', '.next'))
    (out / 'web/node_modules').symlink_to(host.config.web_node_modules)
    env = read_env_file(host.config.public_web_env)
    host.runner.run([out / 'web/node_modules/.bin/next', 'build'], cwd=out / 'web', env=env,
                    log=logs / 'next-build.log')


def package_owner(host: Host, tree: Path, out: Path, commit: str, logs: Path) -> None:
    """An owner release: the commit's tracked tree, a venv of its own synced
    from its lockfile, and the owner web built against the owner's .env.local
    with the node its unit runs."""
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    # Not _private: the venv's files are hardlinks into the uv cache.
    out.chmod(0o700)
    logs.mkdir(parents=True, exist_ok=True)
    archive = logs / 'owner-tree.tar'
    host.runner.run(['git', '-C', tree, 'archive', '--format=tar', '-o', archive, commit])
    host.runner.run(['tar', '-xf', archive, '-C', out])
    archive.unlink(missing_ok=True)
    # Without -u, the deployer's own venv (VIRTUAL_ENV from `uv run`) would be synced.
    host.runner.run(['env', '-u', 'VIRTUAL_ENV', '-u', 'UV_PROJECT_ENVIRONMENT', 'uv', 'sync', '--locked',
                     '--extra', 'dev'], cwd=out, log=logs / 'owner-uv-sync.log')
    web = out / 'web'
    (web / 'node_modules').symlink_to(host.config.web_node_modules)
    (web / '.env.local').symlink_to(host.config.owner.web_env_local)
    path = owner_machine(host).unit_path(WEB_UNIT)
    host.runner.run(['env', f'PATH={path}', web / 'node_modules/.bin/next', 'build'], cwd=web,
                    log=logs / 'owner-next-build.log')


def owner_machine(host: Host) -> Machine:
    if host.owner_units is None:
        host.owner_units = Machine(host.config.owner.checkout, host.runner, health_host=host.config.health_host,
                                   sleep=host.sleep)
    return host.owner_units


def _package_notes(host: Host, main: Path, data: dict, out: Path) -> None:
    """This release's "What's new" ledger, on every release: the web reads it at
    request time, so a controller-only release's entries show without a rebuild."""
    running = data.get('running')
    log = git(host, main, *notes.log_args(running, data['target'])) if running else ''
    previous = notes.read_previous(Path(data['runningDir']) / notes.WHATS_NEW)
    notes.write(out / notes.WHATS_NEW, notes.merge(data['release'], data['target'][:7],
                                                   notes.parse_entries(log), previous))


def _package_worker(host: Host, tree: Path, state: State, kinds, logs: Path) -> dict:
    """Stage what the worker gets: changed opt files, and the rebuilt image."""
    if plan.WORKER not in kinds:
        return {'workerFiles': [], 'image': None}
    payload = state.path.parent / 'worker'
    shutil.rmtree(payload, ignore_errors=True)
    payload.mkdir(parents=True)
    files = []
    for rel in sorted(plan.WORKER_FILES):
        shutil.copy2(tree / rel, payload / Path(rel).name)
        files.append(Path(rel).name)
    built = None
    if plan.IMAGE in kinds:
        path, sha = image.build(tree, host.config.image_inputs_cache, state.path.parent / 'image', host.runner,
                                log_dir=logs)
        image.require_pin(tree, sha)
        shutil.copy2(path, payload / path.name)
        built = sha
    return {'workerFiles': files, 'image': built}


def _private(path: Path) -> None:
    for item in [path, *path.rglob('*')]:
        if not item.is_symlink():
            item.chmod(item.stat().st_mode & 0o700)


# ── qualify ──────────────────────────────────────────────────────────
def qualify_phase(host: Host, main: Path, state: State) -> None:
    data = state.need('worktree')
    tree, logs = Path(data['worktree']), state.path.parent / 'logs'
    if not (tree / 'scripts/test.sh').exists():
        raise RuntimeError(f"target {data['target'][:7]} has no scripts/test.sh; it predates the deployer's suite")
    dsn = os.environ.get('GMS_TEST_PG_DSN')
    if not dsn:
        raise RuntimeError('qualify needs GMS_TEST_PG_DSN pointing at the disposable test database')
    # The target's own locked dependencies, in the build worktree's venv.
    host.runner.run(['uv', 'sync', '--locked', '--extra', 'dev'], cwd=tree, log=logs / 'uv-sync.log')
    python = tree / '.venv/bin/python'
    env = {'GMS_PYTHON': str(python), 'GMS_TEST_PG_DSN': dsn,
           'GMS_GATEWAY_TEST_DSN': os.environ.get('GMS_GATEWAY_TEST_DSN', dsn)}
    with lock.held(main / '.runtime/issue-loop/land.lock', f"deploy {state.data['release']}", wait_seconds=1800):
        host.runner.run([tree / 'scripts/test.sh'], cwd=tree, env=env, log=logs / 'test.log')
    host.runner.run([python, '-m', 'ruff', 'check', 'src', 'tests'], cwd=tree, log=logs / 'ruff.log')
    web = tree / 'web'
    if not (web / 'node_modules').exists():
        (web / 'node_modules').symlink_to(host.config.web_node_modules)
    host.runner.run([web / 'node_modules/.bin/tsc', '--noEmit', '-p', '.'], cwd=web, log=logs / 'tsc.log')
    tests = sorted(glob.glob(str(web / 'scripts/test-*.mjs')))
    host.runner.run(['node', '--import', 'tsx', '--test', *tests], cwd=web, log=logs / 'web-tests.log')
    state.set(qualifiedAt=now())
    say('qualified: scripts/test.sh, ruff, web tsc and script tests')


# ── preflight ────────────────────────────────────────────────────────
def preflight_phase(host: Host, main: Path, state: State) -> None:
    data = state.need('worktree', 'releaseDir', 'runningDir', 'qualifiedAt')
    tree = Path(data['worktree'])
    if tree.exists():
        dirty = git(host, tree, 'status', '--porcelain', '--untracked-files=no')
        if dirty:
            raise RuntimeError(f'build worktree {tree} is dirty; a release is built from a commit')
    if str(running_release(host)) != data['runningDir']:
        raise RuntimeError('the running release changed since plan; plan again')
    owner_unchanged_since_plan(host, data)
    active = host.active_runs(host.config.registry)
    if active:
        raise RuntimeError(f'{active} run(s) active; deploy when idle')
    qualified = {'commit': data['target'], 'release': data['release'], 'kinds': data['kinds'],
                 'image': data.get('image'), 'qualifiedAt': data['qualifiedAt']}
    for out in (data['releaseDir'], data.get('ownerReleaseDir')):
        if out:
            Path(out, 'QUALIFIED.json').write_text(json.dumps(qualified, indent=2) + '\n')
    if tree.exists():
        host.runner.run(['git', '-C', main, 'worktree', 'remove', '--force', tree])
    say('preflight passed; QUALIFIED.json written')


# ── activate / rollback ──────────────────────────────────────────────
def swap_link(link: Path, target: Path) -> None:
    tmp = link.with_name(link.name + '.swap')
    tmp.unlink(missing_ok=True)
    tmp.symlink_to(target)
    os.replace(tmp, link)


def swap_links(host: Host, target: Path) -> None:
    for link in host.config.current_links:
        swap_link(link, target)


def worker_services_start(config) -> str:
    """(Re)start the worker's services. The guest does not start them at boot,
    so everything that boots or updates the worker runs this."""
    q = shlex.quote
    return f'systemctl restart {q(config.services["manager"])} {q(config.services["clock_sync"])}'


def worker_install_script(config, files, with_image: bool, remote: str) -> str:
    w, q = config.worker, shlex.quote
    lines = ['set -e']
    if with_image:
        lines += [f'if [ -f {q(w.image_path)} ]; then cp -a {q(w.image_path)} {q(w.image_path + ".prev")}; fi',
                  f'install -o root -g root -m 0444 {q(remote + "/agent-full.squashfs")} {q(w.image_path + ".tmp")}',
                  f'mv -f {q(w.image_path + ".tmp")} {q(w.image_path)}']
    for name in files:
        dest = f'{w.opt_dir}/{name}'
        lines += [f'if [ -f {q(dest)} ]; then cp -a {q(dest)} {q(dest + ".prev")}; fi',
                  f'install -o root -g root -m 0644 {q(remote + "/" + name)} {q(dest)}']
    lines += [worker_services_start(config), 'sleep 2', f'systemctl is-active {q(config.services["manager"])}', f'rm -rf {q(remote)}']
    return 'sudo -n bash -c ' + q('\n'.join(lines))


def worker_restore_script(config, files, with_image: bool) -> str:
    w, q = config.worker, shlex.quote
    paths = ([w.image_path] if with_image else []) + [f'{w.opt_dir}/{name}' for name in files]
    lines = ['set -e'] + [f'if [ -f {q(p + ".prev")} ]; then mv -f {q(p + ".prev")} {q(p)}; fi' for p in paths]
    lines += [worker_services_start(config)]
    return 'sudo -n bash -c ' + q('\n'.join(lines))


def activate_phase(host: Host, state: State) -> None:
    data = state.need('releaseDir', 'runningDir', 'release')
    files, with_image = data.get('workerFiles', []), bool(data.get('image'))
    logs, worker_installed = state.path.parent / 'logs', False
    owner_unchanged_since_plan(host, data)
    if data.get('ownerReleaseDir'):
        state.set(ownerBaseline=owner_baseline(host))  # refuses before anything changes
    try:
        if files:
            payload = state.path.parent / 'worker'
            uploads = [payload / n for n in files] + ([payload / 'agent-full.squashfs'] if with_image else [])
            remote = f"/tmp/gms-deploy-{data['release']}"
            host.worker_upload(uploads, remote)
            worker_installed = True
            host.worker_ssh(worker_install_script(host.config, files, with_image, remote), log=logs / 'worker.log')
        if ships_invited(data):
            swap_links(host, Path(data['releaseDir']))
            host.restart(host.config.services['controller'], host.config.services['web'])
            host.wait_healthy()
        if data.get('ownerReleaseDir'):
            state.set(ownerSwapped=True)
            restart_owner(host, Path(data['ownerReleaseDir']), state.data['ownerBaseline'])
    except Exception as error:
        restored = _rollback(host, state.data, worker_installed)
        raise RuntimeError(f'activate failed ({error}); rollback {"succeeded" if restored else "FAILED"}') from error
    state.set(activatedAt=now(), workerInstalled=worker_installed)
    say(f"activated {data['release']}")


def owner_baseline(host: Host) -> dict:
    """Each owner unit's answer now. activate judges every restart by "answers
    like before", so all must be healthy first."""
    before = baseline(owner_machine(host), OWNER_UNITS)
    bad = [f.line() for f in healthy_baseline(before) if not f.ok]
    if bad:
        raise RuntimeError('owner units not healthy before the deploy; nothing changed: ' + '; '.join(bad))
    return before


def restart_owner(host: Host, release: Path, before: dict) -> None:
    """Point owner-current at `release` and restart the owner units on it,
    the watchdog held from before the swap."""
    m = owner_machine(host)
    with watchdog_held(m):
        swap_link(host.config.owner_current, release)
        restart_each(m, before)


def _rollback(host: Host, data: dict, worker_installed: bool) -> bool:
    """Both tracks back on their previous releases; each is tried even when
    the other fails."""
    restored = True
    if data.get('ownerSwapped'):
        restored = _attempt(lambda: restart_owner(host, Path(data['ownerRunningDir']), data['ownerBaseline']))
    if ships_invited(data):
        restored = _attempt(lambda: _rollback_invited(host, data, worker_installed)) and restored
    return restored


def ships_invited(data: dict) -> bool:
    """False for an owner-only release (e.g. the owner track catching up):
    the invited stack is left running untouched."""
    return bool(set(data.get('kinds', [])) & plan.INVITED_KINDS)


def _attempt(step) -> bool:
    try:
        step()
        return True
    except Exception as error:  # report, never mask the original failure
        say(f'ROLLBACK FAILED: {error}')
        return False


def _rollback_invited(host: Host, data: dict, worker_installed: bool) -> None:
    if worker_installed:
        host.worker_ssh(worker_restore_script(host.config, data.get('workerFiles', []), bool(data.get('image'))))
    swap_links(host, Path(data['runningDir']))
    host.restart(host.config.services['controller'], host.config.services['web'])
    host.wait_healthy()


def rollback_phase(host: Host, state: State) -> None:
    data = state.need('runningDir', 'releaseDir')
    if not _rollback(host, data, data.get('workerInstalled', False)):
        raise RuntimeError('rollback failed; see output')
    state.set(rolledBackAt=now())
    say(f"rolled back to {Path(data['runningDir']).name}")


# ── postcheck ────────────────────────────────────────────────────────
def postcheck_phase(host: Host, main: Path, state: State) -> None:
    data = state.need('activatedAt', 'release')
    expected = {'/': 200, VALID_CONVERSATION: 200, INVALID_CONVERSATION: 404}
    got = {path: host.http_status(host.config.web_local_url + path, host.config.public_host) for path in expected}
    ports = {p: host.port_open(host.config.health_host, p) for p in host.config.health_ports}
    owner = owner_health(host, Path(data['ownerReleaseDir'])) if data.get('ownerReleaseDir') else {}
    ok = got == expected and all(ports.values()) and all(ok for ok, _ in owner.values())
    record = {'release': data['release'], 'commit': data['target'], 'kinds': data['kinds'], 'at': now(),
              'result': 'deployed' if ok else 'activated-postcheck-failed', 'http': got,
              **({'owner': {name: detail for name, (_, detail) in owner.items()}} if owner else {})}
    deploys = main / '.runtime/issue-loop/deploys.jsonl'
    deploys.parent.mkdir(parents=True, exist_ok=True)
    with open(deploys, 'a') as handle:
        handle.write(json.dumps(record) + '\n')
    state.set(postcheck=record)
    if not ok:
        raise RuntimeError(f'POSTCHECK FAILED (release left up): http={got} ports={ports} owner={owner}. '
                           f"Roll back with: scripts/deploy.sh --phase rollback --release {data['release']}")
    say(f"postcheck passed: {data['release']} is live")


def owner_health(host: Host, release: Path) -> dict[str, tuple[bool, str]]:
    """owner-current on this release, and every owner unit answering its own
    probe: serve ready, mcp's unauthenticated POST refused with 401 (up, OAuth
    gate on; no tool call, no mail), web 200, supervise with children."""
    link = host.config.owner_current
    found = {'owner-current': (link.resolve() == release.resolve(), str(link.resolve()))}
    for f in healthy_baseline(baseline(owner_machine(host), OWNER_UNITS)):
        found[f.name] = (f.ok, f.detail)
    return found


def clean_phase(host: Host, main: Path, state: State) -> None:
    tree = state.data.get('worktree')
    if tree and Path(tree).exists():
        host.runner.run(['git', '-C', main, 'worktree', 'remove', '--force', tree])
    say('build worktree removed')
