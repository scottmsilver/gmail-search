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

from . import image, lock, plan
from .config import read_env_file
from .host import Host

say = lambda message: print(f'==> {message}', flush=True)  # noqa: E731
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
        self.path.write_text(json.dumps(self.data, indent=2) + '\n')

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
    batch = plan.classify(paths, guest) if running else plan.Batch(manual=['<unknown running commit>'])
    action = plan.decide(running, target_sha, batch=batch,
                         target_contains_running=bool(running) and is_ancestor(host, main, running, target_sha),
                         running_contains_target=bool(running) and is_ancestor(host, main, target_sha, running))
    taken = {p.name for p in host.config.releases.iterdir()} if host.config.releases.exists() else set()
    return {'action': action, 'target': target_sha, 'running': running, 'runningDir': str(running_dir),
            'kinds': sorted(batch.kinds), 'manual': batch.manual, 'paths': paths,
            'release': plan.release_name(word, today, taken)}


# ── package ──────────────────────────────────────────────────────────
def build_worktree(release: str, dry_run: bool) -> Path:
    return Path('~/.wt').expanduser() / (f'deploy-dry-run-{release}' if dry_run else f'deploy-{release}')


def release_dir(host: Host, main: Path, release: str, dry_run: bool) -> Path:
    return (main / '.runtime/deploy/dry-run' / release) if dry_run else host.config.releases / release


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
    worker = _package_worker(host, tree, state, data['kinds'], logs)
    (out / 'RELEASE').write_text(f"{data['target'][:7]}: {data['release']} {now()}\n")
    _private(out)
    state.set(worktree=str(tree), releaseDir=str(out), **worker)
    say(f"packaged {data['release']} ({', '.join(data['kinds'])}) at {out}")


def _package_web(host: Host, tree: Path, out: Path, logs: Path) -> None:
    shutil.rmtree(out / 'web', ignore_errors=True)
    shutil.copytree(tree / 'web', out / 'web', ignore=shutil.ignore_patterns('node_modules', '.next'))
    (out / 'web/node_modules').symlink_to(host.config.web_node_modules)
    env = read_env_file(host.config.public_web_env)
    host.runner.run([out / 'web/node_modules/.bin/next', 'build'], cwd=out / 'web', env=env,
                    log=logs / 'next-build.log')


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
    active = host.active_runs(host.config.registry)
    if active:
        raise RuntimeError(f'{active} run(s) active; deploy when idle')
    qualified = {'commit': data['target'], 'release': data['release'], 'kinds': data['kinds'],
                 'image': data.get('image'), 'qualifiedAt': data['qualifiedAt']}
    Path(data['releaseDir'], 'QUALIFIED.json').write_text(json.dumps(qualified, indent=2) + '\n')
    if tree.exists():
        host.runner.run(['git', '-C', main, 'worktree', 'remove', '--force', tree])
    say('preflight passed; QUALIFIED.json written')


# ── activate / rollback ──────────────────────────────────────────────
def swap_links(host: Host, target: Path) -> None:
    for link in host.config.current_links:
        tmp = link.with_name(link.name + '.swap')
        tmp.unlink(missing_ok=True)
        tmp.symlink_to(target)
        os.replace(tmp, link)


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
    try:
        if files:
            payload = state.path.parent / 'worker'
            uploads = [payload / n for n in files] + ([payload / 'agent-full.squashfs'] if with_image else [])
            remote = f"/tmp/gms-deploy-{data['release']}"
            host.worker_upload(uploads, remote)
            worker_installed = True
            host.worker_ssh(worker_install_script(host.config, files, with_image, remote), log=logs / 'worker.log')
        swap_links(host, Path(data['releaseDir']))
        host.restart(host.config.services['controller'], host.config.services['web'])
        host.wait_healthy()
    except Exception as error:
        restored = _rollback(host, data, worker_installed)
        raise RuntimeError(f'activate failed ({error}); rollback {"succeeded" if restored else "FAILED"}') from error
    state.set(activatedAt=now(), workerInstalled=worker_installed)
    say(f"activated {data['release']}")


def _rollback(host: Host, data: dict, worker_installed: bool) -> bool:
    try:
        if worker_installed:
            host.worker_ssh(worker_restore_script(host.config, data.get('workerFiles', []), bool(data.get('image'))))
        swap_links(host, Path(data['runningDir']))
        host.restart(host.config.services['controller'], host.config.services['web'])
        host.wait_healthy()
        return True
    except Exception as error:  # report, never mask the original failure
        say(f'ROLLBACK FAILED: {error}')
        return False


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
    ok = got == expected and all(ports.values())
    record = {'release': data['release'], 'commit': data['target'], 'kinds': data['kinds'], 'at': now(),
              'result': 'deployed' if ok else 'activated-postcheck-failed', 'http': got}
    deploys = main / '.runtime/issue-loop/deploys.jsonl'
    deploys.parent.mkdir(parents=True, exist_ok=True)
    with open(deploys, 'a') as handle:
        handle.write(json.dumps(record) + '\n')
    state.set(postcheck=record)
    if not ok:
        raise RuntimeError(f'POSTCHECK FAILED (release left up): http={got} ports={ports}. '
                           f"Roll back with: scripts/deploy.sh --phase rollback --release {data['release']}")
    say(f"postcheck passed: {data['release']} is live")


def clean_phase(host: Host, main: Path, state: State) -> None:
    tree = state.data.get('worktree')
    if tree and Path(tree).exists():
        host.runner.run(['git', '-C', main, 'worktree', 'remove', '--force', tree])
    say('build worktree removed')
