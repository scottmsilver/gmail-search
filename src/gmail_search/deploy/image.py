"""The worker image, rebuilt reproducibly from repo inputs.

Inputs: the runtime trees fingerprinted in
deploy/public/worker/pi-mcp-runtime-inputs.json (node `bin/`, `lib/`, and the
locked Pi packages), cached once under `image_inputs_cache/<manifest digest>/`
and re-verified against the manifest on every build; plus the committed guest
files (the list in prepare-full-agent-runtime.sh) and workflow agents. The jiti
cache is warmed at the guest's paths and the squashfs is built with fixed
timestamps, so the same commit always gives the same bytes. The result must
equal AGENT_FULL_PIN committed in firecracker_backend.py: the pin is decided
when a change is made (`scripts/deploy.sh --update-pin`) and committed with it,
never written by a deploy.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil

WORKER = Path('deploy/public/worker')
PREPARE = WORKER / 'prepare-full-agent-runtime.sh'
MANIFEST = WORKER / 'pi-mcp-runtime-inputs.json'
PIN_FILES = (WORKER / 'firecracker_backend.py', Path('tests/test_production_worker_profile.py'),
             WORKER / 'production/README.md')
PIN = re.compile(r"AGENT_FULL_PIN = '([0-9a-f]{64})'")
SQUASH_ARGS = ('-all-root', '-noappend', '-comp', 'zstd', '-reproducible', '-all-time', '0',
               '-mkfs-time', '0', '-no-progress')


class ImageError(RuntimeError):
    pass


def parse_guest_files(builder_script: str) -> list[str]:
    """The guest file list, read from the root image builder so there is one list."""
    match = re.search(r'^for name in (.+?); do$', builder_script, re.M)
    if not match:
        raise ImageError(f'no guest file list in {PREPARE}')
    return match.group(1).split()


def guest_files(repo: Path) -> list[str]:
    return parse_guest_files((repo / PREPARE).read_text())


def committed_pin(repo: Path) -> str:
    match = PIN.search((repo / PIN_FILES[0]).read_text())
    if not match:
        raise ImageError(f'no AGENT_FULL_PIN in {PIN_FILES[0]}')
    return match.group(1)


def write_pin(repo: Path, new: str) -> None:
    """Replace the pin everywhere it is recorded (working tree only; never commits)."""
    old = committed_pin(repo)
    for rel in PIN_FILES:
        path = repo / rel
        text = path.read_text()
        if old in text:
            path.write_text(text.replace(old, new))


def _verifier(repo: Path):
    spec = importlib.util.spec_from_file_location('verify_inputs', repo / WORKER / 'verify_pi_mcp_runtime_inputs.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest_digest(repo: Path) -> str:
    return hashlib.sha256((repo / MANIFEST).read_bytes()).hexdigest()[:16]


def inputs_dir(repo: Path, cache: Path) -> Path:
    return cache / manifest_digest(repo)


def verify_inputs(repo: Path, root: Path) -> None:
    """`root` holds bin/, lib/ and pi-pkgs/ matching the committed manifest."""
    trees = json.loads((repo / MANIFEST).read_text())['trees']
    found = _verifier(repo).fingerprints(root, root / 'pi-pkgs', trusted=False)
    bad = sorted(name for name in trees if found.get(name) != trees[name])
    if bad:
        raise ImageError(f'image inputs at {root} do not match {MANIFEST}: {", ".join(bad)}')


def seed_inputs(repo: Path, cache: Path, source_root: Path) -> Path:
    """Copy bin/, lib/, pi-pkgs/ from an extracted image into the cache, verified.
    Modes are normalized first (as every build does): an image built by hand
    may carry narrower modes than the qualified one, with the same content."""
    target = inputs_dir(repo, cache)
    staging = target.with_name(target.name + '.partial')
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    for name in ('bin', 'lib', 'pi-pkgs'):
        shutil.copytree(source_root / name, staging / name, symlinks=True)
    normalize(staging)
    verify_inputs(repo, staging)
    shutil.rmtree(target, ignore_errors=True)
    staging.rename(target)
    return target


def stage_root(repo: Path, cache: Path, root: Path) -> None:
    """Assemble the image tree: verified inputs, guest files, workflow agents."""
    inputs = inputs_dir(repo, cache)
    if not inputs.is_dir():
        raise ImageError(f'no verified image inputs at {inputs}; seed them with '
                         f'`scripts/deploy.sh --seed-image-inputs <extracted image root>`')
    verify_inputs(repo, inputs)
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    for name in ('bin', 'lib', 'pi-pkgs'):
        shutil.copytree(inputs / name, root / name, symlinks=True)
    for name in guest_files(repo):
        shutil.copy2(repo / WORKER / name, root / name)
    shutil.copytree(repo / WORKER / 'workflow-agents', root / 'workflow-agents')


def normalize(root: Path) -> None:
    for cache in root.rglob('__pycache__'):
        shutil.rmtree(cache, ignore_errors=True)
    for path in [root, *root.rglob('*')]:
        if path.is_symlink():
            continue
        mode = path.stat().st_mode
        path.chmod(mode | 0o444 | (0o111 if path.is_dir() or mode & 0o100 else 0))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def build(repo: Path, cache: Path, workdir: Path, runner, *, log_dir: Path) -> tuple[Path, str]:
    """Build the image from `repo`; return (squashfs path, sha256)."""
    root, image = workdir / 'root', workdir / 'agent-full.squashfs'
    stage_root(repo, cache, root)
    runner.run([repo / WORKER / 'warm-jiti-cache.sh', root], log=log_dir / 'warm-jiti-cache.log')
    normalize(root)
    image.unlink(missing_ok=True)
    runner.run(['mksquashfs', root, image, *SQUASH_ARGS], log=log_dir / 'mksquashfs.log')
    return image, sha256_file(image)


def require_pin(repo: Path, sha: str) -> None:
    pin = committed_pin(repo)
    if sha != pin:
        raise ImageError(f'worker image pin mismatch: built {sha[:12]}, committed {pin[:12]}. '
                         'Run `scripts/deploy.sh --update-pin` in the change\'s worktree and commit the pin.')
