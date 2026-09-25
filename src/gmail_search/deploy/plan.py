"""What a batch is and whether it ships. Pure functions of the changed paths,
so every decision is testable without a release, a service or a network."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import json
from pathlib import Path
import re

CONTROLLER, WEB, IMAGE, WORKER, OWNER = 'controller', 'web', 'image', 'worker', 'owner'
# The kinds that swap and restart the invited stack; an owner-only release does not.
INVITED_KINDS = {CONTROLLER, WEB, IMAGE, WORKER}

WORKER_DIR = 'deploy/public/worker/'
# Files installed in the worker's opt dir, by repo path.
WORKER_FILES = {
    'deploy/public/worker/firecracker_backend.py', 'deploy/public/worker/full_agent_manager.py',
    'src/gmail_search/gateway/full_agent_rpc.py', 'deploy/public/worker/full_agent_rpc_frontend.py',
    'deploy/public/worker/full_agent_rpc_server.py', 'deploy/public/worker/guest_agent_bootstrap.py',
    'deploy/public/worker/guest_run_bootstrap.py', 'deploy/public/worker/guest_tool_config.py',
    'deploy/public/worker/production/phc_clock_sync.py',
    'deploy/public/worker/production_worker_profile.py', 'deploy/public/worker/vsock_http_relay.py',
}
# Paths that change the worker image besides the guest files themselves.
IMAGE_INPUTS = (
    'deploy/public/worker/workflow-agents/', 'deploy/public/worker/warm-jiti-cache.sh',
    'deploy/public/worker/warm_jiti_cache.py', 'deploy/public/worker/pi-mcp-runtime-inputs.json',
    'deploy/pi/pi-pkgs/package.json', 'deploy/pi/pi-pkgs/package-lock.json',
)
NOT_SHIPPED = re.compile(r'^(docs/|tests/|scripts/|\.claude/|\.github/|\.githooks/|web/scripts/)|\.md$|^\.gitignore$'
                         # Example configs and test-only probes: nothing a release or the worker reads.
                         r'|^deploy/examples/|^deploy/[^/]+\.example\.json$|^deploy/public/probe_[^/]+\.py$'
                         # Owner-run migration scripts (deploy/public/OWNER_KEY_MIGRATION.md): the
                         # package phase copies only src/, web/, and WORKER_FILES into a release
                         # (phases.py:package_phase), so these are never shipped and a change to one
                         # must not permanently refuse every future deploy (#70).
                         r'|^deploy/public/migrate_[^/]+\.py$')
# The controller runs from the main checkout's venv: a dependency change needs
# `uv sync` there, which is the owner's step, not the deployer's.
DEPENDENCIES = {'pyproject.toml', 'uv.lock', 'web/package.json', 'web/package-lock.json', 'web/bun.lock'}
# What an owner release (#52) carries besides src/ and web/: serve's page
# templates (server.py finds them beside src/) and the watchdog the timer runs.
OWNER_PATHS = ('templates/', 'scripts/serve_watchdog.sh')
REVIEWED_SCHEMA = re.compile(r"^REVIEWED_SCHEMA_BLOB = '([0-9a-f]{40})'$", re.M)


@dataclass
class Batch:
    kinds: set[str] = field(default_factory=set)
    manual: list[str] = field(default_factory=list)

    @property
    def shippable(self) -> bool:
        return bool(self.kinds) and not self.manual


def classify(paths, guest_files, *, owner_track: bool = False) -> Batch:
    """Kinds a batch needs, plus the paths no kind covers (those need the owner).
    With an owner track, its own paths are its business (classify_owner)."""
    batch = Batch()
    guest = {WORKER_DIR + name for name in guest_files}
    for path in paths:
        if owner_track and path.startswith(OWNER_PATHS):
            continue
        if path in DEPENDENCIES:
            batch.manual.append(path)
        elif path in WORKER_FILES or path in guest or path.startswith(IMAGE_INPUTS):
            if path in WORKER_FILES:
                batch.kinds.add(WORKER)
            if path in guest or path.startswith(IMAGE_INPUTS):
                batch.kinds.add(IMAGE)
            if path.startswith('src/'):
                batch.kinds.add(CONTROLLER)
        elif NOT_SHIPPED.search(path):
            continue  # after the shipped paths: workflow agents are .md files
        elif path.startswith('src/'):
            batch.kinds.add(CONTROLLER)
        elif path.startswith('web/'):
            batch.kinds.add(WEB)
        else:
            batch.manual.append(path)
    # The manager verifies the image against the pin in firecracker_backend.py,
    # so a new image always ships with the worker files.
    if IMAGE in batch.kinds:
        batch.kinds.add(WORKER)
    return batch


def classify_owner(paths) -> bool:
    """Whether the owner daemons run anything these paths change: any shipped
    src/ or web/ path (serve, mcp and supervise import widely), or OWNER_PATHS."""
    return any(p.startswith(OWNER_PATHS) or (p.startswith(('src/', 'web/')) and not NOT_SHIPPED.search(p))
               for p in paths)


def schema_reviewed(schema_blob: str, checks_source: str) -> bool:
    """Every owner daemon runs pg_schema.sql against the live database at boot,
    so an owner release may carry only the blob its own commit declares
    reviewed (checkout_checks.REVIEWED_SCHEMA_BLOB)."""
    found = REVIEWED_SCHEMA.search(checks_source)
    return bool(found) and found.group(1) == schema_blob


def decide(running_commit, target_commit, *, target_contains_running, running_contains_target, batch) -> str:
    """ship | skip | superseded | refuse:<why>."""
    if running_commit == target_commit:
        return 'skip'
    if running_commit and running_contains_target:
        return 'superseded'
    if running_commit and not target_contains_running:
        return 'refuse:target does not contain the running release commit'
    if batch.manual:
        return 'refuse:needs the owner: ' + ', '.join(sorted(batch.manual))
    return 'ship' if batch.kinds else 'skip'


NAME = re.compile(r'^[a-z][a-z0-9]{0,23}$')


def release_name(word: str, today: date, taken) -> str:
    if not NAME.match(word):
        raise ValueError(f'--name must match {NAME.pattern}')
    base = f'{word}-{today:%Y%m%d}'
    name, n = base, 1
    while name in taken:
        n += 1
        name = f'{base}-{n}'
    return name


def running_commit(release_dir: Path) -> str | None:
    """The commit a release was built from: QUALIFIED.json, else a leading sha in
    RELEASE (hand deploys wrote `<sha>: ...`). None when neither records one."""
    qualified = release_dir / 'QUALIFIED.json'
    if qualified.exists():
        return json.loads(qualified.read_text()).get('commit')
    release = release_dir / 'RELEASE'
    if release.exists():
        head = re.match(r'([0-9a-f]{7,40})\b', release.read_text())
        return head.group(1) if head else None
    return None
