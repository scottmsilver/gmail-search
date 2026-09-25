"""The read-only questions scripts/one-checkout.sh asks before it moves the
production checkout onto `main` (#57). Each answers from git objects or a
parsed lockfile, never from the running system, so tests need only a repo."""
from __future__ import annotations

from dataclasses import dataclass
import json
import tomllib

# The owner's decision (2026-09-25): these four uncommitted edits are saved as a
# patch and discarded once #55 has landed. Anything else dirty stops the switch.
EXPECTED_DIRTY = frozenset({
    '.github/workflows/ci.yml',
    'deploy/public/probe_bm25_deleted_statistics.py',
    'src/gmail_search/store/pg_schema.sql',
    'tests/test_bm25_deleted_statistics.py',
})

# Every daemon runs pg_schema.sql against the live database at boot
# (store/db.py:_init_db_pg). This is the blob of main's copy whose statements
# were checked against the live catalog on 2026-09-25: the one index main adds
# (idx_messages_thread) already exists and the tenant_isolation policies are
# already scoped to gmail_search_reader and gmail_analyst, so starting main's
# daemons creates nothing new. A different blob needs that review again.
REVIEWED_SCHEMA_BLOB = 'd3cef4aadf4799ad2eb57a539038b1b0ad38a6e2'
SCHEMA_PATH = 'src/gmail_search/store/pg_schema.sql'

# public-web runs `next` out of the production checkout's node_modules, so a
# change here would need `npm ci` under a service this switch does not stop.
WEB_LOCKFILES = ('web/package.json', 'web/package-lock.json')

PROJECT = 'gmail-search'


@dataclass(frozen=True)
class Finding:
    name: str
    ok: bool
    detail: str

    def line(self) -> str:
        return f"{'ok  ' if self.ok else 'FAIL'} {self.name}: {self.detail}"


def _locked_packages(lock_text: str) -> dict[str, list[dict]]:
    """Every locked entry by name; a lock can hold one name more than once
    (different versions for different markers)."""
    packages: dict[str, list[dict]] = {}
    for entry in tomllib.loads(lock_text).get('package', []):
        packages.setdefault(entry['name'], []).append(entry)
    return packages


def _requests(deps: list[dict]) -> list[tuple[str, tuple]]:
    return [(d['name'], tuple(d.get('extra', []))) for d in deps]


def _runtime_closure(packages: dict[str, list[dict]]) -> set[str]:
    """Names reachable from the project's own dependencies, following every
    extra a dependency requests (`psycopg[binary,pool]`, and extras those ask
    for in turn), the project's own extras excluded: what the services import,
    as opposed to what only tests import."""
    seen: set[tuple[str, tuple]] = set()
    todo = [r for root in packages[PROJECT] for r in _requests(root.get('dependencies', []))]
    while todo:
        name, extras = todo.pop()
        if (name, extras) in seen or name not in packages:
            continue
        seen.add((name, extras))
        for entry in packages[name]:
            todo.extend(_requests(entry.get('dependencies', [])))
            for extra in extras:
                todo.extend(_requests(entry.get('optional-dependencies', {}).get(extra, [])))
    return {name for name, _ in seen}


def _identity(entries: list[dict]) -> tuple[str, str]:
    """(what to show, what to compare) for every entry of one name: versions
    are shown, versions and sources (registry URL included) are compared."""
    shown = ' | '.join(sorted(e.get('version', '?') for e in entries))
    full = json.dumps(sorted((e.get('version', '?'), json.dumps(e.get('source', {}), sort_keys=True))
                             for e in entries))
    return shown, full


def lock_changes(old_lock: str, new_lock: str) -> tuple[list[str], list[str]]:
    """(runtime changes, dev-only changes) between two uv.lock texts, each as
    `+name ver`, `-name ver`, `name old -> new`, `name ver (source changed)`, or
    `name now runtime` / `name no longer runtime`."""
    old, new = _locked_packages(old_lock), _locked_packages(new_lock)
    old_rt, new_rt = _runtime_closure(old), _runtime_closure(new)
    old_v = {n: _identity(e) for n, e in old.items() if n != PROJECT}
    new_v = {n: _identity(e) for n, e in new.items() if n != PROJECT}
    runtime_changes = [f'{n} now runtime' for n in sorted(new_rt - old_rt) if n in old_v]
    runtime_changes += [f'{n} no longer runtime' for n in sorted(old_rt - new_rt) if n in new_v]
    dev_changes = []
    for name in sorted(set(old_v) | set(new_v)):
        before, after = old_v.get(name), new_v.get(name)
        if before == after:
            continue
        text = (f'+{name} {after[0]}' if before is None else f'-{name} {before[0]}' if after is None
                else f'{name} {before[0]} (source changed)' if before[0] == after[0]
                else f'{name} {before[0]} -> {after[0]}')
        (runtime_changes if name in old_rt | new_rt else dev_changes).append(text)
    return runtime_changes, dev_changes


def check_lock(old_lock: str, new_lock: str) -> Finding:
    runtime, dev = lock_changes(old_lock, new_lock)
    if runtime:
        return Finding('dependencies', False, 'runtime packages change (the services share this venv; '
                       'hand to the owner): ' + ', '.join(runtime))
    return Finding('dependencies', True, 'runtime unchanged; dev-only: ' + (', '.join(dev) or 'none'))


def check_dirty(dirty: set[str], expected=EXPECTED_DIRTY) -> Finding:
    unexpected = sorted(dirty - expected)
    if unexpected:
        return Finding('uncommitted', False, 'unexpected uncommitted paths: ' + ', '.join(unexpected))
    return Finding('uncommitted', True, f'{len(dirty)} of the {len(expected)} expected paths dirty'
                   + (': ' + ', '.join(sorted(dirty)) if dirty else ''))


def check_unmerged(unmerged: list[str], allowed: set[str]) -> Finding:
    """`unmerged` is `git log --format='%H %s' target..HEAD`; each needs an
    explicit acknowledgement by sha prefix, and a prefix may match only one."""
    shas = [line.split()[0] for line in unmerged]
    ambiguous = sorted(a for a in allowed if sum(s.startswith(a) for s in shas) > 1)
    if ambiguous:
        return Finding('unmerged commits', False, 'acknowledgement matches more than one commit: '
                       + ', '.join(ambiguous))
    missing = [line[:60] for line, sha in zip(unmerged, shas) if not any(sha.startswith(a) for a in allowed)]
    if missing:
        return Finding('unmerged commits', False, 'on the old branch, not on the target, not acknowledged '
                       '(--allow-unmerged): ' + '; '.join(missing))
    return Finding('unmerged commits', True, f'{len(unmerged)} acknowledged')


def check_schema(target_blob: str, reviewed: str = REVIEWED_SCHEMA_BLOB) -> Finding:
    if target_blob != reviewed:
        return Finding('startup schema', False, f'{SCHEMA_PATH} on the target is {target_blob[:12]}, '
                       f'reviewed {reviewed[:12]}: review what it runs at boot, then --schema-reviewed '
                       f'{target_blob}')
    return Finding('startup schema', True, f'{SCHEMA_PATH} is the reviewed blob {reviewed[:12]}')


def check_web_lockfiles(old_blobs: dict[str, str], new_blobs: dict[str, str]) -> Finding:
    changed = [p for p in WEB_LOCKFILES if old_blobs.get(p) != new_blobs.get(p)]
    if changed:
        return Finding('web dependencies', False, 'changed (public-web shares node_modules; hand to the '
                       'owner): ' + ', '.join(changed))
    return Finding('web dependencies', True, 'package.json and package-lock.json unchanged')


def check_issue(number: int, state: str, labels: list[str]) -> Finding:
    """Closed, or landed by the loop (merged or deployed)."""
    done = state == 'CLOSED' or {'loop:merged', 'loop:deployed'} & set(labels)
    return Finding(f'#{number}', bool(done), f"{state.lower()}, labels {','.join(labels) or 'none'}")
