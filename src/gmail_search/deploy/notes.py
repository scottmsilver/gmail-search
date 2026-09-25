"""The "What's new" notes a release ships (#73), read by the invited/public web's
dialog from `<release>/web/whats-new.json`.

They come from local git history only: the commits between the running release
and the target that close an issue (`Fixes #N`), titled by the squash subject.
No GitHub call at deploy time. An issue labelled `documentation` is left out;
land.sh records the issue's labels on its commit as an `Issue-Labels:` trailer,
and a commit without one (landed by hand, or before the trailer existed) is
kept.

The file is a rolling ledger: each package reads the running release's copy
and prepends this release's entries, capped at MAX_RELEASES, so a returning
visitor sees what they missed. A release with no entries carries it forward
unchanged."""
from __future__ import annotations

import json
from pathlib import Path
import re

WHATS_NEW = 'web/whats-new.json'
MAX_RELEASES = 12
HIDDEN_LABELS = {'documentation'}

# `git log -z` with this format: subject NUL body NUL, per commit. Git refuses
# a NUL in a commit message, so no message can forge a field or a commit.
LOG_ARGS = ('-z', '--format=%s%x00%b')
# The web route refuses a larger file, so the ledger drops its oldest releases
# to stay under it.
MAX_BYTES = 128 * 1024
MAX_TITLE = 300
MAX_ISSUES = 50
_FIXES = re.compile(r'^Fixes #(\d+)\s*$', re.M)
_LABELS = re.compile(r'^Issue-Labels:[ \t]*(.*)$', re.M)
_SHA = re.compile(r'[0-9a-f]{40}')
_TITLE_PR = re.compile(r'^(.*?)\s+\(#(\d+)\)$')


def log_args(running: str, target: str) -> list[str]:
    """`git log` arguments for the batch. Both ends must be full shas (plan
    resolved them), so a tampered state file cannot pass git an option."""
    if not (_SHA.fullmatch(running) and _SHA.fullmatch(target)):
        raise ValueError(f'release notes need full commit shas, got {running!r}..{target!r}')
    return ['log', *LOG_ARGS, f'{running}..{target}']


def parse_entries(log: str) -> list[dict]:
    """`git log` output (log_args) to one {number, pr, title} per commit that
    closes a shown issue, in log order (newest first)."""
    fields = log.split('\0')
    entries = [_entry(subject.strip(), body) for subject, body in zip(fields[0::2], fields[1::2])]
    return [entry for entry in entries if entry]


def _entry(subject: str, body: str) -> dict | None:
    fixes = _FIXES.search(body)
    if not fixes or _labels(body) & HIDDEN_LABELS:
        return None
    titled = _TITLE_PR.match(subject)
    return {'number': int(fixes.group(1)), 'pr': int(titled.group(2)) if titled else None,
            'title': (titled.group(1).strip() if titled else subject)[:MAX_TITLE]}


def _labels(body: str) -> set[str]:
    return {label.strip().lower() for line in _LABELS.findall(body) for label in line.split(',')}


def read_previous(path: Path) -> list[dict]:
    """The running release's ledger; missing or malformed reads as no history,
    and a malformed entry drops just that entry."""
    try:
        with open(path, 'rb') as handle:
            text = handle.read(MAX_BYTES + 1)
        if len(text) > MAX_BYTES:
            return []
        releases = json.loads(text).get('releases')
    except (OSError, ValueError, AttributeError):
        return []
    return [clean for clean in map(_clean_release, releases if isinstance(releases, list) else []) if clean]


def _clean_release(value) -> dict | None:
    if not isinstance(value, dict) or not isinstance(value.get('release'), str):
        return None
    issues = value.get('issues')
    if not isinstance(issues, list):
        return None
    return {'release': value['release'], 'target': str(value.get('target', '')),
            'issues': [clean for clean in map(_clean_issue, issues) if clean][:MAX_ISSUES]}


def _clean_issue(value) -> dict | None:
    if not isinstance(value, dict) or type(value.get('number')) is not int or not isinstance(value.get('title'), str):
        return None
    pr = value.get('pr')
    return {'number': value['number'], 'pr': pr if type(pr) is int else None, 'title': value['title'][:MAX_TITLE]}


def merge(release: str, target: str, entries: list[dict], previous: list[dict]) -> dict:
    carried = [r for r in previous if r['release'] != release]
    releases = ([{'release': release, 'target': target, 'issues': entries[:MAX_ISSUES]}] if entries else []) + carried
    releases = releases[:MAX_RELEASES]
    while len(releases) > 1 and len(_serialize({'releases': releases}).encode()) > MAX_BYTES:
        releases.pop()
    return {'releases': releases}


def _serialize(doc: dict) -> str:
    return json.dumps(doc, indent=2) + '\n'


def write(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_serialize(doc))
