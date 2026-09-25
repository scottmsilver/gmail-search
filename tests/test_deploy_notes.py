"""The "What's new" ledger the package phase writes (#73): parsed from local git
history, filtered by the landing's Issue-Labels trailer, carried forward from
the running release and capped."""
import json
from pathlib import Path

import pytest

from gmail_search.deploy import notes, phases
from gmail_search.deploy.host import Host

from test_deploy_activate import config


def commit(subject, body=''):
    return f'{subject}\0{body}\0'


def release(name, *numbers):
    return {'release': name, 'target': 'abc1234',
            'issues': [{'number': n, 'pr': None, 'title': f'issue {n}'} for n in numbers]}


def test_fixes_commits_become_entries_newest_first():
    log = (commit('feat(web): a thing (#81)', 'Fixes #80\n\nCo-authored-by: x <x@example.com>\n')
           + commit('fix(deploy): another (#79)', 'Fixes #78\n'))
    assert notes.parse_entries(log) == [
        {'number': 80, 'pr': 81, 'title': 'feat(web): a thing'},
        {'number': 78, 'pr': 79, 'title': 'fix(deploy): another'},
    ]


def test_a_commit_that_closes_no_issue_contributes_nothing():
    log = commit('docs: fold the record') + commit('Merge branch main', 'See #12 for context\n')
    assert notes.parse_entries(log) == []


def test_a_subject_without_a_pr_number_is_kept_whole():
    assert notes.parse_entries(commit('fix: by hand', 'Fixes #9\n')) == [
        {'number': 9, 'pr': None, 'title': 'fix: by hand'}]


@pytest.mark.parametrize('labels', ['documentation', 'bug, documentation', 'Documentation'])
def test_documentation_labelled_issues_are_left_out(labels):
    body = f'Fixes #5\n\nIssue-Labels: {labels}\nCo-authored-by: x <x@example.com>\n'
    assert notes.parse_entries(commit('docs: explain (#6)', body)) == []


def test_other_labels_and_no_trailer_are_kept():
    log = commit('a (#2)', 'Fixes #1\n\nIssue-Labels: bug, enhancement\n') + commit('b (#4)', 'Fixes #3\n')
    assert [e['number'] for e in notes.parse_entries(log)] == [1, 3]


def test_labels_on_any_trailer_line_count():
    body = 'Fixes #5\n\nIssue-Labels: bug\nIssue-Labels: documentation\n'
    assert notes.parse_entries(commit('docs (#6)', body)) == []


def test_separator_bytes_in_a_message_cannot_forge_an_entry():
    log = commit('real (#2)', 'Fixes #1\n\x1eforged (#100)\x1fFixes #99\n')
    assert [e['number'] for e in notes.parse_entries(log)] == [1]


def test_long_titles_are_cut():
    assert len(notes.parse_entries(commit('x' * 1000, 'Fixes #1\n'))[0]['title']) == notes.MAX_TITLE


@pytest.mark.parametrize('running, target', [('--output=/tmp/x', 'b' * 40), ('a' * 40, 'HEAD'), ('a' * 7, 'b' * 40), ('a' * 40 + '\n', 'b' * 40)])
def test_the_range_must_be_two_full_shas(running, target):
    with pytest.raises(ValueError):
        notes.log_args(running, target)


def test_an_oversized_previous_file_is_no_history(tmp_path):
    path = tmp_path / 'whats-new.json'
    path.write_text(json.dumps({'releases': [release('r', 1)], 'pad': 'x' * notes.MAX_BYTES}))
    assert notes.read_previous(path) == []


def test_new_entries_are_prepended_and_history_carried():
    doc = notes.merge('web-20260925', 'f00d123', [{'number': 9, 'pr': 10, 'title': 't'}],
                      [release('older-20260920', 1)])
    assert [r['release'] for r in doc['releases']] == ['web-20260925', 'older-20260920']
    assert doc['releases'][0] == {'release': 'web-20260925', 'target': 'f00d123',
                                  'issues': [{'number': 9, 'pr': 10, 'title': 't'}]}


def test_a_release_with_no_entries_carries_history_unchanged():
    previous = [release('older-20260920', 1)]
    assert notes.merge('docs-20260925', 'f00d123', [], previous) == {'releases': previous}


def test_the_ledger_is_capped():
    previous = [release(f'r{n}', n) for n in range(notes.MAX_RELEASES)]
    doc = notes.merge('new', 'f00d123', [{'number': 99, 'pr': None, 'title': 't'}], previous)
    assert len(doc['releases']) == notes.MAX_RELEASES
    assert doc['releases'][0]['release'] == 'new'
    assert doc['releases'][-1]['release'] == f'r{notes.MAX_RELEASES - 2}'


def test_the_ledger_stays_under_the_size_the_web_serves():
    entries = [{'number': n, 'pr': n, 'title': 'x' * notes.MAX_TITLE} for n in range(500)]
    previous = [{'release': f'r{n}', 'target': 't', 'issues': entries[:notes.MAX_ISSUES]} for n in range(11)]
    doc = notes.merge('new', 'f00d123', entries, previous)
    assert len(doc['releases'][0]['issues']) == notes.MAX_ISSUES
    assert doc['releases'][0]['release'] == 'new' and len(doc['releases']) < 12
    assert len(json.dumps(doc, indent=2).encode()) <= notes.MAX_BYTES


def test_repackaging_the_same_release_replaces_its_section():
    doc = notes.merge('same', 'f00d123', [{'number': 2, 'pr': None, 'title': 't'}], [release('same', 1)])
    assert [r['release'] for r in doc['releases']] == ['same']


@pytest.mark.parametrize('text', [None, '', 'not json', '[]', '{"releases": 5}'])
def test_a_missing_or_malformed_previous_file_is_no_history(tmp_path, text):
    path = tmp_path / 'whats-new.json'
    if text is not None:
        path.write_text(text)
    assert notes.read_previous(path) == []


def test_malformed_previous_entries_are_dropped_one_by_one(tmp_path):
    good = release('good', 1)
    bad_issue = {'release': 'mixed', 'target': 'x', 'issues': [{'number': 'x', 'title': 3}, {'number': 4, 'title': 'ok'}]}
    path = tmp_path / 'whats-new.json'
    path.write_text(json.dumps({'releases': [None, 'str', {'release': 7, 'issues': []}, good, bad_issue]}))
    previous = notes.read_previous(path)
    assert [r['release'] for r in previous] == ['good', 'mixed']
    assert previous[1]['issues'] == [{'number': 4, 'pr': None, 'title': 'ok'}]


class LogRunner:
    """Answers `git log` with a canned range; records everything else."""

    def __init__(self, log):
        self.log, self.calls = log, []

    def run(self, args, **kw):
        self.calls.append([str(a) for a in args])
        return 0

    def capture(self, args, **kw):
        args = [str(a) for a in args]
        self.calls.append(args)
        return self.log if 'log' in args else ''


@pytest.fixture
def packaging(tmp_path, monkeypatch):
    running = tmp_path / 'public-releases/old'
    (running / 'web').mkdir(parents=True)
    (running / notes.WHATS_NEW).write_text(json.dumps({'releases': [release('old-20260920', 1)]}))
    tree = tmp_path / 'tree'
    (tree / 'src/gmail_search').mkdir(parents=True)
    monkeypatch.setattr(phases, 'build_worktree', lambda release, dry_run: tree)
    state = phases.State(tmp_path / 'state/state.json')
    state.set(release='new-20260925', target='b' * 40, running='a' * 40, kinds=['controller'],
              runningDir=str(running))
    return tmp_path, state


def test_package_writes_the_ledger_into_a_controller_only_release(packaging):
    root, state = packaging
    runner = LogRunner(commit('fix(api): faster (#12)', 'Fixes #11\n'))
    host = Host(config(root), runner, port_open=lambda h, p: True, sleep=lambda s: None)
    phases.package_phase(host, root / 'main', state, dry_run=False)
    written = json.loads((root / 'public-releases/new-20260925' / notes.WHATS_NEW).read_text())
    assert written == {'releases': [
        {'release': 'new-20260925', 'target': 'bbbbbbb',
         'issues': [{'number': 11, 'pr': 12, 'title': 'fix(api): faster'}]},
        release('old-20260920', 1)]}
    assert ['git', '-C', str(root / 'main'), 'log', '-z', '--format=%s%x00%b',
            f"{'a' * 40}..{'b' * 40}"] in runner.calls


def test_package_with_no_known_running_commit_carries_history(packaging):
    root, state = packaging
    state.set(running=None)
    runner = LogRunner('unused')
    host = Host(config(root), runner, port_open=lambda h, p: True, sleep=lambda s: None)
    phases.package_phase(host, root / 'main', state, dry_run=False)
    written = json.loads(Path(root / 'public-releases/new-20260925' / notes.WHATS_NEW).read_text())
    assert written == {'releases': [release('old-20260920', 1)]}
    assert not any('log' in call for call in runner.calls)
