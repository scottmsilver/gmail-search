"""The deployer's decisions: what a batch is and whether it ships."""
from datetime import date

import pytest

from gmail_search.deploy import plan
from gmail_search.deploy.image import guest_files, parse_guest_files

GUEST = ['guest_agent_pi.py', 'guest-mail-server.ts']


def kinds(*paths):
    batch = plan.classify(paths, GUEST)
    return sorted(batch.kinds), batch.manual


def test_controller_web_and_docs_classify():
    assert kinds('src/gmail_search/gateway/search_service.py') == (['controller'], [])
    assert kinds('web/app/page.tsx') == (['web'], [])
    assert kinds('README.md', 'tests/test_x.py', 'scripts/test.sh', 'web/scripts/test-a.mjs', 'docs/a.txt') == ([], [])


def test_guest_files_need_the_image_and_so_the_worker():
    assert kinds('deploy/public/worker/guest_agent_pi.py') == (['image', 'worker'], [])
    assert kinds('deploy/public/worker/workflow-agents/mail-researcher.md') == (['image', 'worker'], [])
    assert kinds('deploy/pi/pi-pkgs/package-lock.json') == (['image', 'worker'], [])


def test_worker_files_and_a_shared_module():
    assert kinds('deploy/public/worker/full_agent_manager.py') == (['worker'], [])
    assert kinds('src/gmail_search/gateway/full_agent_rpc.py') == (['controller', 'worker'], [])


def test_dependency_and_unknown_deploy_paths_need_the_owner():
    assert kinds('uv.lock')[1] == ['uv.lock']
    assert kinds('deploy/systemd/x.service')[1] == ['deploy/systemd/x.service']


@pytest.mark.parametrize('path', [
    'deploy/examples/test.env.example',
    'deploy/deploy.example.json',
    'deploy/public/probe_bm25_deleted_statistics.py',
])
def test_example_configs_and_test_probes_are_not_shipped(path):
    assert kinds(path) == ([], [])


@pytest.mark.parametrize('path', [
    'deploy/sub/deploy.example.json',
    'deploy/public/invited-runtime.json.example',
    'deploy/public/provision_database.py',
    'deploy/public/worker/probe_x.py',
    'deploy/public/worker/probe.sh',
])
def test_paths_near_the_not_shipped_shapes_still_need_the_owner(path):
    assert kinds(path) == ([], [path])


@pytest.mark.parametrize(('running', 'target', 'contains', 'within', 'paths', 'action'), [
    ('a', 'a', True, True, ['src/x.py'], 'skip'),
    ('b', 'a', False, True, ['src/x.py'], 'superseded'),
    ('a', 'b', True, False, ['src/x.py'], 'ship'),
    ('a', 'b', True, False, ['README.md'], 'skip'),
    ('a', 'b', False, False, ['src/x.py'], 'refuse:target does not contain the running release commit'),
    ('a', 'b', True, False, ['uv.lock'], 'refuse:needs the owner: uv.lock'),
])
def test_decide(running, target, contains, within, paths, action):
    batch = plan.classify(paths, GUEST)
    assert plan.decide(running, target, target_contains_running=contains, running_contains_target=within,
                       batch=batch) == action


def test_release_names_are_safe_and_unique():
    assert plan.release_name('loop', date(2026, 9, 25), set()) == 'loop-20260925'
    assert plan.release_name('loop', date(2026, 9, 25), {'loop-20260925'}) == 'loop-20260925-2'
    with pytest.raises(ValueError):
        plan.release_name('../x', date(2026, 9, 25), set())


def test_running_commit_from_qualified_then_release(tmp_path):
    (tmp_path / 'RELEASE').write_text('d937117: ADK removal\n')
    assert plan.running_commit(tmp_path) == 'd937117'
    (tmp_path / 'QUALIFIED.json').write_text('{"commit": "abc1234def"}')
    assert plan.running_commit(tmp_path) == 'abc1234def'
    (tmp_path / 'QUALIFIED.json').unlink()
    (tmp_path / 'RELEASE').write_text('hand deploy without a sha\n')
    assert plan.running_commit(tmp_path) is None


def test_guest_list_is_read_from_the_root_builder():
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    files = guest_files(repo)
    assert 'guest_agent_pi.py' in files and 'guest-agent-workflow.ts' in files
    assert all((repo / 'deploy/public/worker' / name).exists() for name in files)
    assert parse_guest_files('x\nfor name in a b; do\n') == ['a', 'b']
