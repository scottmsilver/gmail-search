"""Opaque workspace snapshots advance only under the current writer fence."""
import asyncio

import pytest

from gmail_search.gateway.artifacts import ArtifactStore
from gmail_search.gateway.capabilities import Capabilities
from gmail_search.gateway.registry import Registry, AccessDenied
from gmail_search.gateway.workspaces import Workspaces


@pytest.fixture
def state(tmp_path):
    registry = Registry(tmp_path/'registry', is_active=lambda owner: owner in {'alice', 'bob'})
    caps = Capabilities(registry)
    root = tmp_path/'objects'
    root.mkdir(mode=0o700)
    artifacts = ArtifactStore(root, caps)
    return registry, caps, artifacts, Workspaces(artifacts)


def run(state, owner='alice', key='first', writer=True):
    registry, caps, _, _ = state
    lease = registry.start_run(owner, 'conversation', request_key=key, writer=writer)
    token = caps.issue(lease.run_id, audience='artifact', operations={'artifact.commit', 'workspace.commit', 'workspace.read'})
    return lease, token.secret


def snapshot(state, secret, value=b'opaque guest snapshot'):
    async def chunks():
        yield value
    return asyncio.run(state[2].publish(secret, chunks(), filename='workspace.bin'))


def test_commit_and_restore_exact_previous_version_after_restart(state):
    registry, _, artifacts, workspaces = state
    first, secret = run(state)
    assert workspaces.restore(secret) is None
    item = snapshot(state, secret)
    assert workspaces.commit(secret, item.id) == 1
    with pytest.raises(AccessDenied):
        workspaces.commit(secret, item.id)
    second, next_secret = run(state, key='second')
    assert second.workspace_version == 1
    restored = Workspaces(artifacts).restore(next_secret)
    assert restored == b'opaque guest snapshot'
    registry.finish(second.run_id, status='failed')
    _, third_secret = run(state, key='third')
    assert workspaces.restore(third_secret) == restored


def test_foreign_owner_or_previous_run_snapshot_cannot_be_committed(state):
    registry, _, _, workspaces = state
    alice, secret = run(state)
    item = snapshot(state, secret)
    _, bob_secret = run(state, owner='bob')
    with pytest.raises(AccessDenied):
        workspaces.commit(bob_secret, item.id)
    registry.finish(alice.run_id, status='failed')
    _, next_secret = run(state, key='next')
    with pytest.raises(AccessDenied):
        workspaces.commit(next_secret, item.id)
    assert workspaces.restore(next_secret) is None


def test_battle_branch_does_not_implicitly_advance_workspace(state):
    registry, _, _, workspaces = state
    branch, secret = run(state, writer=False)
    item = snapshot(state, secret)
    with pytest.raises(AccessDenied):
        workspaces.commit(secret, item.id)
    registry.finish(branch.run_id, status='completed')
    _, next_secret = run(state, key='next')
    assert workspaces.restore(next_secret) is None


def test_stale_writer_cannot_publish_workspace(state):
    registry, _, _, workspaces = state
    first, secret = run(state)
    item = snapshot(state, secret)
    registry.cancel(first.run_id)
    _, next_secret = run(state, key='next')
    with pytest.raises(AccessDenied):
        workspaces.commit(secret, item.id)
    assert workspaces.restore(next_secret) is None


def test_only_selected_completed_battle_becomes_next_workspace(state):
    registry, _, _, workspaces = state
    first, first_token = run(state, writer=False)
    second, second_token = run(state, key='second', writer=False)
    for branch, secret, data in [(first, first_token, b'first'), (second, second_token, b'second')]:
        workspaces.stage_branch(secret, snapshot(state, secret, data).id)
        registry.finish(branch.run_id, status='completed')
    with pytest.raises(AccessDenied):
        workspaces.select_branch('bob', 'conversation', second.run_id)
    with pytest.raises(AccessDenied):
        workspaces.select_branch('alice', 'foreign', second.run_id)
    assert workspaces.select_branch('alice', 'conversation', second.run_id) == 1
    with pytest.raises(AccessDenied):
        workspaces.select_branch('alice', 'conversation', first.run_id)
    _, next_secret = run(state, key='next')
    assert workspaces.restore(next_secret) == b'second'


def test_failed_branch_and_active_writer_block_selection(state):
    registry, _, _, workspaces = state
    branch, secret = run(state, writer=False)
    workspaces.stage_branch(secret, snapshot(state, secret).id)
    with pytest.raises(AccessDenied):
        workspaces.select_branch('alice', 'conversation', branch.run_id)
    registry.finish(branch.run_id, status='failed')
    with pytest.raises(AccessDenied):
        workspaces.select_branch('alice', 'conversation', branch.run_id)
    branch, secret = run(state, key='other', writer=False)
    workspaces.stage_branch(secret, snapshot(state, secret).id)
    registry.finish(branch.run_id, status='completed')
    run(state, key='writer')
    with pytest.raises(AccessDenied):
        workspaces.select_branch('alice', 'conversation', branch.run_id)
