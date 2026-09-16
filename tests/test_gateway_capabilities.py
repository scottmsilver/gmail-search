"""Durable capability, writer fencing and budget boundaries with synthetic IDs."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
import sqlite3

import pytest

from gmail_search.gateway.capabilities import Capabilities
from gmail_search.gateway.registry import Registry, AccessDenied


@pytest.fixture
def state(tmp_path):
    now = [1000.0]
    active = {'alice', 'bob'}
    path = tmp_path / 'registry.sqlite'
    registry = Registry(path, is_active=lambda owner: owner in active, clock=lambda: now[0])
    return registry, Capabilities(registry), now, active, path


def start(registry, **kwargs):
    return registry.start_run('alice', 'conversation', request_key='request', **kwargs)


def test_capability_derives_immutable_identity_and_hides_token(state):
    registry, capabilities, _, _, _ = state
    run = start(registry)
    token = capabilities.issue(run.run_id, audience='mail', operations={'search'}, ttl=30)
    grant = capabilities.authorize(token.secret, audience='mail', operation='search')
    assert (grant.owner_id, grant.conversation_id, grant.run_id) == ('alice', 'conversation', run.run_id)
    assert grant.fence == 1
    assert token.secret not in repr(token)
    with pytest.raises(FrozenInstanceError):
        grant.owner_id = 'bob'
    with pytest.raises(TypeError):
        capabilities.authorize(token.secret, audience='mail', operation='search', owner_id='bob')


@pytest.mark.parametrize('change', ['forged', 'audience', 'operation', 'expired', 'cancelled', 'owner', 'lease'])
def test_denies_invalid_or_inactive_capabilities(state, change):
    registry, capabilities, now, active, _ = state
    run = start(registry, lease_ttl=20, deadline_ttl=100)
    token = capabilities.issue(run.run_id, audience='mail', operations={'search'}, ttl=10)
    secret, audience, operation = token.secret, 'mail', 'search'
    if change == 'forged':
        secret = 'unrecognized-secret'
    elif change == 'audience':
        audience = 'inference'
    elif change == 'operation':
        operation = 'artifact.commit'
    elif change == 'expired':
        now[0] += 10
    elif change == 'cancelled':
        registry.cancel(run.run_id)
    elif change == 'owner':
        active.remove('alice')
    elif change == 'lease':
        token = capabilities.issue(run.run_id, audience='mail', operations={'search'}, ttl=80)
        secret = token.secret
        now[0] += 21
    with pytest.raises(AccessDenied) as caught:
        capabilities.authorize(secret, audience=audience, operation=operation)
    assert secret not in str(caught.value)


def test_tokens_are_hashes_on_disk_and_survive_restart(state):
    registry, capabilities, now, active, path = state
    run = start(registry)
    token = capabilities.issue(run.run_id, audience='mail', operations={'search'}, ttl=30)
    with sqlite3.connect(path) as db:
        rows = db.execute('SELECT * FROM capabilities').fetchall()
    assert token.secret not in repr(rows)
    restarted = Registry(path, is_active=lambda owner: owner in active, clock=lambda: now[0])
    assert Capabilities(restarted).authorize(token.secret, audience='mail', operation='search').run_id == run.run_id


def test_one_writer_and_monotonic_fencing(state):
    registry, capabilities, _, _, _ = state
    first = start(registry)
    with pytest.raises(AccessDenied):
        registry.start_run('alice', 'conversation', request_key='next')
    token = capabilities.issue(first.run_id, audience='artifact', operations={'workspace.commit'}, ttl=30)
    assert capabilities.commit_workspace(token.secret) == 1
    second = registry.start_run('alice', 'conversation', request_key='next')
    assert second.fence == first.fence + 1
    assert second.workspace_version == 1
    with pytest.raises(AccessDenied):
        capabilities.commit_workspace(token.secret)


def test_expired_writer_is_replaced_and_cannot_heartbeat(state):
    registry, _, now, _, _ = state
    first = start(registry, lease_ttl=10)
    now[0] += 11
    second = registry.start_run('alice', 'conversation', request_key='replacement')
    assert second.fence == first.fence + 1
    with pytest.raises(AccessDenied):
        registry.heartbeat(first.run_id)


def test_idempotent_run_creation_and_cross_owner_budget_denied(state):
    registry, _, _, _, _ = state
    budget = registry.create_budget('alice', 100)
    first = start(registry, budget_id=budget)
    assert start(registry, budget_id=budget) == first
    with pytest.raises(AccessDenied):
        registry.start_run('bob', 'conversation', request_key='b', budget_id=budget)


def test_atomic_shared_budget_reservations_across_battle_runs(state):
    registry, _, _, _, _ = state
    budget = registry.create_budget('alice', 100)
    runs = [registry.start_run('alice', 'conversation', request_key=str(i), writer=False, budget_id=budget) for i in range(2)]

    def reserve(run):
        try:
            return registry.reserve(run.run_id, 'call', 70)
        except AccessDenied:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(reserve, runs))
    assert sum(outcome is not None for outcome in outcomes) == 1
    winner = runs[next(i for i, outcome in enumerate(outcomes) if outcome is not None)]
    assert registry.reserve(winner.run_id, 'call', 70) is not None
    with pytest.raises(AccessDenied):
        registry.reserve(winner.run_id, 'call', 71)
    registry.settle(winner.run_id, 'call', 20)
    registry.settle(winner.run_id, 'call', 20)
    assert registry.reserve(runs[1].run_id, 'another', 80) is not None


def test_cancel_keeps_inflight_budget_reserved_until_settlement(state):
    registry, _, _, _, _ = state
    budget = registry.create_budget('alice', 100)
    run = start(registry, budget_id=budget)
    registry.reserve(run.run_id, 'inflight', 100)
    registry.cancel(run.run_id)
    next_run = registry.start_run('alice', 'conversation', request_key='next', budget_id=budget)
    with pytest.raises(AccessDenied):
        registry.reserve(next_run.run_id, 'call', 1)
    registry.settle(run.run_id, 'inflight', 60)
    registry.reserve(next_run.run_id, 'call', 40)


def test_revocation_rechecked_at_commit_and_budget_operation(state):
    registry, capabilities, _, active, _ = state
    budget = registry.create_budget('alice', 100)
    run = start(registry, budget_id=budget)
    token = capabilities.issue(run.run_id, audience='artifact', operations={'workspace.commit'}, ttl=30)
    capabilities.authorize(token.secret, audience='artifact', operation='workspace.commit')
    active.remove('alice')
    with pytest.raises(AccessDenied):
        capabilities.commit_workspace(token.secret)
    with pytest.raises(AccessDenied):
        registry.reserve(run.run_id, 'call', 10)


def test_deadline_bounds_token_and_heartbeat(state):
    registry, capabilities, now, _, _ = state
    run = start(registry, lease_ttl=5, deadline_ttl=10)
    token = capabilities.issue(run.run_id, audience='mail', operations={'search'}, ttl=100)
    assert token.expires_at == run.deadline
    now[0] += 4
    registry.heartbeat(run.run_id, ttl=100)
    now[0] += 6
    with pytest.raises(AccessDenied):
        capabilities.authorize(token.secret, audience='mail', operation='search')


def test_battle_branch_artifact_cannot_commit_after_workspace_advances(state):
    registry, capabilities, _, _, _ = state
    branch = start(registry, writer=False)
    token = capabilities.issue(branch.run_id, audience='artifact', operations={'artifact.commit', 'workspace.commit'}, ttl=30)
    with pytest.raises(AccessDenied):
        capabilities.commit_workspace(token.secret)
    registry.start_run('alice', 'conversation', request_key='writer')
    with pytest.raises(AccessDenied):
        capabilities.authorize(token.secret, audience='artifact', operation='artifact.commit')


def test_reservation_retry_is_not_fresh_permission_to_spend(state):
    registry, _, _, _, _ = state
    run = start(registry, budget_id=registry.create_budget('alice', 100))
    first = registry.reserve(run.run_id, 'request', 40)
    repeat = registry.reserve(run.run_id, 'request', 40)
    assert first.created is True
    assert repeat.created is False
    assert repeat.charged is None
    registry.settle(run.run_id, 'request', 25)
    settled = registry.reserve(run.run_id, 'request', 40)
    assert settled.created is False
    assert settled.charged == 25


def test_revoke_one_token_leaves_other_token_active(state):
    registry, capabilities, _, _, _ = state
    run = start(registry)
    first = capabilities.issue(run.run_id, audience='mail', operations={'search'})
    second = capabilities.issue(run.run_id, audience='mail', operations={'search'})
    capabilities.revoke(first.secret)
    with pytest.raises(AccessDenied):
        capabilities.authorize(first.secret, audience='mail', operation='search')
    assert capabilities.authorize(second.secret, audience='mail', operation='search').run_id == run.run_id


def test_expiry_sweep_and_owner_cancellation_are_idempotent(state):
    registry, _, now, _, _ = state
    expired = start(registry, lease_ttl=5, writer=False)
    active = registry.start_run('alice', 'other', request_key='active', writer=False)
    now[0] += 6
    assert registry.expire() == (expired.run_id,)
    assert registry.expire() == ()
    assert registry.cancel_owner('alice') == (active.run_id,)
    assert registry.cancel_owner('alice') == ()


def test_failed_and_read_only_completion_revoke_tokens(state):
    registry, capabilities, _, _, _ = state
    for status in ('completed', 'failed'):
        run = registry.start_run('alice', 'conversation', request_key=status, writer=False)
        token = capabilities.issue(run.run_id, audience='mail', operations={'search'})
        registry.finish(run.run_id, status=status)
        with pytest.raises(AccessDenied):
            capabilities.authorize(token.secret, audience='mail', operation='search')


def test_distinct_registry_instances_serialize_writer_creation(state):
    registry, _, now, active, path = state
    other = Registry(path, is_active=lambda owner: owner in active, clock=lambda: now[0])

    def create(pair):
        instance, key = pair
        try:
            return instance.start_run('alice', 'conversation', request_key=key)
        except AccessDenied:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(create, [(registry, 'one'), (other, 'two')]))
    assert sum(result is not None for result in results) == 1


def test_invitation_lookup_failure_is_closed_and_sanitized(state):
    registry, capabilities, _, _, _ = state
    run = start(registry)
    token = capabilities.issue(run.run_id, audience='mail', operations={'search'})

    def unavailable(owner):
        raise RuntimeError('sensitive invitation backend diagnostics')

    registry.is_active = unavailable
    with pytest.raises(AccessDenied) as caught:
        capabilities.authorize(token.secret, audience='mail', operation='search')
    assert 'sensitive' not in str(caught.value)


@pytest.mark.parametrize('units', [-1, True, 1.5, 10**13])
def test_invalid_budget_units_are_rejected(state, units):
    registry, _, _, _, _ = state
    with pytest.raises(AccessDenied):
        registry.create_budget('alice', units)


def test_failed_commit_does_not_change_workspace_version(state):
    registry, capabilities, _, _, path = state
    run = start(registry)
    token = capabilities.issue(run.run_id, audience='artifact', operations={'workspace.commit'})
    registry.cancel(run.run_id)
    with pytest.raises(AccessDenied):
        capabilities.commit_workspace(token.secret)
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT version FROM conversations').fetchone()[0] == 0


@pytest.mark.parametrize('mode', [0o755, 0o777])
def test_registry_rejects_nonprivate_parent_directory(tmp_path, mode):
    parent = tmp_path / 'unsafe'
    parent.mkdir(mode=mode)
    parent.chmod(mode)
    with pytest.raises(AccessDenied):
        Registry(parent / 'registry.sqlite', is_active=lambda owner: True)
    assert not (parent / 'registry.sqlite').exists()


def test_registry_accepts_private_owned_parent(tmp_path):
    parent = tmp_path / 'private'
    parent.mkdir(mode=0o700)
    registry = Registry(parent / 'registry.sqlite', is_active=lambda owner: True)
    assert registry.start_run('alice', 'conversation', request_key='new').owner_id == 'alice'
