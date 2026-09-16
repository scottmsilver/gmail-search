"""Disposable controller metadata only; verifier doubles do not qualify PostgreSQL."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
import importlib
import os
import sqlite3
import stat
import subprocess
import sys
import threading
import time

import pytest

from gmail_search.gateway.capabilities import Capabilities
from gmail_search.gateway.partition_profiles import TEXT_OWNER_PARTITIONS_V1, NUMERIC_OWNER_PARTITIONS_V1
from gmail_search.gateway.registry import AccessDenied, Registry


def api():
    return importlib.import_module('gmail_search.gateway.maintenance')


def identity(epoch=1, store='synthetic-store', profile=TEXT_OWNER_PARTITIONS_V1):
    return api().ReleaseIdentity(store, profile, epoch)


def verified(snapshot, target, *, deadline):
    assert time.monotonic() < deadline
    return ('a' if target == 'INDEX_PENDING' else 'b') * 64


def old_registry(tmp_path):
    tmp_path.chmod(0o700)
    return Registry(tmp_path / 'registry.sqlite', is_active=lambda owner: True)


def closed(tmp_path, *, verifier=verified):
    old = old_registry(tmp_path)
    admin = api().MaintenanceAdmin(old.path, verifier=verifier)
    snapshot = admin.initialize_closed(identity(), migration_id='migration-1',
        owner_set_digest='c' * 64, procedure_digest='d' * 64)
    return old, admin, snapshot


def ready(tmp_path):
    old, admin, snapshot = closed(tmp_path)
    snapshot = admin.record_index_pending(snapshot)
    snapshot = admin.publish_ready(snapshot)
    registry = Registry(old.path, is_active=lambda owner: True, release_identity=identity())
    return registry, admin, snapshot


def next_release(admin, snapshot):
    return admin.begin_maintenance(identity(snapshot.identity.release_epoch + 1),
        migration_id='migration-2', owner_set_digest='e' * 64,
        procedure_digest='f' * 64, expected_revision=snapshot.revision)


def test_runtime_missing_file_never_creates_it(tmp_path):
    module = api()
    tmp_path.chmod(0o700)
    path = tmp_path / 'absent.sqlite'
    with pytest.raises(AccessDenied):
        module.GateReader(path, identity()).require_ready()
    with pytest.raises(AccessDenied):
        Registry(path, is_active=lambda _: True, release_identity=identity())
    assert list(tmp_path.iterdir()) == []


def test_initialize_revokes_existing_authority_and_reopen_never_revives_it(tmp_path):
    registry = old_registry(tmp_path)
    budget = registry.create_budget('alice', 100)
    run = registry.start_run('alice', 'conversation', request_key='old', budget_id=budget)
    caps = Capabilities(registry)
    token = caps.issue(run.run_id, audience='sql', operations=['query'])
    registry.reserve(run.run_id, 'provider', 60)
    admin = api().MaintenanceAdmin(registry.path, verifier=verified)
    snapshot = admin.initialize_closed(identity(), migration_id='migration-1',
        owner_set_digest='c' * 64, procedure_digest='d' * 64)
    with registry._transaction() as db:
        assert db.execute('SELECT status FROM runs').fetchone()[0] == 'cancelled'
        assert db.execute('SELECT revoked FROM capabilities').fetchone()[0] == 1
        assert tuple(db.execute('SELECT fence,writer FROM conversations').fetchone()) == (2, None)
        assert tuple(db.execute('SELECT reserved,spent FROM budgets').fetchone()) == (60, 0)
    snapshot = admin.publish_ready(admin.record_index_pending(snapshot))
    new = Registry(registry.path, is_active=lambda _: True, release_identity=identity())
    with pytest.raises(AccessDenied):
        Capabilities(new).authorize(token.secret, audience='sql', operation='query')
    with pytest.raises(AccessDenied):
        new.start_run('alice', 'conversation', request_key='old', budget_id=budget)
    new.settle(run.run_id, 'provider', 40)
    fresh = new.start_run('alice', 'conversation', request_key='new', budget_id=budget)
    assert fresh.run_id != run.run_id


def test_closed_and_wrong_release_deny_but_cleanup_remains_available(tmp_path):
    registry, admin, snapshot = ready(tmp_path)
    budget = registry.create_budget('alice', 100)
    run = registry.start_run('alice', 'conversation', request_key='old', budget_id=budget)
    token = Capabilities(registry).issue(run.run_id, audience='sql', operations=['query'])
    registry.reserve(run.run_id, 'provider', 60)
    pending = next_release(admin, snapshot)
    operations = [lambda: registry.start_run('alice', 'c2', request_key='new'),
        lambda: registry.heartbeat(run.run_id),
        lambda: registry.reserve(run.run_id, 'new-call', 1),
        lambda: Capabilities(registry).issue(run.run_id, audience='sql', operations=['query']),
        lambda: Capabilities(registry).authorize(token.secret, audience='sql', operation='query')]
    for operation in operations:
        with pytest.raises(AccessDenied):
            operation()
    registry.settle(run.run_id, 'provider', 60)
    registry.cancel(run.run_id)
    admin.publish_ready(admin.record_index_pending(pending))
    for operation in operations:
        with pytest.raises(AccessDenied):
            operation()
    newer = Registry(registry.path, is_active=lambda _: True, release_identity=identity(2))
    assert newer.start_run('bob', 'c', request_key='new').owner_id == 'bob'


def test_updated_legacy_default_cannot_bypass_enabled_gate(tmp_path):
    registry, _, _ = ready(tmp_path)
    legacy = Registry(registry.path, is_active=lambda _: True)
    with pytest.raises(AccessDenied):
        legacy.start_run('alice', 'c', request_key='new')
    run = registry.start_run('alice', 'c', request_key='gated')
    token = Capabilities(registry).issue(run.run_id, audience='sql', operations=['query'])
    with pytest.raises(AccessDenied):
        Capabilities(legacy).authorize(token.secret, audience='sql', operation='query')


def test_gate_callback_does_not_recursively_take_sqlite_writer_lock(tmp_path):
    registry, _, _ = ready(tmp_path)
    reader = api().GateReader(registry.path, identity())
    guarded = Registry(registry.path, release_identity=identity(),
        is_active=lambda _: reader.require_ready() is not None)
    started = time.monotonic()
    assert guarded.start_run('alice', 'c', request_key='new')
    assert time.monotonic() - started < 1


@pytest.mark.parametrize('wrong', [lambda: identity(2), lambda: identity(store='other'),
    lambda: identity(profile=NUMERIC_OWNER_PARTITIONS_V1)])
def test_reader_pins_exact_identity(tmp_path, wrong):
    registry, _, _ = ready(tmp_path)
    with pytest.raises(AccessDenied):
        api().GateReader(registry.path, wrong()).require_ready()
    with pytest.raises(FrozenInstanceError):
        identity().release_epoch = 2


def test_missing_verifier_and_invalid_receipt_cannot_advance(tmp_path):
    _, admin, snapshot = closed(tmp_path, verifier=None)
    with pytest.raises(AccessDenied):
        admin.record_index_pending(snapshot)
    for verifier in (lambda *a, **k: True, lambda *a, **k: 'bad'):
        rejected = api().MaintenanceAdmin(admin.path, verifier=verifier)
        with pytest.raises(AccessDenied):
            rejected.record_index_pending(snapshot)
    assert admin.status() == snapshot
    with pytest.raises(AccessDenied):
        api().MaintenanceAdmin(admin.path, verifier=verified).publish_ready(snapshot)


def test_exact_retries_and_stale_transition_bindings(tmp_path):
    _, admin, snapshot = closed(tmp_path)
    assert admin.initialize_closed(identity(), migration_id='migration-1',
        owner_set_digest='c' * 64, procedure_digest='d' * 64) == snapshot
    pending = admin.record_index_pending(snapshot)
    assert admin.record_index_pending(snapshot) == pending
    current = admin.publish_ready(pending)
    assert admin.publish_ready(pending) == current
    with pytest.raises(AccessDenied):
        admin.initialize_closed(identity(), migration_id='changed',
            owner_set_digest='c' * 64, procedure_digest='d' * 64)
    following = next_release(admin, current)
    assert next_release(admin, current) == following
    with pytest.raises(AccessDenied):
        admin.publish_ready(pending)
    with pytest.raises(AccessDenied):
        admin.begin_maintenance(identity(2), migration_id='other-operation',
            owner_set_digest='e' * 64, procedure_digest='f' * 64,
            expected_revision=current.revision)


def test_failed_atomic_initialization_restores_old_registry(tmp_path):
    registry = old_registry(tmp_path)
    run = registry.start_run('alice', 'c', request_key='old')
    token = Capabilities(registry).issue(run.run_id, audience='sql', operations=['query'])
    with registry._transaction() as db:
        db.execute("CREATE TRIGGER reject_revoke BEFORE UPDATE ON capabilities BEGIN SELECT RAISE(ABORT,'synthetic'); END")
    with pytest.raises(AccessDenied):
        api().MaintenanceAdmin(registry.path).initialize_closed(identity(), migration_id='m',
            owner_set_digest='c' * 64, procedure_digest='d' * 64)
    assert Capabilities(registry).authorize(token.secret, audience='sql', operation='query').run_id == run.run_id
    with registry._transaction() as db:
        assert not db.execute("SELECT 1 FROM sqlite_master WHERE name='store_gate'").fetchone()


def test_publisher_finishes_before_closure_and_timeout_leaves_ready(tmp_path):
    registry, admin, snapshot = ready(tmp_path)
    reader = api().GateReader(registry.path, identity(), lock_timeout=.1)
    entered, release = threading.Event(), threading.Event()
    def publisher():
        with reader.publication_guard():
            entered.set()
            assert release.wait(3)
    with ThreadPoolExecutor(1) as pool:
        task = pool.submit(publisher)
        assert entered.wait(2)
        short = api().MaintenanceAdmin(registry.path, verifier=verified, lock_timeout=.05)
        with pytest.raises(AccessDenied):
            next_release(short, snapshot)
        assert reader.require_ready().state == 'READY'
        release.set()
        task.result(timeout=2)
    next_release(admin, snapshot)
    with pytest.raises(AccessDenied):
        with reader.publication_guard():
            pytest.fail('publisher admitted while closed')


def test_verifier_outside_write_transaction_and_failure_preserves_state(tmp_path):
    registry, admin, snapshot = closed(tmp_path)
    def verifier(current, target, *, deadline):
        # A different connection can acquire the writer lock during verification.
        with sqlite3.connect(registry.path, timeout=.05) as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute("UPDATE conversations SET writer=NULL")
        raise RuntimeError('synthetic verifier failure')
    failing = api().MaintenanceAdmin(registry.path, verifier=verifier)
    with pytest.raises(AccessDenied):
        failing.record_index_pending(snapshot)
    assert admin.status() == snapshot


def test_deleted_gate_tables_cannot_reenable_legacy_authority(tmp_path):
    registry, _, _ = ready(tmp_path)
    with registry._transaction() as db:
        db.execute('DROP TABLE store_gate')
        db.execute('DROP TABLE store_releases')
    with pytest.raises(AccessDenied):
        legacy = Registry(registry.path, is_active=lambda _: True)
        legacy.start_run('alice', 'c', request_key='bypass')


def test_verifier_cas_rechecks_catalog_after_external_work(tmp_path):
    registry, admin, snapshot = closed(tmp_path)
    def verifier(current, target, *, deadline):
        # Simulate another administrative writer outside the required file-lock
        # composition. Even that stale observation cannot overwrite a new state.
        with sqlite3.connect(registry.path) as db:
            db.execute('UPDATE store_gate SET revision=revision+1')
        return 'a' * 64
    conflicting = api().MaintenanceAdmin(registry.path, verifier=verifier)
    with pytest.raises(AccessDenied):
        conflicting.record_index_pending(snapshot)
    assert admin.status().state == 'MAINTENANCE'


@pytest.mark.parametrize('field,value', [('migration_id', 'forged'),
    ('owner_set_digest', '1' * 64), ('procedure_digest', '2' * 64),
    ('identity', None), ('revision', 100)])
def test_forged_transition_snapshot_refused_before_verification(tmp_path, field, value):
    _, admin, snapshot = closed(tmp_path)
    calls = []
    guarded = api().MaintenanceAdmin(admin.path, verifier=lambda *a, **k: calls.append(a))
    with pytest.raises(AccessDenied):
        guarded.record_index_pending(replace(snapshot, **{field: value}))
    assert calls == []


def test_atomic_closure_rolls_back_gate_runs_caps_and_fences_on_failure(tmp_path):
    registry, admin, snapshot = ready(tmp_path)
    run = registry.start_run('alice', 'c', request_key='old')
    token = Capabilities(registry).issue(run.run_id, audience='sql', operations=['query'])
    with registry._transaction() as db:
        db.execute("CREATE TRIGGER reject_fence BEFORE UPDATE ON conversations BEGIN SELECT RAISE(ABORT,'synthetic'); END")
    with pytest.raises(AccessDenied):
        next_release(admin, snapshot)
    assert admin.status() == snapshot
    assert Capabilities(registry).authorize(token.secret, audience='sql', operation='query').run_id == run.run_id
    with registry._transaction() as db:
        assert db.execute('SELECT COUNT(*) FROM store_releases').fetchone()[0] == 1
        assert db.execute('SELECT fence FROM conversations').fetchone()[0] == run.fence


def test_admission_transaction_racing_close_cannot_leave_live_old_run(tmp_path):
    registry, admin, snapshot = ready(tmp_path)
    entered, release, closing = threading.Event(), threading.Event(), threading.Event()
    def active(owner):
        entered.set()
        assert release.wait(3)
        return True
    registry.is_active = active
    def close():
        closing.set()
        return next_release(admin, snapshot)
    with ThreadPoolExecutor(2) as pool:
        start = pool.submit(registry.start_run, 'alice', 'c', request_key='race')
        assert entered.wait(2)
        closure = pool.submit(close)
        assert closing.wait(2)
        release.set()
        run = start.result(timeout=2)
        closure.result(timeout=2)
    with registry._transaction() as db:
        assert db.execute('SELECT status FROM runs WHERE run_id=?', (run.run_id,)).fetchone()[0] == 'cancelled'
    with pytest.raises(AccessDenied):
        registry.heartbeat(run.run_id)


def test_committed_close_survives_abrupt_controller_exit(tmp_path):
    registry, _, snapshot = ready(tmp_path)
    run = registry.start_run('alice', 'c', request_key='old')
    token = Capabilities(registry).issue(run.run_id, audience='sql', operations=['query'])
    code = '''
import os,sys
from gmail_search.gateway.maintenance import MaintenanceAdmin, ReleaseIdentity
from gmail_search.gateway.partition_profiles import TEXT_OWNER_PARTITIONS_V1
a=MaintenanceAdmin(sys.argv[1])
s=a.status()
a.begin_maintenance(ReleaseIdentity('synthetic-store',TEXT_OWNER_PARTITIONS_V1,2),
    migration_id='migration-2',owner_set_digest='e'*64,procedure_digest='f'*64,
    expected_revision=s.revision)
os._exit(17)
'''
    process = subprocess.run([sys.executable, '-c', code, str(registry.path)], timeout=5)
    assert process.returncode == 17
    restarted = api().MaintenanceAdmin(registry.path, verifier=verified)
    pending = restarted.status()
    assert pending.state == 'MAINTENANCE' and pending.revision == snapshot.revision + 1
    restarted.publish_ready(restarted.record_index_pending(pending))
    current = Registry(registry.path, is_active=lambda _: True, release_identity=identity(2))
    with pytest.raises(AccessDenied):
        Capabilities(current).authorize(token.secret, audience='sql', operation='query')


def test_expired_external_verification_cannot_advance(tmp_path):
    _, admin, snapshot = closed(tmp_path)
    def slow(*args, **kwargs):
        time.sleep(.03)
        return 'a' * 64
    timed = api().MaintenanceAdmin(admin.path, verifier=slow, verification_timeout=.01)
    with pytest.raises(AccessDenied):
        timed.record_index_pending(snapshot)
    assert admin.status() == snapshot


def test_administrator_creation_syncs_file_and_directory_entries(tmp_path, monkeypatch):
    tmp_path.chmod(0o700)
    observed = []
    original = os.fsync
    def synced(fd):
        observed.append(os.fstat(fd).st_mode)
        original(fd)
    monkeypatch.setattr(os, 'fsync', synced)
    admin = api().MaintenanceAdmin(tmp_path / 'new.sqlite')
    admin.initialize_closed(identity(), migration_id='m',
        owner_set_digest='c' * 64, procedure_digest='d' * 64)
    assert sum(stat.S_ISREG(mode) for mode in observed) == 2
    assert sum(stat.S_ISDIR(mode) for mode in observed) == 2
    observed.clear()
    with pytest.raises(AccessDenied):
        api().GateReader(admin.path, identity()).require_ready()
    assert observed == []


def test_failed_creation_sync_cannot_report_initialized_gate(tmp_path, monkeypatch):
    tmp_path.chmod(0o700)
    def failed(fd):
        raise OSError('synthetic durability failure')
    monkeypatch.setattr(os, 'fsync', failed)
    admin = api().MaintenanceAdmin(tmp_path / 'new.sqlite')
    with pytest.raises(AccessDenied):
        admin.initialize_closed(identity(), migration_id='m',
            owner_set_digest='c' * 64, procedure_digest='d' * 64)
    with pytest.raises(AccessDenied):
        api().GateReader(admin.path, identity()).require_ready()


def test_existing_runtime_denies_journal_mode_drift(tmp_path):
    registry, _, _ = ready(tmp_path)
    run = registry.start_run('alice', 'c', request_key='old')
    token = Capabilities(registry).issue(run.run_id, audience='sql', operations=['query'])
    with sqlite3.connect(registry.path) as db:
        assert db.execute('PRAGMA journal_mode=WAL').fetchone()[0] == 'wal'
    with pytest.raises(AccessDenied):
        Capabilities(registry).authorize(token.secret, audience='sql', operation='query')
    with pytest.raises(AccessDenied):
        registry.start_run('alice', 'new', request_key='new')


def test_closed_gate_preserves_failed_worker_stop_binding_until_ack(tmp_path):
    from test_gateway_worker import Backend
    from gmail_search.gateway.worker import WorkerController
    registry, admin, snapshot = ready(tmp_path)
    backend = Backend()
    controller = WorkerController(registry, backend)
    run = registry.start_run('alice', 'c', request_key='worker')
    handle = controller.start(run.run_id)
    pending = next_release(admin, snapshot)
    backend.fail_stop = True
    with pytest.raises(AccessDenied):
        controller.reconcile()
    assert handle in backend.inventory()
    with registry._transaction() as db:
        assert db.execute('SELECT state FROM workers').fetchone()[0] == 'stopping'
    assert admin.status() == pending
    backend.fail_stop = False
    assert controller.reconcile() == 1
    assert backend.inventory() == set()
    with registry._transaction() as db:
        assert db.execute('SELECT state FROM workers').fetchone()[0] == 'stopped'


def test_runtime_open_executes_only_readonly_statements(tmp_path, monkeypatch):
    registry, _, _ = ready(tmp_path)
    statements = []
    opened = []
    connect = sqlite3.connect
    def tracing(database, *args, **kwargs):
        opened.append(database)
        db = connect(database, *args, **kwargs)
        db.set_trace_callback(statements.append)
        return db
    monkeypatch.setattr(sqlite3, 'connect', tracing)
    Registry(registry.path, is_active=lambda _: True, release_identity=identity())
    api().GateReader(registry.path, identity()).require_ready()
    assert all(path.endswith('?mode=ro') for path in opened)
    assert all(statement.lstrip().split()[0].upper() in ('SELECT', 'PRAGMA') for statement in statements)


@pytest.mark.parametrize('change', ["DELETE FROM store_gate",
    "UPDATE store_gate SET qualification_digest='bad'",
    "UPDATE store_releases SET profile='unknown'",
    "UPDATE store_releases SET store_id='other'"])
def test_corrupted_gate_metadata_denies_existing_runtime(tmp_path, change):
    registry, _, _ = ready(tmp_path)
    with sqlite3.connect(registry.path) as db:
        db.execute(change)
    with pytest.raises(AccessDenied):
        registry.start_run('alice', 'c', request_key='new')
    with pytest.raises(AccessDenied):
        api().GateReader(registry.path, identity()).require_ready()
