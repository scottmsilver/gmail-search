"""Independent crash checks for synthetic controller metadata only."""

import os
import subprocess
import sys

import pytest

from gmail_search.gateway.capabilities import Capabilities
from gmail_search.gateway.maintenance import MaintenanceAdmin, ReleaseIdentity
from gmail_search.gateway.partition_profiles import TEXT_OWNER_PARTITIONS_V1
from gmail_search.gateway.registry import Registry


@pytest.mark.parametrize('after_invalidation', [False, True])
def test_process_death_before_closure_commit_rolls_back_all_authority(tmp_path, after_invalidation):
    tmp_path.chmod(0o700)
    path = tmp_path / 'registry.sqlite'
    identity = ReleaseIdentity('review-store', TEXT_OWNER_PARTITIONS_V1, 1)
    admin = MaintenanceAdmin(path, verifier=lambda *args, **kwargs: 'a' * 64)
    initial = admin.initialize_closed(identity, migration_id='review-1',
        owner_set_digest='b' * 64, procedure_digest='c' * 64)
    original = admin.publish_ready(admin.record_index_pending(initial))
    registry = Registry(path, is_active=lambda _: True, release_identity=identity)
    run = registry.start_run('alice', 'conversation', request_key='old')
    token = Capabilities(registry).issue(run.run_id, audience='sql', operations=['query'])
    code = '''
import os, sys
from gmail_search.gateway.maintenance import MaintenanceAdmin, ReleaseIdentity
from gmail_search.gateway.partition_profiles import TEXT_OWNER_PARTITIONS_V1
original = MaintenanceAdmin._invalidate
def crash(db):
    if sys.argv[2] == 'yes':
        original(db)
    os._exit(23)
MaintenanceAdmin._invalidate = staticmethod(crash)
admin = MaintenanceAdmin(sys.argv[1])
admin.begin_maintenance(ReleaseIdentity('review-store', TEXT_OWNER_PARTITIONS_V1, 2),
    migration_id='review-2', owner_set_digest='d'*64, procedure_digest='e'*64,
    expected_revision=admin.status().revision)
'''
    result = subprocess.run([sys.executable, '-c', code, os.fspath(path),
        'yes' if after_invalidation else 'no'], timeout=10, check=False)
    assert result.returncode == 23
    # SQLite may need a writable connection to recover a hot journal. A runtime
    # read-only readiness check may deny until administrative recovery completes.
    with registry._transaction() as db:
        assert db.execute('SELECT COUNT(*) FROM store_releases').fetchone()[0] == 1
        assert tuple(db.execute('SELECT fence,writer FROM conversations').fetchone()) == (run.fence, run.run_id)
    assert admin.status() == original
    assert Capabilities(registry).authorize(token.secret,
        audience='sql', operation='query').run_id == run.run_id
