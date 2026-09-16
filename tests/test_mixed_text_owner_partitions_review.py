"""Independent process-death checks in a disposable synthetic database."""

import os
import subprocess
import sys

from test_mixed_text_owner_partitions import mixed_source as mixed_source, setup
from test_text_owner_partitions import source as source, rows


def test_process_exit_after_pg_commit_resumes_without_recopy(mixed_source, tmp_path):
    implementation, controller, admin, snapshot, fence = setup(mixed_source, tmp_path)
    before = rows(mixed_source)
    code = '''
from contextlib import contextmanager
import os, sys
sys.path.insert(0, 'tests')
import psycopg
from psycopg.conninfo import make_conninfo
from gmail_search.gateway.maintenance import MaintenanceAdmin
from test_mixed_text_owner_partitions import module

implementation = module()
dsn = make_conninfo(os.environ['GMS_TEST_PG_DSN'], dbname=sys.argv[1])
admin = MaintenanceAdmin(sys.argv[2])
snapshot = admin.status()
with psycopg.connect(dsn, autocommit=True, connect_timeout=3) as conn:
    plan = implementation.capture_plan(conn, identity=snapshot.identity,
        migration_id=snapshot.migration_id, dominant_owner='alice',
        expected_owners=('alice', 'bob', 'charlie'))

class SyntheticFenceDouble:
    # This models the fixture owner's fence. It does not prove deployment drain.
    @contextmanager
    def hold(self, plan, *, deadline):
        yield

def checkpoint(stage):
    if stage == 'phase1_committed':
        os._exit(29)

controller = implementation.MixedTextPhaseOne(
    lambda: psycopg.connect(dsn, autocommit=True, connect_timeout=3),
    admin.path, plan, fence=SyntheticFenceDouble(), _checkpoint=checkpoint)
controller.advance(snapshot)
raise AssertionError('The committed-gap checkpoint was not reached')
'''
    child = subprocess.run([sys.executable, '-c', code, mixed_source.info.dbname,
        os.fspath(admin.path)], timeout=40, check=False, capture_output=True, text=True)
    assert child.returncode == 29, child.stderr[-3000:]
    assert admin.status() == snapshot
    assert rows(mixed_source) == before
    assert mixed_source.execute('SELECT to_regnamespace(%s)',
        (implementation.WITNESS_SCHEMA,)).fetchone()[0] is not None

    checkpoints = []
    restarted = implementation.MixedTextPhaseOne(controller._connect, admin.path,
        controller.plan, fence=fence, _checkpoint=checkpoints.append)
    result = restarted.advance(snapshot)
    assert result.state == 'INDEX_PENDING'
    assert admin.status() == result
    assert not any(stage.startswith(('copied:', 'deleted:')) for stage in checkpoints)
    assert rows(mixed_source) == before
    assert fence.closed
