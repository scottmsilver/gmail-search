"""Phase-two rehearsal: retained BM25 rebuild and READY publication.

Synthetic disposable PostgreSQL only. Requires the patched pg_search build —
an unpatched engine trips `assertion failed: item_pointer_is_valid(ctid)` when
a retained leaf is reindexed and then searched. See
docs/qualification/retained-reader-root-cause.md.
"""
from psycopg import sql
import pytest

# Shares the migration gate/witness with the other mixed-text phase file.
pytestmark = pytest.mark.pg_exclusive

from gmail_search.gateway.partitions import PARTITION_SCHEMA, partition_name
from gmail_search.gateway.registry import AccessDenied

from test_mixed_text_owner_partitions import (
    FixtureFence, mixed_source as mixed_source, module, setup, source as source,
)

TABLES = ('messages', 'attachments', 'propositions')


def leaf_bm25(conn, table, owner):
    """The (relname, relfilenode) of one owner leaf's BM25 index."""
    leaf = sql.Identifier(PARTITION_SCHEMA, partition_name(table, owner)).as_string(conn)
    rows = conn.execute(
        """SELECT c.relname,c.relfilenode FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid
           JOIN pg_am am ON am.oid=c.relam WHERE i.indrelid=to_regclass(%s) AND am.amname='bm25'""",
        (leaf,)).fetchall()
    assert len(rows) == 1, f'expected one BM25 index on {table}/{owner}'
    return rows[0]


def heap_identity(conn, table, owner):
    child = PARTITION_SCHEMA + '.' + partition_name(table, owner)
    return conn.execute(
        'SELECT oid,relfilenode,reltoastrelid FROM pg_class WHERE oid=%s::regclass', (child,)).fetchone()


def owner_counts(conn):
    return {table: conn.execute(sql.SQL(
        'SELECT user_id,count(*) FROM {} GROUP BY user_id ORDER BY user_id'
    ).format(sql.Identifier('public', table))).fetchall() for table in TABLES}


def test_phase_two_rebuilds_retained_indexes_and_publishes_ready(mixed_source, tmp_path):
    source = mixed_source
    _, controller, admin, snapshot, fence = setup(source, tmp_path)

    pending = controller.advance(snapshot)
    assert pending.state == 'INDEX_PENDING'

    before_index = {t: leaf_bm25(source, t, 'alice') for t in TABLES}
    before_heap = {t: heap_identity(source, t, 'alice') for t in TABLES}
    before_counts = owner_counts(source)

    ready = controller.publish(pending)

    assert ready.state == 'READY'
    assert admin.status() == ready
    assert ready.revision == pending.revision + 1
    assert ready.phase1_digest == pending.phase1_digest
    assert ready.qualification_digest and ready.qualification_digest != ready.phase1_digest

    for table in TABLES:
        # The retained index is rebuilt on purpose: same name, new storage.
        after = leaf_bm25(source, table, 'alice')
        assert after[0] == before_index[table][0]
        assert after[1] != before_index[table][1], f'{table} retained BM25 index was not rebuilt'
        # Its heap, TOAST and rows are untouched by the rebuild.
        assert heap_identity(source, table, 'alice') == before_heap[table]

    assert owner_counts(source) == before_counts
    assert fence.closed


def test_phase_two_is_idempotent_and_refuses_stale_input(mixed_source, tmp_path):
    source = mixed_source
    _, controller, admin, snapshot, fence = setup(source, tmp_path)
    pending = controller.advance(snapshot)

    ready = controller.publish(pending)
    filenodes = {t: leaf_bm25(source, t, 'alice')[1] for t in TABLES}

    # Re-publishing returns the same gate and does not reindex again.
    assert controller.publish(pending) == ready
    assert {t: leaf_bm25(source, t, 'alice')[1] for t in TABLES} == filenodes
    assert admin.status() == ready

    # A READY snapshot is not a valid phase-two input.
    with pytest.raises(AccessDenied):
        controller.publish(ready)
    # Neither is the pre-phase-one gate.
    with pytest.raises(AccessDenied):
        controller.publish(snapshot)
    assert admin.status() == ready
    assert fence.closed


def test_phase_two_requires_committed_phase_one(mixed_source, tmp_path):
    _, controller, admin, snapshot, _ = setup(mixed_source, tmp_path)
    with pytest.raises(AccessDenied):
        controller.publish(snapshot)
    assert admin.status() == snapshot


def resumed_controller(source, registry_path, plan, *, checkpoint=lambda stage: None):
    """Another controller over the same registry, without re-initialising the gate."""
    import os
    import psycopg
    from psycopg.conninfo import make_conninfo
    fence = FixtureFence(source)

    def connect():
        return psycopg.connect(make_conninfo(os.environ['GMS_TEST_PG_DSN'], dbname=source.info.dbname),
                               autocommit=True)
    controller = module().MixedTextPhaseOne(connect, registry_path, plan, fence=fence, _checkpoint=checkpoint)
    return controller, fence


@pytest.mark.parametrize('stage', ['retained_vacuumed', 'indexes_rebuilt', 'phase2_committed'])
def test_phase_two_failure_leaves_gate_and_witness_recoverable(mixed_source, tmp_path, stage):
    """A crash mid-phase-two must not publish READY, and must stay resumable."""
    source = mixed_source
    implementation, controller, admin, snapshot, _ = setup(source, tmp_path)
    pending = controller.advance(snapshot)
    registry_path = tmp_path / 'registry.sqlite'

    def fail(current):
        if current == stage:
            raise RuntimeError('synthetic phase-two failure')

    crashing, fence = resumed_controller(source, registry_path, controller.plan, checkpoint=fail)
    with pytest.raises(AccessDenied):
        crashing.publish(pending)

    assert admin.status() == pending, 'gate advanced despite a phase-two failure'
    assert fence.closed
    witness_phase, = source.execute(sql.SQL('SELECT phase FROM {}').format(sql.Identifier(
        implementation.WITNESS_SCHEMA, implementation.WITNESS_TABLE))).fetchone()
    # The rebuild commits together with the witness phase, so only a stage after
    # that commit may legitimately have advanced it.
    expected = 'INDEXES_REBUILT' if stage == 'phase2_committed' else 'LAYOUT_COMMITTED'
    assert witness_phase == expected

    resumed, _ = resumed_controller(source, registry_path, controller.plan)
    ready = resumed.publish(pending)
    assert ready.state == 'READY' and admin.status() == ready
