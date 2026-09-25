"""Phase-one rehearsal only: synthetic mail, private Registry, no READY path."""
from contextlib import contextmanager
from dataclasses import replace
import importlib.util
import hashlib
import json
import os
import time
from pathlib import Path

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo
import pytest

# Shares the migration gate/witness with the other mixed-text phase file.
pytestmark = pytest.mark.pg_exclusive

from gmail_search.gateway.maintenance import MaintenanceAdmin, ReleaseIdentity
from gmail_search.gateway.partition_profiles import TEXT_OWNER_PARTITIONS_V1 as TEXT
from gmail_search.gateway.partitions import PARTITION_SCHEMA, partition_name
from gmail_search.gateway.registry import AccessDenied
from test_text_owner_partitions import source as source, inventory, rows


ROOT = Path(__file__).resolve().parents[1]


def module():
    path = ROOT / 'deploy/public/migrate_mixed_text_owner_partitions.py'
    assert path.exists(), 'The approved phase-one module is not implemented yet'
    spec = importlib.util.spec_from_file_location('mixed_text_phase_one', path)
    result = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = result
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def mixed_source(source):
    source.execute("INSERT INTO users(id,email) VALUES('charlie','charlie@example.test')")
    for owner in ('bob', 'charlie'):
        source.execute("INSERT INTO messages(id,user_id,thread_id,from_addr,to_addr,date,subject,body_text) VALUES(%s,%s,'thread','sender','recipient','2026','needle',%s)",
            (owner+'-message', owner, owner+' synthetic mail'))
        source.execute("INSERT INTO attachments(user_id,message_id,filename,mime_type,extracted_text) VALUES(%s,%s,'file','text/plain',%s)",
            (owner,owner+'-message',owner+' attachment'))
        source.execute("INSERT INTO propositions(user_id,message_id,text,model) VALUES(%s,%s,%s,'fixture')", (owner,owner+'-message',owner+' fact'))
        source.execute("INSERT INTO message_summaries(user_id,message_id,summary,model) VALUES(%s,%s,%s,'fixture')",(owner,owner+'-message',owner+' summary'))
        source.execute("INSERT INTO prop_processed VALUES(%s,%s)",(owner,owner+'-message'))
    # Keep the dominant heap's out-of-line storage meaningful without copying it.
    source.execute("UPDATE messages SET body_text=(SELECT string_agg(md5(x::text),'') FROM generate_series(1,10000) x) WHERE user_id='alice'")
    for owner,count in (('alice',3),('bob',1)):
        for number in range(count):
            mid = owner+'-extra-'+str(number)
            source.execute("INSERT INTO messages(id,user_id,thread_id,from_addr,to_addr,date,body_text) VALUES(%s,%s,'t','s','r','2026',%s)",(mid,owner,mid))
            source.execute("INSERT INTO attachments(user_id,message_id,filename,mime_type,extracted_text) VALUES(%s,%s,'extra','application/octet-stream',NULL)",(owner,mid))
            source.execute("INSERT INTO propositions(user_id,message_id,text,model,embedding) VALUES(%s,%s,'extra','fixture',%s)",(owner,mid,b'\x00\xffsynthetic\x00'))
    source.execute("UPDATE messages SET body_text=(SELECT string_agg(md5(('minority-'||x)::text),'') FROM generate_series(1,10000) x) WHERE id='bob-message'")
    source.execute("INSERT INTO embeddings(user_id,message_id,attachment_id,chunk_type,chunk_text,embedding,model) SELECT 'bob','bob-message',id,'attachment','synthetic',%s,'fixture' FROM attachments WHERE user_id='bob' AND message_id='bob-message'",(b'\x00\xff\x01\x00',))
    return source


class FixtureFence:
    """Fence test double plus real session check; NOT durable deployment fencing."""
    def __init__(self, source):
        self.source = source
        self.closed = True
        self.entries = 0

    @contextmanager
    def hold(self, plan, *, deadline):
        assert self.closed
        assert self.source.info.dbname.startswith('gms_owner_partitions_test_')
        # The fixture owns exactly its source connection when each phase begins.
        pids = self.source.execute('SELECT pid FROM pg_stat_activity WHERE datname=current_database()').fetchall()
        assert pids == [(self.source.info.backend_pid,)]
        self.entries += 1
        yield
        # Releasing inspections must NEVER undo the durable external fence.
        assert self.closed


def setup(source, tmp_path, *, checkpoint=lambda stage: None):
    implementation = module()
    identity = ReleaseIdentity('synthetic-mixed-store', TEXT, 1)
    plan = implementation.capture_plan(source, identity=identity, migration_id='mixed-1',
        dominant_owner='alice', expected_owners=('alice','bob','charlie'))
    tmp_path.chmod(0o700)
    path = tmp_path / 'registry.sqlite'
    admin = MaintenanceAdmin(path)
    snapshot = admin.initialize_closed(identity, migration_id=plan.migration_id,
        owner_set_digest=plan.owner_set_digest, procedure_digest=plan.procedure_digest)
    fence = FixtureFence(source)
    def connect():
        return psycopg.connect(make_conninfo(os.environ['GMS_TEST_PG_DSN'],dbname=source.info.dbname), autocommit=True)
    controller = implementation.MixedTextPhaseOne(connect, path, plan, fence=fence,
        _checkpoint=checkpoint)
    return implementation, controller, admin, snapshot, fence


def test_phase_one_module_boundary_exists():
    assert module().MixedTextPhaseOne


def test_phase_one_preserves_rows_storage_and_leaves_readiness_closed(mixed_source,tmp_path):
    source = mixed_source
    before, data = inventory(source), rows(source)
    implementation, controller, admin, snapshot, fence = setup(source,tmp_path)
    result = controller.advance(snapshot)
    assert result.state == 'INDEX_PENDING' and admin.status() == result
    assert rows(source) == data
    assert fence.closed and fence.entries == 1
    for table in ('messages','attachments','propositions'):
        child = PARTITION_SCHEMA+'.'+partition_name(table,'alice')
        assert source.execute('SELECT oid,relfilenode,reltoastrelid FROM pg_class WHERE oid=%s::regclass',(child,)).fetchone() == before[table][:3]
        actual = source.execute(sql.SQL('SELECT user_id,count(*) FROM {} GROUP BY user_id ORDER BY user_id').format(sql.Identifier('public',table))).fetchall()
        assert actual == [('alice',4),('bob',2),('charlie',1)]
    for name in ('messages_bm25_idx','attachments_bm25_idx','props_bm25_idx'):
        oid,node = before[name]
        assert source.execute('SELECT relfilenode FROM pg_class WHERE oid=%s',(oid,)).fetchone() == (node,)
    witness = json.loads(source.execute(sql.SQL('SELECT inventory FROM {}').format(
        sql.Identifier(implementation.WITNESS_SCHEMA,implementation.WITNESS_TABLE))).fetchone()[0])
    assert witness['source']['messages']['heap'] == list(before['messages'])
    with pytest.raises(AccessDenied):
        admin.publish_ready(result)  # No phase-two verifier exists.
    assert controller.advance(snapshot) == result
    assert controller.advance(result) == result
    assert fence.closed


@pytest.mark.parametrize('stage',['copied:messages:bob','deleted:messages:bob','renamed','attached','witness_written'])
def test_phase_one_failure_rolls_back_source_and_witness(mixed_source,tmp_path,stage):
    before, data = inventory(mixed_source), rows(mixed_source)
    def fail(current):
        if current == stage:
            raise RuntimeError('synthetic phase-one failure')
    implementation, controller, admin, snapshot, fence = setup(mixed_source,tmp_path,checkpoint=fail)
    with pytest.raises(AccessDenied):
        controller.advance(snapshot)
    assert inventory(mixed_source) == before and rows(mixed_source) == data
    assert mixed_source.execute('SELECT to_regnamespace(%s)',(implementation.WITNESS_SCHEMA,)).fetchone() == (None,)
    assert admin.status() == snapshot and fence.closed


def test_committed_pg_sqlite_gap_resumes_without_copying_again(mixed_source,tmp_path):
    checkpoints = []
    def fail(stage):
        checkpoints.append(stage)
        if stage == 'phase1_committed':
            raise RuntimeError('synthetic lost phase-one acknowledgement')
    implementation, controller, admin, snapshot, fence = setup(mixed_source,tmp_path,checkpoint=fail)
    before = rows(mixed_source)
    with pytest.raises(AccessDenied):
        controller.advance(snapshot)
    assert admin.status() == snapshot
    assert rows(mixed_source) == before
    controller = implementation.MixedTextPhaseOne(lambda:psycopg.connect(make_conninfo(os.environ['GMS_TEST_PG_DSN'],dbname=mixed_source.info.dbname),autocommit=True),
        admin.path,controller.plan,fence=fence,_checkpoint=checkpoints.append)
    result = controller.advance(snapshot)
    assert result.state == 'INDEX_PENDING'
    assert sum(stage == 'copied:messages:bob' for stage in checkpoints) == 1
    assert rows(mixed_source) == before and fence.closed


@pytest.mark.parametrize('fault',['unknown_owner','wrong_database','wrong_procedure','missing_fence'])
def test_binding_or_fence_failure_precedes_layout_changes(mixed_source,tmp_path,fault):
    implementation, controller, admin, snapshot, _ = setup(mixed_source,tmp_path)
    before = inventory(mixed_source)
    if fault == 'unknown_owner':
        mixed_source.execute("INSERT INTO users(id,email) VALUES('outside','outside@example.test')")
        mixed_source.execute("INSERT INTO messages(id,user_id,thread_id,from_addr,to_addr,date) VALUES('outside','outside','t','s','r','2026')")
    elif fault == 'wrong_database':
        controller = implementation.MixedTextPhaseOne(controller._connect,admin.path,
            replace(controller.plan,database_oid=controller.plan.database_oid+1),fence=controller._fence)
    elif fault == 'wrong_procedure':
        controller = implementation.MixedTextPhaseOne(controller._connect,admin.path,
            replace(controller.plan,procedure_digest='0'*64),fence=controller._fence)
    else:
        controller._fence = None
    with pytest.raises(AccessDenied):
        controller.advance(snapshot)
    assert inventory(mixed_source) == before and admin.status() == snapshot


def test_procedure_digest_pins_actual_module_and_helper_bytes():
    implementation = module()
    expected = hashlib.sha256((ROOT/'deploy/public/migrate_mixed_text_owner_partitions.py').read_bytes()
        + b'\x00' + (ROOT/'deploy/public/migrate_text_owner_partitions.py').read_bytes()).hexdigest()
    assert implementation.PROCEDURE_DIGEST == expected


def test_closed_phase_resume_refuses_sequence_allocation_drift(mixed_source,tmp_path):
    _, controller, admin, snapshot, _ = setup(mixed_source,tmp_path)
    pending = controller.advance(snapshot)
    mixed_source.execute("SELECT nextval('public.attachments_id_seq')")
    with pytest.raises(AccessDenied):
        controller.advance(pending)
    assert admin.status() == pending


def test_controller_plan_is_pinned(mixed_source,tmp_path):
    _, controller, _, _, _ = setup(mixed_source,tmp_path)
    with pytest.raises(AttributeError):
        controller.plan = replace(controller.plan,database_oid=0)


def test_factory_cannot_reuse_same_connection(mixed_source,tmp_path):
    _, controller, _, _, _ = setup(mixed_source,tmp_path)
    connection = controller._connect()
    class Reused:
        def __getattr__(self,name):return getattr(connection,name)
        def close(self):pass
    reused = Reused()
    controller._connect = lambda:reused
    try:
        with controller._fresh():pass
        with pytest.raises(ValueError,match='[Ff]resh'):
            with controller._fresh():pass
    finally:
        connection.close()


def test_unknown_connection_close_ack_poison_prevents_retry(mixed_source,tmp_path):
    _, controller, _, snapshot, _ = setup(mixed_source,tmp_path)
    factory = controller._connect
    opened = []
    class LostCloseAck:
        def __init__(self):
            self.connection = factory()
            opened.append(self.connection)
        def __getattr__(self,name):return getattr(self.connection,name)
        def close(self):
            self.connection.close()
            raise OSError('synthetic lost close ACK')
    controller._connect = LostCloseAck
    with pytest.raises(OSError):
        with controller._fresh():pass
    with pytest.raises(AccessDenied):
        controller.advance(snapshot)
    assert len(opened) == 1


def test_fence_exit_failure_poison_prevents_retry(mixed_source,tmp_path):
    _, controller, admin, snapshot, fence = setup(mixed_source,tmp_path)
    class LostFenceAck:
        @contextmanager
        def hold(self,*args,**kwargs):
            with fence.hold(*args,**kwargs):
                yield
            raise OSError('synthetic fence inspection cleanup ACK lost')
    controller._fence = LostFenceAck()
    with pytest.raises(AccessDenied):controller.advance(snapshot)
    with pytest.raises(AccessDenied):controller.advance(snapshot)
    assert fence.entries == 1
    assert admin.status() == snapshot and fence.closed


def test_phase_budget_applies_across_individually_fast_statements(mixed_source,tmp_path):
    _, controller, _, _, _ = setup(mixed_source,tmp_path)
    deadline = time.monotonic()+1
    with pytest.raises((TimeoutError,psycopg.errors.QueryCanceled)):
        with controller._fresh(deadline=deadline) as conn:
            with conn.transaction():
                for _ in range(20):
                    conn.execute('SELECT pg_sleep(.1)')
    assert time.monotonic()-deadline < .5


@pytest.mark.parametrize('fault',['grant','column','default','constraint','index','schema_object','dependency','extra_leaf','extra_leaf_index','missing_index','allocation'])
def test_existing_witness_and_inventory_drift_refuses_resume(mixed_source,tmp_path,fault):
    implementation, controller, admin, snapshot, _ = setup(mixed_source,tmp_path)
    pending = controller.advance(snapshot)
    witness = sql.Identifier(implementation.WITNESS_SCHEMA,implementation.WITNESS_TABLE)
    child = sql.Identifier(PARTITION_SCHEMA,partition_name('messages','alice'))
    if fault=='grant':mixed_source.execute(sql.SQL('GRANT SELECT(binding) ON {} TO PUBLIC').format(witness))
    elif fault=='column':mixed_source.execute(sql.SQL('ALTER TABLE {} ADD COLUMN extra text').format(witness))
    elif fault=='default':mixed_source.execute(sql.SQL("ALTER TABLE {} ALTER COLUMN phase SET DEFAULT 'LAYOUT_COMMITTED'").format(witness))
    elif fault=='constraint':mixed_source.execute(sql.SQL('ALTER TABLE {} DROP CONSTRAINT text_phase_one_phase_check').format(witness))
    elif fault=='index':mixed_source.execute(sql.SQL('CREATE INDEX extra_witness ON {}(phase)').format(witness))
    elif fault=='schema_object':mixed_source.execute(sql.SQL('CREATE TYPE {} AS ENUM (\'extra\')').format(sql.Identifier(implementation.WITNESS_SCHEMA,'extra')))
    elif fault=='dependency':mixed_source.execute(sql.SQL('CREATE VIEW public.witness_dependency AS SELECT binding FROM {}').format(witness))
    elif fault=='extra_leaf':mixed_source.execute(sql.SQL('CREATE TABLE {}(id text)').format(sql.Identifier(PARTITION_SCHEMA,'extra')))
    elif fault=='extra_leaf_index':mixed_source.execute(sql.SQL('CREATE INDEX extra_leaf_index ON {}(date)').format(child))
    elif fault=='missing_index':mixed_source.execute('DROP INDEX public.idx_messages_user_date')
    else:mixed_source.execute("SELECT nextval('public.propositions_id_seq')")
    with pytest.raises(AccessDenied):controller.advance(pending)
    assert admin.status() == pending
