"""Historical shared BM25 statistics require an explicit rebuild boundary."""
import importlib.util
import os
from pathlib import Path

import pytest

# BM25 statistics / plan assertions: needs a quiet database (see scripts/test.sh).
pytestmark = pytest.mark.pg_exclusive

SOURCE=Path(__file__).parents[1]/'deploy/public/probe_bm25_deleted_statistics.py'


def module():
    spec=importlib.util.spec_from_file_location('deleted_statistics_probe',SOURCE)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def _require_approved_fixture():
    """Skip unless GMS_TEST_PG_DSN names the disposable fixture the probe accepts.

    Being merely set is not enough: the probe refuses any other DSN, and these
    tests need the *patched* pg_search build besides (see the churn test's
    docstring). Off-fixture runs must skip, not error."""
    dsn=os.getenv('GMS_TEST_PG_DSN')
    if not dsn:pytest.skip('Explicit synthetic ParadeDB required')
    if not module().dsn_is_approved_fixture(dsn):
        pytest.skip('GMS_TEST_PG_DSN is not the approved disposable ParadeDB fixture')


@pytest.fixture(scope='module')
def report():
    _require_approved_fixture()
    return module().run()


@pytest.mark.parametrize('profile',['message_text','message_numeric','attachment'])
def test_historical_foreign_statistics_survive_delete_vacuum_and_attach(report,profile):
    evidence=report['profiles'][profile]
    clean=evidence['clean']['force_custom_plan']
    polluted=evidence['mixed_before_delete']['force_custom_plan']
    assert clean!=polluted and [row[0] for row in clean]!=[row[0] for row in polluted]
    for step in ('after_delete','after_analyze','after_vacuum','after_attach','attached_vacuum','after_merge_vacuum'):
        value=evidence[step]
        assert value['force_custom_plan']==value['force_generic_plan']==value['fresh_backend']==polluted
        assert value['foreign_only']==[] and value['live_rows']==4
    assert evidence['direct_child_denied']
    segments=evidence['after_vacuum']['segments']
    assert sum(int(row['num_docs']) for row in segments)==4
    assert sum(int(row['num_deleted']) for row in segments)==200
    assert evidence['after_vacuum']['objects']==evidence['leaf_after_attach_objects']


@pytest.mark.parametrize('profile',['message_text','message_numeric','attachment'])
def test_reindex_restores_clean_scores_preserves_heap_and_can_rollback(report,profile):
    evidence=report['profiles'][profile]
    clean=evidence['clean']['force_custom_plan']
    rebuilt=evidence['reindex']
    assert rebuilt['force_custom_plan']==rebuilt['force_generic_plan']==rebuilt['fresh_backend']==clean
    old=evidence['after_attach']['objects'];new=rebuilt['objects']
    assert old[:6]==new[:6] and old[6]!=new[6]
    assert sum(int(row['num_deleted']) for row in rebuilt['segments'])==0
    rollback=evidence['reindex_rollback']
    assert rollback['before']['objects']==rollback['after']['objects']
    assert rollback['during_objects'][:6]==old[:6] and rollback['during_objects'][6]!=old[6]
    assert rollback['before']['force_custom_plan']==rollback['after']['fresh_backend']


@pytest.mark.parametrize('dsn',[
    'host=127.0.0.1 port=5544 dbname=postgres user=postgres',
    'host=127.0.0.1 hostaddr=192.0.2.1 port=55440 dbname=postgres user=postgres',
    'host=127.0.0.1 port=55440 dbname=postgres user=postgres options=-csearch_path=evil',
])
def test_probe_rejects_nonfixture_dsn_before_connect(monkeypatch,dsn):
    probe=module();monkeypatch.setenv('GMS_TEST_PG_DSN',dsn)
    def forbidden(*args,**kwargs):pytest.fail('Unsafe DSN attempted connection')
    with monkeypatch.context() as guard:
        guard.setattr(probe.psycopg,'connect',forbidden)
        with pytest.raises(ValueError):probe.run()


@pytest.fixture(scope='module')
def atomic_report():
    _require_approved_fixture()
    return module().run(atomic=True)


@pytest.mark.parametrize('profile',['message_text','message_numeric','attachment'])
def test_same_transaction_delete_attach_reindex_keeps_foreign_statistics(atomic_report,profile):
    evidence=atomic_report['profiles'][profile]
    clean=evidence['clean']['fresh_backend']
    polluted=evidence['mixed_before_delete']['fresh_backend']
    committed=evidence['atomic_commit']
    assert committed['all_owner_counts']==[('alice',4)]
    assert committed['force_custom_plan']==committed['force_generic_plan']==committed['fresh_backend']==polluted
    assert committed['during_owner_rows']==polluted and polluted!=clean
    assert sum(int(row['num_docs']) for row in committed['segments'])==204
    rollback=evidence['atomic_rollback']
    assert rollback['all_owner_counts']==[('alice',4),('bob',200)]
    assert rollback['objects']==evidence['mixed_before_delete']['objects']
    assert rollback['fresh_backend']==polluted
    assert evidence['separate_post_commit_reindex']['fresh_backend']==clean


@pytest.fixture(scope='module')
def staged_report():
    _require_approved_fixture()
    return module().run(staged=True)


@pytest.mark.parametrize('profile',['message_text','message_numeric','attachment'])
def test_committed_delete_attach_then_restart_reindex_is_clean_and_retryable(staged_report,profile):
    evidence=staged_report['profiles'][profile]
    old=evidence['mixed_before_delete'];clean=evidence['clean']['fresh_backend']
    phase1=evidence['phase1_commit'];rollback=evidence['phase2_rollback'];retry=evidence['phase2_retry_commit']
    assert phase1['all_owner_counts']==rollback['all_owner_counts']==[('alice',4)]
    assert phase1['fresh_backend']==rollback['fresh_backend']==old['fresh_backend']
    assert phase1['objects'][:7]==rollback['objects'][:7]==old['objects'][:7]
    assert rollback['during_objects'][:6]==old['objects'][:6]
    assert rollback['during_objects'][6]!=old['objects'][6]
    assert retry['fresh_backend']==retry['force_custom_plan']==retry['force_generic_plan']==clean
    assert retry['objects'][:6]==old['objects'][:6] and retry['objects'][6]!=old['objects'][6]
    assert retry['live_rows']==4 and retry['foreign_only']==[]


@pytest.fixture(scope='module')
def churn_reports():
    _require_approved_fixture()
    return {mode:module().run(churn=mode) for mode in ('observed','unobserved')}


@pytest.mark.parametrize('mode',['observed','unobserved'])
@pytest.mark.parametrize('profile',['message_text','message_numeric','attachment'])
def test_generic_plan_after_owner_churn_no_longer_asserts(churn_reports,mode,profile):
    """This test used to pin the crash. It now pins its absence.

    Until the patched engine, these two steps reproduced
    `assertion failed: item_pointer_is_valid(ctid)` under a generic plan — the
    upstream pg_search defect rooted out in
    docs/qualification/retained-reader-root-cause.md. The test asserted that
    error *as the expected result*, which was right while it was the behaviour
    of the engine we ran.

    We now run the patched build, so the generic plan returns rows. Asserting
    that is the regression guard: if the patch is ever lost — a rebuilt image
    that pulls the floating tag, a rollback to stock 0.23.0 — the assertion
    comes back and this fails, naming the reason.
    """
    evidence=churn_reports[mode]['profiles'][profile]
    clean=evidence['clean']['fresh_backend']
    for step in ('after_vacuum','read_retry'):
        value=evidence['post_reindex_churn']['stages'][step]
        for plan in ('force_generic_plan','fresh_generic_backend'):
            assert isinstance(value[plan],list), (
                f'{plan} at {step} is not rows but {value[plan]!r} — if this is the '
                'item_pointer_is_valid assertion, the engine has lost the ctid patch')
        assert value['force_generic_plan']==value['fresh_generic_backend']
        assert value['force_custom_plan']==value['fresh_backend']==clean
    if mode=='observed':
        for step in ('after_insert','after_delete'):
            assert isinstance(evidence['post_reindex_churn']['stages'][step]['force_generic_plan'],list)


@pytest.fixture(scope='module')
def custom_report():
    _require_approved_fixture()
    return module().run(churn='custom_cycles')


@pytest.mark.parametrize('profile',['message_text','message_numeric','attachment'])
def test_candidate_custom_unprepared_profile_survives_staged_commit_and_cycles(custom_report,profile):
    evidence=custom_report['profiles'][profile]
    assert evidence['phase2_retry_commit']['fresh_backend']==evidence['clean']['fresh_backend']
    cycles=evidence['custom_unprepared_cycles']
    assert len(cycles['cycles'])==3
    for cycle in cycles['cycles']:
        for step in ('vacuum','foreign_insert','foreign_update','foreign_delete_vacuum'):
            assert cycle[step]['rows']==cycles['baseline']['rows']
            assert cycle[step]['direct']==cycles['baseline']['direct']
