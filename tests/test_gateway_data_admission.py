"""SQL and fixed retrieval share capacity until cancellation cleanup settles."""
import asyncio
import importlib

import pytest

from gmail_search.gateway.database import QueryGateway, QueryLimits, ReaderCredential, ReaderRegistry


def admission(**limits):
    assert importlib.util.find_spec('gmail_search.gateway.data_admission') is not None
    return importlib.import_module('gmail_search.gateway.data_admission').DataAdmission(**limits)


def test_capacity_is_shared_by_owner_and_global_and_release_is_idempotent():
    capacity = admission(global_concurrency=2,owner_concurrency=1)
    alice = capacity.acquire('alice')
    with pytest.raises(RuntimeError,match='capacity'):
        capacity.acquire('alice')
    bob = capacity.acquire('bob')
    with pytest.raises(RuntimeError,match='capacity'):
        capacity.acquire('charlie')
    alice.release()
    alice.release()
    charlie = capacity.acquire('charlie')
    assert dict(capacity.active) == {'bob':1,'charlie':1}
    bob.release()
    charlie.release()
    assert not capacity.active


@pytest.mark.parametrize('options', [{'global_concurrency':0},{'owner_concurrency':0},{'global_concurrency':True},{'global_concurrency':33},{'global_concurrency':1,'owner_concurrency':2}])
def test_invalid_capacity_is_refused(options):
    with pytest.raises(ValueError):
        admission(**options)


def gateway(capacity):
    from gmail_search.gateway.database import reader_role
    credential = ReaderCredential('alice','dbname=synthetic user='+reader_role('alice'))
    return QueryGateway(ReaderRegistry({'alice':credential},is_active=lambda _:True), admission=capacity)


@pytest.mark.asyncio
async def test_sql_respects_capacity_held_by_another_data_operation():
    capacity = admission(global_concurrency=1,owner_concurrency=1)
    held = capacity.acquire('alice')
    api = gateway(capacity)
    with pytest.raises(RuntimeError,match='capacity'):
        await api.query('alice','SELECT count(*) FROM messages')
    assert dict(capacity.active) == {'alice':1}
    held.release()


@pytest.mark.asyncio
async def test_repeated_cancellation_keeps_slot_until_owned_cleanup_finishes(monkeypatch):
    capacity = admission(global_concurrency=1,owner_concurrency=1)
    api = gateway(capacity)
    running, cleanup, release = asyncio.Event(),asyncio.Event(),asyncio.Event()
    async def execute(*args):
        running.set()
        try:
            await asyncio.Future()
        finally:
            cleanup.set()
            await release.wait()
    monkeypatch.setattr(api,'_execute',execute)
    task = asyncio.create_task(api.query('alice','SELECT count(*) FROM messages'))
    await running.wait()
    task.cancel()
    await cleanup.wait()
    task.cancel()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    try:
        assert not task.done()
        with pytest.raises(RuntimeError,match='capacity'):
            capacity.acquire('bob')
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert not capacity.active


def test_shared_capacity_cannot_weaken_sql_profile():
    from gmail_search.gateway.database import reader_role
    credential = ReaderCredential('alice','dbname=synthetic user='+reader_role('alice'))
    with pytest.raises(ValueError):
        QueryGateway(ReaderRegistry({'alice':credential},is_active=lambda _:True),
                     limits=QueryLimits(global_concurrency=1,owner_concurrency=1),
                     admission=admission(global_concurrency=2,owner_concurrency=1))


def test_capacity_profile_and_active_snapshot_are_read_only():
    capacity = admission(global_concurrency=1,owner_concurrency=1)
    with pytest.raises(AttributeError):
        capacity.global_concurrency = 32
    held = capacity.acquire('alice')
    try:
        with pytest.raises(TypeError):
            capacity.active['alice'] = 0
    finally:
        held.release()
