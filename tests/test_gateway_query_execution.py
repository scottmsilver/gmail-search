import asyncio

import psycopg
import pytest

from gmail_search.gateway.database import ReaderCredential, ReaderRegistry, QueryGateway, QueryLimits
from test_gateway_database_integration import database as database_fixture, reader_dsn

database = database_fixture


def gateway(dsn, owner, *, active=None, **limits):
    cred = ReaderCredential(owner, reader_dsn(dsn, owner))
    return QueryGateway(ReaderRegistry({owner: cred}, is_active=lambda uid: active is None or uid in active), limits=QueryLimits(**limits))


@pytest.mark.asyncio
async def test_gateway_whole_mailbox_aggregate_and_bounded_extract(database):
    dsn, (alice, bob) = database
    with psycopg.connect(dsn, autocommit=True) as admin:
        admin.execute('INSERT INTO public.messages (id,user_id) VALUES (%s,%s)', (alice+'2',alice))
    api = gateway(dsn, alice, max_rows=1)
    result = await api.query(alice, 'SELECT count(*) AS total FROM messages')
    assert result.rows == ((2,),)
    assert result.complete
    result = await api.query(alice, 'SELECT id FROM messages ORDER BY id')
    assert result.rows == ((alice,),)
    assert not result.complete
    assert result.columns == ('id',)
    assert not api.active_queries


@pytest.mark.asyncio
async def test_gateway_rejects_revoked_unknown_users_and_arbitrary_sql(database):
    from gmail_search.gateway.analytics import QueryRejected
    dsn, (alice, bob) = database
    active = {alice}
    api = gateway(dsn, alice, active=active)
    with pytest.raises(PermissionError):
        await api.query(bob, 'SELECT count(*) FROM messages')
    with pytest.raises(QueryRejected):
        await api.query(alice, 'SELECT pg_read_file(\'/etc/passwd\')')
    active.clear()
    with pytest.raises(PermissionError):
        await api.query(alice, 'SELECT count(*) FROM messages')


@pytest.mark.asyncio
async def test_gateway_byte_limit_marks_incomplete(database):
    dsn, (alice, bob) = database
    api = gateway(dsn, alice, max_bytes=4)
    result = await api.query(alice, 'SELECT subject FROM messages')
    assert result.rows == () and not result.complete


@pytest.mark.asyncio
async def test_gateway_cancellation_terminates_query_and_frees_slot(database):
    dsn, (alice, bob) = database
    api = gateway(dsn, alice, deadline_seconds=3, lock_timeout_ms=2500)
    with psycopg.connect(dsn) as blocker:
        blocker.execute('LOCK TABLE public.messages IN ACCESS EXCLUSIVE MODE')
        task = asyncio.create_task(api.query(alice, 'SELECT count(*) FROM messages'))
        # Wait for server-visible query, not an arbitrary sleep.
        for _ in range(100):
            with psycopg.connect(dsn, autocommit=True) as admin:
                waiting = admin.execute("SELECT 1 FROM pg_stat_activity WHERE datname=current_database() AND application_name='gms-analytical-gateway' AND wait_event_type='Lock'").fetchone()
            if waiting:
                break
            await asyncio.sleep(.02)
        assert waiting
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with psycopg.connect(dsn, autocommit=True) as admin:
            assert not admin.execute("SELECT 1 FROM pg_stat_activity WHERE datname=current_database() AND application_name='gms-analytical-gateway'").fetchone()
    assert not api.active_queries
    assert (await api.query(alice, 'SELECT count(*) FROM messages')).rows == ((1,),)


@pytest.mark.asyncio
async def test_gateway_deadline_and_revocation_cancel_waiting_query(database):
    dsn, (alice, bob) = database
    active = {alice}
    api = gateway(dsn, alice, active=active, deadline_seconds=.3, lock_timeout_ms=2000)
    with psycopg.connect(dsn) as blocker:
        blocker.execute('LOCK TABLE public.messages IN ACCESS EXCLUSIVE MODE')
        with pytest.raises(TimeoutError):
            await api.query(alice, 'SELECT count(*) FROM messages')
    assert not api.active_queries
    assert (await api.query(alice, 'SELECT count(*) FROM messages')).rows == ((1,),)


@pytest.mark.asyncio
async def test_oversized_row_is_bounded_before_transport(database):
    import tracemalloc
    dsn, (alice, bob) = database
    with psycopg.connect(dsn, autocommit=True) as admin:
        admin.execute('UPDATE public.messages SET body_text=%s WHERE user_id=%s', ('x' * 8_000_000, alice))
    api = gateway(dsn, alice, max_bytes=1024)
    tracemalloc.start()
    try:
        result = await api.query(alice, 'SELECT body_text FROM messages')
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert not result.complete and not result.rows
    assert peak < 2_000_000, 'Gateway must not decode an oversized private row before applying its byte limit'


@pytest.mark.asyncio
async def test_inflight_owner_revocation_closes_connection(database):
    dsn, (alice, bob) = database
    active = {alice}
    api = gateway(dsn, alice, active=active, deadline_seconds=3, lock_timeout_ms=2500)
    with psycopg.connect(dsn) as blocker:
        blocker.execute('LOCK TABLE public.messages IN ACCESS EXCLUSIVE MODE')
        task = asyncio.create_task(api.query(alice, 'SELECT count(*) FROM messages'))
        while not api.active_queries:
            await asyncio.sleep(.01)
        active.clear()
        with pytest.raises(PermissionError):
            await asyncio.wait_for(task, 1)
    assert not api.active_queries
    with psycopg.connect(dsn, autocommit=True) as admin:
        assert not admin.execute("SELECT 1 FROM pg_stat_activity WHERE datname=current_database() AND application_name='gms-analytical-gateway'").fetchone()


@pytest.mark.asyncio
async def test_concurrent_capacity_is_released_after_cancellation(database):
    dsn, (alice, bob) = database
    credentials = {uid: ReaderCredential(uid, reader_dsn(dsn, uid)) for uid in (alice, bob)}
    api = QueryGateway(ReaderRegistry(credentials, is_active=lambda _: True), limits=QueryLimits(owner_concurrency=1, global_concurrency=1, deadline_seconds=3, lock_timeout_ms=2500))
    with psycopg.connect(dsn) as blocker:
        blocker.execute('LOCK TABLE public.messages IN ACCESS EXCLUSIVE MODE')
        task = asyncio.create_task(api.query(alice, 'SELECT count(*) FROM messages'))
        while not api.active_queries:
            await asyncio.sleep(.01)
        for uid in (alice, bob):
            with pytest.raises(RuntimeError, match='capacity'):
                await api.query(uid, 'SELECT count(*) FROM messages')
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert not api.active_queries
    assert (await api.query(bob, 'SELECT count(*) FROM messages')).rows == ((1,),)
