"""Run-scoped analytical service, including real PostgreSQL cancellation."""
import asyncio

import psycopg
import pytest

from gmail_search.gateway.analytics import QueryRejected
from gmail_search.gateway.capabilities import Capabilities
from gmail_search.gateway.database import QueryGateway, QueryLimits, QueryResult, ReaderCredential, ReaderRegistry
from gmail_search.gateway.registry import AccessDenied, Registry
from gmail_search.gateway.service import RunQueryService
from test_gateway_database_integration import database as database_fixture, reader_dsn

database = database_fixture


def scoped(tmp_path, gateway, owner='alice'):
    registry = Registry(tmp_path / 'service.sqlite', is_active=lambda _: True)
    capabilities = Capabilities(registry)
    run = registry.start_run(owner, 'conversation', request_key='query', writer=False)
    token = capabilities.issue(run.run_id, audience='sql', operations={'query', 'schema'})
    return RunQueryService(capabilities, gateway), registry, capabilities, run, token


def gateway(database):
    dsn, owners = database
    readers = {owner: ReaderCredential(owner, reader_dsn(dsn, owner)) for owner in owners}
    return QueryGateway(ReaderRegistry(readers, is_active=lambda _: True), limits=QueryLimits(deadline_seconds=4, lock_timeout_ms=3000))


@pytest.mark.asyncio
async def test_token_selects_only_its_owners_rows(database, tmp_path):
    _, (alice, bob) = database
    service, _, _, _, token = scoped(tmp_path, gateway(database), alice)
    result = await service.query(token.secret, 'SELECT user_id FROM messages')
    assert result.rows == ((alice,),)
    assert result.complete
    foreign = await service.query(token.secret, f"SELECT user_id FROM messages WHERE user_id='{bob}'")
    assert foreign.rows == ()
    with pytest.raises(TypeError):
        await service.query(token.secret, 'SELECT user_id FROM messages', owner_id=bob)


@pytest.mark.asyncio
@pytest.mark.parametrize('reason', ['unknown', 'audience', 'revoked'])
async def test_rejected_token_does_not_start_query(database, tmp_path, reason):
    _, (alice, _) = database
    api = gateway(database)
    service, _, capabilities, run, token = scoped(tmp_path, api, alice)
    secret = token.secret
    if reason == 'unknown':
        secret = '0' * 64
    elif reason == 'audience':
        secret = capabilities.issue(run.run_id, audience='mail', operations={'query'}).secret
    else:
        capabilities.revoke(secret)
    with pytest.raises(AccessDenied):
        await service.query(secret, 'SELECT user_id FROM messages')
    assert not api.active_queries


@pytest.mark.asyncio
async def test_schema_has_separate_operation_and_is_immutable(tmp_path):
    service, _, capabilities, run, token = scoped(tmp_path, None)
    schema = await service.schema(token.secret)
    assert schema['messages']['user_id'] == 'text'
    assert 'users' not in schema
    with pytest.raises(TypeError):
        schema['messages']['user_id'] = 'int8'
    query_only = capabilities.issue(run.run_id, audience='sql', operations={'query'})
    with pytest.raises(AccessDenied):
        await service.schema(query_only.secret)


@pytest.mark.asyncio
@pytest.mark.parametrize('query', [None, 1, {}, 'DELETE FROM messages'])
async def test_bad_query_types_and_writes_are_rejected(database, tmp_path, query):
    _, (alice, _) = database
    api = gateway(database)
    service, _, _, _, token = scoped(tmp_path, api, alice)
    with pytest.raises(QueryRejected):
        await service.query(token.secret, query)
    assert not api.active_queries


@pytest.mark.asyncio
@pytest.mark.parametrize('cancel_caller', [False, True])
async def test_inflight_revocation_or_caller_cancellation_closes_database_connection(database, tmp_path, cancel_caller):
    dsn, (alice, _) = database
    api = gateway(database)
    service, registry, _, run, token = scoped(tmp_path, api, alice)
    with psycopg.connect(dsn) as blocker:
        blocker.execute('LOCK TABLE public.messages IN ACCESS EXCLUSIVE MODE')
        task = asyncio.create_task(service.query(token.secret, 'SELECT count(*) AS total FROM messages'))
        try:
            waiting = False
            for _ in range(100):
                with psycopg.connect(dsn, autocommit=True) as admin:
                    waiting = bool(admin.execute("SELECT 1 FROM pg_stat_activity WHERE datname=current_database() AND application_name='gms-analytical-gateway' AND wait_event_type='Lock'").fetchone())
                if waiting:
                    break
                await asyncio.sleep(.02)
            assert waiting
            if cancel_caller:
                task.cancel()
            else:
                registry.cancel(run.run_id)
            with pytest.raises(asyncio.CancelledError if cancel_caller else AccessDenied):
                await asyncio.wait_for(task, timeout=2)
            with psycopg.connect(dsn, autocommit=True) as admin:
                assert not admin.execute("SELECT 1 FROM pg_stat_activity WHERE datname=current_database() AND application_name='gms-analytical-gateway'").fetchone()
            assert not api.active_queries
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_revalidates_before_returning_completed_query(tmp_path):
    class CompletingGateway:
        async def query(self, owner, text):
            registry.cancel(run.run_id)
            return QueryResult(('value',), ((1,),), True)

    service, registry, _, run, token = scoped(tmp_path, CompletingGateway())
    with pytest.raises(AccessDenied):
        await service.query(token.secret, 'SELECT 1 AS value')


@pytest.mark.asyncio
async def test_final_watcher_cleanup_precedes_fresh_sql_authorization(tmp_path,monkeypatch):
    started,closing,release=(asyncio.Event() for _ in range(3))
    class ReadyGateway:
        async def query(self,owner,query):
            await started.wait()
            return QueryResult(('value',),(('private',),),True)
    service,_,caps,_,token=scoped(tmp_path,ReadyGateway())
    async def watcher(value):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            closing.set()
            await release.wait()
    monkeypatch.setattr(service,'_watch',watcher)
    task=asyncio.create_task(service.query(token.secret,'SELECT subject FROM messages'))
    try:
        await asyncio.wait_for(closing.wait(),1)
        caps.revoke(token.secret)
        release.set()
        with pytest.raises(AccessDenied):
            await task
    finally:
        release.set()
        await asyncio.gather(task,return_exceptions=True)


@pytest.mark.asyncio
async def test_repeated_sql_cancellation_retains_database_cleanup(tmp_path):
    from gmail_search.gateway.data_admission import DataAdmission
    admission=DataAdmission(global_concurrency=1,owner_concurrency=1)
    started,closing,release,closed=(asyncio.Event() for _ in range(4))
    class ClosingGateway:
        async def query(self,owner,query):
            capacity=admission.acquire(owner)
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                closing.set()
                await release.wait()
                closed.set()
                capacity.release()
    service,_,_,_,token=scoped(tmp_path,ClosingGateway())
    task=asyncio.create_task(service.query(token.secret,'SELECT subject FROM messages'))
    try:
        await asyncio.wait_for(started.wait(),1)
        task.cancel()
        await asyncio.wait_for(closing.wait(),1)
        task.cancel()
        await asyncio.sleep(.02)
        held=(not task.done(),closed.is_set(),dict(admission.active))
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert held==(True,False,{'alice':1})
        assert closed.is_set() and not admission.active
    finally:
        release.set()
        await asyncio.gather(task,return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('operation',['query','schema'])
async def test_cancelled_sql_authorization_drains_registry_thread(tmp_path,monkeypatch,operation):
    import threading
    started,release,finished=threading.Event(),threading.Event(),threading.Event()
    service,_,caps,_,token=scoped(tmp_path,None)
    authorize=caps.authorize
    def gated_authorization(*args,**kwargs):
        lease=authorize(*args,**kwargs)
        started.set()
        assert release.wait(3)
        finished.set()
        return lease
    monkeypatch.setattr(caps,'authorize',gated_authorization)
    work=service.schema(token.secret) if operation=='schema' else service.query(token.secret,'SELECT subject FROM messages')
    task=asyncio.create_task(work)
    try:
        assert await asyncio.to_thread(started.wait,1)
        task.cancel()
        await asyncio.sleep(.01)
        task.cancel()
        await asyncio.sleep(.01)
        held=not task.done() and not finished.is_set()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert held and finished.is_set()
    finally:
        release.set()
        await asyncio.gather(task,return_exceptions=True)
        await asyncio.to_thread(finished.wait,1)
