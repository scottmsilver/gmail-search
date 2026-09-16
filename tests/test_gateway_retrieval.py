"""Typed retrieval uses real owner readers and has no legacy API fallback."""
import asyncio

import httpx
import psycopg
import pytest

from gmail_search.gateway import http
from gmail_search.gateway.analytics import QueryRejected
from gmail_search.gateway.registry import AccessDenied
from gmail_search.gateway.database import QueryGateway, QueryLimits, QueryResult, ReaderCredential, ReaderRegistry
from test_gateway_service import scoped, gateway
from test_gateway_database_integration import database as database_fixture, reader_dsn

database = database_fixture


def retrieval(tmp_path, api, owner='alice'):
    from gmail_search.gateway.retrieval import RunRetrievalService
    sql, registry, capabilities, run, _ = scoped(tmp_path, api, owner)
    token = capabilities.issue(run.run_id, audience='retrieval', operations={'thread.get'})
    service = RunRetrievalService(capabilities, api)
    return service, registry, capabilities, run, token, sql


@pytest.mark.asyncio
async def test_thread_route_requires_explicit_retrieval_service(tmp_path):
    sql, *_ = scoped(tmp_path, None)
    # This optional seam must exist without changing the legacy app factory.
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=http.create_gateway_app(sql, retrieval=None)), base_url='http://gateway') as client:
        assert (await client.post('/v1/thread', json={'thread_id': 'shared'})).status_code == 404


@pytest.mark.asyncio
async def test_large_body_page_is_owner_scoped_and_bounded_before_gateway_result(database, tmp_path):
    dsn, (alice, bob) = database
    offset, limit = 2_000_000, 80
    bodies = {
        owner: 'x' * offset + f'|{owner}|'.ljust(200, owner[-1])
        for owner in (alice, bob)
    }
    with psycopg.connect(dsn, autocommit=True) as conn:
        for owner, body in bodies.items():
            conn.execute('INSERT INTO messages (id,thread_id,body_text,subject,date,user_id) VALUES (%s,%s,%s,%s,%s,%s)',
                         ('same-id', 'large-shared', body, owner + '-subject', '2026-09-15', owner))
    api = QueryGateway(
        ReaderRegistry({owner: ReaderCredential(owner, reader_dsn(dsn, owner)) for owner in (alice, bob)},
                       is_active=lambda _: True),
        limits=QueryLimits(max_bytes=512, deadline_seconds=4, lock_timeout_ms=3000),
    )
    service, _, _, _, token, _ = retrieval(tmp_path, api, alice)
    result = await service.thread(token.secret, thread_id='large-shared', body_offset=offset, body_limit=limit)
    message = result['messages'][0]
    assert message['body_text'] == bodies[alice][offset:offset + limit]
    assert message['body_total_chars'] == len(bodies[alice])
    assert message['body_next_offset'] == offset + limit
    assert message['body_pagination_limited'] is False
    assert bob not in str(result)


@pytest.mark.asyncio
async def test_colliding_thread_ids_use_token_owner_and_text_paging(database, tmp_path):
    dsn, (alice, bob) = database
    with psycopg.connect(dsn, autocommit=True) as conn:
        for owner in (alice, bob):
            conn.execute('INSERT INTO messages (id,thread_id,body_text,subject,date,user_id) VALUES (%s,%s,%s,%s,%s,%s)',
                         ('same-id', 'shared', owner + '-body', owner + '-subject', '2026-09-15', owner))
    service, _, _, _, token, sql = retrieval(tmp_path, gateway(database), alice)
    app = http.create_gateway_app(sql, retrieval=service)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://gateway') as client:
        headers = {'authorization': 'Bearer ' + token.secret}
        response = await client.post('/v1/thread', headers=headers, json={'thread_id': 'shared', 'body_offset': 1, 'body_limit': 4})
        assert response.status_code == 200, response.text
        result = response.json()
        assert result['messages'][0]['subject'] == alice + '-subject'
        assert result['messages'][0]['body_text'] == (alice + '-body')[1:5]
        assert result['messages'][0]['body_text_truncated'] is True
        assert result['messages'][0]['body_next_offset'] == 5
        assert result['messages'][0]['body_format'] == 'text'
        assert result['body_format'] == 'text' and not result['complete']
        assert result['source_complete'] and result['thread_message_count'] == 1
        assert bob not in response.text
        assert response.headers['cache-control'] == 'private, no-store'


@pytest.mark.asyncio
async def test_thread_http_rejects_owner_injection_and_wrong_audience_before_body(tmp_path):
    service, _, capabilities, run, token, sql = retrieval(tmp_path, None)
    app = http.create_gateway_app(sql, retrieval=service)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://gateway') as client:
        headers = {'authorization': 'Bearer ' + token.secret}
        for value in ({'thread_id': 'x', 'owner_id': 'bob'}, {'thread_id': "x' OR true--"},
                      {'thread_id': 'x', 'body_limit': True}, {'thread_id': 'x', 'message_limit': 101}):
            assert (await client.post('/v1/thread', headers=headers, json=value)).status_code == 400
        wrong = capabilities.issue(run.run_id, audience='sql', operations={'thread.get'})
        assert (await client.post('/v1/thread', headers={'authorization': 'Bearer ' + wrong.secret}, content=b'x' * 40000)).status_code == 403
        assert (await client.post('/v1/thread', headers=headers, content=b'x' * 40000)).status_code == 413
        assert (await client.post('/v1/thread', headers={**headers, 'content-type': 'application/json'}, content='{"thread_id":"a","thread_id":"b"}')).status_code == 400


@pytest.mark.asyncio
async def test_body_page_uses_database_slice_and_total_length(tmp_path):
    captured = {}

    class BoundedGateway:
        async def query(self, owner, query):
            captured['owner'] = owner
            captured['query'] = query
            return QueryResult(('id', 'body_text', 'body_total_chars'),
                               (('one', 'server-page', 100),), True)

    service, _, _, _, token, _ = retrieval(tmp_path, BoundedGateway())
    result = await service.thread(token.secret, thread_id='thread', body_offset=5, body_limit=10)
    message = result['messages'][0]
    assert captured['owner'] == 'alice'
    assert "pg_catalog.substr(body_text, 6, 10) AS body_text" in captured['query']
    assert 'length(body_text) AS body_total_chars' in captured['query']
    assert message['body_text'] == 'server-page'
    assert message['body_total_chars'] == 100
    assert message['body_next_offset'] == 15
    await service.thread(token.secret, thread_id='thread', body_offset=2_000_001, body_limit=10)
    assert "pg_catalog.substr(body_text, 2000002, 10) AS body_text" in captured['query']


@pytest.mark.asyncio
@pytest.mark.parametrize('cancel', [True, False])
async def test_inflight_retrieval_cancellation_is_awaited(tmp_path, cancel):
    started, stopped = asyncio.Event(), asyncio.Event()
    class BlockingGateway:
        async def query(self, owner, query):
            assert owner == 'alice'
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(.01)
                stopped.set()
    service, registry, _, run, token, _ = retrieval(tmp_path, BlockingGateway())
    task = asyncio.create_task(service.thread(token.secret, thread_id='shared'))
    await asyncio.wait_for(started.wait(), 1)
    if cancel:
        task.cancel()
    else:
        registry.cancel(run.run_id)
    with pytest.raises(asyncio.CancelledError if cancel else AccessDenied):
        await asyncio.wait_for(task, 2)
    assert stopped.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize('source_complete,offset,expected_count', [(True, 0, 1), (False, 0, None), (True, 1, None)])
async def test_completeness_does_not_invent_counts(tmp_path, source_complete, offset, expected_count):
    class BoundedGateway:
        async def query(self, owner, query):
            return QueryResult(('id', 'body_text', 'body_total_chars'), (('one', 'body', 4),), source_complete)
    service, _, _, _, token, _ = retrieval(tmp_path, BoundedGateway())
    result = await service.thread(token.secret, thread_id='thread', message_offset=offset)
    assert result['thread_message_count'] == expected_count
    assert result['complete'] is (expected_count is not None)


@pytest.mark.asyncio
async def test_message_limit_reports_continuation(tmp_path):
    class BoundedGateway:
        async def query(self, owner, query):
            return QueryResult(('id', 'body_text', 'body_total_chars'), (('one', 'body', 4), ('two', 'body', 4)), True)
    service, _, _, _, token, _ = retrieval(tmp_path, BoundedGateway())
    result = await service.thread(token.secret, thread_id='thread', message_limit=1)
    assert [m['id'] for m in result['messages']] == ['one']
    assert result['next_message_offset'] == 1 and not result['complete']
    assert result['thread_message_count'] is None


@pytest.mark.asyncio
async def test_offset_ceiling_never_returns_unusable_continuation(tmp_path):
    class BoundedGateway:
        async def query(self, owner, query):
            return QueryResult(('id', 'body_text', 'body_total_chars'), (('one', 'body', 4), ('two', 'body', 4)), True)
    service, _, _, _, token, _ = retrieval(tmp_path, BoundedGateway())
    result = await service.thread(token.secret, thread_id='thread', message_offset=10000, message_limit=1)
    assert result['next_message_offset'] is None
    assert result['pagination_limited'] is True


@pytest.mark.asyncio
async def test_byte_limited_page_offers_continuation_only_after_progress(tmp_path):
    class BoundedGateway:
        rows = (('one', 'body', 4),)
        async def query(self, owner, query):
            return QueryResult(('id', 'body_text', 'body_total_chars'), self.rows, False)
    api = BoundedGateway()
    service, _, _, _, token, _ = retrieval(tmp_path, api)
    result = await service.thread(token.secret, thread_id='thread')
    assert result['next_message_offset'] == 1 and not result['complete']
    api.rows = ()
    result = await service.thread(token.secret, thread_id='thread', message_offset=1)
    assert result['next_message_offset'] is None and not result['complete']


@pytest.mark.asyncio
async def test_body_page_above_two_megabytes_offers_a_usable_continuation(tmp_path):
    class BoundedGateway:
        async def query(self, owner, query):
            return QueryResult(('id', 'body_text', 'body_total_chars'), (('one', 'database-page', 2_200_000),), True)
    service, _, _, _, token, _ = retrieval(tmp_path, BoundedGateway())
    result = await service.thread(token.secret, thread_id='thread', body_offset=2_000_000, body_limit=100000)
    message = result['messages'][0]
    assert message['body_next_offset'] == 2_100_000
    assert message['body_pagination_limited'] is False
    assert not result['complete']


@pytest.mark.asyncio
async def test_body_offset_keeps_postgres_substring_start_within_int4(tmp_path):
    captured = {}

    class BoundedGateway:
        async def query(self, owner, query):
            captured['query'] = query
            return QueryResult(('id', 'body_text', 'body_total_chars'),
                               (('one', 'z', 2_147_483_647),), True)

    service, _, _, _, token, _ = retrieval(tmp_path, BoundedGateway())
    result = await service.thread(token.secret, thread_id='thread',
                                  body_offset=2_147_483_646, body_limit=1)
    assert result['messages'][0]['body_next_offset'] is None
    assert 'pg_catalog.substr(body_text, 2147483647, 1) AS body_text' in captured['query']
    with pytest.raises(QueryRejected):
        await service.thread(token.secret, thread_id='thread',
                             body_offset=2_147_483_647, body_limit=1)


@pytest.mark.asyncio
async def test_final_retrieval_authorization_prevents_completed_data_release(tmp_path):
    class RevokingGateway:
        async def query(self, owner, query):
            registry.cancel(run.run_id)
            return QueryResult(('id', 'body_text', 'body_total_chars'), (('one', 'private', 7),), True)
    service, registry, _, run, token, _ = retrieval(tmp_path, RevokingGateway())
    with pytest.raises(AccessDenied):
        await service.thread(token.secret, thread_id='thread')


@pytest.mark.asyncio
async def test_http_disconnect_cancels_underlying_retrieval(tmp_path):
    started, stopped = asyncio.Event(), asyncio.Event()
    class BlockingGateway:
        async def query(self, owner, query):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
    service, _, _, _, token, sql = retrieval(tmp_path, BlockingGateway())
    app = http.create_gateway_app(sql, retrieval=service)
    events = asyncio.Queue()
    events.put_nowait({'type': 'http.request', 'body': b'{"thread_id":"shared"}', 'more_body': False})
    sent = []
    async def send(event):
        sent.append(event)
    scope = {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.4'},
             'method': 'POST', 'path': '/v1/thread', 'raw_path': b'/v1/thread',
             'query_string': b'', 'root_path': '', 'scheme': 'http', 'http_version': '1.1',
             'server': ('gateway', 80), 'client': ('127.0.0.1', 1),
             'headers': [(b'authorization', ('Bearer ' + token.secret).encode()),
                         (b'content-type', b'application/json')]}
    task = asyncio.create_task(app(scope, events.get, send))
    try:
        await asyncio.wait_for(started.wait(), 1)
        events.put_nowait({'type': 'http.disconnect'})
        await asyncio.wait_for(stopped.wait(), 1)
        await asyncio.wait_for(task, 1)
        assert not any(b'messages' in event.get('body', b'') for event in sent)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_http_repeated_cancellation_drains_service_cleanup(tmp_path):
    from gmail_search.gateway.retrieval_http import run_while_connected
    started, cleanup_started, allow_cleanup, cleaned = (asyncio.Event() for _ in range(4))
    disconnect = asyncio.Event()
    class Request:
        async def receive(self):
            await disconnect.wait()
            return {'type': 'http.disconnect'}
    class BlockingGateway:
        async def query(self, owner, query):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleanup_started.set()
                await allow_cleanup.wait()
                cleaned.set()
    service, _, _, _, token, _ = retrieval(tmp_path, BlockingGateway())
    task = asyncio.create_task(run_while_connected(Request(), service.thread(token.secret, thread_id='thread')))
    try:
        await asyncio.wait_for(started.wait(), 1)
        disconnect.set()
        await asyncio.wait_for(cleanup_started.wait(), 1)
        task.cancel()
        await asyncio.sleep(.02)
        assert not task.done()
        allow_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
        assert cleaned.is_set()
    finally:
        allow_cleanup.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_final_watcher_cleanup_precedes_fresh_retrieval_authorization(tmp_path,monkeypatch):
    started,closing,release=(asyncio.Event() for _ in range(3))
    class ReadyGateway:
        async def query(self,owner,query):
            await started.wait()
            return QueryResult(('id','body_text','body_total_chars'),(('one','private',7),),True)
    service,_,caps,_,token,_=retrieval(tmp_path,ReadyGateway())
    async def watcher(value):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            closing.set()
            await release.wait()
    monkeypatch.setattr(service,'_watch',watcher)
    task=asyncio.create_task(service.thread(token.secret,thread_id='thread'))
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
async def test_direct_repeated_cancellation_retains_database_cleanup(tmp_path):
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
    service,_,_,_,token,_=retrieval(tmp_path,ClosingGateway())
    task=asyncio.create_task(service.thread(token.secret,thread_id='thread'))
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
async def test_cancelled_authorization_drains_its_registry_thread(tmp_path,monkeypatch):
    import threading
    started,release,finished=threading.Event(),threading.Event(),threading.Event()
    service,_,caps,_,token,_=retrieval(tmp_path,None)
    authorize=caps.authorize
    def gated_authorization(*args,**kwargs):
        lease=authorize(*args,**kwargs)
        started.set()
        assert release.wait(3)
        finished.set()
        return lease
    monkeypatch.setattr(caps,'authorize',gated_authorization)
    task=asyncio.create_task(service.thread(token.secret,thread_id='thread'))
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


def manifest_reader(api,items,*,complete=True,next_id=None):
    from gmail_search.gateway.attachment_reader import AttachmentMetadataPage
    class Reader:
        gateway=api
        calls=[]
        async def list_for_thread(self,owner,thread,*,after_attachment_id,limit,deadline,check_active):
            await check_active()
            self.calls.append((owner,thread,after_attachment_id,limit,deadline))
            return AttachmentMetadataPage(tuple(items),after_attachment_id,limit,complete,
                complete and after_attachment_id==0,next_id,False)
    return Reader()


def manifest(aid,message='one',owner='alice'):
    from gmail_search.gateway.attachment_reader import AttachmentMetadata
    return AttachmentMetadata(owner,aid,message,'thread',None,'application/x-unknown',0,
                              'unfetched',None,'missing')


class ManifestGateway:
    def __init__(self,body='body'):
        self.body=body
        self.calls=0
    async def query(self,owner,query):
        self.calls+=1
        return QueryResult(('id','body_text','body_total_chars'),(('one',self.body,len(self.body)),),True)


@pytest.mark.asyncio
async def test_optional_manifest_keeps_offpage_messages_and_explicit_inventory(tmp_path):
    from gmail_search.gateway.retrieval import RunRetrievalService
    api=ManifestGateway()
    _,_,caps,_,token,_=retrieval(tmp_path,api)
    reader=manifest_reader(api,[manifest(2),manifest(3,'outside')],complete=False,next_id=3)
    service=RunRetrievalService(caps,api,attachment_reader=reader)
    result=await service.thread(token.secret,thread_id='thread',attachment_after_id=1,attachment_limit=2)
    inventory=result['attachment_inventory']
    assert inventory==dict(items=[dict(id=2,message_id='one',thread_id='thread',filename=None,
        mime_type='application/x-unknown',size_bytes=0,fetch_status='unfetched',text_chars=None,
        stored_text_state='missing',extraction_complete=None),dict(id=3,message_id='outside',thread_id='thread',filename=None,
        mime_type='application/x-unknown',size_bytes=0,fetch_status='unfetched',text_chars=None,
        stored_text_state='missing',extraction_complete=None)],after_attachment_id=1,limit=2,
        source_complete=False,complete=False,next_attachment_id=3,pagination_limited=False,
        same_snapshot_as_messages=False)
    assert result['messages'][0]['attachments']==inventory['items'][:1]
    assert result['messages'][0]['attachments_complete'] is False
    assert result['complete'] is True
    assert reader.calls[0][:4]==('alice','thread',1,2)
    assert 'owner_id' not in str(inventory) and 'raw_path' not in str(inventory)


@pytest.mark.asyncio
async def test_no_reader_preserves_shape_and_rejects_attachment_paging(tmp_path):
    api=ManifestGateway()
    service,_,_,_,token,_=retrieval(tmp_path,api)
    result=await service.thread(token.secret,thread_id='thread')
    assert 'attachment_inventory' not in result and 'attachments' not in result['messages'][0]
    for changes in ({'attachment_after_id':1},{'attachment_limit':1},{'attachment_after_id':True},
                    {'attachment_limit':101},{'attachment_after_id':2**63}):
        with pytest.raises(QueryRejected):
            await service.thread(token.secret,thread_id='thread',**changes)
    assert api.calls==1


def test_manifest_reader_requires_same_gateway(tmp_path):
    from gmail_search.gateway.retrieval import RunRetrievalService
    api=ManifestGateway()
    _,_,caps,_,_,_=retrieval(tmp_path,api)
    with pytest.raises(ValueError):
        RunRetrievalService(caps,api,attachment_reader=manifest_reader(ManifestGateway(),[]))


@pytest.mark.asyncio
async def test_manifest_foreign_binding_and_combined_response_limit_fail_closed(tmp_path):
    from gmail_search.gateway.retrieval import RunRetrievalService
    api=ManifestGateway()
    _,_,caps,_,token,_=retrieval(tmp_path,api)
    reader=manifest_reader(api,[manifest(1,owner='bob')])
    service=RunRetrievalService(caps,api,attachment_reader=reader)
    with pytest.raises(RuntimeError):
        await service.thread(token.secret,thread_id='thread')
    reader=manifest_reader(api,[manifest(1)])
    service=RunRetrievalService(caps,api,attachment_reader=reader)
    api.body='x'*(4*1024*1024)
    with pytest.raises(RuntimeError,match='Thread response exceeds its byte limit'):
        await service.thread(token.secret,thread_id='thread')


@pytest.mark.asyncio
async def test_manifest_listing_cleanup_survives_repeated_cancel(tmp_path):
    from gmail_search.gateway.retrieval import RunRetrievalService
    api=ManifestGateway()
    _,_,caps,_,token,_=retrieval(tmp_path,api)
    reader=manifest_reader(api,[])
    entered,closing,release,closed=(asyncio.Event() for _ in range(4))
    async def blocked(*args,**kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            closing.set()
            await release.wait()
            closed.set()
    reader.list_for_thread=blocked
    service=RunRetrievalService(caps,api,attachment_reader=reader)
    task=asyncio.create_task(service.thread(token.secret,thread_id='thread'))
    try:
        await asyncio.wait_for(entered.wait(),1)
        task.cancel()
        await asyncio.wait_for(closing.wait(),1)
        task.cancel()
        await asyncio.sleep(.01)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert closed.is_set()
    finally:
        release.set()
        await asyncio.gather(task,return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('owner_index',[0,1])
async def test_real_manifest_owner_scope_inventory_over_100_and_offpage_messages(database,tmp_path,owner_index):
    from gmail_search.gateway.attachment_reader import OwnerAttachmentReader
    from gmail_search.gateway.retrieval import RunRetrievalService
    dsn,owners=database
    with psycopg.connect(dsn,autocommit=True) as conn:
        for owner in owners:
            for mid,day in (('one','2026-01-01'),('outside','2026-01-02')):
                conn.execute('INSERT INTO messages(id,thread_id,body_text,date,user_id) VALUES(%s,%s,%s,%s,%s)',
                    (mid,'thread',owner+' body',day,owner))
            for aid in range(1,103):
                conn.execute('INSERT INTO attachments(id,message_id,filename,mime_type,size_bytes,fetch_status,user_id) VALUES(%s,%s,%s,%s,0,%s,%s)',
                    (aid,'one' if aid%2 else 'outside',owner+'.file','application/x-unknown','unfetched',owner))
    api=gateway(database)
    _,_,caps,_,token,_=retrieval(tmp_path,api,owners[owner_index])
    service=RunRetrievalService(caps,api,attachment_reader=OwnerAttachmentReader(api))
    result=await service.thread(token.secret,thread_id='thread',message_limit=1)
    inventory=result['attachment_inventory']
    assert len(inventory['items'])==100 and inventory['next_attachment_id']==100
    assert not inventory['complete'] and not inventory['source_complete']
    assert inventory['same_snapshot_as_messages'] is False
    assert len(result['messages'][0]['attachments'])==50
    assert result['messages'][0]['attachments_complete'] is False
    assert owners[1-owner_index] not in str(result)
    final=await service.thread(token.secret,thread_id='thread',message_limit=1,attachment_after_id=100)
    tail=final['attachment_inventory']
    assert [item['id'] for item in tail['items']]==[101,102]
    assert tail['source_complete'] is True and tail['complete'] is False
    assert tail['next_attachment_id'] is None
    assert final['messages'][0]['attachments'][0]['id']==101
    assert not api.active_queries


@pytest.mark.asyncio
async def test_complete_empty_inventory_is_explicit_without_changing_body_flags(tmp_path):
    from gmail_search.gateway.retrieval import RunRetrievalService
    api=ManifestGateway()
    _,_,caps,_,token,_=retrieval(tmp_path,api)
    service=RunRetrievalService(caps,api,attachment_reader=manifest_reader(api,[]))
    result=await service.thread(token.secret,thread_id='thread')
    assert result['messages'][0]['attachments']==[]
    assert result['messages'][0]['attachments_complete'] is True
    assert result['attachment_inventory']['complete'] is True
    assert result['complete'] is True


@pytest.mark.asyncio
async def test_revocation_while_listing_prevents_manifest_publication(tmp_path):
    from gmail_search.gateway.retrieval import RunRetrievalService
    api=ManifestGateway()
    _,_,caps,_,token,_=retrieval(tmp_path,api)
    reader=manifest_reader(api,[manifest(1)])
    original=reader.list_for_thread
    async def revoke(*args,**kwargs):
        result=await original(*args,**kwargs)
        caps.revoke(token.secret)
        return result
    reader.list_for_thread=revoke
    service=RunRetrievalService(caps,api,attachment_reader=reader)
    with pytest.raises(AccessDenied):
        await service.thread(token.secret,thread_id='thread')


@pytest.mark.asyncio
async def test_thread_deadline_includes_final_watcher_cleanup(tmp_path,monkeypatch):
    started,closing,release=(asyncio.Event() for _ in range(3))
    api=ManifestGateway()
    service,_,_,_,token,_=retrieval(tmp_path,api)
    original_query=api.query
    async def query(*args):
        await started.wait()
        return await original_query(*args)
    async def watcher(value):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            closing.set()
            await release.wait()
    monkeypatch.setattr(api,'query',query)
    monkeypatch.setattr(service,'_watch',watcher)
    loop=asyncio.get_running_loop()
    original_time=loop.time
    task=asyncio.create_task(service.thread(token.secret,thread_id='thread'))
    try:
        await asyncio.wait_for(closing.wait(),1)
        monkeypatch.setattr(loop,'time',lambda:original_time()+31)
        release.set()
        with pytest.raises(TimeoutError):
            await task
    finally:
        monkeypatch.setattr(loop,'time',original_time)
        release.set()
        await asyncio.gather(task,return_exceptions=True)
