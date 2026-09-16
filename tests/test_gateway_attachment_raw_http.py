import asyncio
import hashlib
import json
from dataclasses import replace

import pytest
from fastapi import FastAPI

from gmail_search.gateway.attachment_raw_http import RawAttachmentMiddleware, RawStreamAborted
from gmail_search.gateway.attachment_raw_service import RunRawAttachmentService
from gmail_search.gateway.attachment_source import RawAttachmentInput
from gmail_search.gateway.capabilities import Capabilities
from gmail_search.gateway.data_admission import DataAdmission
from gmail_search.gateway.registry import Registry


class Source:
    def __init__(self, data=b'synthetic'):
        self.result=RawAttachmentInput('alice',1,'application/octet-stream',data)
        self.calls=[]
        self.started=asyncio.Event()
        self.closing=asyncio.Event()
        self.closed=False
        self.wait=False
        self.close_gate=None

    async def load_raw(self,owner,aid):
        self.calls.append((owner,aid))
        self.started.set()
        try:
            if self.wait:
                await asyncio.Event().wait()
            return self.result
        finally:
            self.closing.set()
            if self.close_gate:
                await self.close_gate.wait()
            self.closed=True


@pytest.fixture
def setup(tmp_path):
    tmp_path.chmod(0o700)
    registry=Registry(tmp_path/'registry.sqlite',is_active=lambda owner: owner in ('alice','bob'))
    caps=Capabilities(registry)
    run=registry.start_run('alice','conversation',request_key='run')
    token=caps.issue(run.run_id,audience='attachment',operations=['raw']).secret
    source=Source()
    admission=DataAdmission(global_concurrency=2,owner_concurrency=1)
    service=RunRawAttachmentService(caps,source,admission=admission)
    return registry,caps,token,source,admission,service


class Connection:
    def __init__(self,token,body=b'{"attachment_id":1}',**scope):
        self.scope=dict(type='http',asgi={'version':'3.0','spec_version':'2.4'},http_version='1.1',
            method='POST',scheme='http',path='/v1/attachment/raw',raw_path=b'/v1/attachment/raw',
            root_path='',query_string=b'',server=('test',80),client=('test',1),
            headers=[(b'authorization',b'Bearer '+token.encode()),(b'content-type',b'application/json'),
                     (b'content-length',str(len(body)).encode())])
        self.scope.update(scope)
        self.queue=asyncio.Queue()
        self.queue.put_nowait(dict(type='http.request',body=body,more_body=False))
        self.sent=[]
        self.block_type=None
        self.sending=asyncio.Event()
        self.send_closing=asyncio.Event()
        self.send_gate=None
        self.send_closed=False

    async def receive(self):
        return await self.queue.get()

    async def send(self,message):
        self.sent.append(message.copy())
        if message['type']==self.block_type:
            self.sending.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.send_closing.set()
                if self.send_gate:
                    await self.send_gate.wait()
                self.send_closed=True

    def disconnect(self):
        self.queue.put_nowait({'type':'http.disconnect'})


async def fallback(scope,receive,send):
    await send({'type':'http.response.start','status':404,'headers':[]})
    await send({'type':'http.response.body','body':b''})


def middleware(setup):
    return RawAttachmentMiddleware(fallback,service=setup[-1])


def packet(connection):
    start=connection.sent[0]
    assert start['status']==200
    data=b''.join(m.get('body',b'') for m in connection.sent[1:])
    size=int.from_bytes(data[:4],'big')
    assert 0<size<=4096
    return start,json.loads(data[4:4+size]),data[4+size:]


@pytest.mark.asyncio
@pytest.mark.parametrize('data',[b'',b'synthetic',b'x'*(10*1024*1024)])
async def test_exact_frame_zero_and_max_payload(setup,data):
    source=setup[3]; source.result=replace(source.result,data=data)
    conn=Connection(setup[2]); await middleware(setup)(conn.scope,conn.receive,conn.send)
    start,header,body=packet(conn)
    assert header==dict(version=1,operation='raw',attachment_id=1,mime_type='application/octet-stream',
                        size_bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    assert body==data and not conn.sent[-1].get('more_body',False)
    assert int(dict(start['headers'])[b'content-length'])==len(b''.join(m.get('body',b'') for m in conn.sent))
    assert all(len(m.get('body',b''))<=65536 for m in conn.sent)
    assert setup[4].active=={} and source.closed


def test_uncalled_middleware_owns_no_payload_or_admission(setup):
    middleware(setup)
    assert not setup[3].calls and setup[4].active=={}


@pytest.mark.asyncio
@pytest.mark.parametrize('body',[b'{"attachment_id":1,"attachment_id":1}',b'{"attachment_id":true}',
    b'{"attachment_id":1,"owner_id":"bob"}',b'[]',b'{"attachment_id":0}',b'{"attachment_id":9223372036854775808}',
    b'{"attachment_id":1,"path":"/etc/passwd"}',b'['*3000+b']'*3000])
async def test_invalid_body_rejected_before_load(setup,body):
    conn=Connection(setup[2],body)
    await middleware(setup)(conn.scope,conn.receive,conn.send)
    assert conn.sent[0]['status'] in (400,413)
    assert not setup[3].calls and setup[4].active=={}


@pytest.mark.asyncio
@pytest.mark.parametrize('change',[{'query_string':b'raw_path=secret'}, {'raw_path':b'/v1/attachment/%72aw'},
    {'path':'/v1/attachment/raw/','raw_path':b'/v1/attachment/raw/'}, {'method':'GET'}])
async def test_path_query_method_aliases_refused(setup,change):
    conn=Connection(setup[2],**change)
    await middleware(setup)(conn.scope,conn.receive,conn.send)
    assert conn.sent[0]['status'] in (400,405)
    assert not setup[3].calls


@pytest.mark.asyncio
@pytest.mark.parametrize('header',[(b'range',b'bytes=0-1'),(b'x-user-id',b'bob'),(b'cookie',b'x=1'),
    (b'content-encoding',b'gzip'),(b'authorization',b'Bearer '+'b'.encode()*64),
    (b'content-length',b'1'),(b'raw_path',b'/etc/passwd')])
async def test_duplicate_or_routing_headers_refused(setup,header):
    conn=Connection(setup[2]); conn.scope['headers'].append(header)
    await middleware(setup)(conn.scope,conn.receive,conn.send)
    assert conn.sent[0]['status'] in (400,401)
    assert not setup[3].calls


@pytest.mark.asyncio
@pytest.mark.parametrize('change',[dict(owner_id='bob'),dict(attachment_id=2),dict(data=b'x'*(10*1024*1024+1)),
    dict(data=bytearray(b'x')),dict(mime_type='text/plain\r\nX: y')])
async def test_invalid_source_is_fixed_error_before_200(setup,change):
    setup[3].result=replace(setup[3].result,**change)
    conn=Connection(setup[2]); await middleware(setup)(conn.scope,conn.receive,conn.send)
    assert conn.sent[0]['status']==404 and setup[4].active=={}
    assert b'bob' not in b''.join(m.get('body',b'') for m in conn.sent)


@pytest.mark.asyncio
@pytest.mark.parametrize('kind',['http.response.start','http.response.body'])
@pytest.mark.parametrize('stop',['cancel','disconnect','revoke'])
async def test_gated_send_teardown_holds_admission_through_repeated_cancel(setup,kind,stop):
    conn=Connection(setup[2]); conn.block_type=kind; conn.send_gate=asyncio.Event()
    task=asyncio.create_task(middleware(setup)(conn.scope,conn.receive,conn.send))
    await asyncio.wait_for(conn.sending.wait(),2)
    if stop=='cancel': task.cancel()
    elif stop=='disconnect': conn.disconnect()
    else: setup[1].revoke(setup[2])
    await asyncio.wait_for(conn.send_closing.wait(),2)
    task.cancel(); task.cancel()
    await asyncio.sleep(.03)
    assert not task.done() and setup[4].active=={'alice':1} and not conn.send_closed
    conn.send_gate.set()
    await asyncio.gather(task,return_exceptions=True)
    assert setup[4].active=={} and conn.send_closed


@pytest.mark.asyncio
async def test_disconnect_during_loader_cleanup_holds_lease(setup):
    source=setup[3]; source.wait=True; source.close_gate=asyncio.Event()
    conn=Connection(setup[2])
    task=asyncio.create_task(middleware(setup)(conn.scope,conn.receive,conn.send))
    await source.started.wait(); conn.disconnect()
    await source.closing.wait(); task.cancel(); task.cancel(); await asyncio.sleep(.03)
    assert not task.done() and setup[4].active=={'alice':1} and not conn.sent
    source.close_gate.set(); await asyncio.gather(task,return_exceptions=True)
    assert source.closed and setup[4].active=={}


@pytest.mark.asyncio
async def test_outer_fastapi_composition_retains_lease_at_real_send(setup):
    app=FastAPI()
    @app.middleware('http')
    async def inner(request,call_next):
        raise AssertionError('Raw route must bypass buffering middleware')
    app.add_middleware(RawAttachmentMiddleware,service=setup[-1])
    conn=Connection(setup[2]); conn.block_type='http.response.body'; conn.send_gate=asyncio.Event()
    task=asyncio.create_task(app(conn.scope,conn.receive,conn.send))
    await asyncio.wait_for(conn.sending.wait(),2)
    assert setup[4].active=={'alice':1}
    task.cancel(); await conn.send_closing.wait(); await asyncio.sleep(.02)
    assert setup[4].active=={'alice':1}
    conn.send_gate.set(); await asyncio.gather(task,return_exceptions=True)
    assert setup[4].active=={}


@pytest.mark.asyncio
async def test_revoked_token_denied_without_consuming_body_or_loading(setup):
    setup[1].revoke(setup[2])
    conn=Connection(setup[2])
    await middleware(setup)(conn.scope,conn.receive,conn.send)
    assert conn.sent[0]['status']==403 and conn.queue.qsize()==1
    assert not setup[3].calls and setup[4].active=={}


@pytest.mark.asyncio
async def test_parse_capability_does_not_authorize_raw(setup):
    registry,caps,_,source,_,_=setup
    run=registry.start_run('alice','other',request_key='other')
    token=caps.issue(run.run_id,audience='attachment',operations=['parse']).secret
    conn=Connection(token); await middleware(setup)(conn.scope,conn.receive,conn.send)
    assert conn.sent[0]['status']==403 and not source.calls


@pytest.mark.asyncio
async def test_disconnect_already_queued_before_first_send_reaps_loader(setup):
    setup[3].wait=True
    conn=Connection(setup[2]); conn.disconnect()
    await middleware(setup)(conn.scope,conn.receive,conn.send)
    assert not conn.sent and setup[4].active=={}


@pytest.mark.asyncio
async def test_deadline_during_loader_returns_preheader_error_after_close(setup):
    source=setup[3]; source.wait=True; source.close_gate=asyncio.Event()
    service=RunRawAttachmentService(setup[1],source,admission=setup[4],timeout_seconds=.08)
    conn=Connection(setup[2]); task=asyncio.create_task(RawAttachmentMiddleware(fallback,service=service)(
        conn.scope,conn.receive,conn.send))
    await asyncio.wait_for(source.closing.wait(),2)
    assert setup[4].active=={'alice':1} and not conn.sent and not task.done()
    source.close_gate.set(); await task
    assert conn.sent[0]['status']==504 and setup[4].active=={} and source.closed


@pytest.mark.asyncio
async def test_deadline_during_body_send_aborts_after_cleanup_without_second_headers(setup):
    service=RunRawAttachmentService(setup[1],setup[3],admission=setup[4],timeout_seconds=.08)
    conn=Connection(setup[2]); conn.block_type='http.response.body'; conn.send_gate=asyncio.Event()
    task=asyncio.create_task(RawAttachmentMiddleware(fallback,service=service)(conn.scope,conn.receive,conn.send))
    await asyncio.wait_for(conn.send_closing.wait(),2)
    assert setup[4].active=={'alice':1} and not task.done()
    conn.send_gate.set()
    with pytest.raises(RawStreamAborted): await task
    assert sum(m['type']=='http.response.start' for m in conn.sent)==1 and setup[4].active=={}
    assert conn.sent[-1].get('more_body') is True


@pytest.mark.asyncio
@pytest.mark.parametrize('kind',['http.response.start','http.response.body'])
async def test_send_failure_aborts_without_retrying_response(setup,kind):
    conn=Connection(setup[2])
    async def send(message):
        await conn.send(message)
        if message['type']==kind: raise OSError('sensitive network detail')
    with pytest.raises(RawStreamAborted,match='^Attachment stream aborted.$'):
        await middleware(setup)(conn.scope,conn.receive,send)
    assert len([m for m in conn.sent if m['type']=='http.response.start'])==1
    assert setup[4].active=={}


@pytest.mark.asyncio
async def test_revoke_after_headers_sends_no_binary_body(setup):
    conn=Connection(setup[2])
    async def send(message):
        await conn.send(message)
        if message['type']=='http.response.start': setup[1].revoke(setup[2])
    with pytest.raises(RawStreamAborted): await middleware(setup)(conn.scope,conn.receive,send)
    assert len(conn.sent)==1 and setup[4].active=={}


@pytest.mark.asyncio
async def test_shared_owner_capacity_refuses_second_download_without_load(setup):
    first=Connection(setup[2]); first.block_type='http.response.body'
    task=asyncio.create_task(middleware(setup)(first.scope,first.receive,first.send))
    await first.sending.wait()
    second=Connection(setup[2]); await middleware(setup)(second.scope,second.receive,second.send)
    assert second.sent[0]['status']==429 and setup[3].calls==[('alice',1)]
    assert setup[4].active=={'alice':1}
    task.cancel(); await asyncio.gather(task,return_exceptions=True)
    assert setup[4].active=={}


@pytest.mark.asyncio
async def test_shared_global_capacity_across_service_instances(setup):
    admission=DataAdmission(global_concurrency=1,owner_concurrency=1)
    alice=RunRawAttachmentService(setup[1],setup[3],admission=admission)
    run=setup[0].start_run('bob','bob-conversation',request_key='bob')
    token=setup[1].issue(run.run_id,audience='attachment',operations=['raw']).secret
    bob_source=Source(); bob_source.result=replace(bob_source.result,owner_id='bob')
    bob=RunRawAttachmentService(setup[1],bob_source,admission=admission)
    first=Connection(setup[2]); first.block_type='http.response.body'
    task=asyncio.create_task(RawAttachmentMiddleware(fallback,service=alice)(first.scope,first.receive,first.send))
    await first.sending.wait()
    second=Connection(token); await RawAttachmentMiddleware(fallback,service=bob)(second.scope,second.receive,second.send)
    assert second.sent[0]['status']==429 and not bob_source.calls
    task.cancel(); await asyncio.gather(task,return_exceptions=True)
    second=Connection(token); await RawAttachmentMiddleware(fallback,service=bob)(second.scope,second.receive,second.send)
    assert second.sent[0]['status']==200 and bob_source.calls==[('bob',1)] and admission.active=={}


@pytest.mark.asyncio
async def test_unrelated_path_delegates_without_authorization(setup):
    conn=Connection('not-a-token',path='/other',raw_path=b'/other')
    await middleware(setup)(conn.scope,conn.receive,conn.send)
    assert conn.sent[0]['status']==404 and not setup[3].calls


@pytest.mark.asyncio
async def test_source_failure_drops_private_exception_frames_before_release(setup):
    import weakref
    class PrivateMarker: pass
    observed=[]
    async def bad_source(owner,aid):
        marker=PrivateMarker(); observed.append(weakref.ref(marker))
        raise OSError('sensitive storage failure')
    setup[3].load_raw=bad_source
    conn=Connection(setup[2])
    async def send(message):
        # Source exception contexts must not retain file data after admission.
        assert observed[0]() is None and setup[4].active=={}
        await conn.send(message)
    await middleware(setup)(conn.scope,conn.receive,send)
    assert conn.sent[0]['status']==404
