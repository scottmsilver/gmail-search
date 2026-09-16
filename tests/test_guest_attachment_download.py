"""Standalone persistent guest downloader; synthetic loopback responses only."""
import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys

import pytest
import pytest_asyncio

DIRECTORY=Path(__file__).parents[1]/'deploy/public/worker'


def modules():
    result=[]
    for name in ('guest_attachment_transport','guest_attachment_download'):
        path=DIRECTORY/(name+'.py')
        assert path.exists(), 'Standalone raw downloader is not implemented'
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module
        spec.loader.exec_module(module)
        result.append(module)
    return result


def frame(data=b'synthetic',aid=1,**extra):
    header=dict(version=1,operation='raw',attachment_id=aid,mime_type='application/octet-stream',
                size_bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    header.update(extra)
    encoded=json.dumps(header,separators=(',',':')).encode()
    return len(encoded).to_bytes(4,'big')+encoded+data


def response(body,*,headers=None,status=200):
    if headers is None:
        headers=[(b'Content-Type',b'application/octet-stream'),(b'Content-Length',str(len(body)).encode())]
    return b'HTTP/1.1 '+str(status).encode()+b' OK\r\n'+b''.join(k+b': '+v+b'\r\n' for k,v in headers)+b'\r\n'+body


class Server:
    def __init__(self):
        self.raw=response(frame())
        self.calls=[]
        self.tasks=set()
        self.received=asyncio.Event()
        self.disconnected=asyncio.Event()
        self.hold=False
        self.pause_after=None
        self.keep_open=lambda body:False
        self.active=0
        self.peak=0

    async def handle(self,reader,writer):
        task=asyncio.current_task(); self.tasks.add(task)
        self.active+=1; self.peak=max(self.active,self.peak)
        try:
            head=await reader.readuntil(b'\r\n\r\n')
            lines=head[:-4].split(b'\r\n')
            headers=dict(line.split(b': ',1) for line in lines[1:])
            body=await reader.readexactly(int(headers[b'Content-Length']))
            self.calls.append((lines[0],headers,json.loads(body)))
            self.received.set()
            raw=self.raw(body) if callable(self.raw) else self.raw
            if not self.hold:
                writer.write(raw if self.pause_after is None else raw[:self.pause_after])
                await writer.drain()
            if self.hold or self.pause_after is not None or self.keep_open(body):
                await reader.read(); self.disconnected.set()
        except (OSError,asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()
            try: await writer.wait_closed()
            except OSError: pass
            self.active-=1; self.tasks.discard(task)


@pytest_asyncio.fixture
async def setup(tmp_path,monkeypatch):
    transport,download=modules()
    server=Server()
    listener=await asyncio.start_server(server.handle,'127.0.0.1',0)
    port=listener.sockets[0].getsockname()[1]
    async def connect():
        return await asyncio.open_connection('127.0.0.1',port,limit=16384)
    monkeypatch.setattr(transport,'_open',connect)
    tmp_path.chmod(0o700)
    fd=os.open(tmp_path,os.O_RDONLY|os.O_DIRECTORY)
    slots=asyncio.Semaphore(2)
    client=download.GuestAttachmentDownloader(fd,'a'*64,slots=slots)
    try:
        yield transport,download,server,client,tmp_path,slots
    finally:
        try: await client.aclose()
        finally:
            os.close(fd)
            listener.close(); await listener.wait_closed()
            for task in list(server.tasks): task.cancel()
            await asyncio.gather(*server.tasks,return_exceptions=True)


def deadline(seconds=3):
    return asyncio.get_running_loop().time()+seconds


def files(setup):
    return [p for p in setup[4].rglob('*') if p.is_file()]


@pytest.mark.asyncio
@pytest.mark.parametrize('data',[b'',b'synthetic',b'z'*(10*1024*1024)])
async def test_verified_generated_private_file_only_returns_metadata(setup,data):
    _,_,server,client,root,slots=setup
    server.raw=response(frame(data))
    result=await client.download_many((1,),deadline=deadline())
    assert len(result)==1 and set(result[0])=={'relative_path','size_bytes','sha256'}
    item=result[0]; path=root/item['relative_path']
    assert path.read_bytes()==data and item['size_bytes']==len(data)
    assert item['sha256']==hashlib.sha256(data).hexdigest()
    assert path.name.startswith('attachment-') and path.suffix=='.bin'
    assert stat.S_IMODE(path.stat().st_mode)==0o600
    assert stat.S_IMODE(path.parent.stat().st_mode)==0o700
    assert len(files(setup))==1 and slots._value==2
    assert client.usage=={'files':1,'bytes':len(data),'quarantined':0}
    request,headers,body=server.calls[0]
    assert request==b'POST /v1/attachment/raw HTTP/1.1' and body=={'attachment_id':1}
    assert headers[b'Authorization']==b'Bearer '+b'a'*64
    assert headers[b'Host']==b'127.0.0.1:18080' and headers[b'Accept-Encoding']==b'identity'


@pytest.mark.asyncio
@pytest.mark.parametrize('raw',[response(frame(),status=302),response(frame(),headers=[]),
    response(frame(),headers=[(b'Content-Type',b'application/json'),(b'Content-Length',b'1')]),
    response(frame(),headers=[(b'Content-Type',b'application/octet-stream'),(b'Content-Length',b'1'),(b'content-length',b'1')]),
    response(frame(),headers=[(b'Content-Type',b'application/octet-stream'),(b'Content-Length',b'100'),(b'Content-Encoding',b'gzip')]),
    response(frame(),headers=[(b'Content-Type',b'application/octet-stream'),(b'Content-Length',b'100'),(b'Transfer-Encoding',b'chunked')]),
    response(frame(aid=2)),response(frame(version=True)),response(frame(operation='parse')),
    response(frame(mime_type='text/plain\r\nX: y')),response(frame(sha256='0'*64)),
    response(frame(size_bytes=11*1024*1024)),response(frame()+b'extra'),response(frame()[:-1]),
    response((4097).to_bytes(4,'big')+b'x'*4097),response((0).to_bytes(4,'big')),
    b'HTTP/1.1 200 OK\r\nX: '+b'a'*17000+b'\r\n\r\n'])
async def test_bad_http_or_frame_never_leaves_files(setup,raw):
    setup[2].raw=raw
    result=await setup[3].download_many((1,),deadline=deadline())
    assert set(result[0])=={'error'} and not files(setup)
    assert setup[3].usage=={'files':0,'bytes':0,'quarantined':0} and setup[5]._value==2


@pytest.mark.asyncio
async def test_duplicate_packet_keys_are_rejected(setup):
    header=b'{"version":1,"version":1}'
    setup[2].raw=response(len(header).to_bytes(4,'big')+header)
    result=await setup[3].download_many((1,),deadline=deadline())
    assert 'error' in result[0] and not files(setup)


@pytest.mark.asyncio
async def test_cancel_mid_payload_removes_partial_after_socket_close(setup):
    server=setup[2]; server.raw=response(frame(b'x'*100000)); server.pause_after=1000
    task=asyncio.create_task(setup[3].download_many((1,),deadline=deadline()))
    await server.received.wait()
    for _ in range(100):
        if files(setup): break
        await asyncio.sleep(.005)
    assert files(setup)
    task.cancel(); task.cancel()
    with pytest.raises(asyncio.CancelledError): await task
    await asyncio.wait_for(server.disconnected.wait(),1)
    assert not files(setup) and setup[3].usage['files']==0 and setup[5]._value==2


@pytest.mark.asyncio
async def test_deadline_while_waiting_for_exact_eof_removes_file(setup):
    server=setup[2]; server.pause_after=len(server.raw)
    with pytest.raises(TimeoutError):
        await setup[3].download_many((1,),deadline=deadline(.08))
    assert not files(setup) and setup[5]._value==2


@pytest.mark.asyncio
async def test_shared_socket_admission_includes_other_core_requests(setup):
    slots=setup[5]; await slots.acquire(); await slots.acquire()
    task=asyncio.create_task(setup[3].download_many((1,),deadline=deadline()))
    await asyncio.sleep(.03)
    assert not setup[2].calls
    slots.release(); await task; slots.release()
    assert slots._value==2


@pytest.mark.asyncio
async def test_batch_byte_budget_is_cumulative_after_bad_hash(setup,monkeypatch):
    total=0
    original=os.write
    def write(fd,data):
        nonlocal total
        count=original(fd,data); total+=count; return count
    monkeypatch.setattr(setup[1].os,'write',write)
    def raw(body):
        aid=json.loads(body)['attachment_id']
        return response(frame(b'x'*(10*1024*1024),aid,sha256='0'*64))
    setup[2].raw=raw
    result=await setup[3].download_many((1,2,3),deadline=deadline())
    assert all('error' in value for value in result) and not files(setup)
    assert setup[3].usage['bytes']==0
    assert total==20*1024*1024


@pytest.mark.asyncio
async def test_file_quota_includes_zero_byte_successes_and_has_no_delete_refund(setup):
    setup[2].raw=lambda body:response(frame(b'',json.loads(body)['attachment_id']))
    result=await setup[3].download_many(tuple(range(1,21)),deadline=deadline())
    assert all('relative_path' in item for item in result) and setup[3].usage['files']==20
    (setup[4]/result[0]['relative_path']).unlink()
    result=await setup[3].download_many((21,),deadline=deadline())
    assert 'error' in result[0] and len(setup[2].calls)==20


@pytest.mark.asyncio
async def test_enospc_and_partial_write_cleanup(setup,monkeypatch):
    import errno
    original=os.write
    calls=0
    def write(fd,data):
        nonlocal calls
        calls+=1
        if calls==1: return original(fd,data[:2])
        raise OSError(errno.ENOSPC,'synthetic disk full')
    monkeypatch.setattr(setup[1].os,'write',write)
    result=await setup[3].download_many((1,),deadline=deadline())
    assert 'error' in result[0] and calls==2 and not files(setup)
    assert setup[3].usage['files']==0


@pytest.mark.asyncio
async def test_unlink_failure_retains_quota_and_slot_until_close_retry(setup,monkeypatch):
    original=os.unlink
    setup[2].raw=response(frame(sha256='0'*64))
    def unlink(*args,**kwargs): raise OSError('synthetic cleanup failure')
    monkeypatch.setattr(setup[1].os,'unlink',unlink)
    result=await setup[3].download_many((1,),deadline=deadline())
    assert 'error' in result[0] and setup[3].usage['quarantined']==1
    assert setup[3].usage['files']==1 and setup[5]._value==1 and files(setup)
    monkeypatch.setattr(setup[1].os,'unlink',original)
    await setup[3].aclose()
    assert not files(setup) and setup[5]._value==2


@pytest.mark.asyncio
async def test_final_link_never_overwrites_existing_path(setup,monkeypatch):
    original=os.link
    sentinel=[]
    def link(src,dst,**kwargs):
        fd=os.open(dst,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600,dir_fd=kwargs['dst_dir_fd'])
        os.write(fd,b'keep'); os.close(fd); sentinel.append(dst)
        return original(src,dst,**kwargs)
    monkeypatch.setattr(setup[1].os,'link',link)
    result=await setup[3].download_many((1,),deadline=deadline())
    assert 'error' in result[0]
    assert len(files(setup))==1 and files(setup)[0].read_bytes()==b'keep'
    assert setup[3].usage['files']==0


@pytest.mark.asyncio
@pytest.mark.parametrize('ids',[(True,),(),tuple(range(1,22)),(0,),('1',),(1,1)])
async def test_only_bounded_ids_are_accepted_no_path_arguments(setup,ids):
    with pytest.raises(setup[1].DownloadError):
        await setup[3].download_many(ids,deadline=deadline())
    assert not setup[2].calls and not files(setup)


class Writer:
    def __init__(self):
        self.transport=self
        self.closing=asyncio.Event()
        self.gate=asyncio.Event()
        self.closed=False
        self.aborted=False
        self.failure=False
    def write(self,data): pass
    async def drain(self): pass
    def close(self): pass
    def abort(self): self.aborted=True
    async def wait_closed(self):
        self.closing.set()
        if self.failure: raise RuntimeError('synthetic close failure')
        await self.gate.wait(); self.closed=True


def reader():
    value=asyncio.StreamReader(limit=16384)
    value.feed_data(response(frame())); value.feed_eof()
    return value


@pytest.mark.asyncio
async def test_cancel_during_already_started_normal_eof_close_drains(setup,monkeypatch):
    writer=Writer()
    async def connect(): return reader(),writer
    monkeypatch.setattr(setup[0],'_open',connect)
    task=asyncio.create_task(setup[3].download_many((1,),deadline=deadline()))
    await writer.closing.wait()
    task.cancel(); task.cancel(); await asyncio.sleep(.03)
    assert not task.done() and setup[5]._value==1 and files(setup)
    assert writer.aborted and not writer.closed
    writer.gate.set()
    with pytest.raises(asyncio.CancelledError): await task
    assert not files(setup) and writer.closed and setup[5]._value==2


@pytest.mark.asyncio
async def test_cancelled_connection_returning_socket_is_retrieved_and_closed(setup,monkeypatch):
    writer=Writer(); started=asyncio.Event()
    async def connect():
        started.set()
        try: await asyncio.Event().wait()
        except asyncio.CancelledError: return reader(),writer
    monkeypatch.setattr(setup[0],'_open',connect)
    task=asyncio.create_task(setup[3].download_many((1,),deadline=deadline()))
    await started.wait(); task.cancel()
    await writer.closing.wait(); task.cancel(); await asyncio.sleep(.02)
    assert not task.done() and setup[5]._value==1
    writer.gate.set()
    with pytest.raises(asyncio.CancelledError): await task
    assert writer.closed and setup[5]._value==2 and not files(setup)


@pytest.mark.asyncio
async def test_unacknowledged_socket_close_retains_slot_and_quota(setup,monkeypatch):
    writer=Writer(); writer.failure=True
    async def connect(): return reader(),writer
    monkeypatch.setattr(setup[0],'_open',connect)
    result=await setup[3].download_many((1,),deadline=deadline())
    assert 'error' in result[0] and setup[5]._value==1 and not writer.closed
    assert setup[3].usage['quarantined']==1 and setup[3].usage['files']==1
    writer.failure=False; writer.gate.set(); await setup[3].aclose()
    assert writer.closed and setup[5]._value==2 and setup[3].usage['files']==0


@pytest.mark.asyncio
async def test_cancel_at_final_link_before_return_removes_unreturned_file(setup,monkeypatch):
    original=setup[3]._publish
    def publish(record):
        value=original(record)
        asyncio.current_task().cancel()
        return value
    monkeypatch.setattr(setup[3],'_publish',publish)
    task=asyncio.create_task(setup[3].download_many((1,),deadline=deadline()))
    with pytest.raises(asyncio.CancelledError): await task
    assert not files(setup) and setup[3].usage['files']==0


@pytest.mark.asyncio
async def test_deadline_during_final_publication_rolls_back(setup,monkeypatch):
    import time
    original=setup[3]._publish
    def publish(record):
        result=original(record); time.sleep(.07); return result
    monkeypatch.setattr(setup[3],'_publish',publish)
    with pytest.raises(TimeoutError):
        await setup[3].download_many((1,),deadline=deadline(.05))
    assert not files(setup) and setup[3].usage['files']==0


@pytest.mark.asyncio
async def test_instance_disk_budget_counts_completed_files_across_batches(setup):
    setup[2].raw=lambda body:response(frame(b'x'*(10*1024*1024),json.loads(body)['attachment_id']))
    for ids in ((1,2),(3,4),(5,6)):
        result=await setup[3].download_many(ids,deadline=deadline())
        assert all('relative_path' in item for item in result)
    result=await setup[3].download_many((7,),deadline=deadline())
    assert 'error' in result[0] and len(files(setup))==6
    assert setup[3].usage=={'files':6,'bytes':60*1024*1024,'quarantined':0}


@pytest.mark.asyncio
async def test_batch_cancel_removes_already_completed_temporary_items(setup):
    def raw(body):
        if json.loads(body)['attachment_id']==2:
            return b'HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: 20\r\n\r\n'
        return response(frame())
    setup[2].raw=raw
    setup[2].keep_open=lambda body:json.loads(body)['attachment_id']==2
    task=asyncio.create_task(setup[3].download_many((1,2),deadline=deadline()))
    for _ in range(100):
        if files(setup) and setup[5]._value==1: break
        await asyncio.sleep(.005)
    task.cancel()
    with pytest.raises(asyncio.CancelledError): await task
    assert not files(setup) and setup[3].usage['files']==0


@pytest.mark.asyncio
async def test_publication_uses_original_held_fd_and_holds_socket_slot(setup,monkeypatch):
    observed=[]
    original=setup[3]._publish
    def publish(record):
        observed.append((record.fd,setup[5]._value))
        value=original(record)
        if record.fd is not None:
            info=os.fstat(record.fd)
            assert info.st_nlink==1 and info.st_size==len(b'synthetic')
            assert info.st_ino==(setup[4]/value['relative_path']).stat().st_ino
        return value
    monkeypatch.setattr(setup[3],'_publish',publish)
    result=await setup[3].download_many((1,),deadline=deadline())
    assert 'relative_path' in result[0] and observed[0][0] is not None and observed[0][1]==1


@pytest.mark.asyncio
async def test_file_close_failure_quarantines_fd_disk_and_slot_until_retry(setup,monkeypatch):
    original_publish=setup[3]._publish; original_close=os.close; target=[]
    def publish(record):
        result=original_publish(record); target.append(record.fd); return result
    def close(fd):
        if target and fd==target[0]: raise OSError('synthetic close unavailable')
        return original_close(fd)
    monkeypatch.setattr(setup[3],'_publish',publish)
    monkeypatch.setattr(setup[1].os,'close',close)
    result=await setup[3].download_many((1,),deadline=deadline())
    assert 'error' in result[0] and setup[3].usage['quarantined']==1 and setup[5]._value==1
    assert target[0] is not None and stat.S_ISREG(os.fstat(target[0]).st_mode)
    monkeypatch.setattr(setup[1].os,'close',original_close)
    # An uncertain but valid fd is never blindly closed again. A controller
    # normally tears down the guest; this controlled test closes its known fd.
    with pytest.raises(setup[1].DownloadCleanupError): await setup[3].aclose()
    original_close(target[0])
    await setup[3].aclose()
    assert setup[5]._value==2 and setup[3].usage['files']==0 and not files(setup)


@pytest.mark.asyncio
@pytest.mark.parametrize('same_inode',[False,True])
async def test_lost_file_close_ack_never_closes_reused_unrelated_fd(setup,monkeypatch,same_inode):
    original_publish=setup[3]._publish; original_close=os.close; target=[]; replacement=[]
    outside=setup[4]/'unrelated'; outside.write_bytes(b'keep')
    def publish(record):
        nonlocal outside
        result=original_publish(record); target.append(record.fd)
        if same_inode: outside=setup[4]/result['relative_path']
        return result
    def close(fd):
        if target and fd==target[0] and not replacement:
            original_close(fd)
            new_fd=os.open(outside,os.O_RDONLY)
            if new_fd!=fd:
                os.dup2(new_fd,fd); original_close(new_fd)
            replacement.append(fd)
            raise OSError('synthetic lost close acknowledgment')
        return original_close(fd)
    monkeypatch.setattr(setup[3],'_publish',publish)
    monkeypatch.setattr(setup[1].os,'close',close)
    result=await setup[3].download_many((1,),deadline=deadline())
    try:
        assert 'error' in result[0] and replacement==target
        assert os.read(replacement[0],4)==(b'synt' if same_inode else b'keep')
        assert setup[3].usage['quarantined']==1 and setup[5]._value==1
    finally:
        if replacement: original_close(replacement[0])
    await setup[3].aclose()
    assert setup[3].usage['files']==0 and setup[5]._value==2


@pytest.mark.asyncio
async def test_trailing_bytes_after_declared_http_body_are_not_accepted(setup):
    setup[2].raw=response(frame())+b'extra'
    result=await setup[3].download_many((1,),deadline=deadline())
    assert 'error' in result[0] and not files(setup) and setup[5]._value==2


@pytest.mark.asyncio
async def test_replaced_private_directory_refuses_returned_paths(setup):
    private=next(setup[4].iterdir())
    private.rename(setup[4]/'moved')
    private.symlink_to(setup[4]/'moved',target_is_directory=True)
    result=await setup[3].download_many((1,),deadline=deadline())
    assert 'error' in result[0] and not setup[2].calls


@pytest.mark.asyncio
async def test_final_inode_substitution_is_refused_and_quarantined(setup,monkeypatch):
    original=os.link
    def link(src,dst,**kwargs):
        original(src,dst,**kwargs)
        os.unlink(dst,dir_fd=kwargs['dst_dir_fd'])
        fd=os.open(dst,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600,dir_fd=kwargs['dst_dir_fd'])
        os.write(fd,b'foreign'); os.close(fd)
    monkeypatch.setattr(setup[1].os,'link',link)
    result=await setup[3].download_many((1,),deadline=deadline())
    assert 'error' in result[0] and setup[3].usage['quarantined']==1
    foreign=files(setup)[0]
    assert foreign.read_bytes()==b'foreign'
    # The controlled same-UID mutation is removed by its test owner, not by
    # downloader cleanup pretending it still owns that inode.
    foreign.unlink()
    await setup[3].aclose()
    assert setup[3].usage['files']==0 and setup[5]._value==2


def test_trusted_workspace_and_required_shared_slots(tmp_path):
    _,download=modules()
    fd=os.open(tmp_path,os.O_RDONLY|os.O_DIRECTORY)
    try:
        tmp_path.chmod(0o755)
        with pytest.raises(download.DownloadError):
            download.GuestAttachmentDownloader(fd,'a'*64,slots=asyncio.Semaphore(2))
        tmp_path.chmod(0o700)
        with pytest.raises(TypeError): download.GuestAttachmentDownloader(fd,'a'*64)
        with pytest.raises(download.DownloadError):
            download.GuestAttachmentDownloader(fd,'a'*64,slots=asyncio.Semaphore(3))
    finally: os.close(fd)


@pytest.mark.asyncio
@pytest.mark.parametrize('field',['_directory_fd','_workspace_fd'])
@pytest.mark.parametrize('same_inode',[False,True])
async def test_directory_close_ack_loss_never_closes_reused_handle(setup,monkeypatch,field,same_inode):
    client=setup[3];target=getattr(client,field);original=os.close;replacement=[]
    private=setup[4]/client._directory_name
    other=setup[4]/'unrelated-directory';other.mkdir(mode=0o700)
    selected=(private if field=='_directory_fd' else setup[4]) if same_inode else other
    expected=selected.stat()
    def close(fd):
        if fd==target and not replacement:
            original(fd)
            opened=os.open(selected,os.O_RDONLY|os.O_DIRECTORY)
            if opened!=fd:os.dup2(opened,fd);original(opened)
            replacement.append(fd)
            raise OSError('synthetic lost directory close acknowledgment')
        return original(fd)
    monkeypatch.setattr(setup[1].os,'close',close)
    try:
        with pytest.raises((OSError,setup[1].DownloadCleanupError)):await client.aclose()
        with pytest.raises(setup[1].DownloadCleanupError):await client.aclose()
        observed=os.fstat(replacement[0])
        assert (observed.st_dev,observed.st_ino)==(expected.st_dev,expected.st_ino)
        assert getattr(client,field)==target
    finally:
        if replacement:
            try:original(replacement[0])
            except OSError:pass
    await client.aclose()
    assert client._directory_fd is None and client._workspace_fd is None


@pytest.mark.asyncio
async def test_concurrent_repeated_aclose_shares_one_owned_cleanup_task(setup,monkeypatch):
    client=setup[3];started=asyncio.Event();release=asyncio.Event();calls=0
    async def close_transport():
        nonlocal calls
        calls+=1;started.set();await release.wait()
    monkeypatch.setattr(client._transport,'aclose',close_transport)
    first=asyncio.create_task(client.aclose());await started.wait()
    second=asyncio.create_task(client.aclose());await asyncio.sleep(.02)
    try:
        assert calls==1
        first.cancel();second.cancel();first.cancel();second.cancel();await asyncio.sleep(.02)
        assert not first.done() and not second.done() and client._directory_fd is not None
    finally:
        release.set();await asyncio.gather(first,second,return_exceptions=True)
    assert client._directory_fd is None and client._workspace_fd is None
    await client.aclose();assert calls==1
