"""Fixed full-agent manager lifecycle, without VMs or provider calls."""
import hashlib
import importlib.util
import io
from pathlib import Path
import sys
import threading
import time
import uuid

import pytest

from gmail_search.gateway import full_agent_rpc as rpc

ROOT=Path(__file__).parents[1]/'deploy/public/worker'
sys.path.insert(0,str(ROOT))
sys.modules['full_agent_rpc']=rpc
spec=importlib.util.spec_from_file_location('full_agent_manager',ROOT/'full_agent_manager.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def packet():
    from guest_agent_bootstrap import encode_config
    return encode_config(dict(version=1,profile='mail-agent-pi-v1',prompt='Actual arbitrary question',tool_config=dict(version=3,tool_profile='mail-raw-mcp-v3',capabilities={k:'a'*64 for k in ('sql','retrieval','artifact','attachment')}),inference_capability='b'*64,events_capability='c'*64))


def request(op='launch',handle=None,context=None,payload=None,seq=1):
    context=context or dict(run_id=uuid.uuid4().hex,owner_id='alice',conversation_id='conversation',fence=1,workspace_version=0,deadline=time.time()+120)
    payload=packet() if payload is None and op=='launch' else payload or b''
    h=dict(version=1,op=op,request_id=uuid.uuid4().hex,handle=handle or uuid.uuid4().hex,context_sha256=rpc.context_digest(context),payload_size=len(payload))
    if op=='launch':h.update(context=context,input_sha256=hashlib.sha256(payload).hexdigest(),lease_expires=time.time()+25)
    if op=='renew':h.update(renew_seq=seq,lease_expires=time.time()+25)
    return h,payload


class Backend:
    def __init__(self):self.live=set();self.launches=0;self.stops=0;self.renewals=[];self.fail_stop=False
    def launch(self,h,lease,limits):self.live.add(h);self.launches+=1
    def renew(self,h,lease):self.renewals.append(lease)
    def stop(self,h):
        self.stops+=1
        if self.fail_stop:raise RuntimeError('no ACK')
        self.live.discard(h)
    def inventory(self):return set(self.live)


class Session:
    def __init__(self):self.entered=threading.Event();self.release=threading.Event();self.delivered=[];self.closed=False;self.fail_close=False
    def bootstrap(self,h,payload,cancel):
        self.entered.set()
        while not self.release.wait(.01):
            if cancel.is_set():return
        if not cancel.is_set():self.delivered.append(payload)
    def close(self):
        if self.fail_close:raise RuntimeError('relay pending')
        self.closed=True


def setup(tmp_path):
    b=Backend();s=Session();m=mod.Manager(tmp_path/'state',b,controller_uid=123,session_factory=lambda handle:s)
    return m,b,s


def call(m,h,p):return m.handle(h,p,peer_uid=123)[0]


def test_lost_launch_ack_replay_exact_binding(tmp_path):
    m,b,s=setup(tmp_path);h,p=request()
    try:
        assert call(m,h,p)['status']=='running';assert s.entered.wait(1)
        retry=dict(h,request_id=uuid.uuid4().hex)
        assert call(m,retry,p)['status']=='running';assert b.launches==1
        changed=packet().replace(b'Actual arbitrary question',b'Actual different question')
        rebound=dict(h,input_sha256=hashlib.sha256(changed).hexdigest(),payload_size=len(changed))
        assert call(m,rebound,changed)['code']=='binding_mismatch'
        assert p not in (tmp_path/'state/jobs.sqlite').read_bytes()
    finally:m.close()
    assert s.closed and not b.live


def test_stop_before_launch_and_cancel_before_bootstrap(tmp_path):
    m,b,s=setup(tmp_path);h,p=request();stop,_=request('stop',h['handle'],h['context'])
    try:
        assert call(m,stop,b'')['status']=='stopped'
        assert call(m,h,p)['code']=='stopped';assert b.launches==0
        h,p=request();assert call(m,h,p)['status']=='running';assert s.entered.wait(1)
        stop,_=request('stop',h['handle'],h['context'])
        assert call(m,stop,b'')['status']=='stopped';assert not s.delivered and s.closed and not b.live
    finally:m.close()


def test_restart_reaps_orphans_and_tombstones(tmp_path):
    m,b,s=setup(tmp_path);h,p=request();call(m,h,p);assert s.entered.wait(1);m.close()
    b.live.add(h['handle'])
    n=mod.Manager(tmp_path/'state',b,controller_uid=123,session_factory=lambda handle:Session())
    try:
        assert not b.live;assert call(n,h,p)['code']=='stopped'
    finally:n.close()


def test_renew_replay_and_wrong_binding(tmp_path):
    m,b,s=setup(tmp_path);h,p=request();call(m,h,p);assert s.entered.wait(1)
    try:
        renew,_=request('renew',h['handle'],h['context'])
        assert call(m,renew,b'')['status']=='running'
        assert call(m,renew,b'')['code']=='renewal_replay'
        assert call(m,dict(renew,context_sha256='0'*64,renew_seq=2),b'')['code']=='binding_mismatch'
        assert len(b.renewals)==1
    finally:m.close()


def test_stop_failure_retains_capacity_until_both_acks(tmp_path):
    m,b,s=setup(tmp_path);h,p=request();call(m,h,p);assert s.entered.wait(1)
    stop,_=request('stop',h['handle'],h['context']);b.fail_stop=True;s.fail_close=True
    assert call(m,stop,b'')['code']=='stop_pending'
    h2,p2=request();assert call(m,h2,p2)['code']=='busy'
    b.fail_stop=False;s.fail_close=False
    assert call(m,stop,b'')['status']=='stopped'
    assert s.closed and not b.live;m.close()


def test_protocol_closed_framing():
    h,p=request();wire=rpc.encode_frame(h,p)
    assert rpc.read_frame(io.BytesIO(wire).read)==(h,p)
    with pytest.raises(ValueError):rpc.encode_frame(dict(h,path='/tmp/no'),p)
    with pytest.raises(ValueError):rpc.read_frame(io.BytesIO((4097).to_bytes(4,'big')).read)
    raw=b'{"version":1,"version":1}'
    with pytest.raises(ValueError):rpc.read_frame(io.BytesIO(len(raw).to_bytes(4,'big')+raw).read)
    with pytest.raises(ValueError):rpc.encode_frame(dict(h,payload_size=32773),b'x'*32773)


def test_expired_lease_and_launch_replay_cannot_extend_it(tmp_path):
    m,b,s=setup(tmp_path);h,p=request();call(m,h,p);assert s.entered.wait(1)
    try:
        entry=m.jobs[h['handle']];expires=entry['expires']
        assert call(m,dict(h,lease_expires=h['lease_expires']+10),p)['status']=='running'
        assert entry['expires']==expires
        entry['expires']=time.time()-1
        renew,_=request('renew',h['handle'],h['context'])
        assert call(m,renew,b'')['code']=='expired'
        m.tick();assert entry['cancel'].is_set()
    finally:m.close()


def test_peer_uid_and_invalid_bootstrap_never_launch(tmp_path):
    m,b,s=setup(tmp_path);h,p=request()
    try:
        assert m.handle(h,p,peer_uid=124)[0]['code']=='denied'
        bad=b'\x00\x00\x00\x02{}';h['payload_size']=len(bad);h['input_sha256']=hashlib.sha256(bad).hexdigest()
        assert call(m,h,bad)['code']=='invalid_bootstrap';assert b.launches==0
    finally:m.close()


def test_cancel_while_backend_launch_pending_never_delivers_caps(tmp_path):
    m,b,s=setup(tmp_path);entered=threading.Event();release=threading.Event();original=b.launch
    def launch(*args):entered.set();release.wait(2);original(*args)
    b.launch=launch;h,p=request();call(m,h,p);assert entered.wait(1)
    stop,_=request('stop',h['handle'],h['context']);result=[]
    thread=threading.Thread(target=lambda:result.append(call(m,stop,b'')));thread.start()
    assert m.jobs[h['handle']]['cancel'].wait(1);assert thread.is_alive()
    release.set();thread.join(2)
    assert result[0]['status']=='stopped' and not s.delivered and not b.live;m.close()


def test_restart_stop_failure_prevents_manager_admission(tmp_path):
    m,b,s=setup(tmp_path);m.close();b.live={'1'*32};b.fail_stop=True
    with pytest.raises(RuntimeError):mod.Manager(tmp_path/'state',b,controller_uid=123,session_factory=lambda handle:s)
    b.fail_stop=False
    m=mod.Manager(tmp_path/'state',b,controller_uid=123,session_factory=lambda handle:s);assert not b.live;m.close()


def test_administrative_orphan_stop_cannot_allow_delayed_launch(tmp_path):
    m,b,s=setup(tmp_path);h,p=request();b.live.add(h['handle'])
    stop,_=request('stop',h['handle'],h['context']);stop['context_sha256']='0'*64
    # An unknown handle can have physical inventory, so stop must reap it before ACK.
    try:
        assert call(m,stop,b'')['status']=='stopped'
        assert not b.live
        assert call(m,h,p)['status']=='error';assert b.launches==0
    finally:m.close()


def test_stopped_binding_reaps_reappearing_physical_handle(tmp_path):
    m,b,s=setup(tmp_path);h,p=request();stop,_=request('stop',h['handle'],h['context'])
    try:
        assert call(m,stop,b'')['status']=='stopped';b.live.add(h['handle'])
        assert call(m,stop,b'')['status']=='stopped';assert not b.live
    finally:m.close()


def test_renew_during_launch_is_applied_after_backend_ready(tmp_path):
    m,b,s=setup(tmp_path);entered=threading.Event();release=threading.Event();original=b.launch
    def launch(*args):entered.set();release.wait(2);original(*args)
    b.launch=launch;h,p=request();call(m,h,p);assert entered.wait(1)
    try:
        renew,_=request('renew',h['handle'],h['context'])
        assert call(m,renew,b'')['status']=='running';assert not b.renewals
        release.set();assert s.entered.wait(1)
        assert len(b.renewals)==1 and b.renewals[0].lease_expires==m.jobs[h['handle']]['expires']
    finally:release.set();m.close()


def test_stop_drops_completed_job_entry(tmp_path):
    m,b,s=setup(tmp_path);h,p=request();call(m,h,p);assert s.entered.wait(1)
    entry=m.jobs[h['handle']];entry['cancel'].set();entry['thread'].join(2)
    assert m._row(h['handle'])['state']=='stopped'
    stop,_=request('stop',h['handle'],h['context'])
    assert call(m,stop,b'')['status']=='stopped'
    assert h['handle'] not in m.jobs;m.close()


def test_conservative_host_deadline_allows_worker_clock_one_second_behind(tmp_path,monkeypatch):
    m,b,s=setup(tmp_path);now=time.time();h,p=request();h['context']['deadline']=now+175
    h['context_sha256']=rpc.context_digest(h['context']);h['lease_expires']=now+25
    try:
        with monkeypatch.context() as patch:
            patch.setattr(mod.time,'time',lambda:now-1)
            assert call(m,h,p)['status']=='running';assert s.entered.wait(1)
        assert m.jobs[h['handle']]['context']['deadline']-(now-1)==176
    finally:m.close()
