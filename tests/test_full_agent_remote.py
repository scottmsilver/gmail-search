"""Host WorkerBackend maps Registry authority to fixed manager requests."""
from dataclasses import replace

import pytest

from gmail_search.gateway.registry import Registry,AccessDenied
from gmail_search.gateway.worker import WorkerController
from gmail_search.gateway.full_agent_remote import SSHFullAgentBackend,FULL_LIMITS
from test_full_agent_manager import setup,packet


class Transport:
    def __init__(self,manager):self.manager=manager;self.calls=[];self.lose=False
    def exchange(self,h,p,cancelled):
        self.calls.append(h['op']);result=self.manager.handle(h,p,peer_uid=123)
        if self.lose and h['op']=='launch':self.lose=False;raise OSError('Lost launch ACK')
        return result


def fixture(tmp_path):
    m,b,s=setup(tmp_path);directory=tmp_path/'host';directory.mkdir(mode=0o700)
    registry=Registry(directory/'registry.sqlite',is_active=lambda owner:owner=='alice')
    budget=registry.create_budget('alice',1000000)
    lease=registry.start_run('alice','conversation',request_key='first',budget_id=budget,deadline_ttl=120)
    t=Transport(m);backend=SSHFullAgentBackend(registry,transport=t,envelope_for=lambda lease,prompt,runtime:packet())
    return registry,lease,backend,m,b,s,t


def test_controller_launch_heartbeat_and_stop(tmp_path):
    registry,lease,backend,m,b,s,t=fixture(tmp_path)
    try:
        backend.prepare_input(lease,'Actual arbitrary question')
        controller=WorkerController(registry,backend,limits=FULL_LIMITS,max_workers=1,max_owner_workers=1)
        handle=controller.start(lease.run_id);assert s.entered.wait(1)
        assert handle in backend.inventory();controller.heartbeat(lease.run_id)
        controller.cancel(lease.run_id);assert not backend.inventory();assert s.closed
        assert packet() not in registry.path.read_bytes()
    finally:backend.close();m.close()


def test_lost_launch_ack_compensates_with_stop(tmp_path):
    registry,lease,backend,m,b,s,t=fixture(tmp_path)
    try:
        backend.prepare_input(lease,'Actual arbitrary question');t.lose=True
        controller=WorkerController(registry,backend,limits=FULL_LIMITS)
        with pytest.raises(AccessDenied):controller.start(lease.run_id)
        assert 'stop' in t.calls and not backend.inventory() and not b.live
    finally:backend.close();m.close()


def test_input_binding_fresh_authorization_and_no_rebind(tmp_path):
    registry,lease,backend,m,b,s,t=fixture(tmp_path)
    try:
        with pytest.raises((AccessDenied,ValueError)):backend.prepare_input(replace(lease,owner_id='bob'),'Actual arbitrary question')
        backend.prepare_input(lease,'Actual arbitrary question')
        with pytest.raises((AccessDenied,ValueError)):backend.prepare_input(lease,'Another question')
        registry.cancel(lease.run_id)
        with pytest.raises(AccessDenied):backend.launch('1'*32,lease,FULL_LIMITS)
        assert not b.live
    finally:backend.close();m.close()


def test_restart_refuses_caps_replay_and_reconciles_remote(tmp_path):
    registry,lease,backend,m,b,s,t=fixture(tmp_path)
    backend.prepare_input(lease,'Actual arbitrary question');backend.launch('1'*32,lease,FULL_LIMITS)
    other=SSHFullAgentBackend(registry,transport=t,envelope_for=lambda lease,prompt,runtime:packet())
    try:
        assert '1'*32 in other.inventory()
        other.stop('1'*32);assert not other.inventory()
        with pytest.raises(AccessDenied):other.launch('1'*32,lease,FULL_LIMITS)
    finally:backend.close();other.close();m.close()
