"""Real temporary Unix frontend/server composition, no SSH or VM required."""
import importlib.util
import io
import os
import threading

import pytest

from test_full_agent_manager import ROOT,Backend,Session,mod,request,rpc
from gmail_search.gateway.full_agent_remote import SSHTransport


def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/(name+'.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_frontend_forwards_one_frame_and_manager_reaps(tmp_path):
    b=Backend();session=Session();manager=mod.Manager(tmp_path/'state',b,controller_uid=os.getuid(),session_factory=lambda handle:session)
    server=load('full_agent_rpc_server').Server(manager,tmp_path/'manager.sock',controller_uid=os.getuid())
    stopped=threading.Event();thread=threading.Thread(target=server.serve,args=(stopped,));thread.start()
    frontend=load('full_agent_rpc_frontend');h,p=request()
    reader,writer=os.pipe();outreader,outwriter=os.pipe();errors=[]
    def run():
        try:frontend.relay(reader,outwriter,socket_path=str(tmp_path/'manager.sock'))
        except Exception as exc:errors.append(exc)
        finally:os.close(reader);os.close(outwriter)
    relay=threading.Thread(target=run);relay.start()
    try:
        wire=rpc.encode_frame(h,p);os.write(writer,wire);os.close(writer)
        data=b''
        while part:=os.read(outreader,4096):data+=part
        relay.join(2);assert not errors
        response,payload=rpc.read_frame(io.BytesIO(data).read,response=True)
        assert response['status']=='running' and not payload
    finally:
        os.close(outreader);stopped.set();thread.join(2);server.close();manager.close()
    assert not b.live


def test_frontend_rejects_trailing_bytes_without_manager(tmp_path):
    frontend=load('full_agent_rpc_frontend');h,p=request();reader,writer=os.pipe();outreader,outwriter=os.pipe()
    try:
        os.write(writer,rpc.encode_frame(h,p)+b'!');os.close(writer)
        with pytest.raises(ValueError):frontend.relay(reader,outwriter,socket_path=str(tmp_path/'absent'))
    finally:os.close(reader);os.close(outreader);os.close(outwriter)


def test_ssh_pins_fixed_account_command_and_disables_forwarding(tmp_path):
    key=tmp_path/'key';known=tmp_path/'known';key.write_text('synthetic');known.write_text('synthetic');key.chmod(0o600);known.chmod(0o600)
    transport=SSHTransport(host='127.0.0.1',port=22092,private_key=key,known_hosts=known)
    command=transport.command()
    assert command[-2:]==['gmail-full-agent-rpc@127.0.0.1','full-agent-rpc-v1']
    assert 'ClearAllForwardings=yes' in command and 'IdentityAgent=none' in command and 'StrictHostKeyChecking=yes' in command
