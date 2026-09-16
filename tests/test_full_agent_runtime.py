"""Separate synthetic full-agent image and bounded fixed guest bridge."""
import importlib.util
from pathlib import Path
import subprocess

import pytest

ROOT=Path(__file__).parents[1]/'deploy/public/worker'


def load(name):
    path=ROOT/name
    assert path.exists(), 'Full-agent runtime component missing'
    spec=importlib.util.spec_from_file_location(name.replace('-','_'),path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_full_profile_is_fixed_readonly_and_still_synthetic():
    backend=load('firecracker_backend.py')
    config=backend.fixed_config({'memory_mib':1024,'vcpus':1},profile='agent_full')
    assert config['drives'][1]['path_on_host']=='/agent-full.squashfs'
    assert all(d['is_read_only'] for d in config['drives'])
    assert 'network-interfaces' not in config
    assert backend.AGENT_PI_MCP_PIN=='478461755f2f344572bb0784685205fb173795bb5e3acae6fe523fb73dc62bbc'
    with pytest.raises(RuntimeError,match='clean synthetic'):backend.SyntheticFullAgentBackend()


class Peer:
    def __init__(self,data=()):self.data=list(data);self.sent=b'';self.reads=[];self.connected=None
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def settimeout(self,seconds):assert 0<seconds<=5
    def connect(self,address):self.connected=address
    def recv(self,size):self.reads.append(size);return self.data.pop(0) if self.data else b''
    def sendall(self,data):self.sent+=data


@pytest.mark.parametrize('chunks,expected',[( [b'1234',b'56'],b'123456'),([b'1234',b'567'],b'1234')])
def test_bridge_exact_limit_never_forwards_overrun(monkeypatch,chunks,expected):
    mod=load('guest-agent-vsock-bridge.py')
    assert mod.LIMIT==12*1024**2
    monkeypatch.setattr(mod,'LIMIT',6)
    client=Peer();upstream=Peer(chunks)
    monkeypatch.setattr(mod.socket,'socket',lambda *args:upstream)
    monkeypatch.setattr(mod.select,'select',lambda *args:([upstream],[],[]))
    assert mod.SLOTS.acquire(blocking=False)
    mod.bridge(client)
    assert upstream.connected==(2,8000)
    assert client.sent==expected
    assert all(size<=65536 for size in upstream.reads)


def test_builder_rejects_nonroot_before_creating_image():
    script=ROOT/'prepare-full-agent-runtime.sh'
    assert script.exists()
    result=subprocess.run(['bash',str(script),'/untrusted/runtime','/untrusted/deps'],capture_output=True)
    assert result.returncode!=0 and b'Root-owned trusted' in result.stderr


def test_v3_schema_survives_javascript_numeric_roundtrip_without_relaxing_gateway():
    import json
    import sys
    sys.path.insert(0,str(ROOT))
    try:mod=load('guest_mail_mcp.py')
    finally:sys.path.pop(0)
    from gmail_search.gateway.cli_compat import compile_anthropic_cli_request
    definitions=mod._tool_definitions('mail-raw-mcp-v3')
    tools=[{'name':d['name'],'description':d['description'],'input_schema':d['inputSchema']} for d in definitions]
    body={'model':'claude-sonnet-4-6','max_tokens':4096,'messages':[{'role':'user','content':'synthetic'}],'tools':tools,'stream':True}
    javascript=json.loads(json.dumps(body),parse_int=lambda text:int(float(text)))
    assert compile_anthropic_cli_request(javascript,server_model='claude-sonnet-4-6',max_output_tokens=4096,client_profile='pi-0.84.4')
    older=mod._tool_definitions('mail-read-v2')
    assert next(t for t in older if t['name']=='get_thread_batch')['inputSchema']['properties']['attachment_after_id']['maximum']==9223372036854775807


def test_v3_exact_id_preflight_keeps_legacy_integer_contract():
    import sys
    sys.path.insert(0,str(ROOT))
    try:mod=load('guest_mail_mcp.py')
    finally:sys.path.pop(0)
    for name,args in [('get_attachment_batch',{'items':[{'attachment_id':9007199254740992,'mode':'meta'}]}),
                      ('get_thread_batch',{'thread_ids':['thread'],'attachment_after_id':9007199254740992})]:
        assert mod._valid_arguments(name,args)
        assert not mod._valid_arguments(name,args,allow_raw=True,exact_ids=True)


@pytest.mark.asyncio
async def test_v3_core_rejects_inexact_ids_before_network(tmp_path):
    import sys
    sys.path.insert(0,str(ROOT))
    try:
        from guest_mail_tools import GuestMailTools
    finally:sys.path.pop(0)
    config={'version':3,'tool_profile':'mail-raw-mcp-v3','capabilities':{k:'a'*64 for k in ('sql','retrieval','artifact','attachment')}}
    core=GuestMailTools(tmp_path,config)
    async def forbidden(*args,**kwargs):raise AssertionError('Invalid ID reached network')
    core._request=forbidden
    try:
        value=await core.dispatch('get_attachment_batch',{'items':[{'attachment_id':9007199254740992,'mode':'meta'}]})
        assert 'error' in value['results'][0]['result']
        value=await core.dispatch('get_thread_batch',{'thread_ids':['thread'],'attachment_after_id':9007199254740992})
        assert value == {'error':'Invalid thread paging options.'}
    finally:await core.aclose()


def test_full_extension_keeps_explicit_bounded_model_output_guard():
    source=(ROOT/'guest-agent-mail-mcp.ts').read_text()
    # The adapter's boolean default silently truncates model-facing output at
    # 50 KiB. This fixed profile must explicitly support the core's byte ceiling.
    assert 'outputGuard: true' not in source
    assert 'outputGuard: { maxBytes: 8 * 1024 * 1024, maxLines: 100000, detailsMaxBytes: 16 * 1024 }' in source
