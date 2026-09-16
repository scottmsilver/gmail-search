"""Versioned guest config is trusted bootstrap input, never model arguments."""
import importlib
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).parents[1]/'deploy/public/worker'
V1={'sql':'1'*64,'retrieval':'2'*64,'artifact':'3'*64}
V2={'version':2,'tool_profile':'mail-read-v2','capabilities':{**V1,'attachment':'4'*64}}


def module():
    sys.path.insert(0,str(ROOT))
    try:
        return importlib.import_module('guest_tool_config')
    finally:
        sys.path.pop(0)


def test_exact_profiles_and_immutable_secret_snapshot():
    mod=module()
    old=mod.parse_tool_config(V1)
    new=mod.parse_tool_config(V2)
    assert old.version==1 and old.profile=='legacy-mail-v1' and len(old.tool_names)==4
    assert new.version==2 and new.profile=='mail-read-v2' and len(new.tool_names)==8
    assert new.tool_names[-1]=='get_attachment_batch'
    assert V1['sql'] not in repr(new)
    with pytest.raises(TypeError):new.capabilities['sql']='changed'
    assert mod.persisted_payload(old)==V1 and mod.persisted_payload(new)==V2


@pytest.mark.parametrize('value',[{**V1,'attachment':'4'*64},{'version':1,'capabilities':V1},
    {**V2,'version':True},{**V2,'version':3},{**V2,'tool_profile':'legacy-mail-v1'},
    {**V2,'capabilities':V1},{**V2,'url':'http://elsewhere'},
    {**V2,'capabilities':{**V2['capabilities'],'provider':'5'*64}}])
def test_versions_profiles_and_audiences_cannot_mix(value):
    with pytest.raises(ValueError):module().parse_tool_config(value)


def test_private_persistence_preserves_version_and_never_overwrites(tmp_path):
    mod=module()
    tmp_path.chmod(0o700)
    mod.write_capability_file(tmp_path,mod.parse_tool_config(V2))
    target=tmp_path/'capabilities.json'
    assert json.loads(target.read_bytes())==V2 and target.stat().st_mode&0o777==0o600
    with pytest.raises((ValueError,OSError)):mod.write_capability_file(tmp_path,mod.parse_tool_config(V1))
    assert json.loads(target.read_bytes())==V2


@pytest.mark.parametrize('unsafe',['symlink','public_parent'])
def test_private_persistence_refuses_unsafe_location(tmp_path,unsafe):
    mod=module()
    root=tmp_path/'run'
    root.mkdir(mode=0o700)
    if unsafe=='symlink':
        (root/'capabilities.json').symlink_to(tmp_path/'outside')
    else:root.chmod(0o755)
    with pytest.raises((ValueError,OSError)):mod.write_capability_file(root,mod.parse_tool_config(V2))
    assert not (tmp_path/'outside').exists()


def worker_module(name):
    sys.path.insert(0,str(ROOT))
    try:return importlib.import_module(name)
    finally:sys.path.pop(0)


def test_bootstrap_requires_explicit_v2_entrypoint():
    import io
    import struct
    mod=worker_module('guest_run_bootstrap')
    value={'version':2,'runtime':'pi','tool_profile':'mail-read-v2','capabilities':V2['capabilities']}
    raw=json.dumps(value).encode()
    frame=struct.pack('!I',len(raw))+raw
    with pytest.raises(ValueError):mod.read_config(io.BytesIO(frame).read)
    assert mod.read_config(io.BytesIO(frame).read,expected_profile='mail-read-v2')==value
    with pytest.raises(ValueError):
        mod.validate_config({'version':1,'runtime':'pi','capabilities':V1},expected_profile='mail-read-v2')


@pytest.mark.asyncio
async def test_core_profiles_and_attachment_mapping(tmp_path):
    mod=worker_module('guest_mail_tools')
    old=mod.GuestMailTools(tmp_path,V1)
    new=mod.GuestMailTools(tmp_path,V2)
    calls=[]
    async def request(method,path,audience,body,budget):
        calls.append((method,path,audience,json.loads(body)))
        return {'extracted_text':None,'complete':False}
    old._request=new._request=request
    assert 'error' in await old.dispatch('get_attachment_batch',{'items':[{'attachment_id':1}]})
    assert 'error' in await old.dispatch('get_thread_batch',{'thread_ids':['a'],'attachment_limit':1})
    result=await new.dispatch('get_attachment_batch',{'items':[{'attachment_id':1},{'attachment_id':2,'mode':'meta'}]})
    assert calls==[('POST','/v1/attachment/text','attachment',{'attachment_id':1,'offset':0,'limit':20000}),
                   ('POST','/v1/attachment/meta','attachment',{'attachment_id':2})]
    assert result['results'][0]['result']['extracted_text'] is None
    assert new.tool_profile=='mail-read-v2' and len(old.tool_names)==4


@pytest.mark.asyncio
async def test_attachment_invalid_items_never_open_socket(tmp_path):
    mod=worker_module('guest_mail_tools')
    core=mod.GuestMailTools(tmp_path,V2)
    async def request(*args,**kwargs):pytest.fail('invalid item opened a socket')
    core._request=request
    items=[{'attachment_id':True},{'attachment_id':0},{'attachment_id':2**63},
           {'attachment_id':1,'mode':'raw'},{'attachment_id':1,'path':'secret'},
           {'attachment_id':1,'mode':'meta','offset':0},
           {'attachment_id':1,'offset':2147483647},{'attachment_id':1,'limit':0}]
    result=await core.dispatch('get_attachment_batch',{'items':items})
    assert len(result['results'])==len(items)
    assert all('error' in row['result'] for row in result['results'])


@pytest.mark.asyncio
async def test_mcp_profile_is_loaded_once_for_listing_and_calls(tmp_path):
    mod=worker_module('guest_mail_mcp')
    core=worker_module('guest_mail_tools').GuestMailTools(tmp_path,V1)
    count=0
    def factory():
        nonlocal count
        count+=1
        return core
    messages=[]
    server=mod.GuestMailMCP(factory,messages.append)
    server._ready=True
    await server.receive({'jsonrpc':'2.0','id':1,'method':'tools/list'})
    assert len(messages[-1]['result']['tools'])==4 and count==1
    await server.receive({'jsonrpc':'2.0','id':2,'method':'tools/call','params':{'name':'get_attachment_batch','arguments':{'items':[{'attachment_id':1}]}}})
    await server.wait_idle()
    assert messages[-1]['result']['isError'] is True and count==1
    assert len(mod._tool_definitions('mail-read-v2'))==8
    legacy=next(t for t in messages[0]['result']['tools'] if t['name']=='get_thread_batch')
    assert 'attachment_limit' not in legacy['inputSchema']['properties']
