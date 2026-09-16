"""Explicit v3 persistent MCP raw integration, without runtime image changes."""
import asyncio
import importlib
import io
import json
import os
import sys

import pytest

from test_guest_attachment_download import setup as setup
from test_guest_attachment_download import response, frame, Writer, reader, files
from test_guest_tool_config import ROOT,V1,V2

V3={'version':3,'tool_profile':'mail-raw-mcp-v3','capabilities':{**V1,'attachment':'a'*64}}


def module(name):
    sys.path.insert(0,str(ROOT))
    try:return importlib.import_module(name)
    finally:sys.path.pop(0)


def test_explicit_v3_config_and_bootstrap_preserve_older_profiles():
    config=module('guest_tool_config'); bootstrap=module('guest_run_bootstrap')
    parsed=config.parse_tool_config(V3)
    assert parsed.profile=='mail-raw-mcp-v3' and parsed.version==3
    assert parsed.tool_names==config.parse_tool_config(V2).tool_names and len(parsed.tool_names)==8
    assert config.persisted_payload(parsed)==V3
    value={**V3,'runtime':'claude'}; raw=json.dumps(value).encode(); packet=len(raw).to_bytes(4,'big')+raw
    assert bootstrap.read_config(io.BytesIO(packet).read,expected_profile='mail-raw-mcp-v3')==value
    for profile in ('legacy-mail-v1','mail-read-v2'):
        with pytest.raises(ValueError):bootstrap.read_config(io.BytesIO(packet).read,expected_profile=profile)
    for value in ({**V3,'version':2},{**V3,'tool_profile':'mail-read-v2'},{**V3,'capabilities':V1}):
        with pytest.raises(ValueError):config.parse_tool_config(value)


@pytest.mark.parametrize('options',[dict(port=18081),dict(timeout_seconds=31)])
def test_v3_refuses_split_gateway_or_long_timeout(tmp_path,options):
    with pytest.raises(ValueError):module('guest_mail_tools').GuestMailTools(tmp_path,V3,**options)


@pytest.mark.asyncio
async def test_mixed_batch_groups_raw_once_preserves_order_and_persistent_quota(setup,monkeypatch):
    core=module('guest_mail_tools').GuestMailTools(setup[4],V3)
    setup[2].raw=lambda body:response(frame(b'opaque',json.loads(body)['attachment_id']))
    calls=[]; groups=[]
    async def request(method,path,audience,body,budget):
        calls.append((path,json.loads(body)))
        await asyncio.sleep(.01)
        return {'attachment_id':json.loads(body)['attachment_id'],'kind':path.rsplit('/',1)[-1]}
    core._request=request
    original=setup[1].GuestAttachmentDownloader.download_many
    async def download(self,ids,**kwargs):
        assert self._slots is core._slots
        groups.append(ids)
        return await original(self,ids,**kwargs)
    monkeypatch.setattr(setup[1].GuestAttachmentDownloader,'download_many',download)
    items=[{'attachment_id':1,'mode':'meta'},{'attachment_id':2,'mode':'raw'},
           {'attachment_id':3},{'attachment_id':4,'mode':'raw'}]
    result=await core.dispatch('get_attachment_batch',{'items':items})
    assert [r['input'] for r in result['results']]==items and groups==[(2,4)]
    assert result['results'][0]['result']['kind']=='meta' and result['results'][2]['result']['kind']=='text'
    for index in (1,3):
        value=result['results'][index]['result']
        assert set(value)=={'relative_path','size_bytes','sha256'}
        assert (setup[4]/value['relative_path']).read_bytes()==b'opaque'
    downloader=core._raw_downloader
    assert downloader.usage['files']==2
    await core.dispatch('get_attachment_batch',{'items':[{'attachment_id':5,'mode':'raw'}]})
    assert core._raw_downloader is downloader and downloader.usage['files']==3
    await core.aclose(); assert core._slots._value==2


@pytest.mark.asyncio
async def test_raw_and_json_calls_share_two_slots_without_double_acquire(setup):
    core=module('guest_mail_tools').GuestMailTools(setup[4],V3)
    setup[2].hold=True
    raw=asyncio.create_task(core.dispatch('get_attachment_batch',{'items':[{'attachment_id':1,'mode':'raw'}]}))
    await asyncio.wait_for(setup[2].received.wait(),1)
    entered=asyncio.Event(); release=asyncio.Event()
    async def request(*args,**kwargs):
        assert core._slots._value==0
        entered.set(); await release.wait(); return {'ok':True}
    core._request=request
    sql=asyncio.create_task(core.dispatch('sql_query_batch',{'queries':['SELECT 1']}))
    await asyncio.wait_for(entered.wait(),1)
    assert core._slots._value==0
    release.set(); await sql
    raw.cancel(); await asyncio.gather(raw,return_exceptions=True)
    await core.aclose(); assert core._slots._value==2


@pytest.mark.asyncio
@pytest.mark.parametrize('profile',[V1,V2])
async def test_old_profiles_cannot_guess_raw_mode(setup,profile):
    core=module('guest_mail_tools').GuestMailTools(setup[4],profile)
    result=await core.dispatch('get_attachment_batch',{'items':[{'attachment_id':1,'mode':'raw'}]})
    assert 'error' in result or 'error' in result['results'][0]['result']
    assert not setup[2].calls
    await core.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('unsafe',['mode','symlink'])
async def test_raw_workspace_checked_before_network(setup,unsafe):
    workspace=setup[4]/'core'; workspace.mkdir(mode=0o700)
    if unsafe=='mode':workspace.chmod(0o755)
    else:
        target=setup[4]/'target';target.mkdir(mode=0o700);workspace.rmdir();workspace.symlink_to(target)
    core=module('guest_mail_tools').GuestMailTools(workspace,V3)
    result=await core.dispatch('get_attachment_batch',{'items':[{'attachment_id':1,'mode':'raw'}]})
    assert 'error' in result['results'][0]['result'] and not setup[2].calls
    await core.aclose()


@pytest.mark.asyncio
async def test_duplicate_raw_ids_rejected_before_any_batch_io(setup):
    core=module('guest_mail_tools').GuestMailTools(setup[4],V3)
    async def request(*args,**kwargs):pytest.fail('duplicate raw IDs must fail before I/O')
    core._request=request
    result=await core.dispatch('get_attachment_batch',{'items':[
        {'attachment_id':1,'mode':'raw'},{'attachment_id':2,'mode':'meta'},{'attachment_id':1,'mode':'raw'}]})
    assert 'error' in result and not setup[2].calls
    await core.aclose()


@pytest.mark.asyncio
async def test_raw_union_and_mcp_output_are_profile_specific(setup):
    mcp=module('guest_mail_mcp'); core=module('guest_mail_tools').GuestMailTools(setup[4],V3)
    messages=[]; server=mcp.GuestMailMCP(lambda:core,messages.append);server._ready=True
    await server.receive({'jsonrpc':'2.0','id':1,'method':'tools/list'})
    tools=messages[-1]['result']['tools']
    assert len(tools)==8 and [t['name'] for t in tools]==list(core.tool_names)
    attachment=next(t for t in tools if t['name']=='get_attachment_batch')
    variants=attachment['inputSchema']['properties']['items']['items']['oneOf']
    assert next(v for v in variants if v['properties']['mode'].get('const')=='raw')['additionalProperties'] is False
    assert 'raw' not in json.dumps(mcp._tool_definitions('mail-read-v2')).lower()
    await server.receive({'jsonrpc':'2.0','id':2,'method':'tools/call','params':{
        'name':'get_attachment_batch','arguments':{'items':[{'attachment_id':1,'mode':'raw'}]}}})
    await server.wait_idle()
    value=messages[-1]['result']['structuredContent']['results'][0]['result']
    assert set(value)=={'relative_path','size_bytes','sha256'} and 'synthetic' not in json.dumps(messages)
    await server.close();assert core._closed


@pytest.mark.asyncio
async def test_direct_mcp_close_drains_eof_cleanup_before_core_teardown(setup,monkeypatch):
    writer=Writer()
    async def connect():return reader(),writer
    monkeypatch.setattr(setup[0],'_open',connect)
    core=module('guest_mail_tools').GuestMailTools(setup[4],V3)
    mcp=module('guest_mail_mcp'); server=mcp.GuestMailMCP(lambda:core,lambda value:None);server._ready=True
    await server.receive({'jsonrpc':'2.0','id':1,'method':'tools/call','params':{
        'name':'get_attachment_batch','arguments':{'items':[{'attachment_id':1,'mode':'raw'}]}}})
    await writer.closing.wait()
    task=asyncio.create_task(server.close());await asyncio.sleep(.02);task.cancel();task.cancel()
    await asyncio.sleep(.02)
    assert not task.done() and core._slots._value==1 and core._raw_workspace_fd is not None
    writer.gate.set();await asyncio.gather(task,return_exceptions=True)
    assert writer.closed and core._closed and core._raw_workspace_fd is None and core._slots._value==2
    assert not files(setup)


@pytest.mark.asyncio
async def test_core_close_itself_is_owned_and_denies_new_calls(setup,monkeypatch):
    core=module('guest_mail_tools').GuestMailTools(setup[4],V3)
    writer=Writer()
    async def connect():return reader(),writer
    monkeypatch.setattr(setup[0],'_open',connect)
    call=asyncio.create_task(core.dispatch('get_attachment_batch',{'items':[{'attachment_id':1,'mode':'raw'}]}))
    await writer.closing.wait()
    close=asyncio.create_task(core.aclose());await asyncio.sleep(.02);close.cancel();close.cancel()
    assert 'error' in await core.dispatch('describe_schema',{})
    assert not close.done() and core._slots._value==1
    writer.gate.set();await asyncio.gather(call,close,return_exceptions=True)
    assert core._raw_workspace_fd is None and core._slots._value==2


@pytest.mark.asyncio
async def test_cli_rejects_v3_before_constructing_or_dispatching_core(monkeypatch):
    cli=module('guest_mail_tool_cli');config=module('guest_tool_config').parse_tool_config(V3)
    monkeypatch.setattr(cli,'_capabilities',lambda:config)
    def factory(*args,**kwargs):pytest.fail('v3 must not create one-shot raw core')
    monkeypatch.setattr(cli,'GuestMailTools',factory)
    raw=json.dumps({'name':'get_attachment_batch','arguments':{'items':[{'attachment_id':1,'mode':'raw'}]}}).encode()
    with pytest.raises(ValueError):await cli.invoke(io.BytesIO(raw))


@pytest.mark.asyncio
@pytest.mark.parametrize('extra',[{'offset':0},{'limit':1},{'path':'file'},{'filename':'a.bin'},
    {'url':'http://elsewhere'},{'owner_id':'other'},{'mime_type':'text/plain'}])
async def test_raw_item_cannot_add_paging_paths_or_routing(setup,extra):
    core=module('guest_mail_tools').GuestMailTools(setup[4],V3)
    result=await core.dispatch('get_attachment_batch',{'items':[{'attachment_id':1,'mode':'raw',**extra}]})
    assert 'error' in result['results'][0]['result'] and not setup[2].calls
    assert core._raw_downloader is None
    await core.aclose()


@pytest.mark.asyncio
async def test_close_failure_propagates_and_workspace_owner_is_retained_until_retry(setup,monkeypatch):
    writer=Writer();writer.failure=True
    async def connect():return reader(),writer
    monkeypatch.setattr(setup[0],'_open',connect)
    core=module('guest_mail_tools').GuestMailTools(setup[4],V3)
    server=module('guest_mail_mcp').GuestMailMCP(lambda:core,lambda message:None);server._ready=True
    await server.receive({'jsonrpc':'2.0','id':1,'method':'tools/call','params':{
        'name':'get_attachment_batch','arguments':{'items':[{'attachment_id':1,'mode':'raw'}]}}})
    await server.wait_idle()
    assert core._slots._value==1 and core._raw_downloader.usage['quarantined']==1
    with pytest.raises(setup[1].DownloadCleanupError):await server.close()
    assert core._raw_workspace_fd is not None and core._slots._value==1
    writer.failure=False;writer.gate.set()
    await server.close()
    assert core._raw_workspace_fd is None and core._slots._value==2


@pytest.mark.asyncio
async def test_mcp_v2_cannot_guess_raw_union_even_after_structural_preflight(setup):
    core=module('guest_mail_tools').GuestMailTools(setup[4],V2)
    messages=[];server=module('guest_mail_mcp').GuestMailMCP(lambda:core,messages.append);server._ready=True
    await server.receive({'jsonrpc':'2.0','id':1,'method':'tools/call','params':{
        'name':'get_attachment_batch','arguments':{'items':[{'attachment_id':1,'mode':'raw'}]}}})
    await server.wait_idle()
    assert messages[-1]['result']['isError'] is True and not setup[2].calls
    await server.close()


@pytest.mark.asyncio
async def test_mcp_direct_close_also_drains_core_cleanup_with_no_active_calls():
    started=asyncio.Event();release=asyncio.Event()
    class Core:
        tool_profile='mail-raw-mcp-v3'
        closed=False
        async def aclose(self):
            started.set();await release.wait();self.closed=True
    core=Core();server=module('guest_mail_mcp').GuestMailMCP(lambda:core,lambda value:None)
    server._get_tools()
    task=asyncio.create_task(server.close());await started.wait()
    task.cancel();task.cancel();await asyncio.sleep(.02)
    assert not task.done() and not core.closed
    release.set();await asyncio.gather(task,return_exceptions=True)
    assert core.closed


@pytest.mark.asyncio
async def test_missing_core_lifetime_api_is_not_silently_skipped():
    class InvalidCore:tool_profile='legacy-mail-v1'
    server=module('guest_mail_mcp').GuestMailMCP(InvalidCore,lambda value:None)
    server._get_tools()
    with pytest.raises(AttributeError):await server.close()


@pytest.mark.asyncio
async def test_mixed_batch_does_not_publish_raw_paths_after_json_cleanup_expires(setup):
    core=module('guest_mail_tools').GuestMailTools(setup[4],V3,timeout_seconds=.1)
    closing=asyncio.Event();release=asyncio.Event()
    async def request(*args,**kwargs):
        try:await asyncio.Event().wait()
        finally:
            closing.set();await release.wait()
    core._request=request
    call=asyncio.create_task(core.dispatch('get_attachment_batch',{'items':[
        {'attachment_id':1,'mode':'raw'},{'attachment_id':2,'mode':'meta'}]}))
    await asyncio.wait_for(closing.wait(),1)
    await asyncio.sleep(.02)
    assert core._raw_downloader.usage['files']==1
    release.set();result=await call
    assert 'error' in result and 'relative_path' not in json.dumps(result)
    # Fully verified files whose outer publication was lost stay charged.
    assert core._raw_downloader.usage['files']==1
    await core.aclose()


@pytest.mark.asyncio
async def test_core_workspace_close_uncertainty_never_recloses_replacement(setup,monkeypatch):
    core=module('guest_mail_tools').GuestMailTools(setup[4],V3)
    core._get_raw_downloader()
    target=core._raw_workspace_fd;replacement=[];original=os.close
    def close(fd):
        if fd==target and not replacement:
            original(fd)
            other=os.open(setup[4],os.O_RDONLY|os.O_DIRECTORY)
            if other!=fd:os.dup2(other,fd);original(other)
            replacement.append(fd)
            raise OSError('synthetic uncertain workspace close')
        return original(fd)
    monkeypatch.setattr(module('guest_mail_tools').os,'close',close)
    with pytest.raises(ValueError):await core.aclose()
    with pytest.raises(ValueError):await core.aclose()
    assert os.fstat(replacement[0]).st_ino==setup[4].stat().st_ino
    original(replacement[0]);await core.aclose()
    assert core._raw_workspace_fd is None
