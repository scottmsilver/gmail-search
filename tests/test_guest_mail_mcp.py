"""MCP stdio boundary tests for the guest-only mail tool adapter."""
import asyncio
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).parents[1] / "deploy/public/worker"


def load_module():
    sys.path.insert(0, str(ROOT))
    try:
        spec = importlib.util.spec_from_file_location("guest_mail_mcp_test", ROOT / "guest_mail_mcp.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


class FakeTools:
    tool_profile = "mail-read-v2"
    def __init__(self):
        self.calls = []

    async def dispatch(self, name, arguments):
        self.calls.append((name, arguments))
        return {"ok": {"name": name, "arguments": arguments}}

    async def aclose(self):
        self.closed=True


@pytest.mark.asyncio
async def test_lifecycle_lists_only_guest_tools_and_returns_structured_core_result():
    module = load_module()
    tools = FakeTools()
    emitted = []
    server = module.GuestMailMCP(lambda: tools, emitted.append)

    await server.receive({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-11-25", "capabilities": {},
                   "clientInfo": {"name": "test", "version": "1"}},
    })
    assert emitted.pop() == {
        "jsonrpc": "2.0", "id": 1,
        "result": {"protocolVersion": "2025-11-25", "capabilities": {"tools": {}},
                   "serverInfo": {"name": "guest-mail-tools", "version": "1.0.0"}},
    }

    await server.receive({"jsonrpc": "2.0", "method": "notifications/initialized"})
    await server.receive({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    listed = emitted.pop()["result"]["tools"]
    assert [tool["name"] for tool in listed] == [
        "describe_schema", "sql_query_batch", "get_thread_batch", "publish_artifact_batch", "search_emails_batch", "find_facts", "query_emails_batch", "get_attachment_batch",
    ]
    assert "raw" not in json.dumps(listed).lower()
    assert "markdown" not in json.dumps(listed).lower()
    assert listed[2]["inputSchema"]["properties"]["body_format"] == {"const": "text"}
    assert listed[3]["inputSchema"]["properties"]["items"]["items"]["properties"]["mime_type"] == {
        "const": "application/octet-stream"
    }

    await server.receive({
        "jsonrpc": "2.0", "id": "call-1", "method": "tools/call",
        "params": {"name": "get_thread_batch", "arguments": {"thread_ids": ["abc"], "body_limit": 12}},
    })
    await server.wait_idle()
    result = emitted.pop()
    assert tools.calls == [("get_thread_batch", {"thread_ids": ["abc"], "body_limit": 12})]
    assert result["id"] == "call-1"
    assert result["result"]["isError"] is False
    assert result["result"]["structuredContent"] == {"ok": {"name": "get_thread_batch", "arguments": {"thread_ids": ["abc"], "body_limit": 12}}}
    assert json.loads(result["result"]["content"][0]["text"]) == result["result"]["structuredContent"]


@pytest.mark.asyncio
async def test_cancelled_notification_cancels_and_drains_core_without_response():
    module = load_module()
    cancelled = asyncio.Event()
    started = asyncio.Event()

    class BlockingTools(FakeTools):
        tool_profile = "legacy-mail-v1"
        async def dispatch(self, _name, _arguments):
            try:
                started.set()
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await asyncio.sleep(0)
                cancelled.set()
                raise

    emitted = []
    server = module.GuestMailMCP(BlockingTools, emitted.append)
    await server.receive({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}})
    emitted.clear()
    await server.receive({"jsonrpc": "2.0", "method": "notifications/initialized"})
    await server.receive({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
        "name": "describe_schema", "arguments": {}}})
    await started.wait()
    await server.receive({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 3}})
    await server.wait_idle()
    assert cancelled.is_set()
    assert emitted == []


@pytest.mark.asyncio
async def test_connection_close_cancels_and_drains_every_active_core_call():
    module = load_module()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class BlockingTools(FakeTools):
        tool_profile = "legacy-mail-v1"
        async def dispatch(self, _name, _arguments):
            try:
                started.set()
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

    emitted = []
    server = module.GuestMailMCP(BlockingTools, emitted.append)
    await server.receive({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}})
    emitted.clear()
    await server.receive({"jsonrpc": "2.0", "method": "notifications/initialized"})
    await server.receive({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
        "name": "describe_schema", "arguments": {}}})
    await started.wait()
    await server.close()
    assert cancelled.is_set()
    assert server._tools_instance.closed
    assert emitted == []


@pytest.mark.asyncio
async def test_two_calls_are_admitted_and_the_third_has_a_consistent_error():
    module = load_module()
    release = asyncio.Event()

    class BlockingTools(FakeTools):
        tool_profile = "legacy-mail-v1"
        async def dispatch(self, _name, _arguments):
            await release.wait()
            return {"done": True}

    emitted = []
    server = module.GuestMailMCP(BlockingTools, emitted.append)
    await server.receive({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}})
    emitted.clear()
    await server.receive({"jsonrpc": "2.0", "method": "notifications/initialized"})
    for identifier in (2, 3, 4):
        await server.receive({"jsonrpc": "2.0", "id": identifier, "method": "tools/call", "params": {
            "name": "describe_schema", "arguments": {}}})
    assert emitted == [{"jsonrpc": "2.0", "id": 4, "error": {"code": -32000, "message": "Too many concurrent requests."}}]
    release.set()
    await server.wait_idle()
    assert {item["id"] for item in emitted[1:]} == {2, 3}


def test_strict_line_parser_bounds_before_json_and_rejects_duplicate_or_nonfinite_values():
    module = load_module()
    assert module.parse_message_line(b'{"jsonrpc":"2.0","id":1,"method":"ping"}') == {
        "jsonrpc": "2.0", "id": 1, "method": "ping"
    }
    for raw in (
        b'x' * (module.MAX_INPUT_LINE_BYTES + 1),
        b'{"jsonrpc":"2.0","id":1,"id":2,"method":"ping"}',
        b'{"jsonrpc":"2.0","id":1,"method":"ping","params":NaN}',
        b'{"jsonrpc":"2.0","id":1,"method":"ping","params":"\\ud800"}',
    ):
        with pytest.raises(module.ProtocolError):
            module.parse_message_line(raw)


@pytest.mark.asyncio
async def test_request_ids_are_bounded_before_they_enter_the_active_call_table():
    module = load_module()
    emitted = []
    server = module.GuestMailMCP(FakeTools, emitted.append)
    await server.receive({"jsonrpc": "2.0", "id": "x" * 257, "method": "ping"})
    assert emitted == [{"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid Request."}}]


@pytest.mark.asyncio
async def test_official_mcp_sdk_can_initialize_list_ping_and_call_the_stdio_server(tmp_path):
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    run=tmp_path/'run'
    run.mkdir(mode=0o700)
    config=run/'capabilities.json'
    config.write_text(json.dumps({'version':2,'tool_profile':'mail-read-v2',
                                  'capabilities':{'sql':'1'*64,'retrieval':'2'*64,'artifact':'3'*64,'attachment':'4'*64}}))
    config.chmod(0o600)
    launcher=tmp_path/'trusted_test_launcher.py'
    launcher.write_text(f"import sys, asyncio\nfrom pathlib import Path\nsys.path.insert(0,{str(ROOT)!r})\nimport guest_mail_mcp as m, guest_mail_tool_cli as c\nm.RUN_ROOT=c.RUN_ROOT=Path({str(run)!r})\nasyncio.run(m.main())\n")
    async with stdio_client(StdioServerParameters(command=sys.executable, args=["-I", str(launcher)])) as streams:
        async with ClientSession(*streams) as session:
            initialized = await session.initialize()
            assert initialized.serverInfo.name == "guest-mail-tools"
            assert [tool.name for tool in (await session.list_tools()).tools] == [
                "describe_schema", "sql_query_batch", "get_thread_batch", "publish_artifact_batch", "search_emails_batch", "find_facts", "query_emails_batch", "get_attachment_batch",
            ]
            await session.send_ping()
            result = await session.call_tool("describe_schema", {"unexpected": True})
            assert result.isError is True


@pytest.mark.asyncio
async def test_concurrent_mcp_batches_share_two_http_slots():
    module = load_module()
    release, entered = asyncio.Event(), asyncio.Event()
    active = peak = 0
    class Tools(FakeTools):
        tool_profile = "legacy-mail-v1"
        def __init__(self):
            self.slots = asyncio.Semaphore(2)
        async def dispatch(self, name, arguments):
            async def request():
                nonlocal active, peak
                async with self.slots:
                    active += 1
                    peak = max(peak, active)
                    entered.set()
                    try:
                        await release.wait()
                    finally:
                        active -= 1
            await asyncio.gather(*(request() for _ in range(4)))
            return {}
    server = module.GuestMailMCP(Tools, lambda value: None)
    server._ready = True
    try:
        for identifier in (1, 2):
            await server.receive({'jsonrpc':'2.0', 'id':identifier, 'method':'tools/call',
                                  'params':{'name':'describe_schema', 'arguments':{}}})
        await entered.wait()
        await asyncio.sleep(.02)
        assert peak == 2
    finally:
        release.set()
        await server.wait_idle()


@pytest.mark.asyncio
async def test_cancellation_during_partial_stdout_finishes_frame_before_next(monkeypatch):
    module = load_module()
    written = bytearray()
    blocked, release = asyncio.Event(), asyncio.Event()
    first = True
    def write(fd, data):
        nonlocal first
        if first:
            first = False
            written.extend(data[:3])
            return 3
        if not release.is_set():
            raise BlockingIOError()
        written.extend(data)
        return len(data)
    async def ready(*args, **kwargs):
        blocked.set()
        await release.wait()
    monkeypatch.setattr(module.os, 'set_blocking', lambda *args: None)
    monkeypatch.setattr(module.os, 'write', write)
    monkeypatch.setattr(module, '_ready', ready)
    output = module._Stdout()
    first_task = asyncio.create_task(output({'id':1, 'result':{}}))
    second_task = None
    try:
        await blocked.wait()
        first_task.cancel()
        await asyncio.sleep(0)
        first_task.cancel()
        second_task = asyncio.create_task(output({'id':2, 'result':{}}))
        await asyncio.sleep(.01)
        assert not first_task.done()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await first_task
        if second_task:
            await second_task
    assert [json.loads(line) for line in written.splitlines()] == [{'id':1, 'result':{}}, {'id':2, 'result':{}}]


@pytest.mark.asyncio
async def test_stalled_partial_output_poison_prevents_any_later_frame(monkeypatch):
    module = load_module()
    written, failures = bytearray(), []
    def write(fd, data):
        if not written:
            written.extend(data[:3])
            return 3
        raise BlockingIOError()
    async def ready(*args, **kwargs):
        await asyncio.Event().wait()
    monkeypatch.setattr(module.os, 'set_blocking', lambda *args: None)
    monkeypatch.setattr(module.os, 'write', write)
    monkeypatch.setattr(module, '_ready', ready)
    monkeypatch.setattr(module, 'OUTPUT_SECONDS', .01)
    output = module._Stdout(on_failure=lambda: failures.append(True))
    with pytest.raises(TimeoutError):
        await output({'id':1})
    with pytest.raises(ConnectionError):
        await output({'id':2})
    assert output.failed and failures == [True]
    assert bytes(written) == b'{"i'


@pytest.mark.asyncio
async def test_main_exits_cleanly_when_output_deadline_cancels_reader(monkeypatch):
    module = load_module()
    async def lines():
        yield json.dumps({'jsonrpc':'2.0','id':1,'method':'ping'}).encode()
        await asyncio.Event().wait()
    def write(*args):
        raise BlockingIOError()
    async def ready(*args, **kwargs):
        await asyncio.Event().wait()
    monkeypatch.setattr(module, '_input_lines', lines)
    monkeypatch.setattr(module.os, 'set_blocking', lambda *args: None)
    monkeypatch.setattr(module.os, 'write', write)
    monkeypatch.setattr(module, '_ready', ready)
    monkeypatch.setattr(module, 'OUTPUT_SECONDS', .01)
    monkeypatch.setattr(asyncio.get_running_loop(), 'add_signal_handler', lambda *args: None)
    assert await module.main() == 1


@pytest.mark.asyncio
async def test_main_repeated_shutdown_cancellation_drains_without_traceback(monkeypatch):
    module = load_module()
    entered, release = asyncio.Event(), asyncio.Event()
    original = module.GuestMailMCP
    class Server(original):
        async def close(self):
            entered.set()
            await release.wait()
    async def lines():
        if False:
            yield b''
    monkeypatch.setattr(module, 'GuestMailMCP', Server)
    monkeypatch.setattr(module, '_input_lines', lines)
    monkeypatch.setattr(asyncio.get_running_loop(), 'add_signal_handler', lambda *args: None)
    task = asyncio.create_task(module.main())
    try:
        await entered.wait()
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(.01)
        assert not task.done()
    finally:
        release.set()
    assert await task == 0


@pytest.mark.asyncio
async def test_oversize_result_error_keeps_request_id(monkeypatch):
    module = load_module()
    written = bytearray()
    monkeypatch.setattr(module, 'MAX_OUTPUT_LINE_BYTES', 256)
    monkeypatch.setattr(module.os, 'set_blocking', lambda *args: None)
    def write(fd, data):
        written.extend(data)
        return len(data)
    monkeypatch.setattr(module.os, 'write', write)
    await module._Stdout()(module._result(77, {'text':'x'*1000}))
    message = json.loads(written)
    assert message['id'] == 77 and message['error']['code'] == -32603
    assert len(written) <= 256


def test_search_tool_schema_and_validation_match_closed_gateway_options():
    module=load_module()
    tool=next(tool for tool in module._tool_definitions("mail-read-v2") if tool['name']=='search_emails_batch')
    schema=tool['inputSchema']['properties']['searches']
    assert schema['maxItems']==20 and schema['items']['additionalProperties'] is False
    assert schema['items']['properties']['query']['maxLength']==1000
    assert module._valid_arguments('search_emails_batch',{'searches':[{'query':'draw','detail':'refs'}]})
    for item in ({'query':'draw','owner_id':'bob'},{'query':'draw','top_k':True},{'query':'draw','date_from':'invalid'}):
        assert not module._valid_arguments('search_emails_batch',{'searches':[item]})


def test_facts_schema_and_validation_use_closed_single_query_contract():
    module = load_module()
    tool = next(tool for tool in module._tool_definitions("mail-read-v2") if tool['name'] == 'find_facts')
    schema = tool['inputSchema']
    assert schema['additionalProperties'] is False
    assert schema['properties']['query']['maxLength'] == 1000
    assert schema['properties']['k']['maximum'] == 500
    assert module._valid_arguments('find_facts', {'query': 'cars'})
    for value in ({'query': 'x', 'owner_id': 'bob'}, {'query': 'x', 'exhaustive': 1}, {'query': 'x', 'k': True}):
        assert not module._valid_arguments('find_facts', value)


def test_metadata_schema_and_closed_validation():
    module = load_module()
    tool = next(tool for tool in module._tool_definitions("mail-read-v2") if tool['name'] == 'query_emails_batch')
    schema = tool['inputSchema']['properties']['filters']
    assert schema['maxItems'] == 20 and schema['items']['additionalProperties'] is False
    assert module._valid_arguments('query_emails_batch', {'filters': [{}]})
    assert module._valid_arguments('query_emails_batch', {'filters': [{'sender': "O'Brien", 'has_attachment': False}]})
    for item in ({'limit': True}, {'owner_id': 'bob'}, {'date_to': 'not-a-date'}):
        assert not module._valid_arguments('query_emails_batch', {'filters': [item]})


def test_thread_manifest_paging_schema_and_validation():
    module = load_module()
    schema = next(tool for tool in module._tool_definitions("mail-read-v2") if tool['name'] == 'get_thread_batch')['inputSchema']['properties']
    assert schema['attachment_after_id']['maximum'] == 9223372036854775807
    assert schema['attachment_limit']['maximum'] == 100
    assert module._valid_arguments('get_thread_batch', {'thread_ids': ['thread'], 'attachment_after_id': 10, 'attachment_limit': 5})
    assert not module._valid_arguments('get_thread_batch', {'thread_ids': ['thread'], 'attachment_after_id': True})


@pytest.mark.asyncio
async def test_mcp_unavailable_config_never_advertises_tools():
    module=load_module()
    def factory():raise ValueError('private configuration')
    emitted=[]
    server=module.GuestMailMCP(factory,emitted.append)
    server._ready=True
    await server.receive({'jsonrpc':'2.0','id':1,'method':'tools/list'})
    assert emitted==[{'jsonrpc':'2.0','id':1,'error':{'code':-32000,'message':'Guest tools are unavailable.'}}]


@pytest.mark.asyncio
async def test_mcp_cached_legacy_profile_cannot_expand_after_configuration_changes():
    module=load_module()
    tools=FakeTools()
    tools.tool_profile='legacy-mail-v1'
    calls=0
    def factory():
        nonlocal calls
        calls+=1
        return tools
    emitted=[]
    server=module.GuestMailMCP(factory,emitted.append)
    server._ready=True
    await server.receive({'jsonrpc':'2.0','id':1,'method':'tools/list'})
    tools.tool_profile='mail-read-v2'
    await server.receive({'jsonrpc':'2.0','id':2,'method':'tools/list'})
    assert all(len(row['result']['tools'])==4 for row in emitted) and calls==1
    await server.receive({'jsonrpc':'2.0','id':3,'method':'tools/call','params':{'name':'get_attachment_batch','arguments':{'items':[{'attachment_id':1}]}}})
    await server.wait_idle()
    assert emitted[-1]['result']['isError'] is True and tools.calls==[]


def test_mcp_attachment_schema_matches_closed_modes_and_paging():
    module=load_module()
    import jsonschema
    schema=next(t['inputSchema'] for t in module._tool_definitions('mail-read-v2') if t['name']=='get_attachment_batch')
    valid=[{'attachment_id':1},{'attachment_id':1,'mode':'meta'},{'attachment_id':1,'mode':'text','offset':2147483646,'limit':100000}]
    invalid=[{'attachment_id':True},{'attachment_id':1,'mode':'raw'},{'attachment_id':1,'mode':'meta','offset':0},{'attachment_id':1,'limit':100001}]
    for item in valid:
        jsonschema.validate({'items':[item]},schema)
        assert module._valid_arguments('get_attachment_batch',{'items':[item]})
    for item in invalid:
        with pytest.raises(jsonschema.ValidationError):jsonschema.validate({'items':[item]},schema)
        assert not module._valid_arguments('get_attachment_batch',{'items':[item]})


async def _started(server):
    await server.receive({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                          "params": {"protocolVersion": "2025-11-25", "capabilities": {},
                                     "clientInfo": {"name": "test", "version": "1"}}})
    await server.receive({"jsonrpc": "2.0", "method": "notifications/initialized"})


async def _call(server, identifier, name, arguments):
    await server.receive({"jsonrpc": "2.0", "id": identifier, "method": "tools/call",
                          "params": {"name": name, "arguments": arguments}})
    await server.wait_idle()


@pytest.mark.asyncio
async def test_a_subagent_budget_refuses_mail_calls_past_the_limit():
    module = load_module()
    tools, emitted = FakeTools(), []
    server = module.GuestMailMCP(lambda: tools, emitted.append, call_budget=2)
    await _started(server)
    for identifier in range(3):
        await _call(server, identifier, "get_thread_batch", {"thread_ids": ["abc"], "body_limit": 12})
    assert len(tools.calls) == 2
    refused = emitted[-1]["result"]
    assert refused["isError"] and module.BUDGET_SPENT in json.dumps(refused)


def test_only_a_server_started_for_a_subagent_is_budgeted():
    module = load_module()
    assert module._budget_from_argv(['--subagent']) == module.SUBAGENT_CALL_BUDGET
    assert module._budget_from_argv([]) is None
    with pytest.raises(SystemExit):
        module._budget_from_argv(['--budget=1000'])
