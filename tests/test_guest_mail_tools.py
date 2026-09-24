"""Guest-only tools use synthetic loopback HTTP, never live mail or providers."""
import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import pytest_asyncio

MODULE = Path(__file__).parents[1] / 'deploy/public/worker/guest_mail_tools.py'
TOKENS = {'sql': '1' * 64, 'retrieval': '2' * 64, 'artifact': '3' * 64}
V2_CONFIG = {'version': 2, 'tool_profile': 'mail-read-v2', 'capabilities': {**TOKENS, 'attachment': '4'*64}}
V3_CONFIG = {'version': 3, 'tool_profile': 'mail-raw-mcp-v3', 'capabilities': {**TOKENS, 'attachment': '4'*64}}


def load_module():
    assert MODULE.exists(), 'guest capability tool adapter is not implemented'
    spec = importlib.util.spec_from_file_location('guest_mail_tools', MODULE)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0,str(MODULE.parent))
    try:spec.loader.exec_module(module)
    finally:sys.path.pop(0)
    return module


class Gateway:
    def __init__(self):
        self.calls = []
        self.active = 0
        self.peak = 0
        self.tasks = set()
        self.received = asyncio.Event()
        self.disconnected = asyncio.Event()
        self.hold = False
        self.raw = None
        self.status = 200

    async def handle(self, reader, writer):
        task = asyncio.current_task()
        self.tasks.add(task)
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            head = await reader.readuntil(b'\r\n\r\n')
            lines = head.decode().split('\r\n')
            method, path, _ = lines[0].split()
            headers = dict(line.split(': ', 1) for line in lines[1:] if line)
            body = await reader.readexactly(int(headers.get('Content-Length', '0')))
            self.calls.append((method, path, headers, body))
            self.received.set()
            if self.hold:
                await reader.read()
                self.disconnected.set()
                return
            if self.raw is not None:
                writer.write(self.raw)
            else:
                if path == '/v1/schema':
                    result = {'messages': {'id': 'text', 'thread_id': 'text'}}
                elif path == '/v1/sql':
                    result = {'columns': ['thread_id'], 'rows': [['thread-a']], 'complete': False,
                              'scope': 'query_result'}
                elif path == '/v1/thread':
                    result = {'thread_id': json.loads(body)['thread_id'], 'cite_ref': 'thread-a',
                              'messages': [], 'complete': False, 'source_complete': False,
                              'next_message_offset': 20, 'body_format': 'text'}
                else:
                    result = {'id': 'a' * 32, 'filename': 'report.csv', 'size': len(body)}
                if self.status != 200:
                    result = {'detail': 'private upstream ' + TOKENS['sql']}
                data = json.dumps(result).encode()
                status = 201 if path.startswith('/v1/artifacts?') and self.status == 200 else self.status
                writer.write(f'HTTP/1.1 {status} OK\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n'.encode() + data)
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            self.active -= 1
            self.tasks.discard(task)


@pytest_asyncio.fixture
async def gateway(monkeypatch):
    fixture = Gateway()
    server = await asyncio.start_server(fixture.handle, '127.0.0.1', 0)
    port = server.sockets[0].getsockname()[1]
    original = asyncio.open_connection

    async def connect(host, selected_port, **kwargs):
        assert host == '127.0.0.1' and selected_port in (18080, 18081)
        return await original(host, port, **kwargs)

    monkeypatch.setattr(asyncio, 'open_connection', connect)
    try:
        yield fixture
    finally:
        server.close()
        await server.wait_closed()
        for task in fixture.tasks.copy():
            task.cancel()
        await asyncio.gather(*fixture.tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_tool_routes_capabilities_and_existing_batch_shapes(tmp_path, gateway):
    module = load_module()
    client = module.GuestMailTools(tmp_path, TOKENS)
    schema = await client.dispatch('describe_schema', {})
    sql = await client.dispatch('sql_query_batch', {'queries': ['SELECT thread_id FROM messages']})
    threads = await client.dispatch('get_thread_batch', {'thread_ids': ['thread-a'],
        'message_offset': 20, 'message_limit': 10, 'body_offset': 50, 'body_limit': 100})
    assert schema == {'messages': {'id': 'text', 'thread_id': 'text'}}
    assert sql['results'][0]['query'] == 'SELECT thread_id FROM messages'
    assert sql['results'][0]['result']['complete'] is False
    assert threads['results'][0]['thread_id'] == 'thread-a'
    assert threads['results'][0]['result']['next_message_offset'] == 20
    assert [call[1] for call in gateway.calls] == ['/v1/schema', '/v1/sql', '/v1/thread']
    assert [call[2]['Authorization'] for call in gateway.calls] == [
        'Bearer ' + TOKENS['sql'], 'Bearer ' + TOKENS['sql'], 'Bearer ' + TOKENS['retrieval']]
    assert json.loads(gateway.calls[-1][3]) == {'thread_id': 'thread-a', 'message_offset': 20,
        'message_limit': 10, 'body_offset': 50, 'body_limit': 100}
    assert all('user_id' not in call[3].decode() and 'session_id' not in call[3].decode() for call in gateway.calls)


@pytest.mark.asyncio
async def test_artifact_upload_reads_guest_bytes_and_returns_string_receipt(tmp_path, gateway):
    module = load_module()
    (tmp_path / 'reports').mkdir()
    (tmp_path / 'reports/report.csv').write_bytes(b'value\n42\n')
    client = module.GuestMailTools(tmp_path, TOKENS)
    result = await client.dispatch('publish_artifact_batch', {'items': [
        {'path': 'reports/report.csv', 'name': 'report.csv', 'mime_type': 'application/octet-stream'}]})
    receipt = result['results'][0]['result']
    assert receipt == {'id': 'a' * 32, 'name': 'report.csv', 'size': 9,
                       'mime_type': 'application/octet-stream', 'content_disposition': 'attachment'}
    method, path, headers, body = gateway.calls[0]
    assert (method, path, body) == ('POST', '/v1/artifacts?filename=report.csv', b'value\n42\n')
    assert headers['Authorization'] == 'Bearer ' + TOKENS['artifact']
    assert headers['Content-Type'] == 'application/octet-stream'
    assert str(tmp_path) not in repr(gateway.calls)


@pytest.mark.asyncio
@pytest.mark.parametrize('name,args', [
    ('unknown', {}), ('describe_schema', {'owner_id': 'other'}),
    ('sql_query_batch', {'queries': ['SELECT 1'], 'port': 80}),
    ('sql_query_batch', {'queries': []}), ('sql_query_batch', {'queries': ['SELECT 1'] * 21}),
    ('get_thread_batch', {'thread_ids': ['a'], 'body_format': 'raw'}),
    ('get_thread_batch', {'thread_ids': ['a'], 'body_format': 'markdown'}),
    ('get_thread_batch', {'thread_ids': ['a'], 'message_ids': ['m']}),
    ('get_thread_batch', {'thread_ids': ['a'], 'message_offset': True}),
])
async def test_invalid_dispatch_never_connects(tmp_path, gateway, name, args):
    module = load_module()
    result = await module.GuestMailTools(tmp_path, TOKENS).dispatch(name, args)
    assert 'error' in result and not gateway.calls


@pytest.mark.asyncio
async def test_batch_isolates_errors_and_rejects_guest_file_escape(tmp_path, gateway):
    module = load_module()
    outside = tmp_path.parent / 'outside-synthetic.txt'
    outside.write_text('outside')
    (tmp_path / 'escape').symlink_to(outside)
    (tmp_path / 'dir').symlink_to(tmp_path.parent, target_is_directory=True)
    (tmp_path / 'good').write_text('42')
    items = [{'path': value} for value in ('../outside-synthetic.txt', str(outside), 'escape', 'dir/outside-synthetic.txt', '.')]
    items += [{'path': 'good', 'mime_type': 'text/html'}, {'path': 'good', 'name': 'x\r\nInjected: secret'}, {'path': 'good'}]
    result = await module.GuestMailTools(tmp_path, TOKENS).dispatch('publish_artifact_batch', {'items': items})
    assert all('error' in entry['result'] for entry in result['results'][:-1])
    assert result['results'][-1]['result']['id'] == 'a' * 32
    assert len(gateway.calls) == 1


@pytest.mark.asyncio
async def test_errors_are_sanitized_without_retry(tmp_path, gateway):
    module = load_module()
    gateway.status = 429
    result = await module.GuestMailTools(tmp_path, TOKENS).dispatch('sql_query_batch', {'queries': ['SELECT 1']})
    assert 'error' in result['results'][0]['result']
    assert TOKENS['sql'] not in repr(result) and 'private upstream' not in repr(result)
    assert len(gateway.calls) == 1


@pytest.mark.asyncio
async def test_repeated_cancellation_closes_all_inflight_connections(tmp_path, gateway):
    module = load_module()
    gateway.hold = True
    client = module.GuestMailTools(tmp_path, TOKENS)
    task = asyncio.create_task(client.dispatch('sql_query_batch', {'queries': ['SELECT 1'] * 20}))
    await asyncio.wait_for(gateway.received.wait(), 2)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.wait_for(gateway.disconnected.wait(), 2)
    assert len(gateway.calls) <= 2


@pytest.mark.asyncio
async def test_instance_concurrency_and_batch_deadline_are_bounded(tmp_path, gateway):
    module = load_module()
    gateway.hold = True
    client = module.GuestMailTools(tmp_path, TOKENS, timeout_seconds=.05)
    results = await asyncio.gather(*(client.dispatch('sql_query_batch', {'queries': ['SELECT 1'] * 20}) for _ in range(2)))
    assert gateway.peak <= 2
    assert all('error' in entry['result'] for result in results for entry in result['results'])
    assert len(gateway.calls) <= 2


@pytest.mark.asyncio
@pytest.mark.parametrize('response', [
    b'HTTP/1.1 200 OK\r\nContent-Length: 2\r\nContent-Length: 2\r\n\r\n{}',
    b'HTTP/1.1 200 OK\r\nContent-Length: 2\r\nTransfer-Encoding: chunked\r\n\r\n{}',
    b'HTTP/1.1 302 Found\r\nLocation: http://elsewhere/\r\n\r\n',
    b'HTTP/1.1 200 OK\r\nContent-Length: 9\r\n\r\n{}',
    b'HTTP/1.1 200 OK\r\n\r\n{"a":1,"a":2}',
    b'HTTP/1.1 200 OK\r\n\r\n{"a":NaN}',
])
async def test_rejects_ambiguous_or_invalid_http_and_json(tmp_path, gateway, response):
    module = load_module()
    gateway.raw = response.replace(b'\r\n', b'\r\nContent-Type: application/json\r\n', 1)
    assert 'error' in await module.GuestMailTools(tmp_path, TOKENS).dispatch('describe_schema', {})


@pytest.mark.asyncio
async def test_size_limits_reject_before_large_upload_and_bound_response(tmp_path, gateway, monkeypatch):
    module = load_module()
    monkeypatch.setattr(module, 'MAX_ARTIFACT_BYTES', 4)
    monkeypatch.setattr(module, 'MAX_RESPONSE_BYTES', 8)
    (tmp_path / 'large').write_text('12345')
    client = module.GuestMailTools(tmp_path, TOKENS)
    result = await client.dispatch('publish_artifact_batch', {'items': [{'path': 'large'}]})
    assert 'error' in result['results'][0]['result'] and not gateway.calls
    assert 'error' in await client.dispatch('describe_schema', {})


@pytest.mark.asyncio
async def test_cancel_aborts_blocked_socket_flush_before_returning(tmp_path, monkeypatch):
    module = load_module()
    writing, closed = asyncio.Event(), asyncio.Event()

    class Writer:
        transport = None
        aborted = False

        def __init__(self):
            self.transport = self

        def write(self, data):
            writing.set()

        async def drain(self):
            await asyncio.Event().wait()

        def close(self):
            pass  # A real close can wait indefinitely for buffered bytes.

        def abort(self):
            self.aborted = True
            closed.set()

        async def wait_closed(self):
            await closed.wait()

    writer = Writer()

    async def connect(*args, **kwargs):
        return asyncio.StreamReader(), writer

    monkeypatch.setattr(asyncio, 'open_connection', connect)
    task = asyncio.create_task(module.GuestMailTools(tmp_path, TOKENS).dispatch('describe_schema', {}))
    await writing.wait()
    task.cancel()
    for _ in range(10):
        await asyncio.sleep(0)
    try:
        assert writer.aborted, 'cancellation must abort a stalled socket flush'
    finally:
        closed.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_content_length_response_and_batch_total_byte_budget(tmp_path, gateway, monkeypatch):
    module = load_module()
    data = b'{"messages":{"id":"text"}}'
    gateway.raw = b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: ' + str(len(data)).encode() + b'\r\n\r\n' + data
    client = module.GuestMailTools(tmp_path, TOKENS)
    assert await client.dispatch('describe_schema', {}) == {'messages': {'id': 'text'}}
    monkeypatch.setattr(module, 'MAX_BATCH_RESPONSE_BYTES', len(data))
    result = await client.dispatch('sql_query_batch', {'queries': ['SELECT 1', 'SELECT 2']})
    assert sum('error' in item['result'] for item in result['results']) == 1


@pytest.mark.asyncio
async def test_fifo_and_nonregular_guest_files_are_rejected_without_reading(tmp_path, gateway):
    import os
    module = load_module()
    os.mkfifo(tmp_path / 'pipe')
    (tmp_path / 'directory').mkdir()
    client = module.GuestMailTools(tmp_path, TOKENS)
    result = await client.dispatch('publish_artifact_batch', {'items': [{'path': 'pipe'}, {'path': 'directory'}]})
    assert all('error' in item['result'] for item in result['results'])
    assert not gateway.calls


@pytest.mark.asyncio
async def test_receipt_uses_trusted_mime_and_rejects_forged_size(tmp_path, gateway):
    module = load_module()
    (tmp_path / 'file').write_text('42')
    client = module.GuestMailTools(tmp_path, TOKENS)

    def receipt(size):
        return b'HTTP/1.1 201 OK\r\nContent-Type: application/json\r\n\r\n' + json.dumps({
            'id': 'b' * 32, 'filename': 'trusted.txt', 'size': size, 'mime_type': 'text/plain'}).encode()

    gateway.raw = receipt(2)
    result = await client.dispatch('publish_artifact_batch', {'items': [{'path': 'file'}]})
    assert result['results'][0]['result']['mime_type'] == 'text/plain'
    assert result['results'][0]['result']['name'] == 'trusted.txt'
    gateway.raw = receipt(3)
    result = await client.dispatch('publish_artifact_batch', {'items': [{'path': 'file'}]})
    assert 'error' in result['results'][0]['result']
    assert len(gateway.calls) == 2


@pytest.mark.parametrize('kwargs', [
    {'port': 80}, {'port': True}, {'timeout_seconds': float('nan')},
    {'timeout_seconds': 0}, {'timeout_seconds': 61},
])
def test_trusted_endpoint_configuration_is_fixed_and_bounded(tmp_path, kwargs):
    module = load_module()
    with pytest.raises(module.ToolError):
        module.GuestMailTools(tmp_path, TOKENS, **kwargs)


def test_capability_configuration_rejects_injection_without_disclosing_tokens(tmp_path):
    module = load_module()
    for caps in ({'sql': TOKENS['sql']}, {**TOKENS, 'sql': TOKENS['sql'] + '\r\nInjected: secret'}):
        with pytest.raises(module.ToolError) as caught:
            module.GuestMailTools(tmp_path, caps)
        assert TOKENS['sql'] not in str(caught.value)


@pytest.mark.asyncio
async def test_search_batch_uses_retrieval_capability_and_preserves_item_errors(gateway,tmp_path):
    module=load_module()
    core=module.GuestMailTools(tmp_path,V2_CONFIG)
    items=[{'query':'draw request','detail':'refs'}, {'query':'x','owner_id':'bob'}, {'query':'second','top_k':2}]
    result=await core.dispatch('search_emails_batch',{'searches':items})
    assert [row['input'] for row in result['results']]==items
    assert 'error' in result['results'][1]['result']
    assert len(gateway.calls)==2
    for method,path,headers,body in gateway.calls:
        assert (method,path)==('POST','/v1/search')
        assert headers['Authorization']=='Bearer '+TOKENS['retrieval']
        assert 'owner_id' not in json.loads(body)


@pytest.mark.asyncio
@pytest.mark.parametrize('item',[{'query':''},{'query':'x','top_k':True},{'query':'x','detail':'html'},
    {'query':'x','date_from':'2026-02-30'},{'query':'x','model':'other'},{'query':'x','max_matches':101}])
async def test_search_invalid_options_never_open_connection(gateway,tmp_path,item):
    module=load_module()
    result=await module.GuestMailTools(tmp_path,V2_CONFIG).dispatch('search_emails_batch',{'searches':[item]})
    assert 'error' in result['results'][0]['result']
    assert not gateway.calls


@pytest.mark.asyncio
async def test_find_facts_uses_single_retrieval_request(gateway, tmp_path):
    module = load_module()
    result = await module.GuestMailTools(tmp_path, V2_CONFIG).dispatch('find_facts', {'query': 'cars', 'exhaustive': False, 'k': 7})
    assert 'error' not in result
    assert len(gateway.calls) == 1
    method, path, headers, body = gateway.calls[0]
    assert (method, path) == ('POST', '/v1/find-facts')
    assert headers['Authorization'] == 'Bearer '+TOKENS['retrieval']
    assert json.loads(body) == {'query': 'cars', 'exhaustive': False, 'k': 7}


@pytest.mark.asyncio
@pytest.mark.parametrize('item', [{'query': ''}, {'query': 'x', 'owner_id': 'bob'},
    {'query': 'x', 'model': 'other'}, {'query': 'x', 'exhaustive': 1},
    {'query': 'x', 'k': True}, {'query': 'x', 'k': 501}])
async def test_facts_invalid_options_never_open_connection(gateway, tmp_path, item):
    module = load_module()
    result = await module.GuestMailTools(tmp_path, V2_CONFIG).dispatch('find_facts', item)
    assert 'error' in result
    assert not gateway.calls


@pytest.mark.asyncio
async def test_metadata_batch_uses_retrieval_and_preserves_item_errors(gateway, tmp_path):
    module = load_module()
    items = [{'sender': "O'Brien", 'has_attachment': False}, {'owner_id': 'bob'}, {'order_by': 'date_asc', 'limit': 2}]
    result = await module.GuestMailTools(tmp_path, V2_CONFIG).dispatch('query_emails_batch', {'filters': items})
    assert [row['input'] for row in result['results']] == items
    assert 'error' in result['results'][1]['result']
    assert len(gateway.calls) == 2
    for method, path, headers, body in gateway.calls:
        assert (method, path) == ('POST', '/v1/query-emails')
        assert headers['Authorization'] == 'Bearer '+TOKENS['retrieval']
        assert 'owner_id' not in json.loads(body)


@pytest.mark.asyncio
@pytest.mark.parametrize('item', [{'limit': True}, {'limit': 101}, {'has_attachment': 'false'},
    {'date_from': '2026-02-30'}, {'date_from': '2026-09-15', 'date_to': '2026-09-14'},
    {'order_by': 'score'}, {'sender': 'x'*1001}, {'label': 'x'*257}, {'sql': 'SELECT 1'}])
async def test_invalid_metadata_options_never_connect(gateway, tmp_path, item):
    module = load_module()
    result = await module.GuestMailTools(tmp_path, V2_CONFIG).dispatch('query_emails_batch', {'filters': [item]})
    assert 'error' in result['results'][0]['result']
    assert not gateway.calls


@pytest.mark.asyncio
async def test_thread_attachment_paging_forwards_fixed_fields(gateway, tmp_path):
    module = load_module()
    value = {'thread_ids': ['thread'], 'attachment_after_id': 12, 'attachment_limit': 3}
    result = await module.GuestMailTools(tmp_path, V2_CONFIG).dispatch('get_thread_batch', value)
    assert 'error' not in result
    assert len(gateway.calls) == 1
    assert json.loads(gateway.calls[0][3]) == {'thread_id': 'thread', 'attachment_after_id': 12, 'attachment_limit': 3}


@pytest.mark.asyncio
@pytest.mark.parametrize('options', [{'attachment_after_id': True}, {'attachment_after_id': -1}, {'attachment_after_id': 9223372036854775808}, {'attachment_limit': 0}, {'attachment_limit': 101}])
async def test_thread_attachment_invalid_paging_never_connects(gateway, tmp_path, options):
    module = load_module()
    result = await module.GuestMailTools(tmp_path, V2_CONFIG).dispatch('get_thread_batch', {'thread_ids': ['thread'], **options})
    assert 'error' in result
    assert not gateway.calls


@pytest.mark.asyncio
async def test_v2_attachment_uses_own_audience_and_preserves_null_empty_and_paging(gateway,tmp_path):
    module=load_module()
    core=module.GuestMailTools(tmp_path,V2_CONFIG)
    for text in (None,''):
        payload={'extracted_text':text,'complete':False,'source_complete':None,'next_offset':20000}
        gateway.raw=b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n'+json.dumps(payload).encode()
        item={'attachment_id':9223372036854775807,'offset':2147483646,'limit':100000}
        result=await core.dispatch('get_attachment_batch',{'items':[item]})
        assert result=={'results':[{'input':item,'result':payload}]}
        method,path,headers,body=gateway.calls[-1]
        assert (method,path)==('POST','/v1/attachment/text')
        assert headers['Authorization']=='Bearer '+V2_CONFIG['capabilities']['attachment']
        assert json.loads(body)==item
    await core.dispatch('get_attachment_batch',{'items':[{'attachment_id':1,'mode':'meta'}]})
    assert gateway.calls[-1][1]=='/v1/attachment/meta'
    assert json.loads(gateway.calls[-1][3])=={'attachment_id':1}


@pytest.mark.asyncio
async def test_v2_attachment_batch_cancellation_closes_owned_http(gateway,tmp_path):
    module=load_module()
    core=module.GuestMailTools(tmp_path,V2_CONFIG)
    gateway.hold=True
    task=asyncio.create_task(core.dispatch('get_attachment_batch',{'items':[{'attachment_id':i} for i in range(1,21)]}))
    await gateway.received.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):await task
    await asyncio.wait_for(gateway.disconnected.wait(),1)
    assert core._slots._value==2 and gateway.peak<=2


@pytest.mark.asyncio
async def test_v2_attachment_errors_are_redacted_and_invalid_items_do_not_block_valid(gateway,tmp_path):
    module=load_module()
    core=module.GuestMailTools(tmp_path,V2_CONFIG)
    gateway.status=403
    items=[{'attachment_id':1,'mode':'raw'},{'attachment_id':2,'mode':'meta'}]
    result=await core.dispatch('get_attachment_batch',{'items':items})
    assert len(gateway.calls)==1 and all('error' in row['result'] for row in result['results'])
    assert TOKENS['sql'] not in json.dumps(result) and 'private upstream' not in json.dumps(result)


JUDGE_CALL = {'state': 'Casera confirmed the 4000W W-Series order.', 'questions': [
    {'id': 'answered', 'type': 'noul', 'instructions': 'Does the state say which heaters were ordered?'},
    {'id': 'kind', 'type': 'choice', 'instructions': 'Order or quote?', 'options': ['order: confirmed purchase', 'quote: price only']}]}


@pytest.mark.asyncio
async def test_judge_converts_questions_and_sends_one_retrieval_request(gateway, tmp_path):
    module = load_module()
    result = await module.GuestMailTools(tmp_path, V3_CONFIG).dispatch('judge', JUDGE_CALL)
    assert 'error' not in result
    method, path, headers, body = gateway.calls[0]
    assert (method, path, len(gateway.calls)) == ('POST', '/v1/judge', 1)
    assert headers['Authorization'] == 'Bearer '+TOKENS['retrieval']
    assert json.loads(body)['questions'] == {
        'answered': {'type': 'noul', 'instructions': 'Does the state say which heaters were ordered?'},
        'kind': {'type': 'choice', 'instructions': 'Order or quote?',
                 'criteria': {'order': 'confirmed purchase', 'quote': 'price only'}}}


@pytest.mark.asyncio
@pytest.mark.parametrize('call', [{'state': 'x', 'questions': []},
    {'state': '', 'questions': JUDGE_CALL['questions']},
    {'state': 'x', 'questions': [{'id': 'a', 'type': 'choice', 'instructions': 'pick', 'options': ['only']}]},
    {'state': 'x', 'questions': [{'id': 'a', 'type': 'noul', 'instructions': 'yes?', 'options': ['x']}]}])
async def test_invalid_judge_calls_never_open_connection(gateway, tmp_path, call):
    module = load_module()
    result = await module.GuestMailTools(tmp_path, V3_CONFIG).dispatch('judge', call)
    assert 'error' in result
    assert not gateway.calls
