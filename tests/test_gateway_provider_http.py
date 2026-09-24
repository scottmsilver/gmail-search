import asyncio
import json

import httpx
import pytest

from gmail_search.gateway.provider_http import AnthropicHTTPTransport
from test_gateway_provider import setup as setup_fixture, service, collect, spend

setup = setup_fixture


def event(kind, **data):
    return ('event: ' + kind + '\ndata: ' + json.dumps(dict(type=kind, **data)) + '\n\n').encode()


def response_events():
    return [
        event('message_start', message={'id':'msg_test', 'type':'message','role':'assistant','model':'server-model',
             'content':[], 'stop_reason':None,'stop_sequence':None,'usage':{'input_tokens':7,'output_tokens':1,'cache_read_input_tokens':0,'cache_creation_input_tokens':0}}),
        event('content_block_start', index=0, content_block={'type':'text','text':''}),
        event('ping'),
        event('content_block_delta', index=0, delta={'type':'text_delta','text':'synthetic'}),
        event('content_block_stop', index=0),
        event('message_delta', delta={'stop_reason':'end_turn','stop_sequence':None}, usage={'output_tokens':3}),
        event('message_stop'),
    ]


class Wire(httpx.AsyncByteStream):
    def __init__(self, chunks, *, wait=False):
        self.chunks, self.wait, self.closed = chunks, wait, False
        self.started=asyncio.Event()
    async def __aiter__(self):
        self.started.set()
        for chunk in self.chunks:
            yield chunk
        if self.wait:
            await asyncio.Event().wait()
    async def aclose(self):
        self.closed = True


def client_for(wire, *, status=200, headers=None, requests=None):
    async def handle(request):
        if requests is not None:
            requests.append(request)
        return httpx.Response(status,headers=headers or {'content-type':'text/event-stream'},stream=wire)
    return httpx.AsyncClient(transport=httpx.MockTransport(handle),trust_env=False,
                            headers={'evil-default':'not-forwarded'},cookies={'session':'not-forwarded'},
                            params={'url':'not-forwarded'},auth=('not','forwarded'),follow_redirects=True)


@pytest.mark.asyncio
async def test_a_claude_login_is_sent_as_an_oauth_bearer_not_an_api_key(setup):
    requests=[]
    async with client_for(Wire(response_events()),requests=requests) as client:
        transport=AnthropicHTTPTransport('synthetic-login',kind='oauth',client=client)
        await collect(service(setup,transport),setup[3])
    headers=requests[0].headers
    assert headers['authorization']=='Bearer synthetic-login'
    assert headers['anthropic-beta']=='oauth-2025-04-20'
    assert 'x-api-key' not in headers
    assert 'synthetic-login' not in repr(transport)


def test_an_unknown_credential_kind_is_refused():
    with pytest.raises(ValueError):
        AnthropicHTTPTransport('synthetic-key',kind='cookie')


@pytest.mark.asyncio
async def test_mock_http_stream_uses_fixed_headers_and_settles_actual_usage(setup):
    frames = response_events(); raw = b''.join(frames).replace(b'\n',b'\r\n')
    wire = Wire([raw[i:i+7] for i in range(0,len(raw),7)])
    requests=[]
    async with client_for(wire,requests=requests) as client:
        transport=AnthropicHTTPTransport('synthetic-key',client=client)
        svc=service(setup,transport)
        chunks=await collect(svc,setup[3])
        assert b''.join(chunks) == b''.join(frames)
    request=requests[0]
    assert str(request.url)=='https://api.anthropic.com/v1/messages'
    assert request.headers['x-api-key']=='synthetic-key'
    assert request.headers['anthropic-version']=='2023-06-01'
    assert request.headers['accept-encoding']=='identity'
    assert not {'authorization','cookie','evil-default'} & set(request.headers)
    assert request.extensions['timeout']['connect']==5
    assert wire.closed and spend(setup[0])==(0,23)
    assert 'synthetic-key' not in repr(transport)


@pytest.mark.asyncio
@pytest.mark.parametrize('status', [301,307,429,503])
async def test_no_redirects_or_retries_or_error_body_disclosure(setup,status):
    requests=[];wire=Wire([b'secret-provider-response'])
    async with client_for(wire,status=status,headers={'location':'https://evil.test'},requests=requests) as client:
        svc=service(setup,AnthropicHTTPTransport('synthetic-key',client=client))
        with pytest.raises(Exception) as error:
            await collect(svc,setup[3])
    assert 'secret-provider' not in str(error.value)
    assert len(requests)==1 and wire.closed and spend(setup[0])==(0,230)


@pytest.mark.asyncio
@pytest.mark.parametrize('mutation', ['truncated','extra_after_stop','wrong_event','bad_json','duplicate_json','unknown_event','decreasing_usage','cache_usage','missing_usage','wrong_model','open_block'])
async def test_malformed_sse_fails_closed_and_keeps_full_reservation(setup,mutation):
    frames=response_events()
    if mutation=='truncated':frames.pop()
    elif mutation=='extra_after_stop':frames.append(event('ping'))
    elif mutation=='wrong_event':frames[0]=frames[0].replace(b'event: message_start',b'event: ping')
    elif mutation=='bad_json':frames[0]=b'event: message_start\ndata: {\n\n'
    elif mutation=='duplicate_json':frames[0]=frames[0].replace(b'{"type": "message_start",',b'{"type": "ping", "type": "message_start",')
    elif mutation=='unknown_event':frames.insert(1,event('new_feature'))
    elif mutation=='decreasing_usage':frames.insert(-1,event('message_delta',delta={'stop_reason':'end_turn'},usage={'output_tokens':2}))
    elif mutation=='cache_usage':frames[0]=frames[0].replace(b'"cache_read_input_tokens": 0',b'"cache_read_input_tokens": 8')
    elif mutation=='missing_usage':frames[-2]=event('message_delta',delta={'stop_reason':'end_turn'})
    elif mutation=='wrong_model':frames[0]=frames[0].replace(b'server-model',b'other-model')
    elif mutation=='open_block':del frames[4]
    wire=Wire(frames)
    async with client_for(wire) as client:
        svc=service(setup,AnthropicHTTPTransport('synthetic-key',client=client))
        with pytest.raises(RuntimeError):
            await collect(svc,setup[3])
    assert wire.closed and spend(setup[0])==(0,230)


@pytest.mark.asyncio
async def test_cancel_revoked_service_closes_actual_http_stream(setup):
    wire=Wire(response_events()[:1],wait=True)
    async with client_for(wire) as client:
        svc=service(setup,AnthropicHTTPTransport('synthetic-key',client=client))
        stream=svc.stream(setup[3],'request',{'model':'server-model','messages':[{'role':'user','content':'hello'}],'max_tokens':10,'stream':True})
        await asyncio.wait_for(anext(stream),1)
        await stream.aclose()
    assert wire.closed and spend(setup[0])==(0,230)


@pytest.mark.asyncio
@pytest.mark.parametrize('headers',[{'content-type':'application/json'}, {'content-type':'text/event-stream','content-encoding':'gzip'}])
async def test_unexpected_content_type_or_compression_is_refused(setup,headers):
    wire=Wire(response_events())
    async with client_for(wire,headers=headers) as client:
        svc=service(setup,AnthropicHTTPTransport('synthetic-key',client=client))
        with pytest.raises(RuntimeError):await collect(svc,setup[3])
    assert wire.closed


@pytest.mark.asyncio
async def test_frame_limit_and_total_limit(setup):
    wire=Wire([b'event: ping\ndata: '+b'x'*2048])
    async with client_for(wire) as client:
        svc=service(setup,AnthropicHTTPTransport('synthetic-key',client=client,max_frame_bytes=1024))
        with pytest.raises(RuntimeError):await collect(svc,setup[3])
    assert wire.closed and spend(setup[0])==(0,230)


@pytest.mark.asyncio
async def test_injected_client_must_disable_environment_proxies():
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request:None)) as client:
        with pytest.raises(ValueError):AnthropicHTTPTransport('synthetic-key',client=client)


@pytest.mark.asyncio
@pytest.mark.parametrize('geo',[None,'not_available'])
async def test_standard_optional_usage_and_cumulative_input_settlement(setup,geo):
    frames=response_events()
    start=json.loads(frames[0].split(b'data: ',1)[1])
    start['message'].update(container=None,stop_details=None,diagnostics=None)
    start['message']['usage'].update(cache_creation=None,server_tool_use=None,service_tier='standard',inference_geo=geo,output_tokens_details=None)
    frames[0]=event('message_start',message=start['message'])
    frames[1]=event('content_block_start',index=0,content_block={'type':'text','text':'','citations':None})
    frames[-2]=event('message_delta',delta={'stop_reason':'end_turn','stop_sequence':None,'container':None,'stop_details':None},
            usage={'input_tokens':9,'output_tokens':3,'cache_read_input_tokens':0,'cache_creation_input_tokens':0,'server_tool_use':None,'output_tokens_details':{'thinking_tokens':1}})
    wire=Wire(frames)
    async with client_for(wire) as client:
        await collect(service(setup,AnthropicHTTPTransport('synthetic-key',client=client)),setup[3])
    assert spend(setup[0])==(0,27)


@pytest.mark.asyncio
async def test_total_body_limit_counts_comment_heartbeats(setup):
    wire=Wire([b': heartbeat\n\n']*200)
    async with client_for(wire) as client:
        transport=AnthropicHTTPTransport('synthetic-key',client=client,max_frame_bytes=128,max_body_bytes=1024)
        with pytest.raises(RuntimeError):await collect(service(setup,transport),setup[3])
    assert wire.closed and spend(setup[0])==(0,230)


@pytest.mark.asyncio
async def test_thinking_and_custom_tool_sse_round_trip(setup):
    frames=[response_events()[0],
        event('content_block_start',index=0,content_block={'type':'thinking','thinking':'','signature':''}),
        event('content_block_delta',index=0,delta={'type':'thinking_delta','thinking':'synthetic reasoning'}),
        event('content_block_delta',index=0,delta={'type':'signature_delta','signature':'synthetic-signature'}),
        event('content_block_stop',index=0),
        event('content_block_start',index=1,content_block={'type':'tool_use','id':'tool_1','name':'calculate','input':{},'caller':{'type':'direct'}}),
        event('content_block_delta',index=1,delta={'type':'input_json_delta','partial_json':'{"x":1}'}),
        event('content_block_stop',index=1),
        event('message_delta',delta={'stop_reason':'tool_use'},usage={'output_tokens':3}),event('message_stop')]
    wire=Wire(frames)
    request={'model':'server-model','messages':[{'role':'user','content':'hello'}],'max_tokens':10,'stream':True,
             'tools':[{'name':'calculate','input_schema':{'type':'object','properties':{'x':{'type':'integer'}}}}]}
    async with client_for(wire) as client:
        result=await collect(service(setup,AnthropicHTTPTransport('synthetic-key',client=client)),setup[3],request=request)
    assert result==frames and spend(setup[0])==(0,23)


@pytest.mark.asyncio
async def test_revocation_closes_suspended_http_stream(setup):
    wire=Wire(response_events()[:1],wait=True)
    async with client_for(wire) as client:
        svc=service(setup,AnthropicHTTPTransport('synthetic-key',client=client))
        task=asyncio.create_task(collect(svc,setup[3]))
        await asyncio.wait_for(wire.started.wait(),1)
        setup[1].revoke(setup[3])
        from gmail_search.gateway.registry import AccessDenied
        with pytest.raises(AccessDenied):
            await asyncio.wait_for(task,2)
    assert wire.closed and spend(setup[0])==(0,230)


@pytest.mark.asyncio
@pytest.mark.parametrize('mutation',['error_event','cache_creation_nonzero','server_tool_nonzero','priority_tier','input_decreases','boolean_count','nonfinite','oversized_chunk'])
async def test_extra_protocol_and_billing_rejections(setup,mutation):
    frames=response_events()
    if mutation=='error_event':frames[2]=event('error',error={'type':'overloaded_error','message':'private upstream diagnostics'})
    elif mutation=='cache_creation_nonzero':frames[-2]=event('message_delta',delta={'stop_reason':'end_turn'},usage={'output_tokens':3,'cache_creation':{'ephemeral_5m_input_tokens':2}})
    elif mutation=='server_tool_nonzero':frames[-2]=event('message_delta',delta={'stop_reason':'end_turn'},usage={'output_tokens':3,'server_tool_use':{'web_search_requests':1}})
    elif mutation=='priority_tier':frames[-2]=event('message_delta',delta={'stop_reason':'end_turn'},usage={'output_tokens':3,'service_tier':'priority'})
    elif mutation=='input_decreases':frames[-2]=event('message_delta',delta={'stop_reason':'end_turn'},usage={'output_tokens':3,'input_tokens':1})
    elif mutation=='boolean_count':frames[-2]=event('message_delta',delta={'stop_reason':'end_turn'},usage={'output_tokens':True})
    elif mutation=='nonfinite':frames[-2]=event('message_delta',delta={'stop_reason':'end_turn'},usage={'output_tokens':float('nan')})
    elif mutation=='oversized_chunk':frames=[b'x'*(1024*1024+1)]
    wire=Wire(frames)
    async with client_for(wire) as client:
        with pytest.raises(RuntimeError) as error:
            await collect(service(setup,AnthropicHTTPTransport('synthetic-key',client=client)),setup[3])
    assert 'private upstream' not in str(error.value)
    assert wire.closed and spend(setup[0])==(0,230)



def _login(path, token, expires_at):
    path.write_text(json.dumps({'claudeAiOauth': {'accessToken': token, 'expiresAt': expires_at * 1000,
                                                  'refreshToken': 'never-read'}}))


@pytest.mark.asyncio
async def test_a_borrowed_login_is_reread_for_every_request_and_never_written(setup, tmp_path):
    """The owner's Claude Code refreshes the file; each request must see the new token."""
    from gmail_search.gateway.provider_http import ClaudeLoginFile
    path = tmp_path / 'credentials.json'
    _login(path, 'first-token', 2000)
    source = ClaudeLoginFile(path, clock=lambda: 1000)
    requests = []
    async with client_for(Wire(response_events()), requests=requests) as client:
        transport = AnthropicHTTPTransport(source, kind='oauth', client=client)
        await collect(service(setup, transport), setup[3])
    _login(path, 'refreshed-token', 2000)
    before = path.read_bytes()
    assert source.read() == 'refreshed-token'
    assert path.read_bytes() == before
    assert requests[0].headers['authorization'] == 'Bearer first-token'
    assert 'first-token' not in repr(transport) and 'token' not in repr(source)


@pytest.mark.parametrize('write', [
    lambda path: _login(path, 'expiring-token', 1030),   # inside the 60 s margin
    lambda path: path.write_text('{"claudeAiOauth": {}}'),
    lambda path: None,                                     # no file at all
])
def test_a_borrowed_login_that_is_expiring_or_unreadable_is_refused(tmp_path, write):
    from gmail_search.gateway.provider_http import ClaudeLoginFile, ProviderProtocolError
    path = tmp_path / 'credentials.json'
    write(path)
    with pytest.raises(ProviderProtocolError):
        ClaudeLoginFile(path, clock=lambda: 1000).read()


def test_a_borrowed_login_is_never_sent_as_an_api_key(tmp_path):
    from gmail_search.gateway.provider_http import ClaudeLoginFile
    with pytest.raises(ValueError):
        AnthropicHTTPTransport(ClaudeLoginFile(tmp_path / 'credentials.json'))


@pytest.mark.asyncio
async def test_a_non_null_diagnostics_field_is_refused(setup):
    """Upstream added message_start.diagnostics on 2026-09-24 (null so far);
    every Claude run failed until it was allowed. Only null is qualified."""
    frames=response_events()
    start=json.loads(frames[0].split(b'data: ',1)[1])
    start['message']['diagnostics']={'note':'unqualified'}
    frames[0]=event('message_start',message=start['message'])
    wire=Wire(frames)
    async with client_for(wire) as client:
        svc=service(setup,AnthropicHTTPTransport('synthetic-key',client=client))
        with pytest.raises(RuntimeError):
            await collect(svc,setup[3])
    assert wire.closed
