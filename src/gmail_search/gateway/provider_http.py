"""Fixed Anthropic Messages HTTP/SSE adapter; not a public route.

Protocol references checked 2026-09-15:
https://platform.claude.com/docs/en/build-with-claude/streaming
https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/usage.py
https://www.python-httpx.org/api/

Only standard text/custom-tool/thinking events and zero cache usage are supported.
Unknown events, fallback models, server tools and other billing dimensions fail
closed pending qualification. This is intentionally stricter than SDK forward
compatibility. Pinned Pi/native Claude single-tool exchanges passed a synthetic upstream;
complete sessions and live provider behavior remain unqualified.
"""
from contextlib import asynccontextmanager
import json
import re

import httpx

from .inference import MAX_REQUEST_BYTES, _plain_json
from .provider import ProviderUsage

_ENDPOINT = 'https://api.anthropic.com/v1/messages'
_NEWLINE = re.compile(rb'\r\n|\r|\n')


class ProviderProtocolError(RuntimeError):
    def __init__(self):
        super().__init__('Provider response is unsupported or incomplete.')


def _object(value, required, optional=''):
    if type(value) is not dict or not set(required.split()) <= value.keys() or value.keys()-set((required+' '+optional).split()):
        raise ProviderProtocolError()
    return value


def _integer(value):
    if type(value) is not int or not 0 <= value <= 10**9:
        raise ProviderProtocolError()
    return value


def _text(value):
    if type(value) is not str:
        raise ProviderProtocolError()
    return value


def _unique_object(pairs):
    result = {}
    for key,value in pairs:
        if key in result:
            raise ProviderProtocolError()
        result[key] = value
    return result


def _usage(value, *, start):
    _object(value,'input_tokens output_tokens' if start else 'output_tokens',
            'input_tokens cache_read_input_tokens cache_creation_input_tokens cache_creation '
            'server_tool_use service_tier inference_geo output_tokens_details')
    for key,number in value.items():
        if key in ('input_tokens','output_tokens'):
            _integer(number)
        elif key in ('cache_read_input_tokens','cache_creation_input_tokens'):
            if number is not None and _integer(number)!=0:
                raise ProviderProtocolError()
        elif key in ('cache_creation','server_tool_use') and number is not None:
            allowed = 'ephemeral_5m_input_tokens ephemeral_1h_input_tokens' if key=='cache_creation' else 'web_search_requests web_fetch_requests'
            _object(number,'',allowed)
            if any(_integer(count)!=0 for count in number.values()):
                raise ProviderProtocolError()
        elif key=='output_tokens_details' and number is not None:
            _object(number,'thinking_tokens')
            if _integer(number['thinking_tokens'])>value['output_tokens']:
                raise ProviderProtocolError()
        elif key=='service_tier' and number not in (None,'standard'):
            raise ProviderProtocolError()
        elif key=='inference_geo' and number is not None:
            raise ProviderProtocolError()
    return value


class _Events:
    def __init__(self, body):
        self.model, self.output_limit = body['model'],body['max_tokens']
        self.tools = {tool['name'] for tool in body.get('tools',[])}
        self.started = self.stopped = self.delta_seen = False
        self.block = None
        self.next_index = 0
        self.input_tokens = self.output_tokens = None

    def accept(self, lines):
        if self.stopped:
            raise ProviderProtocolError()
        event, data = None, []
        for line in lines:
            if line.startswith(b':'):
                continue
            key,sep,value = line.partition(b':')
            if not sep:
                raise ProviderProtocolError()
            if value.startswith(b' '):
                value=value[1:]
            if key == b'event' and event is None:
                event=value.decode('utf-8')
            elif key == b'data':
                data.append(value)
            else:
                raise ProviderProtocolError()
        if event is None and not data:
            return None  # SSE comment heartbeat, still counted against byte limits.
        value=json.loads(b'\n'.join(data).decode('utf-8'),object_pairs_hook=_unique_object)
        _plain_json(value)
        if type(value) is not dict or value.get('type')!=event:
            raise ProviderProtocolError()
        if event=='ping':
            _object(value,'type')
        elif event=='message_start':
            _object(value,'type message')
            if self.started:
                raise ProviderProtocolError()
            message=_object(value['message'],'id type role model content usage','stop_reason stop_sequence container stop_details')
            if message['type']!='message' or message['role']!='assistant' or message['model']!=self.model or message['content']!=[]:
                raise ProviderProtocolError()
            _text(message['id'])
            if any(message.get(key) is not None for key in ('stop_reason','stop_sequence','container','stop_details')):
                raise ProviderProtocolError()
            usage=_usage(message['usage'],start=True)
            self.input_tokens,self.output_tokens=usage['input_tokens'],usage['output_tokens']
            self.started=True
        elif event=='content_block_start':
            _object(value,'type index content_block')
            if not self.started or self.delta_seen or self.block is not None or _integer(value['index'])!=self.next_index:
                raise ProviderProtocolError()
            block=value['content_block']
            if type(block) is not dict:
                raise ProviderProtocolError()
            kind=block.get('type')
            if kind=='text':
                _object(block,'type text','citations'); _text(block['text'])
                if block.get('citations') not in (None,[]):
                    raise ProviderProtocolError()
            elif kind=='tool_use':
                _object(block,'type id name input','caller')
                if block.get('caller',{'type':'direct'})!={'type':'direct'}:
                    raise ProviderProtocolError()
                _text(block['id'])
                if block['name'] not in self.tools or type(block['input']) is not dict:
                    raise ProviderProtocolError()
            elif kind=='thinking':
                _object(block,'type thinking signature'); _text(block['thinking']); _text(block['signature'])
            elif kind=='redacted_thinking':
                _object(block,'type data'); _text(block['data'])
            else:
                raise ProviderProtocolError()
            self.block=kind
        elif event=='content_block_delta':
            _object(value,'type index delta')
            if self.block is None or _integer(value['index'])!=self.next_index:
                raise ProviderProtocolError()
            delta=value['delta']
            if type(delta) is not dict:
                raise ProviderProtocolError()
            allowed={'text':{'text_delta':'text'},'tool_use':{'input_json_delta':'partial_json'},
                     'thinking':{'thinking_delta':'thinking','signature_delta':'signature'}}
            field=allowed.get(self.block,{}).get(delta.get('type'))
            if field is None:
                raise ProviderProtocolError()
            _object(delta,'type '+field); _text(delta[field])
        elif event=='content_block_stop':
            _object(value,'type index')
            if self.block is None or _integer(value['index'])!=self.next_index:
                raise ProviderProtocolError()
            self.block=None
            self.next_index+=1
        elif event=='message_delta':
            _object(value,'type delta usage')
            if not self.started or self.block is not None:
                raise ProviderProtocolError()
            delta=_object(value['delta'],'stop_reason','stop_sequence container stop_details')
            if any(delta.get(key) is not None for key in ('container','stop_details')):
                raise ProviderProtocolError()
            if delta['stop_reason'] not in ('end_turn','max_tokens','stop_sequence','tool_use','pause_turn','refusal','model_context_window_exceeded'):
                raise ProviderProtocolError()
            if delta.get('stop_sequence') is not None:
                _text(delta['stop_sequence'])
            usage=_usage(value['usage'],start=False)
            if usage['output_tokens'] < self.output_tokens:
                raise ProviderProtocolError()
            if 'input_tokens' in usage:
                if usage['input_tokens']<self.input_tokens:
                    raise ProviderProtocolError()
                self.input_tokens=usage['input_tokens']
            self.output_tokens=usage['output_tokens']
            self.delta_seen=True
        elif event=='message_stop':
            _object(value,'type')
            if not self.started or not self.delta_seen or self.block is not None:
                raise ProviderProtocolError()
            self.stopped=True
        else:
            raise ProviderProtocolError()  # Includes provider error events; never echo diagnostics.
        if self.output_tokens is not None and self.output_tokens>self.output_limit:
            raise ProviderProtocolError()
        return b'\n'.join(lines)+b'\n\n'


class _SSE:
    def __init__(self,response,body,max_frame,max_body):
        self.status_code=response.status_code
        self.usage=None
        self.response,self.events=response,_Events(body)
        self.max_frame,self.max_body=max_frame,max_body
        self.used=False

    async def __aiter__(self):
        if self.used:
            raise ProviderProtocolError()
        self.used=True
        buffer=b''
        lines=[]
        frame_bytes=total=0
        terminal=None
        try:
            async for chunk in self.response.aiter_raw():
                if len(chunk)>1024*1024:
                    raise ProviderProtocolError()
                total+=len(chunk)
                if total>self.max_body:
                    raise ProviderProtocolError()
                buffer+=chunk
                while match:=_NEWLINE.search(buffer):
                    if match.group()==b'\r' and match.end()==len(buffer):
                        break  # CRLF may straddle HTTP chunks.
                    line,buffer=buffer[:match.start()],buffer[match.end():]
                    frame_bytes+=match.end()
                    if frame_bytes>self.max_frame:
                        raise ProviderProtocolError()
                    if line:
                        lines.append(line)
                    elif lines:
                        frame=self.events.accept(lines)
                        lines=[];frame_bytes=0
                        if frame is not None:
                            if self.events.stopped:
                                terminal=frame
                            else:
                                yield frame
                    else:
                        frame_bytes=0
                if len(buffer)+frame_bytes>self.max_frame:
                    raise ProviderProtocolError()
            if buffer or lines or not self.events.stopped:
                raise ProviderProtocolError()
            self.usage=ProviderUsage(self.events.input_tokens,self.events.output_tokens)
            yield terminal  # Successful terminal publication waits for a clean EOF.
        except (ValueError,TypeError,UnicodeError,RecursionError):
            raise ProviderProtocolError() from None


class AnthropicHTTPTransport:
    def __init__(self,api_key,*,client=None,max_frame_bytes=262144,max_body_bytes=16*1024*1024):
        if type(api_key) is not str or not 1<=len(api_key)<=512 or any(not 33<=ord(char)<=126 for char in api_key):
            raise ValueError('Invalid provider credential configuration.')
        if any(type(n) is not int for n in (max_frame_bytes,max_body_bytes)) or not 128<=max_frame_bytes<=1024*1024 or not max_frame_bytes<=max_body_bytes<=64*1024*1024:
            raise ValueError('Invalid provider response limits.')
        if client is not None and (not isinstance(client,httpx.AsyncClient) or client.trust_env):
            raise ValueError('Injected HTTP client must disable environment configuration.')
        self._key=api_key
        self._owned=client is None
        self._client=client or httpx.AsyncClient(trust_env=False,follow_redirects=False,
                transport=httpx.AsyncHTTPTransport(retries=0,trust_env=False))
        self._max_frame,self._max_body=max_frame_bytes,max_body_bytes

    async def aclose(self):
        if self._owned:
            await self._client.aclose()

    @asynccontextmanager
    async def stream(self,*,url,body,follow_redirects):
        if url!=_ENDPOINT or follow_redirects is not False or type(body) is not dict or body.get('stream') is not True:
            raise ValueError('Unsupported provider transport request.')
        encoded=json.dumps(body,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()
        if len(encoded)>MAX_REQUEST_BYTES:
            raise ValueError('Provider request exceeds configured limits.')
        request=httpx.Request('POST',_ENDPOINT,content=encoded,headers={
            'x-api-key':self._key,'anthropic-version':'2023-06-01','content-type':'application/json',
            'accept':'text/event-stream','accept-encoding':'identity',
        },extensions={'timeout':httpx.Timeout(connect=5,read=30,write=10,pool=5).as_dict()})
        response=None
        try:
            response=await self._client.send(request,stream=True,follow_redirects=False,auth=None)
            if response.status_code!=200 or response.headers.get('content-type','').split(';')[0].strip().lower()!='text/event-stream' or response.headers.get('content-encoding','identity').lower()!='identity':
                raise ProviderProtocolError()
            yield _SSE(response,body,self._max_frame,self._max_body)
        except httpx.HTTPError:
            raise ProviderProtocolError() from None
        finally:
            if response is not None:
                await response.aclose()
