"""Fixed OpenRouter HTTP/SSE transport with bounded, validated terminal usage.

Stream shape observed live 2026-09-18 (anthropic/claude-opus-5): `: OPENROUTER
PROCESSING` comments, chat.completion.chunk frames, the finish reason repeated on
the usage frame, then `data: [DONE]`. Usage must arrive before [DONE] and settles
at prompt x input rate + completion (reasoning included) x output rate.
No automatic retries, redirects, credentials in URLs or environment proxies.
"""
from contextlib import asynccontextmanager
import json

import httpx

from .inference import MAX_REQUEST_BYTES, _plain_json
from .openrouter import ENDPOINT, reasoning_detail
from .provider_http import (AnthropicHTTPTransport, ProviderProtocolError, _SSE,
                            _object, _integer, _text, _unique_object)

_FINISH = ('stop', 'length', 'tool_calls', 'content_filter')
_TIERS = (None, 'default', 'standard')


def _optional_text(value):
    if value is not None:
        _text(value)


def _zero(details, *keys):
    for key in keys:
        if _integer(details.get(key, 0) or 0) != 0:
            raise ProviderProtocolError()


class _OpenRouterEvents:
    def __init__(self, body):
        self.model, self.limit = body['model'], body['max_tokens']
        self.names = {tool['function']['name'] for tool in body.get('tools', [])}
        self.stopped = False
        self.response_id = self.finish = None
        self.input_tokens = self.output_tokens = None

    def accept(self, lines):
        if self.stopped:
            raise ProviderProtocolError()
        data = []
        for line in lines:
            if line.startswith(b':'):
                continue
            if not line.startswith(b'data:'):
                raise ProviderProtocolError()
            data.append(line[5:].lstrip(b' '))
        if not data:
            return None
        raw = b'\n'.join(data)
        if raw == b'[DONE]':
            # Terminal only after a finish reason and usage; publication waits for EOF.
            if self.finish is None or self.input_tokens is None:
                raise ProviderProtocolError()
            self.stopped = True
            return b'\n'.join(lines) + b'\n\n'
        value = json.loads(raw, object_pairs_hook=_unique_object)
        _plain_json(value)
        _object(value, 'id object created model choices', 'provider usage system_fingerprint service_tier')
        if value['object'] != 'chat.completion.chunk' or value['model'] != self.model:
            raise ProviderProtocolError()
        if self.response_id not in (None, _text(value['id'])):
            raise ProviderProtocolError()
        self.response_id = value['id']
        if value.get('service_tier') not in _TIERS:
            raise ProviderProtocolError()
        if type(value['created']) is not int or value['created'] <= 0:  # a Unix time, past 10**9
            raise ProviderProtocolError()
        for key in ('provider', 'system_fingerprint'):
            _optional_text(value.get(key))
        self._choices(value['choices'])
        if 'usage' in value:
            self._usage(value['usage'])
        return b'\n'.join(lines) + b'\n\n'

    def _choices(self, choices):
        if type(choices) is not list or len(choices) > 1:
            raise ProviderProtocolError()
        for choice in choices:
            _object(choice, 'index delta', 'finish_reason native_finish_reason logprobs')
            if choice['index'] != 0 or choice.get('logprobs') is not None:
                raise ProviderProtocolError()
            self._delta(choice['delta'])
            finish = choice.get('finish_reason')
            if finish is not None:
                if finish not in _FINISH or self.finish not in (None, finish):
                    raise ProviderProtocolError()
                self.finish = finish
            _optional_text(choice.get('native_finish_reason'))

    def _delta(self, delta):
        _object(delta, '', 'role content tool_calls reasoning reasoning_details refusal')
        if delta.get('role') not in (None, 'assistant'):
            raise ProviderProtocolError()
        for key in ('content', 'reasoning', 'refusal'):
            _optional_text(delta.get(key))
        for detail in delta.get('reasoning_details') or []:
            reasoning_detail(detail)
        for call in delta.get('tool_calls') or []:
            _object(call, 'index', 'id type function')
            _integer(call['index'])
            if call.get('type', 'function') != 'function':
                raise ProviderProtocolError()
            _optional_text(call.get('id'))
            function = _object(call.get('function', {}), '', 'name arguments')
            if 'name' in function and function['name'] not in self.names:
                raise ProviderProtocolError()
            _optional_text(function.get('arguments'))

    def _usage(self, usage):
        if self.input_tokens is not None:
            raise ProviderProtocolError()
        _object(usage, 'prompt_tokens completion_tokens total_tokens',
                'cost is_byok prompt_tokens_details completion_tokens_details cost_details server_tool_use')
        inputs, outputs = _integer(usage['prompt_tokens']), _integer(usage['completion_tokens'])
        if _integer(usage['total_tokens']) != inputs + outputs or outputs > self.limit:
            raise ProviderProtocolError()
        prompt = usage.get('prompt_tokens_details') or {}
        _object(prompt, '', 'cached_tokens cache_write_tokens audio_tokens video_tokens')
        if _integer(prompt.get('cached_tokens', 0) or 0) + _integer(prompt.get('cache_write_tokens', 0) or 0) > inputs:
            raise ProviderProtocolError()
        _zero(prompt, 'audio_tokens', 'video_tokens')
        completion = usage.get('completion_tokens_details') or {}
        _object(completion, '', 'reasoning_tokens image_tokens audio_tokens')
        if _integer(completion.get('reasoning_tokens', 0) or 0) > outputs:
            raise ProviderProtocolError()
        _zero(completion, 'image_tokens', 'audio_tokens')
        # Hosted tools (web search) are never requested, so never billable.
        if usage.get('server_tool_use'):
            raise ProviderProtocolError()
        self.input_tokens, self.output_tokens = inputs, outputs


class _OpenRouterSSE(_SSE):
    def __init__(self, response, body, max_frame, max_body):
        self.status_code = response.status_code
        self.usage = None
        self.response, self.events = response, _OpenRouterEvents(body)
        self.max_frame, self.max_body = max_frame, max_body
        self.used = False


class OpenRouterHTTPTransport(AnthropicHTTPTransport):
    """Shares only safe HTTP client ownership/limits; all wire logic is OpenRouter's."""

    @asynccontextmanager
    async def stream(self, *, url, body, follow_redirects):
        if url != ENDPOINT or follow_redirects is not False or type(body) is not dict or body.get('stream') is not True:
            raise ValueError('Unsupported provider transport request.')
        _plain_json(body)
        encoded = json.dumps(body, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()
        if len(encoded) > MAX_REQUEST_BYTES:
            raise ValueError('Provider request exceeds configured limits.')
        request = httpx.Request('POST', ENDPOINT, content=encoded, headers={
            'authorization': 'Bearer ' + self._key, 'content-type': 'application/json',
            'accept': 'text/event-stream', 'accept-encoding': 'identity',
        }, extensions={'timeout': httpx.Timeout(connect=5, read=60, write=10, pool=5).as_dict()})
        response = None
        try:
            response = await self._client.send(request, stream=True, follow_redirects=False, auth=None)
            if (response.status_code != 200
                    or response.headers.get('content-type', '').split(';')[0].strip().lower() != 'text/event-stream'
                    or response.headers.get('content-encoding', 'identity').lower() != 'identity'):
                raise ProviderProtocolError()
            yield _OpenRouterSSE(response, body, self._max_frame, self._max_body)
        except httpx.HTTPError:
            raise ProviderProtocolError() from None
        finally:
            if response is not None:
                await response.aclose()
