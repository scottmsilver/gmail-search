"""Fixed Google HTTP/SSE transport with bounded, validated terminal usage.

https://ai.google.dev/api/generate-content (checked 2026-09-15).
Candidate tokens plus thought tokens form billable output. Unsupported cache/tool
usage, fallback models, missing terminal usage and malformed streams fail closed.
No automatic retries, redirects, credentials in URLs or environment proxies.
"""
from contextlib import asynccontextmanager
import json

import httpx

from .gemini import endpoint, _part
from .inference import MAX_REQUEST_BYTES, _plain_json
from .provider_http import (AnthropicHTTPTransport, ProviderProtocolError, _SSE,
                            _object, _integer, _text, _unique_object)


class _GeminiEvents:
    def __init__(self, body, model):
        self.model = model
        self.limit = body['generationConfig']['maxOutputTokens']
        self.names = {item['name'] for tool in body.get('tools', []) for item in tool['functionDeclarations']}
        self.stopped = False
        self.input_tokens = self.output_tokens = None
        self.response_id = None

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
        value = json.loads(b'\n'.join(data), object_pairs_hook=_unique_object)
        _plain_json(value)
        _object(value, 'modelVersion candidates', 'usageMetadata responseId')
        if value['modelVersion'] != self.model:
            raise ProviderProtocolError()
        if 'responseId' in value:
            current = _text(value['responseId'])
            if self.response_id is not None and self.response_id != current:
                raise ProviderProtocolError()
            self.response_id = current
        candidates = value['candidates']
        if type(candidates) is not list or len(candidates) != 1:
            raise ProviderProtocolError()
        candidate = _object(candidates[0], 'index', 'content finishReason safetyRatings')
        if type(candidate['index']) is not int or candidate['index'] != 0:
            raise ProviderProtocolError()
        if 'content' in candidate:
            content = _object(candidate['content'], 'role parts')
            if content['role'] != 'model' or type(content['parts']) is not list or len(content['parts']) > 2048:
                raise ProviderProtocolError()
            for part in content['parts']:
                _part(part, 'model', self.names)
        if 'safetyRatings' in candidate:
            ratings = candidate['safetyRatings']
            if type(ratings) is not list or len(ratings) > 16:
                raise ProviderProtocolError()
            for rating in ratings:
                _object(rating, 'category probability', 'blocked')
                _text(rating['category'])
                if rating['probability'] not in ('NEGLIGIBLE', 'LOW', 'MEDIUM', 'HIGH', 'HARM_PROBABILITY_UNSPECIFIED') or type(rating.get('blocked', False)) is not bool:
                    raise ProviderProtocolError()
        if 'usageMetadata' in value:
            # A thought-only frame omits candidatesTokenCount; totals must still add up.
            usage = _object(value['usageMetadata'], 'promptTokenCount totalTokenCount',
                            'candidatesTokenCount thoughtsTokenCount cachedContentTokenCount toolUsePromptTokenCount promptTokensDetails candidatesTokensDetails cacheTokensDetails serviceTier')
            # Upstream began reporting the tier (2026-09-18). Only standard is
            # billed at the configured rates; any other tier fails closed.
            if usage.get('serviceTier', 'standard') != 'standard':
                raise ProviderProtocolError()
            inputs = _integer(usage['promptTokenCount'])
            outputs = _integer(usage.get('candidatesTokenCount', 0)) + _integer(usage.get('thoughtsTokenCount', 0))
            if (_integer(usage['totalTokenCount']) != inputs + outputs or outputs > self.limit
                    # Implicit caching is automatic upstream (2026-09-18). Cached
                    # tokens are inside promptTokenCount and cost less, so charging
                    # the whole prompt at the input rate over-counts, never under.
                    or _integer(usage.get('cachedContentTokenCount', 0)) > inputs
                    or _integer(usage.get('toolUsePromptTokenCount', 0)) != 0
                    # Live streams report a larger prompt count on the final frame
                    # after a function call (2026-09-18); the terminal frame settles.
                    or (self.input_tokens is not None and inputs < self.input_tokens)
                    or (self.output_tokens is not None and outputs < self.output_tokens)):
                raise ProviderProtocolError()
            for key in ('promptTokensDetails', 'candidatesTokensDetails', 'cacheTokensDetails'):
                if key in usage:
                    details = usage[key]
                    if type(details) is not list or len(details) > 1:
                        raise ProviderProtocolError()
                    for item in details:
                        _object(item, 'modality tokenCount')
                        if item['modality'] != 'TEXT':
                            raise ProviderProtocolError()
                        _integer(item['tokenCount'])
            self.input_tokens, self.output_tokens = inputs, outputs
        if 'finishReason' in candidate:
            if candidate['finishReason'] not in ('STOP', 'MAX_TOKENS', 'SAFETY', 'RECITATION', 'BLOCKLIST', 'PROHIBITED_CONTENT', 'SPII', 'MALFORMED_FUNCTION_CALL') or 'usageMetadata' not in value:
                raise ProviderProtocolError()
            self.stopped = True
        return b'\n'.join(lines) + b'\n\n'


class _GeminiSSE(_SSE):
    def __init__(self, response, body, model, max_frame, max_body):
        self.status_code = response.status_code
        self.usage = None
        self.response, self.events = response, _GeminiEvents(body, model)
        self.max_frame, self.max_body = max_frame, max_body
        self.used = False


class GeminiHTTPTransport(AnthropicHTTPTransport):
    """Shares only safe HTTP client ownership/limits; all wire logic is Google-specific."""
    def __init__(self, api_key, *, model, **kwargs):
        self._endpoint = endpoint(model)
        self._model = model
        super().__init__(api_key, **kwargs)

    @asynccontextmanager
    async def stream(self, *, url, body, follow_redirects):
        if url != self._endpoint or follow_redirects is not False:
            raise ValueError('Unsupported provider transport request.')
        _plain_json(body)
        encoded = json.dumps(body, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()
        if len(encoded) > MAX_REQUEST_BYTES:
            raise ValueError('Provider request exceeds configured limits.')
        request = httpx.Request('POST', self._endpoint, content=encoded, headers={
            'x-goog-api-key': self._key, 'content-type': 'application/json',
            'accept': 'text/event-stream', 'accept-encoding': 'identity',
        }, extensions={'timeout': httpx.Timeout(connect=5, read=30, write=10, pool=5).as_dict()})
        response = None
        try:
            response = await self._client.send(request, stream=True, follow_redirects=False, auth=None)
            if (response.status_code != 200
                    or response.headers.get('content-type', '').split(';')[0].strip().lower() != 'text/event-stream'
                    or response.headers.get('content-encoding', 'identity').lower() != 'identity'):
                raise ProviderProtocolError()
            yield _GeminiSSE(response, body, self._model, self._max_frame, self._max_body)
        except httpx.HTTPError:
            raise ProviderProtocolError() from None
        finally:
            if response is not None:
                await response.aclose()
