"""Fixed internal Gemini query embeddings; no guest route or ambient credentials.

The controller owns transport lifetime and injects one shared provider admission.
No retry/cache/cost-log path. See docs/gateway-search-embedding.md for the reviewed
API, preview compatibility boundary and conservative internal budget units.
"""
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
import math

import httpx

from .provider import _finish
from .search_provider import _SearchProvider

ENDPOINT = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:embedContent'
_PREFIX = 'task: search result | query: '
_MAX_RESPONSE = 256 * 1024


class EmbeddingUnavailable(RuntimeError):
    def __init__(self):
        super().__init__('Query embedding is unavailable.')


@dataclass(frozen=True)
class GeminiEmbeddingProfile:
    input_units_per_token: int
    model: str = 'gemini-embedding-2'
    dimensions: int = 3072

    def __post_init__(self):
        if (type(self.model) is not str or self.model != 'gemini-embedding-2' or type(self.dimensions) is not int
                or self.dimensions != 3072 or type(self.input_units_per_token) is not int
                or not 1 <= self.input_units_per_token <= 10**8):
            raise ValueError('Unsupported embedding profile')


def _body(text):
    if type(text) is not str or not text.strip() or len(text)>4096:
        raise ValueError('Invalid embedding query')
    try:
        encoded = (_PREFIX+text).encode('utf-8')
    except UnicodeError:
        raise ValueError('Invalid embedding query') from None
    if len(encoded)>4096 or '\x00' in text:
        raise ValueError('Invalid embedding query')
    return {'model':'models/gemini-embedding-2',
            'content':{'parts':[{'text':_PREFIX+text}]},
            'embedContentConfig':{'outputDimensionality':3072,'autoTruncate':False}}


def _object(pairs):
    result={}
    for key,value in pairs:
        if key in result:
            raise EmbeddingUnavailable()
        result[key]=value
    return result


def _parse(raw):
    # Bound nesting before invoking the recursive JSON decoder, ignoring quoted
    # braces and escaped quotes. Body bytes are bounded before concatenation.
    depth=0; quoted=False; escaped=False
    for byte in raw:
        if quoted:
            if escaped:
                escaped=False
            elif byte==92:
                escaped=True
            elif byte==34:
                quoted=False
        elif byte==34:
            quoted=True
        elif byte in (91,123):
            depth+=1
            if depth>8:
                raise EmbeddingUnavailable()
        elif byte in (93,125):
            depth-=1
    def invalid(_):
        raise EmbeddingUnavailable()
    try:
        value=json.loads(raw,object_pairs_hook=_object,parse_constant=invalid)
        if type(value) is not dict or value.keys()-{'embedding','usageMetadata'}:
            raise EmbeddingUnavailable()
        item=value.get('embedding')
        if type(item) is not dict or set(item)!={'values'}:
            raise EmbeddingUnavailable()
        vector=item['values']
        if (type(vector) is not list or len(vector)!=3072
                or any(type(v) not in (int,float) or not math.isfinite(v) or abs(v)>1e6 for v in vector)
                or not any(vector)):
            raise EmbeddingUnavailable()
        return [float(v) for v in vector],value.get('usageMetadata')
    except (ValueError,TypeError,OverflowError,RecursionError):
        raise EmbeddingUnavailable() from None


def _tokens(usage):
    if type(usage) is not dict or usage.keys()-{'promptTokenCount','promptTokenDetails'}:
        return 8192
    count=usage.get('promptTokenCount')
    if type(count) is not int or not 1<=count<=8192:
        return 8192
    if 'promptTokenDetails' in usage:
        details=usage['promptTokenDetails']
        if (type(details) is not list or len(details)!=1 or type(details[0]) is not dict
                or set(details[0])!={'modality','tokenCount'} or details[0]['modality']!='TEXT'
                or type(details[0]['tokenCount']) is not int or details[0]['tokenCount']!=count):
            return 8192
    return count


class GeminiQueryEmbedder(_SearchProvider):
    profile_type = GeminiEmbeddingProfile
    error_type = EmbeddingUnavailable
    endpoint = ENDPOINT
    max_response_bytes = _MAX_RESPONSE
    request_prefix = 'embedding-'

    @property
    def dimensions(self):
        return self._profile.dimensions

    def _reservation_units(self):
        return 8192*self._profile.input_units_per_token

    def _charge(self, usage):
        return _tokens(usage)*self._profile.input_units_per_token

    def _parse_response(self, raw, body):
        return _parse(raw)

    async def embed(self, lease, text, *, deadline, check_active):
        return await self._call(lease,_body(text),deadline=deadline,check_active=check_active)


class GeminiEmbeddingHTTPTransport:
    """Fixed authenticated HTTP; optional trusted mock transport for qualification."""
    def __init__(self, api_key, *, transport=None):
        if type(api_key) is not str or not api_key or len(api_key)>4096 or not api_key.isascii() or any(ord(c)<33 or ord(c)>126 for c in api_key):
            raise ValueError('Invalid provider credential')
        self._key=api_key
        self._client=httpx.AsyncClient(trust_env=False,follow_redirects=False,
            transport=transport,limits=httpx.Limits(max_connections=4,max_keepalive_connections=4),
            timeout=httpx.Timeout(connect=5,read=30,write=5,pool=1))

    @asynccontextmanager
    async def stream(self, *, url, body, follow_redirects):
        if url!=ENDPOINT or follow_redirects is not False:
            raise EmbeddingUnavailable()
        try:
            query=body['content']['parts'][0]['text']
            if type(query) is not str or not query.startswith(_PREFIX) or body!=_body(query[len(_PREFIX):]):
                raise EmbeddingUnavailable()
            encoded=json.dumps(body,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()
            if len(encoded)>32768:
                raise EmbeddingUnavailable()
            async with self._client.stream('POST',ENDPOINT,content=encoded,headers={
                    'x-goog-api-key':self._key,'Content-Type':'application/json',
                    'Accept':'application/json','Accept-Encoding':'identity'}) as response:
                if (response.status_code!=200 or response.headers.get('content-type','').split(';')[0].strip().lower()!='application/json'
                        or response.headers.get('content-encoding','identity').lower()!='identity'):
                    raise EmbeddingUnavailable()
                class Stream:
                    status_code=response.status_code
                    def __aiter__(self):
                        return response.aiter_raw(chunk_size=65536)
                yield Stream()
        except (httpx.HTTPError,KeyError,IndexError,TypeError,ValueError):
            raise EmbeddingUnavailable() from None

    async def aclose(self):
        await _finish(self._client.aclose())
