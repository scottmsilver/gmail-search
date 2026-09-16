"""Closed Anthropic request boundary. This module never sends provider requests.

Generation and token counting have separate entry points selected by trusted
transport, never by a body field. The submitted model must equal server_model;
max_tokens must be positive and within the server's reserved output allowance.

Supported: text, bounded base64 image bytes, custom client tools, local tool
results, and original thinking/redacted-thinking history. Text and custom tool
inputs may contain inert URLs; no accepted API field requests a remote fetch.
Image validation checks encoding and MIME magic without running image codecs.

Unsupported: provider tools, documents/files, caches, citations, metadata,
output_config, context_management, beta fields, tool search/deferred loading,
provider accounts/containers/MCP and all unlisted fields. Native Claude/Pi
compatibility is NOT demonstrated by these validators. Claude gateway headers,
transport, streaming, pricing and model-specific limits need separate validation.

Reference contracts checked 2026-09-15:
https://platform.claude.com/docs/en/api/http/messages/create
https://platform.claude.com/docs/en/api/http/messages/count_tokens
https://code.claude.com/docs/en/llm-gateway
https://github.com/anthropics/anthropic-sdk-python/tree/main/src/anthropic/types
"""
import base64
import binascii
import json
import math
import re

MAX_REQUEST_BYTES = 8 * 1024 * 1024
MAX_JSON_DEPTH = 32
MAX_JSON_NODES = 20_000
MAX_MESSAGES = 512
MAX_CONTENT_BLOCKS = 2048
MAX_TEXT_BYTES = 1024 * 1024
MAX_IMAGES = 8
MAX_IMAGE_BYTES = 2 * 1024 * 1024
MAX_TOTAL_IMAGE_BYTES = 4 * 1024 * 1024
MAX_TOOLS = 128
MAX_OUTPUT_TOKENS = 131_072
_NAME = re.compile(r'[a-zA-Z0-9_-]{1,128}\Z', re.ASCII)
_MODEL = re.compile(r'[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,127}\Z', re.ASCII)


class InferenceRequestRejected(ValueError):
    def __init__(self):
        super().__init__('Inference request uses unsupported fields or exceeds configured limits.')


def _reject():
    raise InferenceRequestRejected()


def _object(value, required, optional=''):
    if type(value) is not dict or not set(required.split()) <= value.keys() or value.keys() - set((required + ' ' + optional).split()):
        _reject()
    return value


def _text(value, maximum=MAX_TEXT_BYTES, *, empty=True):
    if type(value) is not str or (not empty and not value) or len(value.encode('utf-8')) > maximum:
        _reject()
    return value


def _name(value):
    if type(value) is not str or not _NAME.fullmatch(value):
        _reject()
    return value


def _integer(value, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        _reject()
    return value


def _boolean(value):
    if type(value) is not bool:
        _reject()
    return value


def _array(value, maximum, *, nonempty=False):
    if type(value) is not list or not int(nonempty) <= len(value) <= maximum:
        _reject()
    return value


def _plain_json(body):
    """Bound shape before serialization; never invoke arbitrary SDK serializers."""
    pending, containers, nodes, string_bytes = [(body, 0)], set(), 0, 0
    while pending:
        value, depth = pending.pop()
        nodes += 1
        if nodes > MAX_JSON_NODES or depth > MAX_JSON_DEPTH:
            _reject()
        kind = type(value)
        if kind in (dict, list):
            if id(value) in containers:
                _reject()  # Parsed JSON is a tree, not a cycle or shared Python object.
            containers.add(id(value))
            if len(value) > MAX_JSON_NODES:
                _reject()
            if kind is dict:
                for key, item in value.items():
                    if type(key) is not str:
                        _reject()
                    string_bytes += len(key.encode('utf-8'))
                    pending.append((item, depth + 1))
            else:
                pending.extend((item, depth + 1) for item in value)
        elif kind is str:
            string_bytes += len(value.encode('utf-8'))
        elif kind is float:
            if not math.isfinite(value):
                _reject()
        elif kind is int:
            if not -(2**63) <= value < 2**63:
                _reject()
        elif kind not in (bool, type(None)):
            _reject()
        if string_bytes > MAX_REQUEST_BYTES:
            _reject()
    # Escaping can increase the wire size beyond raw UTF-8 string lengths.
    encoded = json.dumps(body, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    if len(encoded.encode('utf-8')) > MAX_REQUEST_BYTES:
        _reject()


def _copy(value):
    if type(value) is dict:
        return {key: _copy(item) for key, item in value.items()}
    if type(value) is list:
        return [_copy(item) for item in value]
    return value


def _schema(value):
    """Explicit JSON Schema subset; references remain inside the supplied schema."""
    _object(value, '', 'type properties required additionalProperties items enum const default description title $defs definitions $ref anyOf oneOf allOf not minLength maxLength minItems maxItems minimum maximum exclusiveMinimum exclusiveMaximum multipleOf')
    output = {}
    for key, item in value.items():
        if key == 'type':
            choices = [item] if type(item) is str else _array(item, 7, nonempty=True)
            if any(choice not in ('object', 'array', 'string', 'number', 'integer', 'boolean', 'null') for choice in choices):
                _reject()
            output[key] = _copy(item)
        elif key in ('properties', '$defs', 'definitions'):
            if type(item) is not dict or len(item) > 128:
                _reject()
            output[key] = {_text(name, 256, empty=False): _schema(child) for name, child in item.items()}
        elif key == 'required':
            output[key] = [_text(name, 256, empty=False) for name in _array(item, 128)]
            if len(set(output[key])) != len(output[key]):
                _reject()
        elif key in ('items', 'not'):
            output[key] = _schema(item)
        elif key == 'additionalProperties':
            output[key] = item if type(item) is bool else _schema(item)
        elif key in ('anyOf', 'oneOf', 'allOf'):
            output[key] = [_schema(child) for child in _array(item, 16, nonempty=True)]
        elif key == '$ref':
            if type(item) is not str or not re.fullmatch(r'#/(?:\$defs|definitions)/[A-Za-z0-9_-]{1,128}', item):
                _reject()
            output[key] = item
        elif key in ('description', 'title'):
            output[key] = _text(item, 32_768)
        elif key == 'enum':
            output[key] = _copy(_array(item, 256, nonempty=True))
        elif key in ('const', 'default'):
            output[key] = _copy(item)  # Inert JSON values, not recursively resolved schemas.
        elif key in ('minLength', 'maxLength', 'minItems', 'maxItems'):
            output[key] = _integer(item, 0, 1_000_000)
        else:
            if type(item) not in (int, float) or not math.isfinite(item) or abs(item) > 10**12 or (key == 'multipleOf' and item <= 0):
                _reject()
            output[key] = item
    return output


class _Normalizer:
    def __init__(self):
        self.blocks = self.images = self.image_bytes = 0

    def content(self, value, role, *, tool_result=False):
        if type(value) is str:
            value = [{'type': 'text', 'text': value}]
        return [self.block(block, role, tool_result=tool_result) for block in _array(value, MAX_CONTENT_BLOCKS)]

    def block(self, value, role, *, tool_result=False):
        if type(value) is not dict:
            _reject()
        self.blocks += 1
        if self.blocks > MAX_CONTENT_BLOCKS:
            _reject()
        kind = value.get('type')
        if kind == 'text':
            _object(value, 'type text')
            return {'type': kind, 'text': _text(value['text'])}
        if kind == 'image' and role == 'user':
            _object(value, 'type source')
            source = _object(value['source'], 'type media_type data')
            if source['type'] != 'base64' or source['media_type'] not in ('image/png', 'image/jpeg', 'image/gif', 'image/webp'):
                _reject()
            data = _text(source['data'], 4 * ((MAX_IMAGE_BYTES + 2) // 3), empty=False)
            decoded = base64.b64decode(data, validate=True)
            self.images += 1
            self.image_bytes += len(decoded)
            if self.images > MAX_IMAGES or len(decoded) > MAX_IMAGE_BYTES or self.image_bytes > MAX_TOTAL_IMAGE_BYTES:
                _reject()
            signatures = {
                'image/png': decoded.startswith(b'\x89PNG\r\n\x1a\n'),
                'image/jpeg': decoded.startswith(b'\xff\xd8\xff'),
                'image/gif': decoded.startswith((b'GIF87a', b'GIF89a')),
                'image/webp': decoded.startswith(b'RIFF') and decoded[8:12] == b'WEBP',
            }
            if not signatures[source['media_type']]:
                _reject()
            return {'type': kind, 'source': {'type': 'base64', 'media_type': source['media_type'], 'data': base64.b64encode(decoded).decode('ascii')}}
        if kind == 'tool_use' and role == 'assistant' and not tool_result:
            _object(value, 'type id name input', 'caller')
            if type(value['input']) is not dict:
                _reject()
            output = {'type': kind, 'id': _name(value['id']), 'name': _name(value['name']), 'input': _copy(value['input'])}
            if 'caller' in value:
                _object(value['caller'], 'type')
                if value['caller']['type'] != 'direct':
                    _reject()
                output['caller'] = {'type': 'direct'}
            return output
        if kind == 'tool_result' and role == 'user' and not tool_result:
            _object(value, 'type tool_use_id', 'content is_error')
            output = {'type': kind, 'tool_use_id': _name(value['tool_use_id'])}
            if 'content' in value:
                output['content'] = self.content(value['content'], role, tool_result=True)
            if 'is_error' in value:
                output['is_error'] = _boolean(value['is_error'])
            return output
        if kind == 'thinking' and role == 'assistant' and not tool_result:
            _object(value, 'type thinking signature')
            return {'type': kind, 'thinking': _text(value['thinking']), 'signature': _text(value['signature'], 262_144, empty=False)}
        if kind == 'redacted_thinking' and role == 'assistant' and not tool_result:
            _object(value, 'type data')
            return {'type': kind, 'data': _text(value['data'], 262_144, empty=False)}
        _reject()

    def tools(self, values):
        output, names = [], set()
        for tool in _array(values, MAX_TOOLS):
            _object(tool, 'name input_schema', 'description type strict allowed_callers')
            name = _name(tool['name'])
            if name in names or tool.get('type', 'custom') != 'custom' or tool.get('allowed_callers', ['direct']) != ['direct']:
                _reject()
            names.add(name)
            if type(tool['input_schema']) is not dict or tool['input_schema'].get('type') != 'object':
                _reject()
            normalized = {'name': name, 'input_schema': _schema(tool['input_schema'])}
            if 'description' in tool:
                normalized['description'] = _text(tool['description'], 32_768)
            if 'type' in tool:
                normalized['type'] = 'custom'
            if 'strict' in tool:
                normalized['strict'] = _boolean(tool['strict'])
            if 'allowed_callers' in tool:
                normalized['allowed_callers'] = ['direct']
            output.append(normalized)
        return output, names


def _compile(body, server_model, max_output_tokens, *, count):
    _plain_json(body)
    if type(server_model) is not str or not _MODEL.fullmatch(server_model):
        _reject()
    _integer(max_output_tokens, 1, MAX_OUTPUT_TOKENS)
    common = 'system tools tool_choice thinking'
    _object(body, 'model messages' if count else 'model messages max_tokens', common if count else common + ' stream stop_sequences temperature top_p top_k')
    if body['model'] != server_model:
        _reject()
    output = {'model': server_model}
    allowance = max_output_tokens
    if not count:
        allowance = output['max_tokens'] = _integer(body['max_tokens'], 1, max_output_tokens)
    normalizer = _Normalizer()
    if 'system' in body:
        output['system'] = normalizer.content(body['system'], 'system')
    output['messages'] = []
    for message in _array(body['messages'], MAX_MESSAGES, nonempty=True):
        _object(message, 'role content')
        if message['role'] not in ('user', 'assistant'):
            _reject()
        output['messages'].append({'role': message['role'], 'content': normalizer.content(message['content'], message['role'])})
    names = set()
    if 'tools' in body:
        output['tools'], names = normalizer.tools(body['tools'])
    if 'tool_choice' in body:
        choice = _object(body['tool_choice'], 'type', 'name disable_parallel_tool_use')
        if choice['type'] not in ('auto', 'any', 'tool', 'none') or (choice['type'] != 'none' and not names):
            _reject()
        output['tool_choice'] = {'type': choice['type']}
        if choice['type'] == 'tool':
            if choice.get('name') not in names:
                _reject()
            output['tool_choice']['name'] = choice['name']
        elif 'name' in choice:
            _reject()
        if 'disable_parallel_tool_use' in choice:
            output['tool_choice']['disable_parallel_tool_use'] = _boolean(choice['disable_parallel_tool_use'])
    if 'thinking' in body:
        thinking = _object(body['thinking'], 'type', 'budget_tokens display')
        kind = thinking['type']
        if kind not in ('enabled', 'disabled', 'adaptive'):
            _reject()
        output['thinking'] = {'type': kind}
        if kind == 'enabled':
            output['thinking']['budget_tokens'] = _integer(thinking.get('budget_tokens'), 1024, allowance - 1)
        elif 'budget_tokens' in thinking:
            _reject()
        if 'display' in thinking:
            if kind == 'disabled' or thinking['display'] not in ('summarized', 'omitted'):
                _reject()
            output['thinking']['display'] = thinking['display']
    if 'stream' in body:
        output['stream'] = _boolean(body['stream'])
    for key in ('temperature', 'top_p'):
        if key in body:
            if type(body[key]) not in (int, float) or not 0 <= body[key] <= 1:
                _reject()
            output[key] = body[key]
    if 'top_k' in body:
        output['top_k'] = _integer(body['top_k'], 1, 1000)
    if 'stop_sequences' in body:
        output['stop_sequences'] = [_text(item, 1024, empty=False) for item in _array(body['stop_sequences'], 16)]
    _plain_json(output)  # Text shorthand expansion also counts toward wire limits.
    return output


def validate_server_effort(server_model, server_effort):
    """Closed model capability table, checked against official effort docs 2026-09-15.

    https://platform.claude.com/docs/en/build-with-claude/effort
    Other models require separate qualification before enabling this control.
    """
    if server_effort is not None and (server_model != 'claude-sonnet-4-6'
            or type(server_effort) is not str or server_effort not in ('low','medium','high','max')):
        _reject()


def compile_anthropic_request(body, *, server_model, max_output_tokens, server_effort=None):
    """Compile a /v1/messages body. No transport, headers or credentials accepted."""
    try:
        validate_server_effort(server_model, server_effort)
        output = _compile(body, server_model, max_output_tokens, count=False)
        if server_effort is not None:
            output['output_config'] = {'effort': server_effort}
        return output
    except (TypeError, ValueError, RecursionError, OverflowError, binascii.Error):
        raise InferenceRequestRejected() from None


def compile_anthropic_count_tokens_request(body, *, server_model, max_output_tokens):
    """Compile /v1/messages/count_tokens; generation-only controls are rejected."""
    try:
        return _compile(body, server_model, max_output_tokens, count=True)
    except (TypeError, ValueError, RecursionError, OverflowError, binascii.Error):
        raise InferenceRequestRejected() from None
