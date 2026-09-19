"""Closed OpenRouter chat-completions boundary; server-owned model, reasoning and cap.

Reference: https://openrouter.ai/docs/api-reference/chat-completion (2026-09-18).
Request shape qualified against Pi 0.84.4's `openai-completions` client. Local
function tools and text/reasoning history only: no images, audio, files,
provider routing, plugins, web search or response formats from the guest.
"""
from dataclasses import dataclass

from .inference import (_plain_json, _copy, _object, _array, _text, _name,
                        _integer, _boolean, _schema, _reject, MAX_MESSAGES, MAX_TOOLS,
                        MAX_OUTPUT_TOKENS)
from .provider import AnthropicRunService, ProviderProfile
from .registry import AccessDenied

ENDPOINT = 'https://openrouter.ai/api/v1/chat/completions'
# Models this boundary may call. The account's zero-data-retention guardrail
# still applies upstream; a model with no compliant endpoint is refused there.
MODELS = frozenset(('anthropic/claude-opus-5', 'meta/muse-spark-1.3'))
EFFORTS = ('low', 'medium', 'high')
_REASONING_FIELDS = ('reasoning', 'reasoning_content', 'reasoning_text')
_DETAIL_TYPES = ('reasoning.text', 'reasoning.summary', 'reasoning.encrypted')


def _text_parts(value):
    """A string, or a list of {"type": "text", "text": ...} parts."""
    if type(value) is str:
        return _text(value)
    for part in _array(value, 2048, nonempty=True):
        _object(part, 'type text', 'cache_control')
        if part['type'] != 'text':
            _reject()
        _text(part['text'])
        if 'cache_control' in part:
            _object(part['cache_control'], 'type', 'ttl')
    return value


def reasoning_detail(detail):
    """One OpenRouter reasoning_details entry (replayed so signed thinking survives)."""
    _object(detail, 'type', 'text summary data signature format index id')
    if detail['type'] not in _DETAIL_TYPES:
        _reject()
    for key in ('text', 'summary', 'data', 'signature', 'format', 'id'):
        if key in detail and detail[key] is not None:
            _text(detail[key])
    if 'index' in detail:
        _integer(detail['index'], 0, 4096)


def _tool_call(call, names):
    _object(call, 'id type function')
    _text(call['id'], 256, empty=False)
    if call['type'] != 'function':
        _reject()
    function = _object(call['function'], 'name arguments')
    if _name(function['name']) not in names:
        _reject()
    _text(function['arguments'])


def _message(message, names):
    role = message.get('role') if type(message) is dict else None
    if role in ('system', 'developer', 'user'):
        _object(message, 'role content')
        _text_parts(message['content'])
    elif role == 'assistant':
        _object(message, 'role', 'content tool_calls reasoning_details ' + ' '.join(_REASONING_FIELDS))
        if message.get('content') is not None:
            _text_parts(message['content'])
        for call in _array(message.get('tool_calls', []), 128):
            _tool_call(call, names)
        for detail in _array(message.get('reasoning_details', []), 256):
            reasoning_detail(detail)
        for key in _REASONING_FIELDS:
            if key in message:
                _text(message[key])
    elif role == 'tool':
        _object(message, 'role tool_call_id content')
        _text(message['tool_call_id'], 256, empty=False)
        _text_parts(message['content'])
    else:
        _reject()


def _tools(value):
    names = set()
    for tool in _array(value, MAX_TOOLS):
        _object(tool, 'type function')
        if tool['type'] != 'function':
            _reject()
        function = _object(tool['function'], 'name parameters', 'description strict')
        name = _name(function['name'])
        if name in names:
            _reject()
        names.add(name)
        _schema(function['parameters'])
        if 'description' in function:
            _text(function['description'], 32768)
        if 'strict' in function:
            _boolean(function['strict'])
    return names


def _tool_choice(value, names):
    if value in ('auto', 'none', 'required'):
        return
    choice = _object(value, 'type function')
    if choice['type'] != 'function' or _name(_object(choice['function'], 'name')['name']) not in names:
        _reject()


def compile_openrouter_request(body, *, model, max_output_tokens, reasoning_effort):
    """Validate a guest body and rebuild it around server-owned choices."""
    _plain_json(body)
    if model not in MODELS or reasoning_effort not in EFFORTS:
        _reject()
    _integer(max_output_tokens, 1, MAX_OUTPUT_TOKENS)
    _object(body, 'model messages stream',
            'max_completion_tokens max_tokens tools tool_choice stream_options store '
            'reasoning_effort temperature top_p parallel_tool_calls')
    if body['stream'] is not True or body.get('store', False) is not False:
        _reject()
    if 'stream_options' in body:
        _object(body['stream_options'], 'include_usage')
    requested = [body[key] for key in ('max_completion_tokens', 'max_tokens') if key in body]
    if len(requested) != 1:
        _reject()
    names = _tools(body.get('tools', []))
    if 'tool_choice' in body:
        _tool_choice(body['tool_choice'], names)
    for message in _array(body['messages'], MAX_MESSAGES, nonempty=True):
        _message(message, names)
    if 'reasoning_effort' in body and body['reasoning_effort'] not in EFFORTS + ('minimal', 'none'):
        _reject()
    for key in ('temperature', 'top_p'):
        if key in body and (type(body[key]) not in (int, float) or not 0 <= body[key] <= 2):
            _reject()
    if 'parallel_tool_calls' in body:
        _boolean(body['parallel_tool_calls'])
    output = {key: _copy(body[key]) for key in ('messages', 'tools', 'tool_choice', 'temperature',
                                                  'top_p', 'parallel_tool_calls') if key in body}
    output.update(model=model, stream=True, stream_options={'include_usage': True},
                  max_tokens=_integer(requested[0], 1, max_output_tokens),
                  reasoning={'effort': reasoning_effort})
    return output


@dataclass(frozen=True)
class OpenRouterProfile:
    model: str
    input_token_limit: int
    output_token_limit: int
    input_units_per_token: int
    output_units_per_token: int
    timeout_seconds: float = 120
    max_response_bytes: int = 16 * 1024 * 1024
    reasoning_effort: str = 'medium'
    family: str = 'openrouter-chat-completions'

    def __post_init__(self):
        if (self.family != 'openrouter-chat-completions' or self.model not in MODELS
                or self.reasoning_effort not in EFFORTS):
            raise AccessDenied()
        # Reuse numeric budget limits only; no Anthropic model/client semantics.
        ProviderProfile(self.model, self.input_token_limit, self.output_token_limit,
                        self.input_units_per_token, self.output_units_per_token,
                        self.timeout_seconds, self.max_response_bytes)


class OpenRouterRunService(AnthropicRunService):
    profile_type = OpenRouterProfile

    def _endpoint(self, profile):
        return ENDPOINT

    def _compile(self, body, profile):
        return compile_openrouter_request(body, model=profile.model, max_output_tokens=profile.output_token_limit,
                                          reasoning_effort=profile.reasoning_effort)

    def _output_limit(self, body):
        return body['max_tokens']
