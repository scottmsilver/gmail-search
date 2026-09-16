"""Closed Google generateContent boundary; server-owned model and thinking level.

Reference: https://ai.google.dev/api/generate-content (2026-09-15).
Local function tools and text/thought history only. No external content references,
hosted tools, caches, audio or image generation. Not yet real Pi qualification.
"""
from dataclasses import dataclass

from .inference import (_plain_json, _copy, _object, _array, _text, _name,
                        _integer, _boolean, _schema, _reject, MAX_OUTPUT_TOKENS)
from .provider import AnthropicRunService, ProviderProfile
from .registry import AccessDenied

MODEL = 'gemini-3.8-flash'
LEVELS = ('MINIMAL', 'LOW', 'MEDIUM', 'HIGH')


def endpoint(model):
    if model != MODEL:
        raise AccessDenied()
    return f'https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse'


def _part(part, role, names):
    _object(part, '', 'text thought thoughtSignature functionCall functionResponse')
    kind = set(part) & {'text', 'functionCall', 'functionResponse'}
    if len(kind) != 1:
        _reject()
    if 'thought' in part:
        if role != 'model' or 'text' not in part:
            _reject()
        _boolean(part['thought'])
    if 'thoughtSignature' in part:
        if role != 'model':
            _reject()
        _text(part['thoughtSignature'], 65536, empty=False)
    if 'text' in part:
        _text(part['text'])
    for key, payload in (('functionCall', 'args'), ('functionResponse', 'response')):
        if key not in part:
            continue
        if role != ('model' if key == 'functionCall' else 'user'):
            _reject()
        value = _object(part[key], f'name {payload}', 'id')
        if _name(value['name']) not in names or type(value[payload]) is not dict:
            _reject()
        if 'id' in value:
            _text(value['id'], 256, empty=False)


def compile_gemini_request(body, *, max_output_tokens, thinking_level):
    _plain_json(body)
    if thinking_level not in LEVELS:
        _reject()
    _integer(max_output_tokens, 1, MAX_OUTPUT_TOKENS)
    _object(body, 'contents generationConfig', 'systemInstruction tools toolConfig')
    names = set()
    for tool in _array(body.get('tools', []), 1):
        _object(tool, 'functionDeclarations')
        for declaration in _array(tool['functionDeclarations'], 128, nonempty=True):
            _object(declaration, 'name parametersJsonSchema', 'description')
            name = _name(declaration['name'])
            if name in names:
                _reject()
            names.add(name)
            _schema(declaration['parametersJsonSchema'])
            if 'description' in declaration:
                _text(declaration['description'], 32768)
    for content in _array(body['contents'], 512, nonempty=True):
        _object(content, 'role parts')
        if content['role'] not in ('user', 'model'):
            _reject()
        for part in _array(content['parts'], 2048, nonempty=True):
            _part(part, content['role'], names)
    if 'systemInstruction' in body:
        system = _object(body['systemInstruction'], 'parts', 'role')
        if system.get('role', 'user') not in ('user', 'system'):
            _reject()
        for part in _array(system['parts'], 128, nonempty=True):
            _object(part, 'text')
            _text(part['text'])
    config = _object(body['generationConfig'], 'maxOutputTokens', 'thinkingConfig candidateCount temperature topP topK stopSequences')
    _integer(config['maxOutputTokens'], 1, max_output_tokens)
    if config.get('candidateCount', 1) != 1 or type(config.get('candidateCount', 1)) is not int:
        _reject()
    for key, maximum in (('temperature', 2), ('topP', 1)):
        if key in config and (type(config[key]) not in (int, float) or not 0 <= config[key] <= maximum):
            _reject()
    if 'topK' in config:
        _integer(config['topK'], 1, 100)
    if 'stopSequences' in config:
        for text in _array(config['stopSequences'], 5):
            _text(text, 1024, empty=False)
    if 'thinkingConfig' in config:
        thinking = _object(config['thinkingConfig'], '', 'thinkingLevel includeThoughts')
        if thinking.get('thinkingLevel', 'HIGH') not in LEVELS:
            _reject()
        if 'includeThoughts' in thinking:
            _boolean(thinking['includeThoughts'])
    if 'toolConfig' in body:
        tool_config = _object(body['toolConfig'], 'functionCallingConfig')
        function = _object(tool_config['functionCallingConfig'], 'mode', 'allowedFunctionNames')
        if function['mode'] not in ('AUTO', 'ANY', 'NONE', 'VALIDATED'):
            _reject()
        for name in _array(function.get('allowedFunctionNames', []), 128):
            if _name(name) not in names:
                _reject()
    result = _copy(body)
    result['generationConfig']['thinkingConfig'] = {'thinkingLevel': thinking_level, 'includeThoughts': True}
    return result


@dataclass(frozen=True)
class GeminiProfile:
    model: str
    input_token_limit: int
    output_token_limit: int
    input_units_per_token: int
    output_units_per_token: int
    timeout_seconds: float = 60
    max_response_bytes: int = 16 * 1024 * 1024
    thinking_level: str = 'HIGH'
    family: str = 'google-generate-content'

    def __post_init__(self):
        if self.family != 'google-generate-content' or self.model != MODEL or self.thinking_level not in LEVELS:
            raise AccessDenied()
        # Reuse numeric budget limits only; no Anthropic model/client semantics.
        ProviderProfile(self.model, self.input_token_limit, self.output_token_limit,
                        self.input_units_per_token, self.output_units_per_token,
                        self.timeout_seconds, self.max_response_bytes)


class GeminiRunService(AnthropicRunService):
    profile_type = GeminiProfile

    def _endpoint(self, profile):
        return endpoint(profile.model)

    def _compile(self, body, profile):
        return compile_gemini_request(body, max_output_tokens=profile.output_token_limit,
                                     thinking_level=profile.thinking_level)

    def _output_limit(self, body):
        return body['generationConfig']['maxOutputTokens']
