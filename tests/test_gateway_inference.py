"""Provider request boundary: inert conversation data and local tools only."""
import base64
import copy
import json
import math

import pytest

from gmail_search.gateway.inference import (
    InferenceRequestRejected, compile_anthropic_request,
    compile_anthropic_count_tokens_request,
)

MODEL = 'claude-test-server-config'
PNG = base64.b64encode(b'\x89PNG\r\n\x1a\n' + b'example').decode()


def request():
    return {'model': MODEL, 'max_tokens': 4096, 'messages': [{'role': 'user', 'content': 'Summarize my mail.'}]}


def compile(body):
    return compile_anthropic_request(body, server_model=MODEL, max_output_tokens=4096)


def tool():
    return {'name': 'read_file', 'description': 'Read a file inside the guest.', 'input_schema': {'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path'], 'additionalProperties': False}}


def image():
    return {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': PNG}}


def test_text_is_copied_and_model_is_server_selected():
    body = request()
    result = compile(body)
    assert result['model'] == MODEL
    assert result['max_tokens'] == 4096
    assert result['messages'] == [{'role': 'user', 'content': [{'type': 'text', 'text': 'Summarize my mail.'}]}]
    body['messages'][0]['content'] = 'changed'
    assert result['messages'][0]['content'][0]['text'] == 'Summarize my mail.'


def test_local_tool_roundtrip_and_thinking_are_preserved_without_aliases():
    body = request()
    body['messages'][0]['content'] = [{'type': 'text', 'text': 'Summarize my mail.'}]
    body.update({'system': [{'type': 'text', 'text': 'Use local tools.'}], 'stream': True, 'tools': [tool()], 'tool_choice': {'type': 'auto', 'disable_parallel_tool_use': True}, 'thinking': {'type': 'enabled', 'budget_tokens': 1024}})
    body['messages'].extend([
        {'role': 'assistant', 'content': [{'type': 'thinking', 'thinking': 'Inspect the file.', 'signature': 'provider-signature'}, {'type': 'tool_use', 'id': 'toolu_example', 'name': 'read_file', 'input': {'path': 'report.txt'}}]},
        {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'toolu_example', 'content': [{'type': 'text', 'text': 'Report'}, image()], 'is_error': False}]},
    ])
    result = compile(body)
    assert result == body
    body['tools'][0]['input_schema']['properties']['path']['type'] = 'number'
    assert result['tools'][0]['input_schema']['properties']['path']['type'] == 'string'


def test_custom_tools_can_contain_inert_urls_as_data():
    body = request()
    body['messages'][0]['content'] = 'This email mentions https://example.invalid/private.'
    body['tools'] = [tool()]
    body['messages'].append({'role': 'assistant', 'content': [{'type': 'tool_use', 'id': 'toolu_one', 'name': 'read_file', 'input': {'path': 'https://example.invalid/private'}}]})
    assert compile(body)['messages'][1]['content'][0]['input']['path'].startswith('https://')


@pytest.mark.parametrize('patch', [
    {'model': 'other-model'}, {'max_tokens': 4097}, {'max_tokens': 0}, {'max_tokens': True},
    {'stream': 'yes'}, {'temperature': math.nan}, {'temperature': 2},
    {'container': 'private-provider-container'}, {'mcp_servers': [{'url': 'https://evil.invalid'}]},
    {'metadata': {'user_id': 'foreign-user'}}, {'cache_control': {'type': 'ephemeral'}},
    {'anthropic_beta': ['unsafe']}, {'api_key': 'SECRET'}, {'base_url': 'https://evil.invalid'},
    {'output_config': {'format': {'type': 'json_schema', 'schema': {}}}},
    {'context_management': {'edits': []}}, {'service_tier': 'auto'}, {'extra_body': {}},
])
def test_rejects_unknown_fields_model_override_and_invalid_controls(patch):
    body = request()
    body.update(patch)
    with pytest.raises(InferenceRequestRejected) as caught:
        compile(body)
    assert 'SECRET' not in str(caught.value)
    assert 'evil.invalid' not in str(caught.value)


@pytest.mark.parametrize('block', [
    {'type': 'image', 'source': {'type': 'url', 'url': 'https://evil.invalid/mail'}},
    {'type': 'image', 'source': {'type': 'base64', 'data': PNG, 'media_type': 'image/png', 'url': 'https://evil.invalid'}},
    {'type': 'document', 'source': {'type': 'url', 'url': 'https://evil.invalid/mail.pdf'}},
    {'type': 'document', 'source': {'type': 'file', 'file_id': 'file_foreign'}},
    {'type': 'document', 'source': {'type': 'base64', 'media_type': 'application/pdf', 'data': 'JVBERg=='}},
    {'type': 'server_tool_use', 'id': 'srvtoolu_one', 'name': 'web_search', 'input': {'query': 'private mail'}},
    {'type': 'web_search_tool_result', 'tool_use_id': 'srvtoolu_one', 'content': []},
    {'type': 'container_upload', 'file_id': 'file_foreign'},
    {'type': 'text', 'text': 'hello', 'cache_control': {'type': 'ephemeral'}},
    {'type': 'text', 'text': 'hello', 'citations': [{'url': 'https://evil.invalid'}]},
    {'type': 'tool_result', 'tool_use_id': 'toolu_one', 'content': [{'type': 'image', 'source': {'type': 'url', 'url': 'https://evil.invalid/nested'}}]},
    {'type': 'tool_result', 'tool_use_id': 'toolu_one', 'content': [{'type': 'tool_result', 'tool_use_id': 'toolu_two', 'content': 'nested'}]},
])
def test_rejects_unsafe_content_recursively(block):
    body = request()
    body['messages'][0]['content'] = [block]
    with pytest.raises(InferenceRequestRejected):
        compile(body)


@pytest.mark.parametrize('definition', [
    {'type': 'web_search_20250305', 'name': 'web_search'},
    {'type': 'code_execution_20250522', 'name': 'code_execution'},
    {**tool(), 'allowed_callers': ['code_execution_20250825']},
    {**tool(), 'defer_loading': True},
    {**tool(), 'cache_control': {'type': 'ephemeral'}},
    {**tool(), 'input_schema': {'type': 'object', 'properties': {'x': {'$ref': 'https://evil.invalid/schema'}}}},
    {**tool(), 'input_schema': {'type': 'object', '$schema': 'https://evil.invalid/schema'}},
])
def test_rejects_hosted_tools_and_remote_schema_references(definition):
    body = request()
    body['tools'] = [definition]
    with pytest.raises(InferenceRequestRejected):
        compile(body)


def test_token_count_has_distinct_closed_request_schema():
    body = request()
    body.pop('max_tokens')
    result = compile_anthropic_count_tokens_request(body, server_model=MODEL, max_output_tokens=4096)
    assert 'max_tokens' not in result
    assert result['model'] == MODEL
    with pytest.raises(InferenceRequestRejected):
        compile(body)
    body['stream'] = True
    with pytest.raises(InferenceRequestRejected):
        compile_anthropic_count_tokens_request(body, server_model=MODEL, max_output_tokens=4096)


@pytest.mark.parametrize('data,media_type', [('not-base64!', 'image/png'), (PNG, 'image/jpeg'), (PNG, 'image/svg+xml'), ('', 'image/png')])
def test_image_format_and_magic_are_validated(data, media_type):
    body = request()
    body['messages'][0]['content'] = [{'type': 'image', 'source': {'type': 'base64', 'media_type': media_type, 'data': data}}]
    with pytest.raises(InferenceRequestRejected):
        compile(body)


@pytest.mark.parametrize('mutate', [
    lambda body: body.update({'messages': [{'role': 'system', 'content': 'bad'}]}),
    lambda body: body.update({'thinking': {'type': 'enabled', 'budget_tokens': 4096}}),
    lambda body: body.update({'thinking': {'type': 'enabled', 'budget_tokens': 1}}),
    lambda body: body.update({'messages': [{'role': 'user', 'content': [{'type': 'thinking', 'thinking': 'bad', 'signature': 'x'}]}]}),
    lambda body: body.update({'tools': [tool(), tool()]}),
    lambda body: body.update({'tool_choice': {'type': 'tool', 'name': 'unknown'}}),
])
def test_invalid_shapes_and_tool_or_thinking_controls(mutate):
    body = request()
    mutate(body)
    with pytest.raises(InferenceRequestRejected):
        compile(body)


def test_non_json_cycles_and_excessive_depth_are_rejected():
    for bad in (object(), {'x'}, math.inf):
        body = request()
        body['messages'][0]['content'] = bad
        with pytest.raises(InferenceRequestRejected):
            compile(body)
    cyclic = request()
    cyclic['messages'].append(cyclic)
    with pytest.raises(InferenceRequestRejected):
        compile(cyclic)
    body = request()
    nested = {}
    for _ in range(40):
        nested = {'x': nested}
    body['tools'] = [{**tool(), 'input_schema': nested}]
    with pytest.raises(InferenceRequestRejected):
        compile(body)


def test_request_content_and_image_counts_are_bounded():
    from gmail_search.gateway import inference
    cases = []
    body = request()
    body['messages'][0]['content'] = 'x' * (inference.MAX_TEXT_BYTES + 1)
    cases.append(body)
    body = request()
    body['messages'][0]['content'] = [image() for _ in range(inference.MAX_IMAGES + 1)]
    cases.append(body)
    body = request()
    body['messages'] *= inference.MAX_MESSAGES + 1
    cases.append(body)
    for body in cases:
        with pytest.raises(InferenceRequestRejected):
            compile(copy.deepcopy(body))


def test_normalized_wire_size_is_also_bounded(monkeypatch):
    from gmail_search.gateway import inference
    body = request()
    raw_size = len(json.dumps(body, separators=(',', ':')).encode())
    monkeypatch.setattr(inference, 'MAX_REQUEST_BYTES', raw_size + 5)
    with pytest.raises(InferenceRequestRejected):
        compile(body)


def test_redacted_thinking_and_direct_custom_tools():
    body = request()
    body['tools'] = [{**tool(), 'type': 'custom', 'allowed_callers': ['direct'], 'strict': True}]
    body['thinking'] = {'type': 'adaptive', 'display': 'omitted'}
    body['messages'].append({'role': 'assistant', 'content': [
        {'type': 'redacted_thinking', 'data': 'opaque-provider-data'},
        {'type': 'tool_use', 'id': 'toolu_direct', 'name': 'read_file', 'input': {'path': 'a'}, 'caller': {'type': 'direct'}},
    ]})
    result = compile(body)
    assert result['messages'][-1] == body['messages'][-1]
    assert result['tools'] == body['tools']


def test_total_decoded_images_and_per_image_bytes_are_bounded(monkeypatch):
    from gmail_search.gateway import inference
    body = request()
    body['messages'][0]['content'] = [image(), image()]
    monkeypatch.setattr(inference, 'MAX_TOTAL_IMAGE_BYTES', 20)
    with pytest.raises(InferenceRequestRejected):
        compile(body)
    monkeypatch.setattr(inference, 'MAX_TOTAL_IMAGE_BYTES', 100)
    monkeypatch.setattr(inference, 'MAX_IMAGE_BYTES', 4)
    with pytest.raises(InferenceRequestRejected):
        compile(body)


def test_total_json_nodes_are_bounded(monkeypatch):
    from gmail_search.gateway import inference
    monkeypatch.setattr(inference, 'MAX_JSON_NODES', 5)
    with pytest.raises(InferenceRequestRejected):
        compile(request())
