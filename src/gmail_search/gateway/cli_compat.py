"""Versioned CLI compatibility hints, removed before the closed API boundary.

Only trusted run profiles select these transforms; request bodies cannot opt in.
Qualified captures: Pi 0.84.4 and native Claude 2.1.272, synthetic worker spike.
No metadata identity, cache directive, schema URI or guest effort hint reaches
upstream. A trusted server effort setting is applied by the strict compiler; the
qualified synthetic CLI profile uses high effort for claude-sonnet-4-6.
"""
from .inference import (
    InferenceRequestRejected, _copy, _object, _plain_json, _text,
    compile_anthropic_request,
)

CLIENT_PROFILES=frozenset(('strict','pi-0.84.4','claude-2.1.272'))
_SCHEMA_URI='https://json-schema.org/draft/2020-12/schema'


def _without_cache(block):
    if type(block) is not dict:
        return
    if 'cache_control' in block:
        cache=_object(block['cache_control'],'type','ttl')
        if cache['type']!='ephemeral' or cache.get('ttl','5m') not in ('5m','1h'):
            raise InferenceRequestRejected()
        del block['cache_control']
    # Tool arguments remain inert data: never recursively rewrite tool input.
    if block.get('type')=='tool_result' and type(block.get('content')) is list:
        for child in block['content']:
            _without_cache(child)


def compile_anthropic_cli_request(body,*,server_model,max_output_tokens,client_profile='strict',server_effort=None):
    if type(client_profile) is not str or client_profile not in CLIENT_PROFILES:
        raise InferenceRequestRejected()
    if client_profile=='strict':
        return compile_anthropic_request(body,server_model=server_model,max_output_tokens=max_output_tokens,server_effort=server_effort)
    try:
        _plain_json(body)
        normalized=_copy(body)
        if type(normalized) is not dict:
            raise InferenceRequestRejected()
        if client_profile=='claude-2.1.272':
            if 'metadata' in normalized:
                metadata=_object(normalized.pop('metadata'),'user_id')
                _text(metadata['user_id'],4096)
            if 'output_config' in normalized:
                config=_object(normalized.pop('output_config'),'effort')
                if config['effort'] not in ('low','medium','high','xhigh','max'):
                    raise InferenceRequestRejected()
            for tool in normalized.get('tools',[]):
                schema=tool.get('input_schema') if type(tool) is dict else None
                if type(schema) is dict and '$schema' in schema:
                    if schema.pop('$schema')!=_SCHEMA_URI:
                        raise InferenceRequestRejected()
        for block in normalized.get('system',[]) if type(normalized.get('system')) is list else []:
            _without_cache(block)
        for tool in normalized.get('tools',[]):
            _without_cache(tool)
        for message in normalized.get('messages',[]):
            if type(message) is dict and type(message.get('content')) is list:
                for block in message['content']:
                    _without_cache(block)
        return compile_anthropic_request(normalized,server_model=server_model,max_output_tokens=max_output_tokens,server_effort=server_effort)
    except (TypeError,ValueError,RecursionError,OverflowError,AttributeError):
        raise InferenceRequestRejected() from None
