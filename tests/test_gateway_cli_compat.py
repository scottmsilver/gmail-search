from copy import deepcopy

import pytest

from gmail_search.gateway.inference import InferenceRequestRejected, compile_anthropic_request
from gmail_search.gateway.cli_compat import compile_anthropic_cli_request


def base():
    return {'model':'server-model','max_tokens':32,'stream':True,'messages':[{'role':'user','content':[{'type':'text','text':'synthetic'}]}],
            'tools':[{'name':'Bash','input_schema':{'type':'object','properties':{'command':{'type':'string'}}}}]}


def compile(body,profile):
    return compile_anthropic_cli_request(body,server_model='server-model',max_output_tokens=100,client_profile=profile)


def test_pi_cache_hints_removed_without_mutating_tool_input_or_source():
    request=base();request['system']=[{'type':'text','text':'synthetic','cache_control':{'type':'ephemeral'}}]
    request['messages'][0]['content'][0]['cache_control']={'type':'ephemeral'}
    request['messages'] += [{'role':'assistant','content':[{'type':'tool_use','id':'tool_1','name':'Bash','input':{'cache_control':'inert local input'}}]},
                            {'role':'user','content':[{'type':'tool_result','tool_use_id':'tool_1','content':'42','cache_control':{'type':'ephemeral'}}]}]
    original=deepcopy(request)
    result=compile(request,'pi-0.84.4')
    assert request==original
    assert 'cache_control' not in result['system'][0]
    assert 'cache_control' not in result['messages'][-1]['content'][0]
    assert result['messages'][1]['content'][0]['input']=={'cache_control':'inert local input'}
    compile_anthropic_request(result,server_model='server-model',max_output_tokens=100)
    with pytest.raises(InferenceRequestRejected):compile(request,'strict')


def test_claude_identity_effort_and_schema_uri_never_forwarded():
    request=base()
    request['metadata']={'user_id':'synthetic-device-session'}
    request['output_config']={'effort':'high'}
    request['tools'][0]['input_schema']['$schema']='https://json-schema.org/draft/2020-12/schema'
    result=compile(request,'claude-2.1.272')
    assert not {'metadata','output_config'} & result.keys()
    assert '$schema' not in result['tools'][0]['input_schema']
    with pytest.raises(InferenceRequestRejected):compile(request,'strict')


@pytest.mark.parametrize('profile',['strict','pi-0.84.4','claude-2.1.272'])
@pytest.mark.parametrize('attack',['remote_image','server_tool','unknown_schema_uri','metadata_extra','output_format','bad_cache'])
def test_compatibility_does_not_expand_provider_capabilities(profile,attack):
    request=base()
    if attack=='remote_image':request['messages'][0]['content']=[{'type':'image','source':{'type':'url','url':'https://evil.test'}}]
    elif attack=='server_tool':request['tools']=[{'type':'web_search_20250305','name':'web_search'}]
    elif attack=='unknown_schema_uri':request['tools'][0]['input_schema']['$schema']='https://evil.test/schema'
    elif attack=='metadata_extra':request['metadata']={'user_id':'synthetic','owner':'foreign'}
    elif attack=='output_format':request['output_config']={'format':{'type':'json_schema','schema':{}}}
    elif attack=='bad_cache':request['messages'][0]['content'][0]['cache_control']={'type':'remote','url':'https://evil.test'}
    with pytest.raises(InferenceRequestRejected):compile(request,profile)


def test_runtime_profile_is_closed_and_guest_cannot_supply_it():
    with pytest.raises(InferenceRequestRejected):compile(base(),'latest')
    request=base();request['client_profile']='claude-2.1.272'
    with pytest.raises(InferenceRequestRejected):compile(request,'claude-2.1.272')


@pytest.mark.parametrize('name,profile', [('pi_initial','pi-0.84.4'),('pi_tool_result','pi-0.84.4'),('claude_initial','claude-2.1.272'),('claude_tool_result','claude-2.1.272')])
def test_sanitized_shapes_from_real_cli_gateway_spike(name,profile):
    import json
    from pathlib import Path
    request=json.loads((Path(__file__).parent/'fixtures/cli_compat'/f'{name}.json').read_text())
    result=compile_anthropic_cli_request(request,server_model='claude-sonnet-4-6',max_output_tokens=32768,client_profile=profile)
    assert result['model']=='claude-sonnet-4-6'
    with pytest.raises(InferenceRequestRejected):
        compile_anthropic_request(request,server_model='claude-sonnet-4-6',max_output_tokens=32768)


def test_effort_is_fixed_by_server_profile_not_cli_hint():
    request=base();request['model']='claude-sonnet-4-6';request['output_config']={'effort':'low'}
    result=compile_anthropic_cli_request(request,server_model='claude-sonnet-4-6',max_output_tokens=100,
                                        client_profile='claude-2.1.272',server_effort='high')
    assert result['output_config']=={'effort':'high'}
    strict=base();strict['model']='claude-sonnet-4-6'
    assert compile_anthropic_request(strict,server_model='claude-sonnet-4-6',max_output_tokens=100,
                                     server_effort='high')['output_config']=={'effort':'high'}
    with pytest.raises(InferenceRequestRejected):
        compile_anthropic_request(request,server_model='claude-sonnet-4-6',max_output_tokens=100,server_effort='high')


@pytest.mark.parametrize('model,effort',[('server-model','high'),('claude-sonnet-4-6','xhigh'),('claude-sonnet-4-6',True)])
def test_unqualified_server_effort_configuration_is_rejected(model,effort):
    request=base();request['model']=model
    with pytest.raises(InferenceRequestRejected):
        compile_anthropic_request(request,server_model=model,max_output_tokens=100,server_effort=effort)


@pytest.mark.parametrize('effort',['low','medium','high','max'])
def test_official_sonnet_effort_settings_compile_from_server_only(effort):
    request=base();request['model']='claude-sonnet-4-6'
    result=compile_anthropic_request(request,server_model='claude-sonnet-4-6',max_output_tokens=100,server_effort=effort)
    assert result['output_config']=={'effort':effort}
