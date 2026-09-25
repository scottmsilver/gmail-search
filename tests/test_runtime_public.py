"""Public runtime security boundary, exercised without a provider or database."""
import asyncio
from types import SimpleNamespace

import pytest
from google.genai import types

from gmail_search.agents import runtime_public as rp


@pytest.mark.parametrize('name,args', [
    ('run_code', {'code': 'print(1)'}), ('sql_query', {'query': 'select 1'}),
    ('search_emails', {'query': 'q', 'user_id': 'victim'}),
    ('search_emails', {'query': 'q', 'session_id': 'victim'}),
    ('get_thread', {'thread_id': '../sql_schema'}),
    ('search_emails', {'query': 'q', 'top_k': 100000}),
    ('search_emails', {'query': 'q', 'top_k': True}),
    ('get_attachment', {'attachment_id': 1, 'mode': 'raw'}),
])
def test_rejects_unsafe_calls(name, args):
    with pytest.raises(rp.PublicRuntimeError):
        rp._validate_call(name, args)


def test_dispatch_binds_authenticated_identity(monkeypatch):
    received = []
    async def search(**kwargs):
        received.append(kwargs)
        return {'results': []}
    monkeypatch.setattr(rp.retrieval, 'search_emails', search)
    asyncio.run(rp._dispatch('search_emails', {'query': 'q'}, 'owner'))
    assert received == [{'query': 'q', 'user_id': 'owner'}]


def response(*parts, usage=None):
    return SimpleNamespace(
        candidates=[SimpleNamespace(content=types.Content(role='model', parts=list(parts)))],
        usage_metadata=usage,
    )


@pytest.fixture
def harness(monkeypatch):
    events, final, requests = [], [], []
    conn = SimpleNamespace(close=lambda: events.append('closed'))
    monkeypatch.setattr(rp, 'get_connection', lambda _: conn)
    monkeypatch.setattr(rp, 'emit_plan_event', lambda *a, **k: events.append('plan'))
    monkeypatch.setattr(rp, 'append_event', lambda *a, **k: events.append(k['kind']))
    monkeypatch.setattr(rp, 'emit_retriever_events', lambda *a, **k: events.append('evidence'))
    monkeypatch.setattr(rp, 'emit_writer_and_final', lambda *a, **k: events.append('final'))
    monkeypatch.setattr(rp, 'emit_error', lambda *a, **k: events.append(str(a[2])))
    monkeypatch.setattr(rp, 'finalize_session', lambda *a, **k: final.append(k))
    monkeypatch.setattr(rp, 'session_elapsed_ms', lambda *a, **k: 0)
    script = []
    async def generate(**kwargs):
        requests.append(kwargs)
        item = script.pop(0)
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            return await item()
        return item
    async def close():
        events.append('client closed')
    client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate), aclose=close), close=lambda: None)
    monkeypatch.setattr(rp, '_new_client', lambda: client)
    return script, events, final, requests


def test_manual_loop(harness, monkeypatch):
    script, events, final, requests = harness
    script.extend([response(types.Part(function_call=types.FunctionCall(name='search_emails', args={'query': 'q'}))), response(types.Part(text='Found [abc].'))])
    seen = []
    async def dispatch(name, args, user_id):
        seen.append(user_id)
        return {'results': [{'thread_id': 'abc'}]}
    monkeypatch.setattr(rp, '_dispatch', dispatch)
    asyncio.run(rp.public_run('db', 'session', 'q', 'owner'))
    assert seen == ['owner']
    assert final == [{'status': 'done', 'final_answer': 'Found [abc].'}]
    assert requests[0]['config'].automatic_function_calling.disable is True
    assert 'final' in events and 'client closed' in events and events[-1] == 'closed'


def test_reports_cost_via_sink_with_thoughts_folded_into_output(harness, monkeypatch):
    """Issue #78: gms.oursilverfamily.com always showed "cost unknown"
    because this runtime never read `usage_metadata` or called a cost
    sink. `thoughts_token_count` must land in `output_tokens`, matching
    `gateway/gemini_http.py`'s `candidatesTokenCount + thoughtsTokenCount`
    convention (Gemini bills thoughts as output)."""
    script, events, final, requests = harness
    usage = SimpleNamespace(prompt_token_count=100, candidates_token_count=20, thoughts_token_count=7)
    script.append(response(types.Part(text='Found [abc].'), usage=usage))
    calls = []
    asyncio.run(rp.public_run('db', 'session', 'q', 'owner', cost_sink=lambda **kw: calls.append(kw)))
    assert calls == [{'agent_name': 'public', 'model': rp._model(), 'input_tokens': 100, 'output_tokens': 27}]


def test_no_cost_sink_given_is_a_silent_noop(harness):
    """The default (no `cost_sink`) must behave exactly as before this
    issue: no crash, no cost reported — "cost unknown" is correct when
    the caller doesn't ask for cost tracking."""
    script, events, final, requests = harness
    usage = SimpleNamespace(prompt_token_count=100, candidates_token_count=20, thoughts_token_count=0)
    script.append(response(types.Part(text='Found [abc].'), usage=usage))
    asyncio.run(rp.public_run('db', 'session', 'q', 'owner'))
    assert final == [{'status': 'done', 'final_answer': 'Found [abc].'}]


def test_missing_usage_metadata_skips_cost_report_without_raising(harness):
    """A malformed or absent `usage_metadata` (None) must not crash the
    turn or fabricate a cost — same "report unknown rather than a
    possibly-wrong number" rule the footer reader already follows."""
    script, events, final, requests = harness
    script.append(response(types.Part(text='Found [abc].')))  # usage=None
    calls = []
    asyncio.run(rp.public_run('db', 'session', 'q', 'owner', cost_sink=lambda **kw: calls.append(kw)))
    assert calls == []
    assert final == [{'status': 'done', 'final_answer': 'Found [abc].'}]


def test_cost_sink_exception_does_not_fail_the_turn(harness, monkeypatch):
    """A broken cost sink is a bookkeeping failure, not a user-facing
    one — mirrors `runtime_claude.py`'s `_report_cost`, which logs and
    swallows rather than letting a ledger bug break the answer."""
    script, events, final, requests = harness
    usage = SimpleNamespace(prompt_token_count=1, candidates_token_count=1, thoughts_token_count=0)
    script.append(response(types.Part(text='Found [abc].'), usage=usage))

    def broken_sink(**kw):
        raise RuntimeError('ledger down')

    asyncio.run(rp.public_run('db', 'session', 'q', 'owner', cost_sink=broken_sink))
    assert final == [{'status': 'done', 'final_answer': 'Found [abc].'}]


def test_reports_cost_once_per_round(harness, monkeypatch):
    """A multi-round turn (tool call, then answer) must report cost for
    each model round, not just the last one — every round is a
    separate billable call."""
    script, events, final, requests = harness
    usage1 = SimpleNamespace(prompt_token_count=50, candidates_token_count=5, thoughts_token_count=0)
    usage2 = SimpleNamespace(prompt_token_count=80, candidates_token_count=15, thoughts_token_count=2)
    script.extend([
        response(types.Part(function_call=types.FunctionCall(name='search_emails', args={'query': 'q'})), usage=usage1),
        response(types.Part(text='Found [abc].'), usage=usage2),
    ])
    async def dispatch(name, args, user_id):
        return {'results': [{'thread_id': 'abc'}]}
    monkeypatch.setattr(rp, '_dispatch', dispatch)
    calls = []
    asyncio.run(rp.public_run('db', 'session', 'q', 'owner', cost_sink=lambda **kw: calls.append(kw)))
    assert [(c['input_tokens'], c['output_tokens']) for c in calls] == [(50, 5), (80, 17)]


def test_elapsed_ms_reaches_final_event_before_finalize_commits_done(harness, monkeypatch):
    """`session_elapsed_ms`'s value must reach `emit_writer_and_final` —
    and `finalize_session` (which commits status='done') must still run
    LAST, exactly as before this feature. Reversing that order is a
    replay race: `/api/agent/analyze/<id>/events` treats status='done'
    as "stream complete, stop polling," so a reconnecting client could
    see 'done' and return before the `final` event's own INSERT has
    committed, missing the answer and its elapsed time entirely. This
    test calls `public_run` with no `cost_sink` (as issue #74 shipped
    it, before #78 wired one in from `service.py`), so elapsed time is
    the only thing available to show here."""
    script, events, final, requests = harness
    script.extend([response(types.Part(text='Found [abc].'))])
    order: list[str] = []
    seen_elapsed: list = []

    def fake_elapsed(*a, **k):
        order.append('elapsed')
        return 4242

    def fake_emit(*a, **k):
        order.append('final_event')
        seen_elapsed.append(k.get('elapsed_ms'))
        events.append('final')

    def fake_finalize(*a, **k):
        order.append('finalize')
        final.append(k)

    monkeypatch.setattr(rp, 'session_elapsed_ms', fake_elapsed)
    monkeypatch.setattr(rp, 'emit_writer_and_final', fake_emit)
    monkeypatch.setattr(rp, 'finalize_session', fake_finalize)
    asyncio.run(rp.public_run('db', 'session', 'q', 'owner'))
    assert order == ['elapsed', 'final_event', 'finalize']
    assert seen_elapsed == [4242]


@pytest.mark.parametrize('failure', ['forged', 'rounds', 'calls', 'input', 'output', 'error', 'timeout', 'cancel'])
def test_failures_finalize_without_leaking(harness, monkeypatch, failure):
    script, events, final, requests = harness
    question = 'q'
    if failure == 'forged':
        script.append(response(types.Part(function_call=types.FunctionCall(name='run_code', args={}))))
    elif failure in ('rounds', 'calls'):
        monkeypatch.setattr(rp, 'MAX_ROUNDS' if failure == 'rounds' else 'MAX_TOOL_CALLS', 0)
        script.append(response(types.Part(function_call=types.FunctionCall(name='search_emails', args={'query': 'q'}))))
    elif failure == 'input':
        question = 'x' * (rp.MAX_INPUT_CHARS + 1)
    elif failure == 'output':
        script.append(response(types.Part(text='x' * (rp.MAX_OUTPUT_CHARS + 1))))
    elif failure == 'error':
        script.append(RuntimeError('secret provider credential'))
    elif failure == 'cancel':
        script.append(asyncio.CancelledError())
    else:
        monkeypatch.setattr(rp, 'TURN_TIMEOUT_SECONDS', 0.01)
        async def slow():
            await asyncio.sleep(10)
        script.append(slow)
    if failure == 'cancel':
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(rp.public_run('db', 'session', question, 'owner'))
    else:
        asyncio.run(rp.public_run('db', 'session', question, 'owner'))
    assert final == [{'status': 'error'}]
    assert 'final' not in events
    assert all('secret' not in event for event in events)
    assert events[-1] == 'closed'


def test_schema_exposes_only_fixed_tools():
    config = rp._config()
    declarations = config.tools[0].function_declarations
    assert {d.name for d in declarations} == {'search_emails', 'query_emails', 'get_thread', 'find_facts', 'get_attachment'}
    for declaration in declarations:
        assert not {'user_id', 'session_id', 'base_url'} & declaration.parameters.properties.keys()
    assert config.automatic_function_calling.disable


def test_server_model_configuration(monkeypatch):
    monkeypatch.setenv('GMAIL_PUBLIC_MODEL', 'google/gemini-test')
    assert rp._model() == 'gemini-test'
    monkeypatch.setenv('GMAIL_PUBLIC_MODEL', 'https://attacker.example/model')
    with pytest.raises(rp.PublicRuntimeError):
        rp._model()


def test_result_truncation_is_bounded_and_explicit():
    import json
    value = rp._bounded_result({'text': 'x' * (rp.MAX_TOOL_RESULT_CHARS * 2)})
    assert value['truncated'] is True
    assert len(json.dumps(value)) <= rp.MAX_TOOL_RESULT_CHARS


def test_whole_batch_validated_before_any_dispatch(harness, monkeypatch):
    script, events, final, requests = harness
    script.append(response(
        types.Part(function_call=types.FunctionCall(name='search_emails', args={'query': 'q'})),
        types.Part(function_call=types.FunctionCall(name='search_emails', args={'query': 'q', 'user_id': 'victim'})),
    ))
    async def dispatch(*args):
        pytest.fail('No calls in an invalid batch should execute')
    monkeypatch.setattr(rp, '_dispatch', dispatch)
    asyncio.run(rp.public_run('db', 'session', 'q', 'owner'))
    assert final == [{'status': 'error'}]


def test_total_context_limit(harness, monkeypatch):
    script, events, final, requests = harness
    monkeypatch.setattr(rp, 'MAX_CONTEXT_CHARS', 1)
    asyncio.run(rp.public_run('db', 'session', 'q', 'owner'))
    assert not requests
    assert final == [{'status': 'error'}]


def test_find_facts_default_is_bounded(monkeypatch):
    seen = []
    async def facts(**kwargs):
        seen.append(kwargs)
        return {'facts': []}
    monkeypatch.setattr(rp.retrieval, 'find_facts', facts)
    asyncio.run(rp._dispatch('find_facts', {'query': 'q'}, 'owner'))
    assert seen[0]['k'] == 100
