import asyncio

import pytest

from gmail_search.agents.workflow_eval import analyze_trace, interval_union, run_live, load_jsonl


def test_overlap_is_wall_time_and_missing_is_unknown():
    assert interval_union([(1, 5), (3, 7), (10, 11)]) == 7
    report = analyze_trace({}, [])
    assert report['tool_wall_seconds'] is None
    assert report['model_turns'] is None
    assert report['usage']['children'] is None
    assert report['quality']['status'] == 'pending_review'


def test_nested_batch_errors_repeated_reads_and_bytes():
    response = {'results': [{'error': 'timeout'}, {'content': [{'text': '{"isError": true}'}]}]}
    events = [dict(kind='mcp_tool_call_full', payload={'name': 'get_email', 'args': {'id': 'x'}, 'response': response}) for _ in range(2)]
    result = analyze_trace({}, events)
    assert result['tool_calls'] == 2
    assert result['tool_error_calls'] == 2
    assert result['nested_error_markers'] == 4
    assert result['duplicate_reads'] == 1
    assert result['tool_result_bytes'] > 0


def test_transcript_turns_usage_and_tool_overlap():
    records = [
        {'type': 'message', 'message': {'role': 'assistant', 'usage': {'input': 4, 'output': 6, 'cost': {'total': 0.2}}}},
        {'type': 'tool_execution_start', 'toolCallId': 'a', 'toolName': 'get_email', 'timestamp': 1},
        {'type': 'tool_execution_start', 'toolCallId': 'b', 'toolName': 'search', 'timestamp': 2},
        {'type': 'tool_execution_end', 'toolCallId': 'a', 'timestamp': 4, 'result': {}},
        {'type': 'tool_execution_end', 'toolCallId': 'b', 'timestamp': 5, 'result': {}},
    ]
    result = analyze_trace({}, [], records)
    assert result['model_turns'] == 1
    assert result['tool_calls'] == 2
    assert result['tool_wall_seconds'] == 4
    assert result['usage']['parent']['input_tokens'] == 4
    assert result['usage']['parent']['cost_usd'] == 0.2
    assert result['usage']['total'] is None


def test_malformed_jsonl_is_explicit(tmp_path):
    path = tmp_path / 'trace.jsonl'
    path.write_text('{}\nnot json\n')
    with pytest.raises(ValueError, match='line 2'):
        load_jsonl(path)


def test_live_requires_opt_in_and_isolates_every_run(tmp_path):
    seen = []
    async def invoke(**kwargs):
        seen.append(kwargs)
        if len(seen) == 1:
            raise RuntimeError('synthetic failure')
    cases = [{'id': 'c1', 'question': 'Synthetic lookup', 'kind': 'simple_control'}]
    kwargs = dict(cases=cases, output_dir=tmp_path, repeats=2, invoke=invoke)
    with pytest.raises(ValueError, match='allow_paid'):
        asyncio.run(run_live(**kwargs))
    assert not seen
    results = asyncio.run(run_live(**kwargs, allow_paid=True))
    assert len(results) == 4
    for key in ('session_id', 'conversation_id', 'workspace'):
        assert len({r[key] for r in seen}) == 4
    assert {r['workflow_profile'] for r in seen} == {'baseline', 'workflow'}
    assert all(r['model'] == 'openrouter/meta/muse-spark-1.3' for r in seen)
    assert all('2026-09-10' in r['question'] for r in seen)
    assert results[0]['status'] == 'error'
    assert len(list(tmp_path.glob('*/run.json'))) == 4


def test_workflow_telemetry_separates_parent_children_and_uses_union():
    def ev(parent, typ, timestamp, **extra):
        return {'kind': 'workflow_event', 'payload': dict(parent=parent, type=typ, timestamp=timestamp, pi_session_id='parent' if parent else 'child', **extra)}
    events = [ev(True, 'message_end', 1, usage={'input': 1, 'output': 2, 'cost': {'total': .1}}), ev(False, 'message_end', 2, usage={'input': 3, 'output': 4, 'cost': {'total': .2}}), ev(True, 'tool_start', 1, tool_call_id='a'), ev(False, 'tool_start', 2, tool_call_id='b'), ev(True, 'tool_end', 4, tool_call_id='a'), ev(False, 'tool_end', 6, tool_call_id='b')]
    report = analyze_trace({}, events)
    assert report['tool_wall_seconds'] == 5
    assert report['usage']['parent']['input_tokens'] == 1
    assert report['usage']['children']['input_tokens'] == 3
    assert report['usage']['observed_total']['input_tokens'] == 4
    assert report['usage']['total'] is None  # No lifecycle completeness proof.


def test_db_tool_event_wall_is_union_and_labelled_proxy():
    events = [
        {'kind': 'tool_call', 'created_at': 1, 'payload': {'name': 'a', 'args': {}}},
        {'kind': 'tool_call', 'created_at': 2, 'payload': {'name': 'b', 'args': {}}},
        {'kind': 'tool_call', 'created_at': 4, 'payload': {'name': 'a', 'response': {}}},
        {'kind': 'tool_call', 'created_at': 6, 'payload': {'name': 'b', 'response': {}}},
    ]
    result = analyze_trace({}, events)
    assert result['tool_wall_seconds'] == 5
    assert result['tool_wall_source'] == 'DB tool event timestamps (recording-latency proxy)'


def test_complete_workflow_summary_allows_total_and_deduplicates_events():
    payload = {'event_id': 'e1', 'parent': True, 'type': 'message_end', 'usage': {'input': 3, 'output': 4, 'cost': {'total': .2}}}
    events = [{'kind': 'workflow_event', 'payload': payload}] * 2
    events.append({'kind': 'workflow_trace_summary', 'payload': {'complete': True}})
    result = analyze_trace({}, events)
    assert result['model_turns'] == 1
    assert result['usage']['total']['input_tokens'] == 3
    assert result['usage']['children']['input_tokens'] == 0


def test_redacted_telemetry_preserves_unknowns_and_result_byte_counts():
    events = []
    for i in range(2):
        events.extend([
            {'kind': 'workflow_event', 'payload': {'type': 'tool_start', 'pi_session_id': 'p', 'tool_call_id': str(i), 'tool_name': 'get_email', 'argument_keys': ['id'], 'timestamp': i * 2}},
            {'kind': 'workflow_event', 'payload': {'type': 'tool_end', 'pi_session_id': 'p', 'tool_call_id': str(i), 'tool_name': 'get_email', 'result_bytes': 42, 'timestamp': i * 2 + 1}},
        ])
    result = analyze_trace({}, events)
    assert result['duplicate_reads'] is None
    assert result['duplicate_read_calls_with_args'] == 0
    assert result['tool_result_bytes'] == 84
    assert result['tool_result_bytes_complete'] is True
    assert result['tool_result_payloads_complete'] is False
    assert result['pi_orchestration_calls'] == 2


def test_review_import_requires_evidence_and_reports_verdicts_without_score():
    from gmail_search.agents.workflow_eval import review_template, import_review
    report = {'runs': [{'session_id': 's1', 'profile': 'baseline', 'answer': 'Synthetic claim'}]}
    template = review_template(report)
    assert template['reviews'][0]['status'] == 'pending_review'
    review = {'reviews': [{'session_id': 's1', 'reviewer': 'human', 'status': 'reviewed', 'claims': [{'claim': 'Synthetic claim', 'cited_id': 'email:synthetic', 'checked_excerpt': 'Source excerpt', 'verdict': 'supported'}], 'dimensions': {key: 'checked' for key in ('correctness', 'completeness', 'citation_support', 'temporal_grounding', 'uncertainty')}}]}
    result = import_review(report, review)
    assert result['quality_review']['by_profile']['baseline']['verdicts'] == {'supported': 1}
    assert 'score' not in result['quality_review']
    review['reviews'][0]['claims'][0]['checked_excerpt'] = ''
    with pytest.raises(ValueError, match='checked_excerpt'):
        import_review(report, review)


def test_summary_usage_in_totals_not_model_turns_and_gap_overrides_summary():
    events = [
        {'kind': 'workflow_event', 'payload': {'parent': True, 'type': 'message_end', 'usage': {'input': 2, 'output': 1, 'cost': {'total': .1}}}},
        {'kind': 'workflow_event', 'payload': {'parent': True, 'type': 'summary_usage', 'usage': {'input': 3, 'output': 1, 'cost': {'total': .2}}}},
        {'kind': 'workflow_trace_summary', 'payload': {'complete': True}},
    ]
    report = analyze_trace({'final_answer': 'Claim [ref:abc123] and [email:def456].'}, events)
    assert report['usage']['total']['input_tokens'] == 5
    assert report['model_turns'] == 1
    assert report['summary_model_calls'] == 1
    assert report['citations'] == 2
    assert report['duplicate_reads'] is None
    assert report['duplicate_reads_complete'] is False
    events.append({'kind': 'workflow_event', 'payload': {'type': 'accounting_gap'}})
    assert analyze_trace({}, events)['usage']['total'] is None
