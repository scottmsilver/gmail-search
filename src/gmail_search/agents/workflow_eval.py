"""Private, observational trace metrics and explicitly opted-in Pi comparisons.

Missing telemetry is represented by None. No quality score is inferred.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import re
import time
from uuid import uuid4

MODEL = 'openrouter/meta/muse-spark-1.3'
REFERENCE_DATE = '2026-09-10'
RUBRIC = {
    'status': 'pending_review',
    'dimensions': {
        'correctness': 'Check substantive claims against cited original messages.',
        'completeness': 'Check all requested entities, dates, and follow-up obligations.',
        'citation_support': 'Open citations; verify they support adjacent claims.',
        'temporal_grounding': 'Resolve relative dates against the fixed reference date.',
        'uncertainty': 'Distinguish missing evidence, inference, and confirmed facts.',
    },
    'score': None,
}


def interval_union(intervals):
    """Seconds covered by intervals, counting parallel work only once."""
    total, end = 0.0, None
    for start, stop in sorted(intervals):
        if stop < start:
            continue
        total += max(0, stop - max(start, end if end is not None else start))
        end = max(stop, end if end is not None else stop)
    return total


def _timestamp(value):
    if isinstance(value, (int, float)):
        return value / 1000 if value > 1e11 else float(value)
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()
        except ValueError:
            pass
    return None


def _errors(value):
    if isinstance(value, str):
        try:
            return _errors(json.loads(value))
        except (ValueError, RecursionError):
            return 0
    if isinstance(value, list):
        return sum(_errors(v) for v in value)
    if isinstance(value, dict):
        marker = any(value.get(k) for k in ('error', 'isError', 'is_error'))
        return int(marker) + sum(_errors(v) for k, v in value.items() if k not in ('error', 'isError', 'is_error'))
    return 0


def load_jsonl(path):
    records = []
    for lineno, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError('expected object')
            records.append(record)
        except ValueError as exc:
            raise ValueError(f'{path}: invalid JSONL at line {lineno}') from exc
    return records


def _usage(messages):
    usages = [m['usage'] for m in messages if isinstance(m.get('usage'), dict)]
    if not usages:
        return None
    result = {}
    for output, key in [('input_tokens', 'input'), ('output_tokens', 'output'), ('cache_read_tokens', 'cacheRead'), ('cache_write_tokens', 'cacheWrite')]:
        values = [u.get(key) for u in usages]
        result[output] = sum(values) if all(isinstance(v, (int, float)) for v in values) else None
    costs = [u.get('cost', {}).get('total') if isinstance(u.get('cost'), dict) else u.get('cost') for u in usages]
    result['cost_usd'] = sum(costs) if all(isinstance(v, (int, float)) for v in costs) else None
    result['messages_with_usage'] = len(usages)
    result['complete'] = len(usages) == len(messages)
    return result


def analyze_trace(session, events, transcript=None):
    """Measure event rows plus an optional *single-turn* Pi transcript.

    Durable MCP calls are separate from Pi orchestration calls; counts/bytes
    prefer full MCP events, avoiding double counting clipped UI mirrors.
    """
    telemetry, event_ids = [], set()
    for event in events:
        if event.get('kind') != 'workflow_event':
            continue
        payload = event['payload']
        event_id = payload.get('event_id')
        if event_id is not None and event_id in event_ids:
            continue
        if event_id is not None:
            event_ids.add(event_id)
        telemetry.append(payload)
    summaries = [e['payload'] for e in events if e.get('kind') == 'workflow_trace_summary']
    trace_complete = bool(summaries and summaries[-1].get('complete') is True) and not any(e.get('type') == 'accounting_gap' for e in telemetry)
    records = list(transcript or [])
    # Workflow telemetry is authoritative when present; avoid counting its
    # parent copy again alongside the persisted transcript.
    if telemetry:
        records = []
        for event in telemetry:
            typ = event.get('type')
            if typ == 'message_end' and event.get('parent') is True:
                records.append({'type': 'message_end', 'message': {'role': 'assistant', 'usage': event.get('usage')}})
            elif typ in ('tool_start', 'tool_end'):
                records.append({'type': 'tool_execution_start' if typ == 'tool_start' else 'tool_execution_end', 'toolCallId': (event.get('pi_session_id'), event.get('tool_call_id')), 'timestamp': event.get('timestamp'), 'toolName': event.get('tool_name', 'unknown'), **({'args': event['args']} if 'args' in event else {}), **({'result': event['result']} if 'result' in event else {}), **({'result_bytes': event['result_bytes']} if 'result_bytes' in event else {}), **({'isError': event['is_error']} if 'is_error' in event else {})})
    messages = [r['message'] for r in records if r.get('type') in ('message', 'message_end') and isinstance(r.get('message'), dict) and r['message'].get('role') == 'assistant']
    full = [e['payload'] for e in events if e.get('kind') == 'mcp_tool_call_full' and isinstance(e.get('payload'), dict)]
    starts, intervals, pi_calls = {}, [], []
    incomplete = False
    for r in records:
        typ = r.get('type')
        if typ == 'tool_execution_start':
            starts[r.get('toolCallId')] = r
        elif typ == 'tool_execution_end':
            start = starts.pop(r.get('toolCallId'), {})
            call = {'name': start.get('toolName', r.get('toolName', 'unknown'))}
            if 'args' in start:
                call['args'] = start['args']
            if 'result' in r:
                call['response'] = r['result']
            if 'isError' in r:
                call['is_error'] = r['isError']
            if 'result_bytes' in r:
                call['result_bytes'] = r['result_bytes']
            pi_calls.append(call)
            a, b = _timestamp(start.get('timestamp')), _timestamp(r.get('timestamp'))
            if a is not None and b is not None and b >= a:
                intervals.append((a, b))
            else:
                incomplete = True
    # Persisted Pi session files use assistant toolCall blocks + toolResult messages.
    if not pi_calls:
        calls = {c.get('id'): c for m in messages for c in m.get('content', []) if isinstance(c, dict) and c.get('type') == 'toolCall'}
        for r in records:
            m = r.get('message', {})
            if m.get('role') == 'toolResult':
                c = calls.pop(m.get('toolCallId'), {})
                pi_calls.append({'name': m.get('toolName', c.get('name', 'unknown')), **({'args': c['arguments']} if 'arguments' in c else {}), 'response': m})
    calls = full or pi_calls
    if not calls:
        # UI events cannot prove payload completeness, but remain useful.
        calls = [e['payload'] for e in events if e.get('kind') == 'tool_call' and 'response' in e.get('payload', {})]
    wall_source = 'timestamped Pi execution intervals' if intervals else None
    if not intervals:
        # UI events have timestamps even when persisted Pi JSONL has only
        # messages. This includes event-recording latency; label it explicitly.
        pending, db_intervals, db_incomplete = {}, [], False
        for event in events:
            if event.get('kind') != 'tool_call':
                continue
            payload = event.get('payload', {})
            name, stamp = payload.get('name'), _timestamp(event.get('created_at'))
            if 'args' in payload and 'response' not in payload:
                if name in pending:
                    db_incomplete = True  # Same-name concurrent calls are ambiguous.
                pending[name] = stamp
            elif 'response' in payload:
                first = pending.pop(name, None)
                if first is not None and stamp is not None and stamp >= first:
                    db_intervals.append((first, stamp))
                else:
                    db_incomplete = True
        if db_intervals and not pending and not db_incomplete:
            intervals, incomplete, starts = db_intervals, False, {}
            wall_source = 'DB tool event timestamps (recording-latency proxy)'
    duplicate_reads, seen = 0, set()
    read_calls, readable_calls = 0, 0
    for call in calls:
        name = str(call.get('name', 'unknown'))
        if any(part in name.lower() for part in ('get_email', 'get_thread', 'read', 'fetch')):
            read_calls += 1
            if 'args' not in call:
                continue
            readable_calls += 1
            key = (name, json.dumps(call['args'], sort_keys=True, default=str))
            duplicate_reads += key in seen
            seen.add(key)
    error_counts = [max(int(bool(c.get('is_error'))), _errors(c.get('response'))) for c in calls]
    byte_counts = []
    for call in calls:
        if isinstance(call.get('result_bytes'), int) and call['result_bytes'] >= 0:
            byte_counts.append(call['result_bytes'])
        elif 'response' in call:
            byte_counts.append(len(json.dumps(call['response'], ensure_ascii=False, default=str).encode()))
    error_covered = sum('response' in c or 'is_error' in c for c in calls)
    start, stop = _timestamp(session.get('started_at')), _timestamp(session.get('finished_at'))
    answer = session.get('final_answer') or ''
    parent_summaries = [{'usage': e.get('usage')} for e in telemetry if e.get('type') == 'summary_usage' and e.get('parent') is True]
    if not telemetry:
        parent_summaries = [{'usage': r.get('usage')} for r in records if r.get('type') in ('compaction', 'branch_summary')]
    child_messages = [{'usage': e.get('usage')} for e in telemetry if e.get('type') in ('message_end', 'summary_usage') and e.get('parent') is False]
    children = _usage(child_messages)
    if trace_complete and not child_messages:
        children = dict(input_tokens=0, output_tokens=0, cache_read_tokens=0, cache_write_tokens=0, cost_usd=0, messages_with_usage=0, complete=True)
    observed = _usage(messages + parent_summaries + child_messages)
    return {
        'session_id': session.get('id'), 'status': session.get('status'),
        'elapsed_seconds': stop - start if start is not None and stop is not None else None,
        'tool_wall_seconds': interval_union(intervals) if intervals and not incomplete and not starts else None,
        'tool_wall_source': wall_source,
        'tool_calls': len(calls), 'tool_call_counts': dict(Counter(c.get('name', 'unknown') for c in calls)),
        'tool_count_source': 'full_mcp_events' if full else 'workflow_telemetry' if telemetry and pi_calls else 'pi_transcript' if pi_calls else 'ui_events',
        'pi_orchestration_calls': len(pi_calls) if transcript is not None or telemetry else None,
        'tool_error_calls': sum(n > 0 for n in error_counts) if error_covered else None, 'nested_error_markers': sum(error_counts) if any('response' in c for c in calls) else None, 'tool_error_calls_covered': error_covered,
        'duplicate_reads': duplicate_reads if readable_calls else None, 'duplicate_read_calls_with_args': readable_calls, 'duplicate_read_calls_total': read_calls, 'duplicate_reads_complete': bool(read_calls) and readable_calls == read_calls, 'duplicate_reads_definition': 'same read tool and identical arguments',
        'tool_result_bytes': sum(byte_counts) if byte_counts else None,
        'tool_result_bytes_calls_covered': len(byte_counts),
        'tool_result_bytes_complete': bool(calls) and len(byte_counts) == len(calls) and bool(full or pi_calls),
        'tool_result_payloads_complete': bool(calls) and all('response' in c for c in calls) and bool(full or pi_calls),
        'model_turns': len(messages) if transcript is not None or telemetry else None,
        'summary_model_calls': len(parent_summaries) + sum(e.get('type') == 'summary_usage' and e.get('parent') is False for e in telemetry),
        'child_model_turns': sum(e.get('type') == 'message_end' and e.get('parent') is False for e in telemetry) if telemetry else None,
        'usage': {'parent': _usage(messages + parent_summaries), 'children': children, 'observed_total': observed, 'total': observed if trace_complete and observed and observed['complete'] else None},
        'workflow_trace_complete': trace_complete if summaries else None,
        'child_events': [e for e in events if e.get('kind', '').startswith(('child_', 'subagent_'))],
        'citations': len(re.findall(r'\[(?:ref|email|thread|art):[^\]]+\]', answer)),
        'quality': RUBRIC,
        'answer': answer,
        'limitations': ['Historical observations are not controlled A/B evidence.', 'Child usage and all-agent totals require verified child telemetry.', 'Tool bytes measure serialized recorded results, not network wire bytes.'],
    }


def _connect_readonly():
    import psycopg
    from psycopg.rows import dict_row
    from gmail_search.store.db import _pg_dsn
    conn = psycopg.connect(_pg_dsn(), row_factory=dict_row, connect_timeout=5)
    conn.read_only = True
    conn.execute("SET LOCAL statement_timeout = '10s'")
    return conn


def corpus_status(conn, user_id=None):
    """Cheap bounded metadata query; no index load, reindex, or model calls."""
    try:
        with conn.transaction():
            where = ' WHERE user_id = %s' if user_id else ''
            row = conn.execute('SELECT MAX(history_id) AS max_history_id, MAX(date) AS latest_message_date FROM messages' + where, (user_id,) if user_id else ()).fetchone()
            return {'database': 'reachable', 'watermark': dict(row), 'user_id': user_id, 'retrieval_backends': 'not_probed', 'frozen_snapshot': False}
    except Exception as exc:
        return {'database': 'reachable', 'watermark': None, 'watermark_error': type(exc).__name__, 'retrieval_backends': 'not_probed', 'frozen_snapshot': False}


def _write_private(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open('x', encoding='utf-8') as f:
        os.chmod(path, 0o600)
        json.dump(value, f, indent=2, default=str, ensure_ascii=False)
        f.write('\n')


def historical(output_dir, session_ids=None, limit=5, user_id=None, transcript_dir=None, sessions_root=None):
    with _connect_readonly() as conn:
        where, args = [], []
        if session_ids:
            where.append('id = ANY(%s)')
            args.append(session_ids)
        if user_id:
            where.append('user_id = %s')
            args.append(user_id)
        clause = ' WHERE ' + ' AND '.join(where) if where else ''
        sessions = conn.execute('SELECT * FROM agent_sessions' + clause + ' ORDER BY started_at DESC LIMIT %s', (*args, limit)).fetchall()
        reports, cases = [], []
        for s in sessions:
            events = conn.execute('SELECT * FROM agent_events WHERE session_id = %s ORDER BY seq', (s['id'],)).fetchall()
            # Explicit session-ID filenames prevent silently using an entire multi-turn conversation.
            path = Path(transcript_dir) / f"{s['id']}.jsonl" if transcript_dir else None
            transcript = load_jsonl(path) if path and path.exists() else None
            if transcript is None and sessions_root and s.get('conversation_id'):
                conv = str(s['conversation_id'])
                if re.fullmatch(r'[A-Za-z0-9_-]+', conv):
                    conv_path = Path(sessions_root) / f'{conv}.jsonl'
                    if conv_path.exists():
                        a, b = _timestamp(s['started_at']), _timestamp(s.get('finished_at'))
                        if a is not None and b is not None:
                            transcript = [r for r in load_jsonl(conv_path) if (t := _timestamp(r.get('timestamp'))) is not None and a <= t <= b]
            metrics = analyze_trace(s, events, transcript)
            try:
                with conn.transaction():
                    metrics['recorded_cost_ledger'] = [dict(row) for row in conn.execute('SELECT operation, model, input_tokens, output_tokens, cached_input_tokens, cache_write_tokens, estimated_cost_usd FROM costs WHERE message_id = %s', (f"deep:{s['id']}",)).fetchall()]
            except Exception as exc:
                metrics['recorded_cost_ledger'] = None
                metrics['cost_ledger_error'] = type(exc).__name__
            reports.append(metrics)
            cases.append({'id': s['id'], 'question': s['question'], 'kind': 'recent_question'})
        report = {'mode': 'historical_observational', 'corpus': corpus_status(conn, user_id), 'runs': reports}
    _write_private(Path(output_dir) / 'historical.json', report)
    cases.append({'id': 'simple-first-name', 'question': 'What is my first name?', 'kind': 'simple_control'})
    _write_private(Path(output_dir) / 'cases.json', {'reference_date': REFERENCE_DATE, 'cases': cases})
    return report


async def _invoke_real(**kwargs):
    from gmail_search.agents.runtime_pi import pi_run, _workspaces_root
    from gmail_search.agents.session import create_session
    from gmail_search.store.db import get_connection
    root = _workspaces_root() / kwargs['workspace']
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    conn = get_connection(kwargs['db_path'])
    try:
        create_session(conn, session_id=kwargs['session_id'], conversation_id=kwargs['conversation_id'], mode='deep', question=kwargs['question'], user_id=kwargs['user_id'])
    finally:
        conn.close()
    await pi_run(**kwargs)
    with _connect_readonly() as conn:
        session = conn.execute('SELECT * FROM agent_sessions WHERE id = %s', (kwargs['session_id'],)).fetchone()
        events = conn.execute('SELECT * FROM agent_events WHERE session_id = %s ORDER BY seq', (kwargs['session_id'],)).fetchall()
        trace_path = Path('deploy/pi/sessions') / f"{kwargs['conversation_id']}.jsonl"
        transcript = load_jsonl(trace_path) if trace_path.exists() else None
        return {'metrics': analyze_trace(session, events, transcript), 'answer': session['final_answer'], 'status': session['status']}


async def run_live(*, cases, output_dir, allow_paid=False, repeats=3, profiles=('baseline', 'workflow'), reference_date=REFERENCE_DATE, user_id=None, invoke=None):
    if not allow_paid:
        raise ValueError('Live runs require explicit allow_paid=True / --allow-paid')
    if repeats < 1 or any(p not in ('baseline', 'workflow') for p in profiles):
        raise ValueError('Positive repeats and baseline/workflow profiles required')
    if not cases or any(not c.get('id') or not c.get('question') for c in cases):
        raise ValueError('Cases need nonempty id and question')
    invoke = invoke or _invoke_real
    results = []
    previous = os.environ.get('GMAIL_PI_THINKING')
    os.environ['GMAIL_PI_THINKING'] = 'medium'
    try:
        for repeat in range(repeats):
            # Alternate profile order to reduce consistent warm-cache bias.
            for case in cases:
                for profile in profiles if repeat % 2 == 0 else tuple(reversed(profiles)):
                    uid = uuid4().hex
                    run = {'case_id': case['id'], 'repeat': repeat, 'profile': profile, 'session_id': f'eval-{uid}', 'conversation_id': f'eval-conv-{uid}', 'workspace': f'eval-{uid}', 'model': MODEL, 'thinking': 'medium', 'reference_date': reference_date, 'quality': RUBRIC}
                    path = Path(output_dir) / uid
                    _write_private(path / 'input.json', {'run': run, 'case': case})
                    costs = []
                    started = time.monotonic()
                    try:
                        result = await invoke(db_path=Path('data'), session_id=run['session_id'], conversation_id=run['conversation_id'], workspace=run['workspace'], question=f'Reference date: {reference_date}. Use this as the current date when interpreting relative dates in the user question. Resolve relative dates quoted in source emails against each email’s sent date.\n\n{case["question"]}', model=MODEL, cost_sink=lambda **cost: costs.append(cost), user_id=user_id, workflow_profile=profile)
                        run.update(result or {})
                        run.setdefault('status', 'returned_without_status')
                    except Exception as exc:
                        run.update(status='error', error_type=type(exc).__name__, error=str(exc))
                    run.update(elapsed_seconds=time.monotonic() - started, recorded_costs=costs)
                    _write_private(path / 'run.json', run)
                    results.append(run)
    finally:
        if previous is None:
            os.environ.pop('GMAIL_PI_THINKING', None)
        else:
            os.environ['GMAIL_PI_THINKING'] = previous
    return results


def review_template(report):
    """Create an editable private evidence-check sheet, without judging claims."""
    return {'instructions': 'Inspect original cited sources. Record exact checked excerpts and your verdict; do not infer truth from a prior answer.', 'reviews': [
        {'session_id': run['session_id'], 'case_id': run.get('case_id'), 'profile': run.get('profile'),
         'answer': run.get('answer', run.get('metrics', {}).get('answer')),
         'status': 'pending_review', 'reviewer': '',
         'dimensions': {key: '' for key in RUBRIC['dimensions']},
         'claims': [{'claim': '', 'cited_id': '', 'checked_excerpt': '', 'verdict': 'uncertain', 'notes': ''}]}
        for run in report['runs']
    ]}


def import_review(report, review):
    """Validate human evidence records and compare verdict counts, never scores.

    This validates the review structure, not the truth of supplied excerpts.
    Excerpt provenance and source interpretation remain the reviewer's work.
    """
    import copy
    result = copy.deepcopy(report)
    runs = {run['session_id']: run for run in result['runs']}
    seen = set()
    grouped = {}
    for item in review['reviews']:
        session_id = item.get('session_id')
        if session_id not in runs or session_id in seen:
            raise ValueError('Review session_id must identify one unique report run')
        seen.add(session_id)
        if item.get('status') != 'reviewed':
            raise ValueError('Imported reviews must have status reviewed; leave unfinished runs out')
        if not str(item.get('reviewer', '')).strip():
            raise ValueError('reviewer is required')
        for dimension in RUBRIC['dimensions']:
            if not str(item.get('dimensions', {}).get(dimension, '')).strip():
                raise ValueError(f'Review requires notes for {dimension}')
        if not item.get('claims'):
            raise ValueError('Review requires at least one checked claim')
        for claim in item['claims']:
            for key in ('claim', 'cited_id', 'checked_excerpt'):
                if not isinstance(claim.get(key), str) or not claim[key].strip():
                    raise ValueError(f'Each claim requires {key}; use uncertain for unresolved evidence')
            if claim.get('verdict') not in ('supported', 'contradicted', 'uncertain'):
                raise ValueError('Claim verdict must be supported, contradicted, or uncertain')
        runs[session_id]['quality_review'] = copy.deepcopy(item)
        profile = runs[session_id].get('profile', 'historical_unknown_profile')
        group = grouped.setdefault(profile, {'reviewed_runs': 0, 'checked_claims': 0, 'verdicts': {}})
        group['reviewed_runs'] += 1
        group['checked_claims'] += len(item['claims'])
        for claim in item['claims']:
            verdict = claim['verdict']
            group['verdicts'][verdict] = group['verdicts'].get(verdict, 0) + 1
    result['quality_review'] = {'status': 'reviewed' if len(seen) == len(runs) else 'partially_reviewed', 'pending_runs': len(runs) - len(seen), 'by_profile': grouped, 'limitation': 'Human-entered evidence checks; counts are not quality scores and depend on claim selection. Compare the same cases and rubric notes.'}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('historical', 'live', 'review-template', 'review-import'))
    parser.add_argument('--output', type=Path, default=Path('data/agent-evals') / uuid4().hex)
    parser.add_argument('--session-id', action='append')
    parser.add_argument('--limit', type=int, default=5)
    parser.add_argument('--user-id')
    parser.add_argument('--transcript-dir', type=Path)
    parser.add_argument('--sessions-root', type=Path, help='Pi conversation JSONL directory; slices by DB turn timestamps')
    parser.add_argument('--cases', type=Path)
    parser.add_argument('--report', type=Path, help='Private historical/comparison JSON for offline review')
    parser.add_argument('--review', type=Path, help='Completed private evidence-review JSON')
    parser.add_argument('--allow-paid', action='store_true')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--profile', action='append', choices=('baseline', 'workflow'))
    args = parser.parse_args(argv)
    # Prevent accidental personal-data exports into versioned source directories.
    private_root = Path('data/agent-evals').resolve()
    if not args.output.resolve().is_relative_to(private_root):
        parser.error('--output must be under data/agent-evals')
    if args.mode.startswith('review-'):
        if not args.report:
            parser.error('review modes require --report')
        report = json.loads(args.report.read_text())
        if args.mode == 'review-template':
            _write_private(args.output / 'review-template.json', review_template(report))
        else:
            if not args.review:
                parser.error('review-import requires --review')
            _write_private(args.output / 'reviewed-report.json', import_review(report, json.loads(args.review.read_text())))
    elif args.mode == 'historical':
        historical(args.output, args.session_id, args.limit, args.user_id, args.transcript_dir, args.sessions_root)
    else:
        if not args.allow_paid:
            parser.error('live requires --allow-paid (billed model calls)')
        if not args.cases or not args.user_id:
            parser.error('live requires --cases and --user-id')
        manifest = json.loads(args.cases.read_text())
        cases = manifest['cases']
        if len([c for c in cases if c.get('kind') == 'recent_question']) != 5 or not any(c.get('kind') == 'simple_control' for c in cases):
            parser.error('manifest needs five recent_question cases and a simple_control case')
        with _connect_readonly() as conn:
            before = corpus_status(conn, args.user_id)
        _write_private(args.output / 'corpus-before.json', before)
        results = asyncio.run(run_live(cases=cases, output_dir=args.output, allow_paid=True, repeats=args.repeats, profiles=tuple(args.profile or ('baseline', 'workflow')), reference_date=manifest.get('reference_date', REFERENCE_DATE), user_id=args.user_id))
        with _connect_readonly() as conn:
            after = corpus_status(conn, args.user_id)
        _write_private(args.output / 'comparison.json', {'runs': results, 'corpus_before': before, 'corpus_after': after, 'watermark_changed': before.get('watermark') != after.get('watermark'), 'quality': RUBRIC})
    print(f'Private eval artifacts: {args.output}')
    return 0
