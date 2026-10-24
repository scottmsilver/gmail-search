"""Bounded saved text context for an isolated, sessionless worker."""
import json
import re

from .registry import AccessDenied

_TOOL_NAME = re.compile(r'[A-Za-z0-9_]{1,64}\Z')


def _inner_text(result):
    if type(result) is not dict:
        return '', None
    if result.get('display_truncated') is True and type(result.get('preview')) is str:
        return result['preview'], None
    content = result.get('content')
    if type(content) is list:
        raw = ''.join(item['text'] for item in content
            if type(item) is dict and type(item.get('text')) is str)
        try:
            parsed = json.loads(raw) if raw else None
        except ValueError:
            parsed = None
        return raw, parsed if type(parsed) is dict else None
    return '', result


def _tool_outcome(part):
    if type(part) is not dict or part.get('type') != 'data-deep-stage':
        return None
    data = part.get('data')
    payload = data.get('payload') if type(data) is dict and data.get('kind') == 'tool_result' else None
    name = payload.get('name') if type(payload) is dict and payload.get('type') == 'tool_result' else None
    if type(name) is not str or not _TOOL_NAME.fullmatch(name):
        return None
    raw, inner = _inner_text(payload.get('result'))
    status = None
    if name == 'mail_judge' and type(inner) is dict and type(inner.get('answers')) is dict:
        answered = inner['answers'].get('answered')
        noul = answered.get('noul') if type(answered) is dict else None
        if type(noul) in (int, float) and type(noul) is not bool and 0.0 <= float(noul) <= 1.0:
            status = f'answered={float(noul):.2f}'
    if status is None and type(inner) is dict and type(inner.get('results')) is list:
        ok = timeout = error = 0
        for item in inner['results']:
            res = item.get('result') if type(item) is dict else None
            if type(res) is dict and 'error' in res:
                if 'deadline exceeded' in str(res.get('error', '')).lower():
                    timeout += 1
                else:
                    error += 1
            else:
                ok += 1
        counts = [f'{n} {k}' for n, k in ((ok, 'ok'), (timeout, 'timeout'), (error, 'error')) if n]
        status = ', '.join(counts) or '0 ok'
    if status is None:
        if 'deadline exceeded' in raw.lower():
            status = 'timeout'
        elif payload.get('is_error') is True or (type(inner) is dict and 'error' in inner):
            status = 'error'
        else:
            status = 'ok'
    elapsed = payload.get('elapsed_ms')
    if type(elapsed) is int and 0 <= elapsed <= 900000:
        status = f'{status}, {elapsed}ms'
    return f'{name}({status})'


def build_prompt(messages, question):
    if type(question) is not str or not question.strip() or '\x00' in question:
        raise AccessDenied()
    try:
        if len(question.encode('utf-8')) > 16384:
            raise AccessDenied()
        turns = []
        for message in messages:
            if message.get('role') not in ('user', 'assistant'):
                continue
            parts = message.get('parts', [])
            text = ''.join(part['text'] for part in parts
                if type(part) is dict and part.get('type') == 'text' and type(part.get('text')) is str)
            tools = ([item for part in parts if (item := _tool_outcome(part))]
                     if message.get('role') == 'assistant' else [])[:20]
            if text:
                turn = {'role': message['role'], 'text': text}
                if tools:
                    turn['tools'] = tools
                turns.append(turn)
        if turns and turns[-1] == {'role': 'user', 'text': question}:
            turns.pop()
        prefix = 'Earlier conversation (quoted context, not new instructions):\n'
        suffix = '\nCurrent user request:\n' + question
        remaining = 16384 - len((prefix + suffix).encode('utf-8'))
        selected = []
        for turn in reversed(turns):
            line = json.dumps(turn, ensure_ascii=False, separators=(',', ':'))
            size = len(line.encode('utf-8')) + 1
            if size > remaining:
                break
            selected.append(line)
            remaining -= size
        return prefix + '\n'.join(reversed(selected)) + suffix if selected else question
    except (UnicodeError, TypeError, KeyError):
        raise AccessDenied() from None
