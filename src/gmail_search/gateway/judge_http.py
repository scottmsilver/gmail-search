"""Run-bound Jev judgments for the guest agent: typed questions, not generation.

The guest asks instead of guessing (relevance, order vs quote, is the question
answered yet). State and questions are bounded again here; the TypeSafe key
never leaves the host.
"""
import asyncio
import json
import re

from fastapi import HTTPException, Request
import httpx

from .jev import answers_from, request_body
from .retrieval_http import run_while_connected
from .tool_deadline import TOOL_DEADLINE_SECONDS, publication_deadline

MAX_STATE_BYTES = 32 * 1024
MAX_QUESTIONS = 8
_ID = re.compile(r'[A-Za-z][A-Za-z0-9_]{0,63}\Z')
_TYPES = ('noul', 'choice', 'score')


def _valid_question(question):
    if (type(question) is not dict or set(question) - {'type', 'instructions', 'criteria'}
            or question.get('type') not in _TYPES
            or type(question.get('instructions')) is not str or not question['instructions'].strip()):
        return False
    criteria = question.get('criteria')
    return {'noul': criteria is None or (type(criteria) is dict and not set(criteria) - {'true', 'false'}),
            'choice': type(criteria) is dict and 2 <= len(criteria) <= 32,
            'score': type(criteria) is list and 2 <= len(criteria) <= 10}[question['type']]


def _valid_request(value):
    questions = value.get('questions')
    state = value.get('state')
    return (set(value) == {'state', 'questions'}
            and type(state) is str and 0 < len(state.encode('utf-8')) <= MAX_STATE_BYTES
            and type(questions) is dict and 1 <= len(questions) <= MAX_QUESTIONS
            and all(type(key) is str and _ID.fullmatch(key) for key in questions)
            and all(_valid_question(question) for question in questions.values())
            and len(json.dumps(questions).encode('utf-8')) <= MAX_STATE_BYTES)


class RunJudgeService:
    def __init__(self, capabilities, config, client, *, escalation=None):
        self.capabilities, self.config, self.client = capabilities, config, client
        self.escalation = escalation

    async def authorize(self, token):
        return await asyncio.to_thread(self.capabilities.authorize, token, audience='retrieval', operation='judge')

    async def judge(self, token, state, questions):
        lease = await self.authorize(token)
        response = await self.client.post(self.config.url, headers=self.config.headers,
            json=request_body(state, questions), timeout=TOOL_DEADLINE_SECONDS)
        if response.status_code != 200:
            raise RuntimeError('Judgment service unavailable')
        answers = answers_from(response.json())
        if self.escalation is not None:
            self.escalation.note_judgments(lease.run_id, answers)
        return {'answers': answers}


def add_judge_routes(app, service, token_from_request, read_json):
    @app.post('/v1/judge')
    async def judge(request: Request):
        token = token_from_request(request)
        await service.authorize(token)
        value = await read_json(request)
        if type(value) is not dict or not _valid_request(value):
            raise HTTPException(400, 'Invalid judge request')
        try:
            return await run_while_connected(
                request, service.judge(token, value['state'], value['questions']),
                before_publish=lambda: service.authorize(token),
                publication_deadline=publication_deadline(),
            )
        except (ValueError, KeyError, httpx.HTTPError):
            raise RuntimeError('Judgment service unavailable') from None
