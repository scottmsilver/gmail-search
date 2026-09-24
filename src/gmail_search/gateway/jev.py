"""Shared TypeSafe Jev plumbing: configuration, request body and answer parsing.

The endpoint and key come from the service environment (TYPESAFE_URL,
TYPESAFE_API_KEY); nothing here names a server.
"""
from dataclasses import dataclass
import os

MODEL = 'jev-latest'


@dataclass(frozen=True)
class JevConfig:
    url: str
    api_key: str = ''

    def __repr__(self):
        return f'JevConfig(url={self.url!r})'

    @property
    def headers(self):
        return {'Authorization': 'Bearer ' + self.api_key}


def jev_config_from_env():
    """JevConfig when both variables are set, else None (Jev features off)."""
    url, key = os.environ.get('TYPESAFE_URL'), os.environ.get('TYPESAFE_API_KEY')
    return JevConfig(url, key) if url and key else None


def request_body(state, questions):
    return {'model': MODEL, 'state': state, 'questions': questions}


def answers_from(payload):
    answers = payload['answers']
    if type(answers) is not dict:
        raise ValueError('Invalid Jev answers')
    return answers
