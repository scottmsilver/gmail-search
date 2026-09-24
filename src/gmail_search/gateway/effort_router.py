"""Pick Gemini's thinking level for a run from its question, using Jev.

Jev (TypeSafe's System One model) answers one Choice about how much
investigation the question needs; code maps the probabilities to a level.
Any failure falls back to DEFAULT_LEVEL so routing can never block a run.
The endpoint and key come from configuration, never from code.
"""
from dataclasses import dataclass
import logging
import time

import httpx

from .jev import JevConfig, answers_from, request_body

logger = logging.getLogger(__name__)

DEFAULT_LEVEL = 'MEDIUM'
TIMEOUT_SECONDS = 1.5
# Cheap only when Jev is sure the question is a narrow lookup; deep when it
# needs complete coverage or aggregation. Everything else stays at MEDIUM.
LOW_WHEN_LOOKUP_AT_LEAST = 0.8
HIGH_WHEN_EXHAUSTIVE_AT_LEAST = 0.5

EFFORT_QUESTION = {
    'type': 'choice',
    'instructions': "How much investigation will answering `question` over the user's email archive need?",
    'criteria': {
        'lookup': 'A single narrow fact (a date, a name, which item) that one or two searches and a thread read will settle.',
        'investigate': ('Needs several searches, reading multiple threads or attachments, and reconciling them: '
                        'totals or amounts spread across invoices, comparisons, timelines, or what happened overall.'),
        'exhaustive': 'Needs complete coverage (all/every), aggregation over many emails, or computation.',
    },
}


def level_for(probabilities):
    """Map Jev's effort probabilities to a Gemini thinking level."""
    if probabilities.get('exhaustive', 0.0) >= HIGH_WHEN_EXHAUSTIVE_AT_LEAST:
        return 'HIGH'
    if probabilities.get('lookup', 0.0) >= LOW_WHEN_LOOKUP_AT_LEAST:
        return 'LOW'
    return DEFAULT_LEVEL


@dataclass(frozen=True)
class EffortRouter:
    config: JevConfig
    client: httpx.Client

    def _ask(self, question):
        response = self.client.post(self.config.url, timeout=TIMEOUT_SECONDS, headers=self.config.headers,
            json=request_body({'question': question}, {'effort': EFFORT_QUESTION}))
        response.raise_for_status()
        answer = answers_from(response.json())['effort']
        return answer['probabilities'], answer.get('confidence')

    def thinking_level(self, question):
        started = time.perf_counter()
        try:
            probabilities, confidence = self._ask(question)
        except Exception as error:  # Routing is advisory; never fail the run.
            logger.warning('effort router unavailable (%s); using %s', type(error).__name__, DEFAULT_LEVEL)
            return DEFAULT_LEVEL
        level = level_for(probabilities)
        logger.info('effort router level=%s probabilities=%s confidence=%s %.0fms', level,
            {k: round(v, 2) for k, v in probabilities.items()}, confidence,
            (time.perf_counter() - started) * 1000)
        return level
