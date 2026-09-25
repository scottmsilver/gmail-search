"""Pick a run's starting model and thinking level from its question, using Jev.

Jev (TypeSafe's System One model) answers one Choice about how much
investigation the question needs; code maps the probabilities to a step on
the escalation ladder. Any failure falls back to DEFAULT_STEP so routing can
never block a run.
The endpoint and key come from configuration, never from code.
"""
from dataclasses import dataclass
import logging
import time

import httpx

from .escalation import LADDER
from .jev import JevConfig, answers_from, request_body

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 1.5
# Ladder steps (see escalation.LADDER): flash LOW for a sure lookup, flash
# MEDIUM to investigate, flash HIGH for exhaustive sweeps; the default is the
# investigate step. Lookups do not start on flash-lite (step 0): it answered
# fast but 1 of 3 live runs on 2026-09-24 named the wrong heater model.
LOOKUP_STEP, INVESTIGATE_STEP, EXHAUSTIVE_STEP = 1, 2, 3
DEFAULT_STEP = INVESTIGATE_STEP
LOW_WHEN_LOOKUP_AT_LEAST = 0.8
HIGH_WHEN_EXHAUSTIVE_AT_LEAST = 0.5

# Asked in the same Jev call: whether the question splits into independent
# lookups worth running as parallel subagents.
PARALLEL_QUESTION = {
    'type': 'noul',
    'instructions': ('Does answering `question` need several independent email lookups that could run in '
                     'parallel, such as different senders, months, invoices or sub-questions?'),
    'criteria': {'true': 'Two or more separate lookups whose results are combined at the end.',
                 'false': 'One line of investigation, or steps that each depend on the previous one.'},
}
PARALLEL_AT_LEAST = 0.6
PARALLEL_HINT = ('Planning hint: this question has independent parts. Give each part to a mail-researcher '
                 'subagent, run them in parallel in the foreground (async: false), wait for every result, '
                 'then combine their findings into one answer.')

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


def step_for(probabilities):
    """Map Jev's effort probabilities to a starting ladder step."""
    if probabilities.get('exhaustive', 0.0) >= HIGH_WHEN_EXHAUSTIVE_AT_LEAST:
        return EXHAUSTIVE_STEP
    if probabilities.get('lookup', 0.0) >= LOW_WHEN_LOOKUP_AT_LEAST:
        return LOOKUP_STEP
    return DEFAULT_STEP


@dataclass(frozen=True)
class EffortRouter:
    config: JevConfig
    client: httpx.Client

    def _ask(self, question):
        response = self.client.post(self.config.url, timeout=TIMEOUT_SECONDS, headers=self.config.headers,
            json=request_body({'question': question}, {'effort': EFFORT_QUESTION, 'parallel': PARALLEL_QUESTION}))
        response.raise_for_status()
        answers = answers_from(response.json())
        effort, parallel = answers['effort'], answers.get('parallel', {})
        return effort['probabilities'], effort.get('confidence'), parallel.get('noul', 0.0)

    def start(self, question):
        """(model, thinking level, planning hint or None) the run should start with."""
        started = time.perf_counter()
        try:
            probabilities, confidence, parallel = self._ask(question)
        except Exception as error:  # Routing is advisory; never fail the run.
            logger.warning('effort router unavailable (%s); using %s', type(error).__name__, LADDER[DEFAULT_STEP])
            return (*LADDER[DEFAULT_STEP], None)
        model, level = LADDER[step_for(probabilities)]
        hint = PARALLEL_HINT if type(parallel) in (int, float) and parallel >= PARALLEL_AT_LEAST else None
        logger.info('effort router start=%s/%s parallel=%.2f probabilities=%s confidence=%s %.0fms',
            model, level, parallel if type(parallel) in (int, float) else -1,
            {k: round(v, 2) for k, v in probabilities.items()}, confidence,
            (time.perf_counter() - started) * 1000)
        return model, level, hint
