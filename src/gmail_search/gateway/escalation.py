"""Upgrade-only thinking escalation for a running Gemini agent.

A run starts at the level Jev's effort router picked. Two signals raise it,
never lower it: the agent's own Jev check that the evidence does not yet
answer the question, and a backstop after every CALLS_PER_STEP model calls.
Same model throughout, so mid-conversation changes are safe.
"""
from collections import OrderedDict
import logging
import threading

logger = logging.getLogger(__name__)

LEVELS = ('MINIMAL', 'LOW', 'MEDIUM', 'HIGH')
# Call-count backstop, off: measured 2026-09-24 it raised thinking mid-run and
# made runs slower (31 s -> over 120 s) without making the agent finish.
# Set to a positive number to bump one level per that many model calls.
CALLS_PER_STEP = 0
# The agent's "do we have the answer yet?" check uses this question id; a
# probability below the threshold asks for more reasoning.
ANSWERED_QUESTION_ID = 'answered'
NOT_ANSWERED_BELOW = 0.5
MAX_TRACKED_RUNS = 1024


class ThinkingEscalation:
    def __init__(self):
        self._runs = OrderedDict()
        self._lock = threading.Lock()

    def _state(self, run_id):
        state = self._runs.setdefault(run_id, {'calls': 0, 'bumps': 0, 'level': None})
        self._runs.move_to_end(run_id)
        while len(self._runs) > MAX_TRACKED_RUNS:
            self._runs.popitem(last=False)
        return state

    def level_for_call(self, run_id, base_level):
        """Count one model call and return the level it should use."""
        with self._lock:
            state = self._state(run_id)
            state['calls'] += 1
            backstop = (state['calls'] - 1) // CALLS_PER_STEP if CALLS_PER_STEP else 0
            steps = state['bumps'] + backstop
            level = LEVELS[min(len(LEVELS) - 1, LEVELS.index(base_level) + steps)]
            if level != (state['level'] or base_level):
                logger.info('thinking escalated run=%s %s -> %s at call %d (bumps=%d)',
                    run_id[:8], state['level'] or base_level, level, state['calls'], state['bumps'])
            state['level'] = level
            return level

    def note_judgments(self, run_id, answers):
        """Raise the level when the agent's own check says it is not answered yet."""
        answered = answers.get(ANSWERED_QUESTION_ID) if type(answers) is dict else None
        if type(answered) is not dict or type(answered.get('noul')) not in (int, float):
            return
        if answered['noul'] < NOT_ANSWERED_BELOW:
            with self._lock:
                self._state(run_id)['bumps'] += 1
            logger.info('thinking bump requested run=%s answered=%.2f', run_id[:8], answered['noul'])
