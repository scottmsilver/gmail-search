"""Upgrade-only model/thinking escalation for a running Gemini agent.

A run starts on the ladder step Jev's effort router picked. Two signals move
it up, never down: the agent's own Jev check that the evidence does not yet
answer the question, and (optionally) a backstop every CALLS_PER_STEP calls.
Gemini accepts a conversation continued by a different model, thought
signatures included (checked 2026-09-24 for lite->flash and flash->pro), so a
step may change the model as well as the thinking level.
"""
from collections import OrderedDict
import logging
import threading

logger = logging.getLogger(__name__)

# Cheapest first. Measured per call: lite ~0.7 s, flash ~2 s, pro ~3.4 s.
LADDER = (
    ('gemini-3.5-flash-lite', 'LOW'),
    ('gemini-3.8-flash', 'LOW'),
    ('gemini-3.8-flash', 'MEDIUM'),
    ('gemini-3.8-flash', 'HIGH'),
    ('gemini-3.1-pro-preview', 'HIGH'),
)
LADDER_MODELS = tuple(dict.fromkeys(model for model, _ in LADDER))
# Call-count backstop, off: measured 2026-09-24 it raised thinking mid-run and
# made runs slower (31 s -> over 120 s) without making the agent finish.
# Set to a positive number to climb one step per that many model calls.
CALLS_PER_STEP = 0
# The agent's "do we have the answer yet?" check uses this question id; a
# probability below the threshold asks for a stronger step.
ANSWERED_QUESTION_ID = 'answered'
NOT_ANSWERED_BELOW = 0.5
# Early checks are low by nature (little evidence read yet). Climbing on each
# one put runs on pro by call 5 and made them slower, so a climb needs this
# many consecutive low checks and at least MIN_CALLS_BETWEEN_CLIMBS calls
# since the last climb.
LOW_CHECKS_TO_CLIMB = 2
MIN_CALLS_BETWEEN_CLIMBS = 3
MAX_TRACKED_RUNS = 1024


def step_of(model, level):
    """Ladder index of (model, level), or None when it is not on the ladder."""
    try:
        return LADDER.index((model, level))
    except ValueError:
        return None


class RunEscalation:
    def __init__(self):
        self._runs = OrderedDict()
        self._lock = threading.Lock()

    def _state(self, run_id):
        state = self._runs.setdefault(run_id, {'calls': 0, 'bumps': 0, 'step': None, 'low_checks': 0, 'climbed_at': 0,
                                               'held': False})
        self._runs.move_to_end(run_id)
        while len(self._runs) > MAX_TRACKED_RUNS:
            self._runs.popitem(last=False)
        return state

    def step_for_call(self, run_id, base_step):
        """Count one model call and return the ladder step it should use."""
        with self._lock:
            state = self._state(run_id)
            state['calls'] += 1
            backstop = (state['calls'] - 1) // CALLS_PER_STEP if CALLS_PER_STEP else 0
            step = min(len(LADDER) - 1, base_step + state['bumps'] + backstop)
            previous = base_step if state['step'] is None else state['step']
            if step != previous:
                logger.info('escalated run=%s %s/%s -> %s/%s at call %d', run_id[:8],
                    *LADDER[previous], *LADDER[step], state['calls'])
            state['step'] = step
            return step

    def hold(self, run_id):
        """Keep this run on its starting step. For parallel-subagent runs: each
        child checks only its own part, so its early low checks said nothing
        about the run and put every model call on pro (2026-09-24)."""
        with self._lock:
            self._state(run_id)['held'] = True

    def note_judgments(self, run_id, answers):
        """Climb a step when the agent's own check says it is not answered yet."""
        answered = answers.get(ANSWERED_QUESTION_ID) if type(answers) is dict else None
        if type(answered) is not dict or type(answered.get('noul')) not in (int, float):
            return
        with self._lock:
            state = self._state(run_id)
            if state['held'] or answered['noul'] >= NOT_ANSWERED_BELOW:
                state['low_checks'] = 0
                return
            state['low_checks'] += 1
            if (state['low_checks'] < LOW_CHECKS_TO_CLIMB
                    or state['calls'] - state['climbed_at'] < MIN_CALLS_BETWEEN_CLIMBS):
                return
            state['bumps'] += 1
            state['low_checks'] = 0
            state['climbed_at'] = state['calls']
        logger.info('escalation requested run=%s answered=%.2f', run_id[:8], answered['noul'])
