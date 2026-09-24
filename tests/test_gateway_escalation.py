from gmail_search.gateway import escalation as module
from gmail_search.gateway.escalation import ThinkingEscalation


def _levels(escalation, run, base, calls):
    return [escalation.level_for_call(run, base) for _ in range(calls)]


def test_backstop_is_off_by_default():
    assert set(_levels(ThinkingEscalation(), 'run-a', 'LOW', 40)) == {'LOW'}


def test_enabled_backstop_raises_one_level_per_block_of_calls_and_caps_at_high(monkeypatch):
    monkeypatch.setattr(module, 'CALLS_PER_STEP', 4)
    levels = _levels(ThinkingEscalation(), 'run-a', 'LOW', 13)
    assert levels[:4] == ['LOW'] * 4 and levels[4] == 'MEDIUM' and levels[8] == 'HIGH' and levels[-1] == 'HIGH'


def test_not_answered_judgment_raises_the_next_call_only_for_that_run():
    escalation = ThinkingEscalation()
    escalation.level_for_call('run-a', 'LOW')
    escalation.note_judgments('run-a', {'answered': {'type': 'noul', 'noul': 0.2}})
    assert escalation.level_for_call('run-a', 'LOW') == 'MEDIUM'
    assert escalation.level_for_call('run-b', 'LOW') == 'LOW'


def test_answered_or_unrelated_judgments_do_not_escalate():
    escalation = ThinkingEscalation()
    escalation.note_judgments('run-a', {'answered': {'type': 'noul', 'noul': 0.9}})
    escalation.note_judgments('run-a', {'kind': {'type': 'choice', 'choice': 'order'}})
    assert escalation.level_for_call('run-a', 'LOW') == 'LOW'
