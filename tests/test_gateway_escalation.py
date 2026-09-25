from gmail_search.gateway import escalation as module
from gmail_search.gateway.effort_router import EXHAUSTIVE_STEP, INVESTIGATE_STEP, LOOKUP_STEP, step_for
from gmail_search.gateway.escalation import LADDER, RunEscalation, step_of


def _steps(escalation, run, base, calls):
    return [escalation.step_for_call(run, base) for _ in range(calls)]


def test_ladder_starts_on_flash_lite_and_ends_on_pro():
    assert LADDER[0] == ('gemini-3.5-flash-lite', 'LOW')
    assert LADDER[-1][0] == 'gemini-3.1-pro-preview'
    assert step_of('gemini-3.8-flash', 'MEDIUM') == 2 and step_of('other', 'LOW') is None


def test_router_maps_effort_to_a_starting_step():
    assert step_for({'lookup': 0.9}) == LOOKUP_STEP
    assert step_for({'lookup': 0.6, 'investigate': 0.4}) == INVESTIGATE_STEP
    assert step_for({'exhaustive': 0.7}) == EXHAUSTIVE_STEP


def test_backstop_is_off_by_default():
    assert set(_steps(RunEscalation(), 'run-a', 0, 40)) == {0}


def test_enabled_backstop_climbs_one_step_per_block_of_calls_and_caps(monkeypatch):
    monkeypatch.setattr(module, 'CALLS_PER_STEP', 2)
    steps = _steps(RunEscalation(), 'run-a', 0, 20)
    assert steps[:2] == [0, 0] and steps[2] == 1 and steps[-1] == len(LADDER) - 1


LOW = {'answered': {'type': 'noul', 'noul': 0.2}}


def test_two_consecutive_low_checks_climb_one_step_only_for_that_run():
    escalation = RunEscalation()
    _steps(escalation, 'run-a', 0, 3)
    escalation.note_judgments('run-a', LOW)
    assert escalation.step_for_call('run-a', 0) == 0  # one early low check is expected
    escalation.note_judgments('run-a', LOW)
    assert escalation.step_for_call('run-a', 0) == 1
    assert escalation.step_for_call('run-b', 0) == 0


def test_a_held_run_never_climbs():
    escalation = RunEscalation()
    escalation.hold('run-a')
    _steps(escalation, 'run-a', 2, 3)
    for _ in range(6):
        escalation.note_judgments('run-a', LOW)
        escalation.step_for_call('run-a', 2)
    assert escalation.step_for_call('run-a', 2) == 2


def test_climbs_are_spaced_by_model_calls():
    escalation = RunEscalation()
    _steps(escalation, 'run-a', 0, 3)
    for _ in range(2):
        escalation.note_judgments('run-a', LOW)
    escalation.step_for_call('run-a', 0)
    for _ in range(2):
        escalation.note_judgments('run-a', LOW)
    assert escalation.step_for_call('run-a', 0) == 1  # only one call since the last climb


def test_an_answered_check_resets_the_low_streak():
    escalation = RunEscalation()
    _steps(escalation, 'run-a', 0, 3)
    escalation.note_judgments('run-a', LOW)
    escalation.note_judgments('run-a', {'answered': {'type': 'noul', 'noul': 0.8}})
    escalation.note_judgments('run-a', LOW)
    assert escalation.step_for_call('run-a', 0) == 0


def test_answered_or_unrelated_judgments_do_not_escalate():
    escalation = RunEscalation()
    escalation.note_judgments('run-a', {'answered': {'type': 'noul', 'noul': 0.9}})
    escalation.note_judgments('run-a', {'kind': {'type': 'choice', 'choice': 'order'}})
    assert escalation.step_for_call('run-a', 0) == 0


def test_a_parallel_hint_holds_the_run_on_its_starting_step():
    from types import SimpleNamespace
    from gmail_search.invited_runtime import _bind_routed_gemini

    class Router:
        def __init__(self, hint):
            self.hint = hint

        def start(self, prompt):
            return 'gemini-3.8-flash', 'MEDIUM', self.hint

    escalation = RunEscalation()
    service = SimpleNamespace(escalation=escalation, bind_profile=lambda run_id, profile: None)
    import dataclasses
    profile = dataclasses.make_dataclass('P', ['model', 'thinking_level'])('m', 'LOW')
    assert _bind_routed_gemini(service, profile, Router('Planning hint'))('run-a', 'q') == 'Planning hint'
    _bind_routed_gemini(service, profile, Router(None))('run-b', 'q')
    assert escalation._runs['run-a']['held'] and 'run-b' not in escalation._runs
