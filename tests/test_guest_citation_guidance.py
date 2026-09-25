"""The invited guest must ask its model to cite the mail behind each claim (#40).

The only instruction an invited run's model gets is MAIL_GUIDANCE, appended to
the system prompt by both guest runners. It asked for artifact citations but
never for thread citations, and 0 of 55 stored invited answers carried one.
web/scripts/test-answer-citations.mjs renders these same forms as chips.
"""
import importlib
from pathlib import Path
import sys

import pytest

WORKER = Path(__file__).parents[1] / 'deploy/public/worker'
sys.path.insert(0, str(WORKER))

PI_PROFILES = ('mail-agent-pi-v1', 'mail-agent-pi-gemini-v1', 'mail-agent-pi-opus-v1')


def guidance():
    return importlib.import_module('guest_agent_pi').MAIL_GUIDANCE


def appended_system_prompt(argv):
    return argv[argv.index('--append-system-prompt') + 1]


@pytest.mark.parametrize('form', ['[ref:THREAD_ID]', '[att:ATTACHMENT_ID]'])
def test_guidance_asks_for_mail_citations_in_the_form_the_web_renders(form):
    assert form in guidance()


def test_guidance_says_where_the_ids_come_from_and_forbids_inventing_them():
    text = guidance()
    assert 'thread_id' in text and 'every factual claim' in text
    assert 'Never invent a thread or attachment ID' in text


@pytest.mark.parametrize('profile', PI_PROFILES)
def test_every_pi_profile_gets_the_citation_guidance(profile):
    runner = importlib.import_module('guest_agent_pi')
    assert profile in runner.PI_MODELS
    for workflow in (False, True):
        assert appended_system_prompt(runner.pi_argv(profile, workflow=workflow)) == guidance()


def test_native_claude_gets_the_citation_guidance():
    argv = importlib.import_module('guest_agent_claude').claude_argv()
    assert appended_system_prompt(argv) == guidance()


def test_researcher_subagent_returns_citations_the_parent_can_copy():
    text = (WORKER / 'workflow-agents/mail-researcher.md').read_text()
    assert '[ref:THREAD_ID]' in text
