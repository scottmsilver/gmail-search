#!/usr/bin/env python3
"""The one fixed entrypoint the worker types into an `agent_full` guest.

The supervisor launches a single fixed command per image, so the choice between
agent runtimes cannot be made there. It is made here, from the `profile` of the
one-shot bootstrap the controller delivers over vsock -- a closed set validated
by guest_agent_bootstrap, never a guest- or model-supplied value.
"""
import asyncio
from pathlib import Path
import signal
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from guest_agent_bootstrap import CLAUDE_PROFILE, PROFILE, receive
import guest_agent_claude
import guest_agent_pi

RUNNERS = {PROFILE: guest_agent_pi.run, CLAUDE_PROFILE: guest_agent_claude.run}


def runner_for(config):
    """The runner for a validated bootstrap. An unknown profile cannot reach
    here -- validate_config refuses it -- but a KeyError is still a refusal."""
    return RUNNERS[config['profile']]


async def main():
    config = receive()
    task = asyncio.current_task()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, task.cancel)
    await runner_for(config)(config)


if __name__ == '__main__':
    try:
        asyncio.run(main())
        print('GMS_AGENT_RUNNER_COMPLETE', flush=True)
    except BaseException:
        print('GMS_AGENT_RUNNER_FAILED', flush=True)
        raise SystemExit(1)
