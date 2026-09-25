#!/usr/bin/env python3
"""Precompile the guest's Pi extensions into ROOT/jiti-cache at image build.

Runs inside warm-jiti-cache.sh's private mount namespace, where the image is
at the guest path (/tmp/runtime), because jiti keys its cache by absolute
path. Starts the real Pi exactly as a workflow run does (guest_agent_pi's
argv and agent dir), waits for its RPC reply that follows extension loading,
then stops it and keeps what jiti compiled into guest.JITI_CACHE. No gateway
is contacted: no prompt is sent.
"""
import json
import os
from pathlib import Path
import select
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import guest_agent_pi as guest  # noqa: E402

READY_SECONDS = 120
REQUEST_ID = 'warm-jiti-cache'


def _pi_environment(home):
    """The run environment prepare() builds, with a placeholder capability.
    Build time has no uid 1000 to hand files to, so ownership stays as is."""
    os.chown = lambda *args, **kwargs: None
    pi = home / 'pi'
    guest.write_pi_dir(pi, guest.PI_GEMINI_PROFILE)
    entry = guest.PI_MODELS[guest.PI_GEMINI_PROFILE]
    return {'HOME': str(home), 'PATH': str(guest.ROOT / 'bin') + ':/usr/bin:/bin', 'LANG': 'C.UTF-8',
            'TERM': 'dumb', entry['key']: 'build-time-placeholder', 'PI_CODING_AGENT_DIR': str(pi)}


def _wait_for_reply(process):
    deadline = time.monotonic() + READY_SECONDS
    buffered = b''
    while time.monotonic() < deadline:
        readable, _, _ = select.select([process.stdout], [], [], 1)
        if not readable:
            if process.poll() is not None:
                return False
            continue
        chunk = os.read(process.stdout.fileno(), 65536)
        if not chunk:
            return False
        buffered += chunk
        if REQUEST_ID.encode() in buffered:
            return True
    return False


def main():
    with tempfile.TemporaryDirectory() as scratch:
        home = Path(scratch)
        argv = guest.pi_argv(guest.PI_GEMINI_PROFILE, workflow=True)
        started = time.monotonic()
        process = subprocess.Popen(argv, cwd=home, env=_pi_environment(home), stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        try:
            process.stdin.write((json.dumps({'type': 'get_state', 'id': REQUEST_ID}) + '\n').encode())
            process.stdin.flush()
            ready = _wait_for_reply(process)
        finally:
            process.kill()
            process.wait()
    shutil.rmtree(guest.JITI_CACHE_IMAGE, ignore_errors=True)
    shutil.copytree(guest.JITI_CACHE, guest.JITI_CACHE_IMAGE)
    entries = len(list(guest.JITI_CACHE_IMAGE.iterdir()))
    print(f'jiti cache: {entries} modules in {time.monotonic() - started:.1f}s (ready={ready})')
    return 0 if ready and entries else 1


if __name__ == '__main__':
    raise SystemExit(main())
