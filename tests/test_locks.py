"""Tests for the write_lock coordination primitive.

Two writers (watch + backfill) both want to upsert/reindex on a single
SQLite DB. fcntl.flock is process-level, so we test blocking semantics
via real subprocesses — threads inside a single process wouldn't see
each other blocked.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest


def _hold_lock_script() -> str:
    """A throwaway script that grabs write_lock and sleeps. Used as a
    'busy' peer in tests.
    """
    return (
        "import sys, time;"
        "sys.path.insert(0, 'src');"
        "from pathlib import Path;"
        "from gmail_search.locks import write_lock;"
        "p = Path(sys.argv[1]);"
        "hold = float(sys.argv[2]);"
        "import os;"
        "print('acquiring', flush=True);"
        "lock_ctx = write_lock(p);"
        "lock_ctx.__enter__();"
        "print('held', flush=True);"
        "time.sleep(hold);"
        "lock_ctx.__exit__(None, None, None);"
        "print('released', flush=True)"
    )


def test_write_lock_round_trip(tmp_path):
    from gmail_search.locks import write_lock

    with write_lock(tmp_path):
        pass
    with write_lock(tmp_path):
        pass  # second acquire after release must succeed


def test_write_lock_blocks_concurrent_peer(tmp_path):
    """Child holds the lock for 2s. We try a blocking acquire — must
    wait for the child to release, then succeed. Total wall clock must
    be ≥ child's hold time.
    """
    from gmail_search.locks import write_lock

    child = subprocess.Popen(
        [sys.executable, "-c", _hold_lock_script(), str(tmp_path), "2.0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    # Wait until child prints 'held' so we know the lock is taken.
    assert child.stdout is not None
    line = child.stdout.readline().strip()
    assert line == "acquiring"
    line = child.stdout.readline().strip()
    assert line == "held"

    start = time.monotonic()
    with write_lock(tmp_path):
        pass
    elapsed = time.monotonic() - start
    # Child was told to hold for 2s; we started waiting after 'held'
    # was printed, so elapsed should be ≥ ~1s (generous floor for
    # scheduling jitter).
    assert elapsed >= 0.8, f"acquire returned too fast ({elapsed:.2f}s) — lock didn't block"

    child.wait(timeout=5)
    assert child.returncode == 0


def test_write_lock_nonblocking_raises_when_held(tmp_path):
    """The non-blocking variant is useful for 'is the writer busy?'
    probes without stalling the caller.
    """
    from gmail_search.locks import try_write_lock, write_lock

    with write_lock(tmp_path):
        with pytest.raises(BlockingIOError):
            with try_write_lock(tmp_path):
                pass


def test_try_write_lock_succeeds_when_free(tmp_path):
    from gmail_search.locks import try_write_lock

    with try_write_lock(tmp_path):
        pass  # no peer → acquire immediately
