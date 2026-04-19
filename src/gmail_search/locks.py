"""Process-level advisory file locks for serialising writers.

SQLite is single-writer; if the watch daemon and a backfill job both
try to upsert at once, one of them hits `database is locked` after the
busy_timeout elapses and crashes. We side-step the race with an
explicit advisory lock held around each pipeline cycle. Whichever
process acquires first runs its cycle to completion; the other waits.

fcntl.flock auto-releases when the process exits, so there are no
stale-lock recovery paths to worry about. All callers share a single
lockfile at <data_dir>/.writer.lock.
"""

from __future__ import annotations

import fcntl
from contextlib import contextmanager
from pathlib import Path

LOCK_FILENAME = ".writer.lock"


def _lock_path(data_dir: Path) -> Path:
    return Path(data_dir) / LOCK_FILENAME


def _open_lock_handle(data_dir: Path):
    path = _lock_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    # "a" so we never truncate an existing lockfile held by a peer.
    return open(path, "a")


@contextmanager
def write_lock(data_dir: Path):
    """Blocking acquire of the writer lock. Releases on context exit.

    Use around each cycle of work that writes to the DB or rebuilds
    the on-disk index. Holding is coarse: one watch cycle, or one
    update batch. Peers wait, they don't crash.
    """
    handle = _open_lock_handle(data_dir)
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


@contextmanager
def try_write_lock(data_dir: Path):
    """Non-blocking variant — raises BlockingIOError when the lock is
    held by another process. Useful for "is a write in flight?" probes
    from the server without stalling request handling.
    """
    handle = _open_lock_handle(data_dir)
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            raise
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        if not handle.closed:
            handle.close()
