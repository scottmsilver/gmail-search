"""Process-level write lock — now a no-op under Postgres.

Historically this module used `fcntl.flock` to serialise SQLite writers
(watch daemon + backfill job) so they didn't hit `database is locked`.
Postgres handles writer concurrency via MVCC, so the lock is pure
latency tax and the fcntl machinery was deleted on 2026-04-20.

The context managers still yield so every existing call site keeps
working unchanged — they just don't acquire anything.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path


@contextmanager
def write_lock(data_dir: Path):
    """Blocking-acquire-style context manager. Under Postgres it's a
    no-op — MVCC serialises writers at the row level so there's no
    benefit to an OS-level mutex.
    """
    yield


@contextmanager
def try_write_lock(data_dir: Path):
    """Non-blocking variant. Under Postgres it always "acquires"
    immediately because there is nothing to contend for.
    """
    yield
