"""The landing lock, in the protocol .claude/issue-loop/lib.sh uses, so the
deployer and the issue loop exclude each other over one lock.

The lock is a directory created with mkdir, holding an `owner` file whose line
is `<label> pid=<pid> host=<host> <UTC time>`, written then read back. Every
read-the-holder-then-act step runs under flock on the side file
`<lock>.mutex` (lib.sh takes the same flock). A holder is stale only when its
pid is gone and it was written on this host; no pid recorded means a live
holder. A stale lock is renamed aside, and given back (rename without
clobbering) if what moved is not the holder that was read. Release removes
the directory only while it still names this process.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import re
import shutil
import socket
import time

POLL_SECONDS = 10


class LockHeld(RuntimeError):
    pass


def owner_line(label: str) -> str:
    stamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    return f'{label} pid={os.getpid()} host={socket.gethostname()} {stamp}'


def read_holder(lock: Path) -> str | None:
    if not lock.is_dir():
        return None
    try:
        return (lock / 'owner').read_text().strip()
    except OSError:
        return 'unknown holder'


def holder_is_stale(holder: str, host: str | None = None) -> bool:
    pid = re.search(r'pid=(\d+)', holder)
    if not pid:
        return False  # no pid recorded: assume a live holder, as lib.sh does
    wrote = re.search(r'host=(\S+)', holder)
    if wrote and wrote.group(1) != (host or socket.gethostname()):
        return False  # a pid means nothing on another machine
    try:
        os.kill(int(pid.group(1)), 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


@contextmanager
def _mutex(lock: Path):
    with open(f'{lock}.mutex', 'a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _try_claim(lock: Path, mine: str) -> bool:
    try:
        lock.mkdir()
    except FileExistsError:
        return False
    (lock / 'owner').write_text(mine + '\n')
    return read_holder(lock) == mine


def _clear_stale(lock: Path, holder: str) -> None:
    aside = Path(f'{lock}.stale-{os.getpid()}-{time.time_ns()}')
    try:
        os.rename(lock, aside)
    except OSError:
        return
    if read_holder(aside) != holder:
        # Not the holder we judged stale: give the name back without clobbering.
        try:
            if not lock.exists():
                os.rename(aside, lock)
        except OSError:
            pass
    shutil.rmtree(aside, ignore_errors=True)


def acquire(lock: Path, label: str, *, wait_seconds: int = 0, poll=POLL_SECONDS, sleep=time.sleep) -> str:
    """Take the lock; return our owner line. Raises LockHeld when it stays held."""
    lock.parent.mkdir(parents=True, exist_ok=True)
    waited = 0
    while True:
        with _mutex(lock):
            mine = owner_line(label)
            if _try_claim(lock, mine):
                return mine
            holder = read_holder(lock)
            if holder is not None and holder_is_stale(holder):
                _clear_stale(lock, holder)
                continue
        if holder is None:
            continue
        if waited >= wait_seconds:
            raise LockHeld(f'landing lock held by: {holder}')
        sleep(poll)
        waited += poll


def release(lock: Path, mine: str) -> None:
    """Remove the lock only while it still names this process."""
    with _mutex(lock):
        if read_holder(lock) == mine:
            shutil.rmtree(lock, ignore_errors=True)


@contextmanager
def held(lock: Path, label: str, *, wait_seconds: int = 0):
    mine = acquire(lock, label, wait_seconds=wait_seconds)
    try:
        yield mine
    finally:
        release(lock, mine)
