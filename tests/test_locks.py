"""Tests for the write_lock context managers.

The fcntl-based OS lock was retired on 2026-04-20 when the SQLite
backend went away. Postgres handles writer concurrency via MVCC, so the
context managers are now no-ops that still yield so call sites keep
working unchanged. These tests pin that behaviour.
"""

from __future__ import annotations


def test_write_lock_yields_noop(tmp_path):
    from gmail_search.locks import write_lock

    with write_lock(tmp_path):
        pass
    with write_lock(tmp_path):
        pass  # re-entering after release must succeed


def test_try_write_lock_yields_noop(tmp_path):
    """Under Postgres there's nothing to contend for, so the
    non-blocking variant always 'acquires' immediately.
    """
    from gmail_search.locks import try_write_lock

    with try_write_lock(tmp_path):
        pass
    with try_write_lock(tmp_path):
        pass


def test_write_and_try_dont_interfere(tmp_path):
    """Nested acquisitions across both flavours are fine — they're
    both yield-only, and that's the whole point.
    """
    from gmail_search.locks import try_write_lock, write_lock

    with write_lock(tmp_path):
        with try_write_lock(tmp_path):
            pass
