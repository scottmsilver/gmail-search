"""ScaNN training-thread cap — keeps a rebuild from pinning every core."""

from __future__ import annotations

from gmail_search.index.builder import _scann_training_threads


def test_thread_cap_leaves_headroom_by_default(monkeypatch):
    monkeypatch.delenv("GMS_SCANN_TRAINING_THREADS", raising=False)
    monkeypatch.setattr("os.cpu_count", lambda: 20)
    # ~1/3 of cores → 6 on a 20-core box, so two concurrent rebuilds still
    # leave cores free for Postgres et al. Always >= 1.
    assert _scann_training_threads() == 6

    monkeypatch.setattr("os.cpu_count", lambda: 1)
    assert _scann_training_threads() == 1


def test_thread_cap_env_override(monkeypatch):
    monkeypatch.setenv("GMS_SCANN_TRAINING_THREADS", "3")
    assert _scann_training_threads() == 3
    # Garbage falls back to the core-fraction default, never crashes.
    monkeypatch.setenv("GMS_SCANN_TRAINING_THREADS", "not-a-number")
    monkeypatch.setattr("os.cpu_count", lambda: 12)
    assert _scann_training_threads() == 4
