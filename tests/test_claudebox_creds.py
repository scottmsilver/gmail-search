"""Tests for claudebox credential expiry awareness.

Covers `_read_expires_at` (ms vs s conversion, missing/garbage inputs)
and `credentials_health` (ok/expiring/expired/unknown), with the host
source and container mount paths monkeypatched onto tmp files so no
real `~/.claude/.credentials.json` is touched.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from gmail_search import claudebox_creds as cc


def _write_creds(path: Path, expires_at) -> None:
    """Write a minimal credentials JSON with the given expiresAt. Pass
    expires_at=None to omit the key entirely."""
    oauth: dict = {"accessToken": "tok", "refreshToken": "ref", "scopes": []}
    if expires_at is not None:
        oauth["expiresAt"] = expires_at
    path.write_text(json.dumps({"claudeAiOauth": oauth}))


# ── _read_expires_at ──────────────────────────────────────────────


def test_read_expires_at_milliseconds(tmp_path):
    """A value above ~1e12 is epoch ms and must be divided by 1000."""
    p = tmp_path / "creds.json"
    ms = 1_700_000_000_000  # nice round epoch-ms
    _write_creds(p, ms)
    assert cc._read_expires_at(p) == pytest.approx(1_700_000_000.0)


def test_read_expires_at_seconds_passthrough(tmp_path):
    """A value below the ms threshold is treated as epoch seconds as-is."""
    p = tmp_path / "creds.json"
    _write_creds(p, 1_700_000_000)
    assert cc._read_expires_at(p) == pytest.approx(1_700_000_000.0)


def test_read_expires_at_missing_file(tmp_path):
    assert cc._read_expires_at(tmp_path / "nope.json") is None


def test_read_expires_at_garbage_json(tmp_path):
    p = tmp_path / "creds.json"
    p.write_text("{not valid json")
    assert cc._read_expires_at(p) is None


def test_read_expires_at_missing_key(tmp_path):
    p = tmp_path / "creds.json"
    _write_creds(p, None)
    assert cc._read_expires_at(p) is None


def test_read_expires_at_non_numeric(tmp_path):
    p = tmp_path / "creds.json"
    p.write_text(json.dumps({"claudeAiOauth": {"expiresAt": "soon"}}))
    assert cc._read_expires_at(p) is None


def test_read_expires_at_oauth_not_dict(tmp_path):
    p = tmp_path / "creds.json"
    p.write_text(json.dumps({"claudeAiOauth": "nope"}))
    assert cc._read_expires_at(p) is None


# ── credentials_health ────────────────────────────────────────────


@pytest.fixture
def mount_path(tmp_path, monkeypatch):
    """Point the mount path at a tmp file and stub the sync so it's a
    no-op (we set up the mount file directly per test)."""
    mount = tmp_path / "mount-creds.json"
    monkeypatch.setattr(cc, "_mount_credentials_path", lambda: mount)
    monkeypatch.setattr(cc, "sync_credentials_if_stale", lambda *a, **k: False)
    return mount


def test_credentials_health_ok(mount_path):
    _write_creds(mount_path, (time.time() + 3600) * 1000)
    status, expires_at = cc.credentials_health()
    assert status == "ok"
    assert expires_at > time.time()


def test_credentials_health_expiring(mount_path):
    # 5 minutes out, inside the default 20-minute margin.
    _write_creds(mount_path, (time.time() + 300) * 1000)
    status, _ = cc.credentials_health()
    assert status == "expiring"


def test_credentials_health_expired(mount_path):
    _write_creds(mount_path, (time.time() - 60) * 1000)
    status, expires_at = cc.credentials_health()
    assert status == "expired"
    assert expires_at < time.time()


def test_credentials_health_unknown_missing_file(mount_path):
    # mount_path file never created.
    status, expires_at = cc.credentials_health()
    assert status == "unknown"
    assert expires_at is None


def test_credentials_health_unknown_garbage(mount_path):
    mount_path.write_text("garbage")
    status, expires_at = cc.credentials_health()
    assert status == "unknown"
    assert expires_at is None


def test_credentials_health_custom_margin(mount_path):
    # 10 minutes out: "ok" with a 5-min margin, "expiring" with a 30-min one.
    _write_creds(mount_path, (time.time() + 600) * 1000)
    assert cc.credentials_health(margin_seconds=5 * 60)[0] == "ok"
    assert cc.credentials_health(margin_seconds=30 * 60)[0] == "expiring"


def test_credentials_health_calls_sync_first(tmp_path, monkeypatch):
    """credentials_health must sync host→mount BEFORE classifying so the
    mount reflects the latest host token."""
    mount = tmp_path / "mount-creds.json"
    monkeypatch.setattr(cc, "_mount_credentials_path", lambda: mount)
    calls: list[bool] = []

    def _fake_sync(*a, **k):
        calls.append(True)
        _write_creds(mount, (time.time() + 3600) * 1000)
        return True

    monkeypatch.setattr(cc, "sync_credentials_if_stale", _fake_sync)
    status, _ = cc.credentials_health()
    assert calls == [True]
    assert status == "ok"


def test_credentials_health_survives_sync_failure(mount_path, monkeypatch):
    """A raising sync must not crash health — it classifies whatever is
    already on the mount."""

    def _boom(*a, **k):
        raise RuntimeError("sync blew up")

    monkeypatch.setattr(cc, "sync_credentials_if_stale", _boom)
    _write_creds(mount_path, (time.time() + 3600) * 1000)
    assert cc.credentials_health()[0] == "ok"
