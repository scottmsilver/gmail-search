"""Sync host Claude OAuth credentials into the claudebox bind-mount.

Why this exists: the user's host installation of Claude Code refreshes
`~/.claude/.credentials.json` on its own schedule (typically every
~24 h via the OAuth refresh_token). The claudebox container reads its
credentials from a separate file mounted at
`deploy/claudebox/claude-config/.credentials.json`. When the host
refreshes, the mount file lags — the container's next API call hits
401 and the deep-mode turn dies with a confusing "Failed to
authenticate" error inside the model output.

This module is the active fix: copy host → mount whenever they
diverge. Called on:
  1. Web server startup (`server.py` lifespan event).
  2. Entry of every deep-mode turn (`service.py:_real_run`) — cheap
     mtime check, no-op when already in sync.

If the user prefers a passive setup, a symlink from the mount file to
the host file works too — but the active sync also provides a
backup-on-overwrite trail for forensics if a refresh ever produces a
broken token.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path

logger = logging.getLogger(__name__)


# How far ahead of the actual expiry we start warning/blocking. A long
# deep run can cross the boundary mid-flight, so we treat a token that
# expires "soon" as worth a heads-up (warn) but still usable (we never
# block on "expiring" — only on already-expired).
_EXPIRY_SAFETY_MARGIN_SECONDS = 20 * 60


# Host source-of-truth. Resolved lazily so tests can monkeypatch.
def _host_credentials_path() -> Path:
    return Path.home() / ".claude" / ".credentials.json"


# Container bind-mount target — relative to the project root, which is
# the orchestrator's CWD in dev/test setups.
_MOUNT_CREDENTIALS_PATH = Path("deploy/claudebox/claude-config/.credentials.json")


def _mount_credentials_path() -> Path:
    return _MOUNT_CREDENTIALS_PATH


# Mtime tolerance in seconds. Filesystems quantize mtimes (often to
# 1ns or 1s); a sub-second diff between the freshly-copied source and
# its destination is still "in sync" for our purposes.
_MTIME_TOLERANCE_SECONDS = 1.0


def sync_credentials_if_stale(*, force: bool = False) -> bool:
    """Copy host credentials → claudebox mount if mtimes differ.
    Returns True iff a sync happened, False if already in sync or
    the host file is missing.

    Best-effort by design: a failure logs a warning and returns
    False — never raises. The deep-mode turn that triggered this
    call will still proceed, and if creds are stale claudebox will
    surface the 401 itself, which is at least a louder failure mode
    than "no logs about why anything is broken."

    `force=True` skips the mtime comparison and always copies (used
    by the manual `gmail-search sync-claude-creds` CLI escape hatch
    if we add one later)."""
    src = _host_credentials_path()
    if not src.exists():
        return False
    dst = _mount_credentials_path()
    try:
        if not force and dst.exists():
            host_mtime = src.stat().st_mtime
            mount_mtime = dst.stat().st_mtime
            if abs(host_mtime - mount_mtime) < _MTIME_TOLERANCE_SECONDS:
                return False
        # Back up the stale one before we overwrite it. Lets us see
        # exactly which token bumped which on disk if a refresh ever
        # produces a broken file. Backups accumulate; cleanup is the
        # operator's responsibility (cheap one-liner: `rm
        # *.stale-bak-*`).
        if dst.exists():
            ts = time.strftime("%Y%m%d-%H%M%S")
            backup = dst.with_name(dst.name + f".stale-bak-{ts}")
            try:
                shutil.copy2(dst, backup)
            except OSError as exc:
                logger.warning("claudebox creds backup failed (continuing): %s", exc)
        shutil.copyfile(src, dst)
        # Lock down perms — the file holds an OAuth refresh_token.
        os.chmod(dst, 0o600)
        # Mirror the source mtime so the next mtime-compare matches.
        host_mtime = src.stat().st_mtime
        os.utime(dst, (host_mtime, host_mtime))
        logger.info(
            "synced claudebox credentials from %s to %s (host mtime %s)",
            src,
            dst,
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(host_mtime)),
        )
        return True
    except OSError as exc:
        logger.warning("claudebox credential sync failed: %s", exc)
        return False


def _read_expires_at(path: Path) -> float | None:
    """Read `claudeAiOauth.expiresAt` from the credentials JSON at
    `path` and return it as an epoch value in SECONDS, or None on any
    error (missing file, bad JSON, missing key, non-numeric value).

    The on-disk value is epoch MILLISECONDS (Claude Code's convention).
    Any value above ~1e12 is far beyond a plausible epoch-seconds
    timestamp, so we treat it as milliseconds and divide by 1000."""
    import json  # inline: formatter strips top-level unused imports

    try:
        raw = path.read_text()
        data = json.loads(raw)
        oauth = data.get("claudeAiOauth")
        if not isinstance(oauth, dict):
            return None
        value = oauth.get("expiresAt")
        if not isinstance(value, (int, float)):
            return None
        expires_at = float(value)
        if expires_at > 1e12:
            expires_at = expires_at / 1000.0
        return expires_at
    except (OSError, ValueError, TypeError):
        return None


def credentials_health(
    margin_seconds: float = _EXPIRY_SAFETY_MARGIN_SECONDS,
) -> tuple[str, float | None]:
    """Sync host → mount, then classify the MOUNT token's freshness.

    Returns `(status, expires_at_epoch_or_None)` where status is one of:
      - "expired"  : expiry is at or before now (token is dead).
      - "expiring" : expiry is within `margin_seconds` of now (still
                     usable, but a long run may cross the boundary).
      - "ok"       : expiry is comfortably in the future.
      - "unknown"  : no expiry could be read from the mount file.

    Always calls `sync_credentials_if_stale()` first so the mount
    reflects the latest host credentials before we inspect it.

    Detection only — never refreshes, never touches the host file."""
    try:
        sync_credentials_if_stale()
    except Exception:  # noqa: BLE001 — best-effort; classify on whatever's on disk
        logger.warning("credentials_health: pre-check sync failed (continuing)", exc_info=True)
    expires_at = _read_expires_at(_mount_credentials_path())
    if expires_at is None:
        return ("unknown", None)
    now = time.time()
    if expires_at <= now:
        return ("expired", expires_at)
    if expires_at - now <= margin_seconds:
        return ("expiring", expires_at)
    return ("ok", expires_at)
