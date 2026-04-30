"""Single source of truth for "what user_id should this INSERT carry?"

Phase 2/3 schema requires NOT NULL user_id on every per-user table.
Three write contexts exist:

  1. **Daemon writes** (sync, summarize, reindex, etc.) — no
     authenticated request, runs against a single Gmail account.
     Resolves to the bootstrap user (env GMS_BOOTSTRAP_EMAIL, default
     scottmsilver@gmail.com). Phase 3c rewrites these daemons to be
     per-user and they'll pass `user_id` explicitly; until then the
     bootstrap default keeps existing single-user installs working.

  2. **Authenticated endpoint writes** (model_battles vote, create
     conversation, deep-mode session) — the FastAPI handler has a
     `User` from `Depends(require_user)`. Pass `user.id` directly to
     the helper.

  3. **Tests / migration scripts** — explicit user_id in the call.

Caching: bootstrap user_id is looked up at most once per process and
stashed in a module-level `_BOOTSTRAP_CACHE` keyed by email so a
second lookup is a dict get."""

from __future__ import annotations

import os
from typing import Optional

_BOOTSTRAP_CACHE: dict[str, str] = {}


def get_bootstrap_user_id(conn) -> str:
    """Returns the user_id of the bootstrap user (env
    GMS_BOOTSTRAP_EMAIL, default scott). Cached per-process per-email
    so most calls are dict lookups, not DB round-trips."""
    email = os.environ.get("GMS_BOOTSTRAP_EMAIL", "scottmsilver@gmail.com").lower()
    cached = _BOOTSTRAP_CACHE.get(email)
    if cached:
        return cached
    row = conn.execute("SELECT id FROM users WHERE email = %s", (email,)).fetchone()
    if not row:
        raise RuntimeError(
            f"bootstrap user {email!r} not found in `users` — invite + sign in "
            "first, or set GMS_BOOTSTRAP_EMAIL to a different existing user."
        )
    _BOOTSTRAP_CACHE[email] = row["id"]
    return row["id"]


def resolve_write_user_id(conn, *, user_id: Optional[str] = None) -> str:
    """The one helper every INSERT site calls. Pass `user_id` when the
    caller has it (authenticated endpoint, test, explicit migration);
    omit it for daemon writes that should land in the bootstrap user's
    rows."""
    if user_id is not None:
        return user_id
    return get_bootstrap_user_id(conn)
