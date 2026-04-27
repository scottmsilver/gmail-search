"""Integration-suite-scoped fixtures.

These tests hit REAL services (claudebox, MCP server, Postgres,
gmail-search FastAPI) and cost real API credits. They are gated
behind `pytest -m integration` and skip cleanly if any dependency
is unreachable. See `tests/integration/README.md` for stack bring-up.
"""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

import pytest

from gmail_search.agents.session import new_session_id

from . import _stack_probe

# ── Auto-mark every test under tests/integration/ ─────────────────


def pytest_collection_modifyitems(config, items):
    """Apply `pytest.mark.integration` to every test collected under
    this directory so a developer can't accidentally write an
    integration test without the marker (and have it run by default
    `pytest -q`)."""
    integration_root = Path(__file__).resolve().parent
    for item in items:
        try:
            item_path = Path(item.fspath).resolve()
        except (TypeError, ValueError):
            continue
        # `is_relative_to` would be cleanest but only exists on 3.9+
        # for Path; this loop is a clean fallback that doesn't care.
        try:
            item_path.relative_to(integration_root)
        except ValueError:
            continue
        item.add_marker(pytest.mark.integration)


# ── Live stack probe ──────────────────────────────────────────────


@pytest.fixture(scope="session")
def live_stack():
    """Probe all four services once at session start; skip the entire
    integration suite (cleanly, with a useful message) if anything is
    down. We do NOT try to start services ourselves — that's the
    operator's job (see README in this directory)."""
    probe = _stack_probe.probe_all(timeout=2.0)
    missing = _stack_probe.first_missing(probe)
    if missing:
        pytest.skip(
            "integration stack not running; bring up claudebox + MCP server "
            f"+ gmail-search server first. Missing: {', '.join(missing)}"
        )
    return probe


# ── Per-test conversation/session ids ─────────────────────────────


@pytest.fixture
def fresh_conversation_id():
    """A unique per-test conversation id so parallel runs don't
    collide on data/agent_scratch/<id>/. Prefixed `itest-` so the
    cleanup glob in CI never sweeps a real conversation."""
    return f"itest-{uuid.uuid4().hex[:12]}"


@pytest.fixture
def fresh_session_id():
    """One MCP session_id per test — the same factory the production
    code uses, so collisions with running sessions are negligible."""
    return new_session_id()


# ── Convenience: leftover-scratch finalizer ───────────────────────


@pytest.fixture
def scratch_cleanup():
    """Yield a function the test calls with `conversation_id` to
    register cleanup. Removes `data/agent_scratch/<id>/` in teardown
    even if the test failed mid-way. Idempotent."""
    to_clean: list[str] = []

    def register(conv_id: str) -> None:
        to_clean.append(conv_id)

    yield register

    scratch_root = Path("data/agent_scratch")
    for conv_id in to_clean:
        target = scratch_root / conv_id
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)


# ── MCP admin token resolution ────────────────────────────────────


def _read_admin_token_from_file() -> str | None:
    """The MCP server's auto-generated admin token (when not pinned
    via env) lands at `scripts/mcp_admin_token`. We do NOT log the
    contents — only the path, in the skip message."""
    path = Path("scripts/mcp_admin_token")
    if path.is_file():
        try:
            return path.read_text(encoding="utf-8").strip() or None
        except OSError:
            return None
    return None


@pytest.fixture(scope="session")
def mcp_admin_token(live_stack):
    """The admin bearer token for the running MCP server. Order:
    1) env `GMAIL_MCP_ADMIN_TOKEN`, 2) `scripts/mcp_admin_token`. If
    neither is found, skip — the integration tests cannot register a
    session without it."""
    token = os.environ.get("GMAIL_MCP_ADMIN_TOKEN") or _read_admin_token_from_file()
    if not token:
        pytest.skip(
            "GMAIL_MCP_ADMIN_TOKEN env var not set and scripts/mcp_admin_token "
            "not found; cannot exercise the MCP admin endpoints"
        )
    return token


@pytest.fixture(scope="session")
def claudebox_token():
    """The bearer token for the running claudebox container. We try
    env first, then `deploy/claudebox/.env`. None is allowed — the
    runtime adapter sends no Authorization header in that case (some
    claudebox builds run unauthenticated locally)."""
    token = os.environ.get("CLAUDEBOX_API_TOKEN") or os.environ.get("GMAIL_CLAUDEBOX_TOKEN")
    if token:
        return token
    return _read_claudebox_token_from_env_file()


def _read_claudebox_token_from_env_file() -> str | None:
    """Pull `CLAUDEBOX_API_TOKEN=...` out of `deploy/claudebox/.env`
    if present. Returns None on any error — the caller treats None
    as 'no auth'."""
    env_path = Path("deploy/claudebox/.env")
    if not env_path.is_file():
        return None
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("CLAUDEBOX_API_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'") or None
    except OSError:
        return None
    return None


@pytest.fixture
def integration_env(mcp_admin_token, claudebox_token, monkeypatch):
    """Wire env vars the runtime adapters expect. Tests can rely on
    `claudebox_invoke` / `register_session_via_admin` Just Working
    inside this fixture's scope."""
    monkeypatch.setenv("GMAIL_MCP_ADMIN_TOKEN", mcp_admin_token)
    monkeypatch.setenv("GMAIL_MCP_ADMIN_URL", _stack_probe.MCP_TOOLS_URL)
    monkeypatch.setenv("GMAIL_CLAUDEBOX_URL", _stack_probe.CLAUDEBOX_URL)
    if claudebox_token:
        monkeypatch.setenv("GMAIL_CLAUDEBOX_TOKEN", claudebox_token)
    return {
        "mcp_admin_token": mcp_admin_token,
        "claudebox_token": claudebox_token,
    }
