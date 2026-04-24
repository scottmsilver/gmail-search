"""Tests for the deep-analysis HTTP surface (`gmail_search.agents.service`).

Covers registration-time concerns (the ADK import probe). The streaming
endpoints themselves are exercised against the orchestrator directly in
`test_agent_orchestration.py`, without spinning up FastAPI.
"""

from __future__ import annotations

import builtins
import logging
import sys
from pathlib import Path

from fastapi import FastAPI

from gmail_search.agents import service


def test_register_agent_routes_calls_adk_probe(monkeypatch):
    """`register_agent_routes` must invoke the ADK probe at boot so a
    broken install surfaces in the server logs immediately, not at the
    first /api/agent/analyze request hours later. We don't care here
    HOW the probe checks; we care that it RAN."""
    called: list[bool] = []

    def _spy() -> None:
        called.append(True)

    monkeypatch.setattr(service, "_probe_adk_imports", _spy, raising=True)
    app = FastAPI()
    service.register_agent_routes(app, Path("/tmp/unused.db"))
    assert called == [True], "register_agent_routes did not invoke the ADK probe"


def test_probe_adk_imports_warns_on_broken_submodule(monkeypatch, caplog):
    """The probe's contract: when one of the ADK-touching submodules
    fails to import, log a WARNING and return cleanly. Chat mode must
    stay healthy even though deep mode is wedged.

    We simulate "ADK importable-but-broken" by intercepting the
    builtin `__import__` for the duration of the probe call so that
    any attempt to re-import a key submodule raises ImportError. This
    matches the shape of a real-world failure (e.g. `google.adk` is
    installed but a transitive dep is the wrong version)."""
    # The probe does `from gmail_search.agents import (analyst, ...)`.
    # Drop the cached submodules so the import statement actually
    # executes the loader path (and our patched __import__ sees it).
    cached: dict[str, object] = {}
    target = "gmail_search.agents.retriever"
    if target in sys.modules:
        cached[target] = sys.modules.pop(target)

    real_import = builtins.__import__

    def _patched_import(name, globals=None, locals=None, fromlist=(), level=0):
        # Trip the import we want to fake-break. Any other import goes
        # through normally so the rest of the test machinery works.
        if name == "gmail_search.agents" and "retriever" in (fromlist or ()):
            raise ImportError("simulated ADK breakage: retriever submodule")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _patched_import)

    try:
        with caplog.at_level(logging.WARNING, logger=service.logger.name):
            # Must NOT raise — the contract is wrap-and-warn.
            service._probe_adk_imports()

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert any(
            "ADK imports failed" in r.getMessage() for r in warnings
        ), f"expected ADK-failure warning, got: {[r.getMessage() for r in warnings]}"
    finally:
        # Restore any submodule we evicted so other tests aren't poisoned.
        for k, v in cached.items():
            sys.modules[k] = v


def test_probe_adk_imports_silent_on_healthy_install(caplog):
    """Sanity check: when imports succeed (the dev/CI machine has a
    working ADK install), the probe is silent. No warning, no error,
    no crash."""
    with caplog.at_level(logging.WARNING, logger=service.logger.name):
        service._probe_adk_imports()
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert not any(
        "ADK imports failed" in r.getMessage() for r in warnings
    ), f"healthy install should not warn; got: {[r.getMessage() for r in warnings]}"
