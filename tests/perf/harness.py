"""Reusable plumbing for the search performance gate.

This module holds NO test assertions — only the machinery the gate uses:

  * prerequisite detection (PG reachable? serve running? env present?), so
    the default `pytest` run auto-skips cleanly on a machine without a DB or
    a live `gmail-search serve`;
  * env/config readers that pull the serve process's secrets from
    `/proc/<pid>/environ` and fall back to plain env vars — never a
    hardcoded URL or token (project rule);
  * a `StageTimer` that monkeypatches the stages of
    `SearchEngine.search_threads` from the OUTSIDE, profiler-style, so we
    can attribute latency per stage WITHOUT editing engine.py;
  * the baseline load/save + tolerance comparison used by the gate.

The tolerance policy is "baseline + tolerance": a metric regresses when it
exceeds ``max(baseline * (1 + REL_TOLERANCE), baseline + ABS_TOLERANCE_MS)``.
That dual bound keeps tiny-but-noisy stages (sub-millisecond) from tripping
on relative jitter while still catching a 10ms stage that balloons to 10s.
"""

from __future__ import annotations

import json
import os
import socket
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

# ── Tolerance policy ────────────────────────────────────────────────────────

REL_TOLERANCE = 0.50  # +50%
ABS_TOLERANCE_MS = 150.0  # or +150ms, whichever is larger

BASELINE_PATH = Path(__file__).with_name("baseline.json")

# The representative signed-in user (scottmsilver). Used for HTTP + in-process
# timing. Overridable via env for other corpora.
DEFAULT_TEST_USER_ID = "u_bW4Sa8cN0wT9KPwp"

# Default repo data dir — get_connection ignores db_path (PG-only) but the
# SearchEngine constructor still wants a real path for per-user dictionaries.
REPO_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


# ── Prerequisite detection ──────────────────────────────────────────────────


def pg_reachable(host: str = "127.0.0.1", port: int = 5544, timeout: float = 0.5) -> bool:
    """Cheap TCP probe so DB-backed perf tests skip cleanly when the
    paradedb container isn't up."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _serve_pids() -> list[int]:
    """Return PIDs of running `gmail-search serve` processes (best effort)."""
    pids: list[int] = []
    try:
        import subprocess

        out = subprocess.run(
            ["pgrep", "-f", "gmail-search serve"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in out.stdout.split():
            try:
                pids.append(int(line))
            except ValueError:
                pass
    except Exception:
        pass
    return pids


def _read_env_from_serve(keys: tuple[str, ...]) -> dict[str, str]:
    """Pull selected env vars out of a live serve process's /proc environ.

    The serve process holds DATABASE/secret config that the test shell may
    not. We read it from /proc so the gate can drive the same DB + auth as
    the live server without anyone exporting secrets into the test env.
    Returns only the keys that were found; missing keys are simply absent.
    """
    found: dict[str, str] = {}
    for pid in _serve_pids():
        environ_path = Path(f"/proc/{pid}/environ")
        try:
            raw = environ_path.read_bytes()
        except OSError:
            continue
        for entry in raw.split(b"\0"):
            if b"=" not in entry:
                continue
            k, _, v = entry.partition(b"=")
            ks = k.decode("utf-8", "replace")
            if ks in keys and ks not in found:
                found[ks] = v.decode("utf-8", "replace")
        if all(k in found for k in keys):
            break
    return found


def resolve_serve_env() -> dict[str, str]:
    """Best-effort resolution of the env the gate needs.

    Order of precedence: the test process's own env wins; anything missing
    is back-filled from a live serve process. Never raises — callers check
    for the keys they need and skip if absent.
    """
    keys = (
        "DB_DSN",
        "DATABASE_URL",
        "GEMINI_API_KEY",
        "GMAIL_MCP_ADMIN_TOKEN",
        "GMAIL_SEARCH_API_URL",
    )
    resolved: dict[str, str] = {}
    from_serve = _read_env_from_serve(keys)
    for k in keys:
        val = os.environ.get(k) or from_serve.get(k)
        if val:
            resolved[k] = val
    return resolved


def serve_db_dsn_from_proc() -> str | None:
    """Read the serve process's DB_DSN straight from /proc (NOT the current
    env, which test isolation may have rewritten). Returns None if serve
    doesn't export an explicit DB_DSN — caller then uses the project
    default DSN, which is what serve itself falls back to."""
    found = _read_env_from_serve(("DB_DSN",))
    return found.get("DB_DSN")


def http_base_url(env: dict[str, str] | None = None) -> str:
    """Serve HTTP base. Reads GMAIL_SEARCH_API_URL, defaults to the serve
    command's local bind. Never hardcoded in committed call sites beyond
    this single localhost default."""
    env = env or {}
    return (
        env.get("GMAIL_SEARCH_API_URL") or os.environ.get("GMAIL_SEARCH_API_URL") or "http://127.0.0.1:8090"
    ).rstrip("/")


def test_user_id() -> str:
    return os.environ.get("GMAIL_PERF_USER_ID", DEFAULT_TEST_USER_ID)


def serve_http_ready(base_url: str, timeout: float = 1.0) -> bool:
    """Is the serve HTTP surface up? Hits /healthz."""
    try:
        import httpx

        resp = httpx.get(f"{base_url}/healthz", timeout=timeout)
        return resp.status_code < 500
    except Exception:
        return False


# ── Stage timing via monkeypatch (no engine.py edits) ───────────────────────


class StageTimer:
    """Wrap the stages of ``SearchEngine.search_threads`` to record per-stage
    wall-clock time, profiler-style. We patch instance/class methods from the
    OUTSIDE and restore them on exit — engine.py is never modified.

    The thread_summary bulk fetch is the one stage that isn't its own method
    (it's an inline ``conn.execute`` inside ``search_threads``). We capture it
    by wrapping the connection's ``execute`` for the duration of the call and
    attributing time to whichever query text contains ``FROM thread_summary``.
    Other inline blocks (the scoring loop) are derived as residual time.
    """

    # SearchEngine methods that map 1:1 to a stage.
    _METHODS = (
        "_embed_query",
        "_resolve_candidate_msg_ids",
        "_fetch_embedding_rows",
        "_collapse_repeat_senders",
        "_llm_rerank",
        "_filter_offtopic",
    )

    def __init__(self, engine: Any):
        self.engine = engine
        self.timings: dict[str, float] = {}
        self._originals: dict[str, Callable] = {}
        self._searcher_search_orig: Callable | None = None
        self._search_fts_orig: Callable | None = None
        self._conn_execute_patched = False

    def _record(self, name: str, dt: float) -> None:
        self.timings[name] = self.timings.get(name, 0.0) + dt * 1000.0  # ms

    def _wrap_method(self, name: str) -> None:
        cls = type(self.engine)
        # Distinguish staticmethods (e.g. _collapse_repeat_senders,
        # _filter_offtopic) from instance methods — the former take no
        # `self`, so a self-prepending wrapper would pass one arg too many.
        raw = cls.__dict__.get(name)
        is_static = isinstance(raw, staticmethod)
        orig = getattr(cls, name)  # the underlying callable (already unbound)
        self._originals[name] = raw if raw is not None else orig

        if is_static:

            def wrapped(*args, __name=name, __orig=orig, **kwargs):
                t0 = time.perf_counter()
                try:
                    return __orig(*args, **kwargs)
                finally:
                    self._record(__name, time.perf_counter() - t0)

            setattr(cls, name, staticmethod(wrapped))
        else:

            def wrapped(_self, *args, __name=name, __orig=orig, **kwargs):  # type: ignore[misc]
                t0 = time.perf_counter()
                try:
                    return __orig(_self, *args, **kwargs)
                finally:
                    self._record(__name, time.perf_counter() - t0)

            setattr(cls, name, wrapped)

    def _wrap_searcher_search(self) -> None:
        searcher = self.engine.searcher
        self._searcher_search_orig = searcher.search

        def wrapped(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return self._searcher_search_orig(*args, **kwargs)
            finally:
                self._record("scann_search", time.perf_counter() - t0)

        searcher.search = wrapped  # type: ignore[assignment]

    def _wrap_search_fts(self) -> None:
        # search_fts is imported into the engine module namespace, so patch it
        # there (where search_threads resolves the name).
        import gmail_search.search.engine as engine_mod

        self._search_fts_orig = engine_mod.search_fts

        def wrapped(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return self._search_fts_orig(*args, **kwargs)
            finally:
                self._record("bm25_search_fts", time.perf_counter() - t0)

        engine_mod.search_fts = wrapped  # type: ignore[assignment]

    def _wrap_conn_execute(self) -> None:
        """Patch the PG wrapper's execute to time the thread_summary bulk
        fetch (and the bm25-only messages IN-fetch) by inspecting SQL text."""
        import gmail_search.store.db as db_mod

        wrapper_cls = getattr(db_mod, "_PgConnWrapper", None)
        if wrapper_cls is None or not hasattr(wrapper_cls, "execute"):
            return
        orig_execute = wrapper_cls.execute
        self._originals["__conn_execute__"] = orig_execute
        self._conn_execute_patched = True

        def wrapped_execute(_self, sql, params=None):
            label = None
            if isinstance(sql, str):
                flat = " ".join(sql.split())
                if "FROM thread_summary" in flat:
                    label = "thread_summary_fetch"
                elif "FROM messages WHERE id IN" in flat:
                    label = "bm25_messages_fetch"
            t0 = time.perf_counter()
            try:
                return orig_execute(_self, sql, params)
            finally:
                if label:
                    self._record(label, time.perf_counter() - t0)

        wrapper_cls.execute = wrapped_execute  # type: ignore[assignment]
        self._wrapper_cls = wrapper_cls

    def __enter__(self) -> "StageTimer":
        for name in self._METHODS:
            self._wrap_method(name)
        self._wrap_searcher_search()
        self._wrap_search_fts()
        self._wrap_conn_execute()
        return self

    def __exit__(self, *exc) -> None:
        for name, orig in self._originals.items():
            if name == "__conn_execute__":
                self._wrapper_cls.execute = orig  # type: ignore[assignment]
            else:
                setattr(type(self.engine), name, orig)
        if self._searcher_search_orig is not None:
            self.engine.searcher.search = self._searcher_search_orig  # type: ignore[assignment]
        if self._search_fts_orig is not None:
            import gmail_search.search.engine as engine_mod

            engine_mod.search_fts = self._search_fts_orig  # type: ignore[assignment]

    def time_search(self, query: str, **kwargs) -> tuple[list, dict[str, float], float]:
        """Run one search and return (results, per_stage_ms, total_ms).

        The "scoring_loop" stage is the residual: total minus the sum of the
        attributed stages. It captures the inline thread-scoring + blend loop
        that isn't its own method.
        """
        self.timings = {}
        t0 = time.perf_counter()
        results = self.engine.search_threads(query, **kwargs)
        total_ms = (time.perf_counter() - t0) * 1000.0
        stages = dict(self.timings)
        attributed = sum(stages.values())
        stages["scoring_loop_residual"] = max(0.0, total_ms - attributed)
        stages["total"] = total_ms
        return results, stages, total_ms


# ── Baseline load / save / compare ──────────────────────────────────────────


def load_baseline(path: Path = BASELINE_PATH) -> dict[str, Any]:
    return json.loads(path.read_text())


def save_baseline(data: dict[str, Any], path: Path = BASELINE_PATH) -> None:
    data = dict(data)
    data.setdefault("_meta", {})
    data["_meta"]["needs_blessed_regen"] = False
    data["_meta"]["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def tolerance_ceiling(baseline_ms: float) -> float:
    """The max allowed value for a metric before it counts as a regression."""
    return max(baseline_ms * (1.0 + REL_TOLERANCE), baseline_ms + ABS_TOLERANCE_MS)


def check_regression(name: str, baseline_ms: float, measured_ms: float) -> tuple[bool, str]:
    """Return (ok, message). ok=False means a regression beyond tolerance."""
    ceiling = tolerance_ceiling(baseline_ms)
    ok = measured_ms <= ceiling
    msg = (
        f"{name}: measured={measured_ms:.1f}ms baseline={baseline_ms:.1f}ms "
        f"ceiling={ceiling:.1f}ms ({'OK' if ok else 'REGRESSION'})"
    )
    return ok, msg


@contextmanager
def closing_connection():
    """Yield a get_connection() and ALWAYS close it. Keeps the per-run
    connection count tiny (project rule: leaking exhausts PG)."""
    from gmail_search.store.db import get_connection

    conn = get_connection(REPO_DATA_DIR)
    try:
        yield conn
    finally:
        conn.close()
