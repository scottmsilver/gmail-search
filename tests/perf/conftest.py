"""Perf-gate pytest config.

  * registers the --update-baseline flag (pytest_addoption must live in a
    conftest.py, not a test module);
  * points perf tests at the REAL database/schema.

The repo's top-level conftest has an autouse `_isolated_pg_schema` fixture
that rewrites DB_DSN to a fresh EMPTY `test_<uuid>` schema per test, so
ordinary tests never touch production data. The perf gate is the deliberate
exception: it must measure the real corpus. The autouse `_perf_real_db`
fixture below runs AFTER that isolation fixture and re-points DB_DSN at the
DSN the live `gmail-search serve` uses (read from /proc or the env), so
perf tests see the real `public` schema. It does NOT mutate any data —
the gate only runs SELECT/EXPLAIN and read-only searches.
"""

import os
import sys
from pathlib import Path

import pytest

# Make `import harness` resolve regardless of pytest import mode.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness as H  # noqa: E402


def pytest_addoption(parser):
    parser.addoption(
        "--update-baseline",
        action="store_true",
        default=False,
        help="Regenerate tests/perf/baseline.json from this run instead of asserting against it.",
    )


@pytest.fixture(autouse=True)
def _perf_real_db(_isolated_pg_schema, monkeypatch):
    """Override the top-level autouse schema isolation: perf tests measure
    the live corpus, so point DB_DSN at the serve process's real DSN. If no
    real DSN can be resolved we leave the env as-is (tests then skip on the
    empty-corpus guard)."""
    # IMPORTANT: by the time this runs, the top-level `_isolated_pg_schema`
    # has already overwritten os.environ["DB_DSN"] with its empty test
    # schema. So we must NOT consult the live env for the DSN here — we read
    # the serve process's DSN straight from /proc (never the polluted env),
    # and otherwise use the project default (public schema).
    serve_dsn = H.serve_db_dsn_from_proc()
    real_dsn = (
        os.environ.get("GMAIL_PERF_DB_DSN")
        or serve_dsn
        or "postgresql://gmail_search:gmail_search@127.0.0.1:5544/gmail_search"
    )
    monkeypatch.setenv("DB_DSN", real_dsn)
    monkeypatch.setenv("DB_BACKEND", "postgres")
    yield
