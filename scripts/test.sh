#!/usr/bin/env bash
# The fast suite: parallel workers for everything, then the few tests that
# measure Postgres statistics or plans (marker pg_exclusive) on a quiet
# database. Integration and perf tests stay excluded, as in pyproject.
# GMS_TEST_WORKERS sets the worker count (default 8). Extra args pass to both.
set -euo pipefail
# A hook exports GIT_DIR, GIT_INDEX_FILE, …: inherited, tests' `git -C <tmp>` would act on
# that repository, and ROOT would be its work tree.
unset $(compgen -e | grep '^GIT_')
ROOT="$(git rev-parse --show-toplevel)"
PY="${GMS_PYTHON:-$ROOT/.venv/bin/python}"
WORKERS="${GMS_TEST_WORKERS:-8}"
BASE='not integration and not perf_slow'
cd "$ROOT"
"$PY" -m pytest -q -n "$WORKERS" --dist loadfile -m "$BASE and not pg_exclusive" "$@"
"$PY" -m pytest -q -m "$BASE and pg_exclusive" "$@"
