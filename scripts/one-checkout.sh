#!/usr/bin/env bash
# Move the production checkout onto main in one window, and back (#57):
# `scripts/one-checkout.sh --help` and docs/one-checkout-runbook.md.
set -euo pipefail
here="$(git rev-parse --show-toplevel)"
production="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")"
cd "$here"
# Never let the tool sync the production venv as a side effect of starting:
# that sync is a step of the switch, not of `uv run`.
if [ "$here" = "$production" ]; then
  exec uv run --no-sync python -m gmail_search.deploy.one_checkout "$@"
fi
exec uv run python -m gmail_search.deploy.one_checkout "$@"
