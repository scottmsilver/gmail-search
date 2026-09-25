#!/usr/bin/env bash
# Move the owner daemons onto the deployer's owner track, once (#52):
# `scripts/owner-cutover.sh --help` and docs/one-checkout-runbook.md.
set -euo pipefail
here="$(git rev-parse --show-toplevel)"
production="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")"
cd "$here"
# Never let the tool sync the production venv as a side effect of starting.
if [ "$here" = "$production" ]; then
  exec uv run --no-sync python -m gmail_search.deploy.owner_cutover "$@"
fi
exec uv run python -m gmail_search.deploy.owner_cutover "$@"
