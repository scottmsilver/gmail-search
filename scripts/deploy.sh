#!/usr/bin/env bash
# The deployer: see `scripts/deploy.sh --help` and README ("Deploying").
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
exec uv run python -m gmail_search.deploy "$@"
