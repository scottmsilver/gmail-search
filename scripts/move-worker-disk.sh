#!/usr/bin/env bash
# Move the worker VM's disk to deploy.json worker.vm_dir: see
# `scripts/move-worker-disk.sh --help` and docs/worker-vm-move.md (#27).
# Runs the checkout's venv directly, not `uv run`, and writes no bytecode, so
# --dry-run and status write nothing at all.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
python=.venv/bin/python
[[ -x $python ]] || { echo "no $python here: run 'uv sync --extra dev' first" >&2; exit 1; }
PYTHONDONTWRITEBYTECODE=1 exec "$python" -m gmail_search.deploy.worker_disk "$@"
