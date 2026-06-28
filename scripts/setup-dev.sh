#!/usr/bin/env bash
# One-time dev setup: turn on the structural commit gate (.githooks/pre-commit).
# Git can't auto-enable a tracked hooks dir on clone, so each clone runs this
# once. CI (.github/workflows/ci.yml) is the backstop for anyone who doesn't.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

git config core.hooksPath .githooks
chmod +x .githooks/* 2>/dev/null || true

echo "✓ core.hooksPath = .githooks"
echo "  pre-commit (fast): ruff + smoke tests   |  pre-push (full): ruff + pytest"
echo "  Override: git commit/push --no-verify   |  SKIP_HOOKS=1"
command -v ruff >/dev/null || echo "  ⚠ install ruff:  pip install -e '.[dev]'"
