#!/usr/bin/env bash
# Stop and remove the local claudebox container.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR"

docker compose -f ./docker-compose.yml down
echo "[stop] claudebox stopped."
