#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR"
say() { printf '[start] %s\n' "$*"; }
bash ./setup.sh
say "docker compose up -d --build"
docker compose -f ./docker-compose.yml up -d --build
if docker exec pi-sandbox pi --version >/dev/null 2>&1; then
  say "pi $(docker exec pi-sandbox pi --version) — Ready."
else
  say "pi did not answer --version. recent logs:"
  docker compose -f ./docker-compose.yml logs --tail 40 pi-sandbox || true
  exit 1
fi
