#!/usr/bin/env bash
# Start the local claudebox container and wait until it is healthy.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR"

bash ./setup.sh

# load token for the post-start health/auth check
set -a
# shellcheck disable=SC1091
source ./.env
set +a

say() { printf '[start] %s\n' "$*"; }

say "docker compose up -d"
docker compose -f ./docker-compose.yml up -d

wait_for_health() {
  local deadline=$((SECONDS + 60))
  while (( SECONDS < deadline )); do
    if curl -fsS http://localhost:8765/health >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

register_gmail_tools_mcp() {
  # `claude mcp add` writes to /home/claude/.claude.json inside the
  # container's image — that file isn't mounted, so it gets wiped on
  # `docker compose down` and we need to re-register on every fresh up.
  # Idempotent: re-running with the same name + URL is a no-op (claude
  # errors with "already exists" which we swallow).
  local url="${GMAIL_MCP_TOOLS_URL_INSIDE_CONTAINER:-http://host.docker.internal:7878/mcp}"
  docker exec -u claude claudebox /home/claude/.local/bin/claude mcp add \
    --transport http --scope user gmail-tools "$url" >/dev/null 2>&1 || true
}

if wait_for_health; then
  # /health is unauthenticated; do a follow-up authenticated probe so we
  # surface bad tokens immediately instead of at first /run.
  if curl -fsS -o /dev/null -H "Authorization: Bearer ${CLAUDEBOX_API_TOKEN:-}" \
      http://localhost:8765/status; then
    say "registering gmail-tools MCP server inside container"
    register_gmail_tools_mcp
    say "Ready."
  else
    say "health ok but /status auth failed — check CLAUDEBOX_API_TOKEN."
    exit 1
  fi
else
  say "health endpoint never came up. recent logs:"
  docker compose -f ./docker-compose.yml logs --tail 80 claudebox || true
  exit 1
fi
