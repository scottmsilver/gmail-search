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
  # The user-scope MCP config lives at /home/claude/.claude.json inside the
  # container's image — that file isn't mounted, so it gets wiped on
  # `docker compose down` and we re-register on every fresh up.
  #
  # SERVICE TOKEN: once the MCP server enforces transport auth
  # (GMAIL_MCP_REQUIRE_TRANSPORT_AUTH=1), an unauthenticated /mcp client
  # gets 401. claudebox is a TRUSTED, multi-tenant container, so it uses a
  # tenantless SERVICE token (aud=mcp-service) — NOT a tenant-bound
  # transport token (that would pin every run to one owner, a cross-tenant
  # bug). Per-run scoping stays on the session the orchestrator registers.
  # Mint the token once via `POST /admin/service-tokens` and put it in
  # deploy/claudebox/.env as GMAIL_MCP_SERVICE_TOKEN; docker-compose passes
  # it into the CONTAINER env. When unset, we register with no header so
  # nothing breaks before enforcement is on.
  #
  # SECURITY: the token is NEVER placed on a command line. `claude mcp add
  # --header` and `mcp add-json` would expose the bearer in the process arg
  # list (visible via `ps` in the container, transiently host-side). Instead
  # an in-container python3 reads GMAIL_MCP_SERVICE_TOKEN from its OWN
  # os.environ and writes the mcpServers entry into ~/.claude.json directly.
  # Idempotent: it overwrites the gmail-tools entry to the desired state.
  local url="${GMAIL_MCP_TOOLS_URL_INSIDE_CONTAINER:-http://host.docker.internal:7878/mcp}"
  docker exec -i -u claude -e GMAIL_MCP_TOOLS_URL="$url" claudebox python3 - <<'PY' >/dev/null 2>&1 || true
import json, os

cfg_path = os.path.expanduser("~/.claude.json")
url = os.environ["GMAIL_MCP_TOOLS_URL"]
# Read the bearer from the CONTAINER env — never from argv.
token = os.environ.get("GMAIL_MCP_SERVICE_TOKEN", "").strip()

entry = {"type": "http", "url": url}
if token:
    entry["headers"] = {"Authorization": "Bearer " + token}

try:
    with open(cfg_path) as f:
        cfg = json.load(f)
except (FileNotFoundError, ValueError):
    cfg = {}

servers = cfg.get("mcpServers")
if not isinstance(servers, dict):
    servers = {}
servers["gmail-tools"] = entry
cfg["mcpServers"] = servers

tmp = cfg_path + ".tmp"
with open(tmp, "w") as f:
    json.dump(cfg, f, indent=2)
os.replace(tmp, cfg_path)
os.chmod(cfg_path, 0o600)
PY
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
