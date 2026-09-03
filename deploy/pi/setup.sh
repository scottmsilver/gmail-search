#!/usr/bin/env bash
# Prepare deploy/pi: sessions + pi-agent dirs, .env with GMS_MCP_URL.
# Never copies Claude Code credentials — pi authenticates on its own
# (ANTHROPIC_API_KEY / GEMINI_API_KEY in .env, or `/login` inside the
# container).
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR"
say() { printf '[setup] %s\n' "$*"; }

ADMIN_ENV_FILE="${HOME}/.config/gmail-search/mcp.env"

ensure_dirs() {
  mkdir -p ./sessions ./pi-agent ../claudebox/workspaces
  say "sessions/, pi-agent/, workspaces ready"
}

ensure_env() {
  if [[ ! -f ./.env ]]; then
    printf 'GMS_MCP_URL=http://host.docker.internal:7878/mcp\nGMAIL_MCP_SERVICE_TOKEN=\nANTHROPIC_API_KEY=\nGEMINI_API_KEY=\n' > ./.env
    chmod 600 ./.env
    say "wrote .env template — fill in GEMINI_API_KEY (or ANTHROPIC_API_KEY) unless you /login"
  else
    say ".env present"
  fi
}

# Mint a fresh tenantless service token via the MCP admin API and write
# it into .env, if .env doesn't already have one. Never copies another
# deploy's token (e.g. claudebox's) — those are scoped to that deploy
# and may have expired.
ensure_service_token() {
  if grep -q '^GMAIL_MCP_SERVICE_TOKEN=.\+' ./.env 2>/dev/null; then
    say ".env already has GMAIL_MCP_SERVICE_TOKEN"
    return
  fi
  if [[ ! -f "$ADMIN_ENV_FILE" ]]; then
    say "WARNING: $ADMIN_ENV_FILE not found — cannot mint a service token; leaving GMAIL_MCP_SERVICE_TOKEN empty"
    return
  fi
  local admin_token
  admin_token=$(grep '^GMAIL_MCP_ADMIN_TOKEN=' "$ADMIN_ENV_FILE" | cut -d= -f2-)
  if [[ -z "$admin_token" ]]; then
    say "WARNING: GMAIL_MCP_ADMIN_TOKEN not found in $ADMIN_ENV_FILE — leaving GMAIL_MCP_SERVICE_TOKEN empty"
    return
  fi
  local response token
  response=$(curl -sS -X POST http://localhost:7878/admin/service-tokens \
    -H "Authorization: Bearer ${admin_token}" \
    -H "Content-Type: application/json" -d '{}')
  token=$(printf '%s' "$response" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("token",""))' 2>/dev/null || true)
  if [[ -z "$token" ]]; then
    say "WARNING: failed to mint a service token — leaving GMAIL_MCP_SERVICE_TOKEN empty"
    return
  fi
  {
    grep -v '^GMAIL_MCP_SERVICE_TOKEN=' ./.env || true
    printf 'GMAIL_MCP_SERVICE_TOKEN=%s\n' "$token"
  } > ./.env.tmp
  mv ./.env.tmp ./.env
  chmod 600 ./.env
  say "minted service token"
}

ensure_dirs
ensure_env
ensure_service_token
