#!/usr/bin/env bash
# Prepare deploy/pi: sessions + pi-agent dirs, .env with GMS_MCP_URL.
# Never copies Claude Code credentials — pi authenticates on its own
# (ANTHROPIC_API_KEY / GEMINI_API_KEY in .env, or `/login` inside the
# container).
set -euo pipefail
umask 077
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

SERVICE_TOKEN_URL="http://localhost:7878/admin/service-tokens"

# Shared wording for every "couldn't mint a token" exit — keeps the
# admin-env-missing, token-missing, and server-down paths consistent.
_warn_no_service_token() {
  say "WARNING: $1 — leaving GMAIL_MCP_SERVICE_TOKEN empty"
}

# POST to $SERVICE_TOKEN_URL with the admin bearer token, without ever
# putting the token on curl's argv (it would otherwise be visible via
# /proc/<pid>/cmdline for the life of the process). The header is fed
# through curl's config-from-stdin (`-K -`); only the URL and method
# stay on argv.
_mint_service_token_response() {
  local admin_token="$1"
  printf 'header = "Authorization: Bearer %s"\n' "$admin_token" \
    | curl -sS -K - -X POST "$SERVICE_TOKEN_URL" -H "Content-Type: application/json" -d '{}'
}

_write_service_token() {
  local token="$1"
  {
    grep -v '^GMAIL_MCP_SERVICE_TOKEN=' ./.env || true
    printf 'GMAIL_MCP_SERVICE_TOKEN=%s\n' "$token"
  } > ./.env.tmp
  mv ./.env.tmp ./.env
  chmod 600 ./.env
}

# Mint a fresh tenantless service token via the MCP admin API and write
# it into .env, if .env doesn't already have one. Never copies another
# deploy's token (e.g. claudebox's) — those are scoped to that deploy
# and may have expired. Any failure (missing admin credentials,
# unreachable server, malformed response) warns and leaves
# GMAIL_MCP_SERVICE_TOKEN empty rather than aborting setup.sh.
ensure_service_token() {
  if grep -q '^GMAIL_MCP_SERVICE_TOKEN=.\+' ./.env 2>/dev/null; then
    say ".env already has GMAIL_MCP_SERVICE_TOKEN"
    return
  fi
  if [[ ! -f "$ADMIN_ENV_FILE" ]]; then
    _warn_no_service_token "$ADMIN_ENV_FILE not found"
    return
  fi
  local admin_token
  admin_token=$(grep '^GMAIL_MCP_ADMIN_TOKEN=' "$ADMIN_ENV_FILE" | cut -d= -f2-)
  if [[ -z "$admin_token" ]]; then
    _warn_no_service_token "GMAIL_MCP_ADMIN_TOKEN not found in $ADMIN_ENV_FILE"
    return
  fi
  local response token
  if ! response=$(_mint_service_token_response "$admin_token"); then
    _warn_no_service_token "could not reach the MCP admin API at $SERVICE_TOKEN_URL"
    return
  fi
  token=$(printf '%s' "$response" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("token",""))' 2>/dev/null || true)
  if [[ -z "$token" ]]; then
    _warn_no_service_token "MCP admin API did not return a service token"
    return
  fi
  _write_service_token "$token"
  say "minted service token"
}

ensure_dirs
ensure_env
ensure_service_token
