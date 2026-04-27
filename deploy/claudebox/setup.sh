#!/usr/bin/env bash
# Prepare the local claudebox deploy:
#   - copies the user's Claude OAuth credentials into ./claude-config
#   - ensures the .claude.json template is in place
#   - creates ./workspaces
#   - generates a fresh CLAUDEBOX_API_TOKEN if .env doesn't have one
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR"

CONFIG_DIR="./claude-config"
WORKSPACES_DIR="./workspaces"
ENV_FILE="./.env"
HOST_CREDS="${HOME}/.claude/.credentials.json"

say() { printf '[setup] %s\n' "$*"; }

ensure_credentials() {
  if [[ ! -f "$HOST_CREDS" ]]; then
    echo "[setup] ERROR: $HOST_CREDS not found." >&2
    echo "[setup] Log in to Claude Code on the host first (claude login)." >&2
    exit 1
  fi
  mkdir -p "$CONFIG_DIR"
  cp "$HOST_CREDS" "$CONFIG_DIR/.credentials.json"
  chmod 600 "$CONFIG_DIR/.credentials.json"
  say "copied .credentials.json (mode 600)"
}

ensure_claude_json() {
  local target="$CONFIG_DIR/.claude.json"
  if [[ ! -f "$target" ]]; then
    echo "[setup] ERROR: $target template missing from repo." >&2
    exit 1
  fi
  say ".claude.json present"
}

ensure_workspaces_dir() {
  mkdir -p "$WORKSPACES_DIR"
  say "workspaces dir ready at $WORKSPACES_DIR"
}

ensure_env_token() {
  if [[ -f "$ENV_FILE" ]] && grep -q '^CLAUDEBOX_API_TOKEN=' "$ENV_FILE"; then
    say ".env already has CLAUDEBOX_API_TOKEN"
    return
  fi
  local token
  token=$(openssl rand -hex 32)
  if [[ -f "$ENV_FILE" ]]; then
    {
      grep -v '^CLAUDEBOX_API_TOKEN=' "$ENV_FILE" || true
      printf 'CLAUDEBOX_API_TOKEN=%s\n' "$token"
    } > "$ENV_FILE.tmp"
    mv "$ENV_FILE.tmp" "$ENV_FILE"
  else
    printf 'CLAUDEBOX_API_TOKEN=%s\n' "$token" > "$ENV_FILE"
  fi
  chmod 600 "$ENV_FILE"
  say "generated fresh CLAUDEBOX_API_TOKEN in .env"
}

main() {
  say "preparing $SCRIPT_DIR"
  ensure_credentials
  ensure_claude_json
  ensure_workspaces_dir
  ensure_env_token
  say "done."
}

main "$@"
