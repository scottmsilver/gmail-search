#!/usr/bin/env bash
# Bring up the claude_code backend stack, run scripts/run_deep_compare.py,
# tear it down. ADK path needs nothing extra (just GEMINI_API_KEY).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MCP_LOG="$REPO_ROOT/scripts/mcp_tools_server.log"
MCP_PID_FILE="$REPO_ROOT/scripts/mcp_tools_server.pid"

cleanup() {
  if [ -f "$MCP_PID_FILE" ]; then
    PID="$(cat "$MCP_PID_FILE")"
    if kill -0 "$PID" 2>/dev/null; then
      echo "[stack] stopping MCP server (pid $PID)"
      kill "$PID" 2>/dev/null || true
      sleep 1
      kill -9 "$PID" 2>/dev/null || true
    fi
    rm -f "$MCP_PID_FILE"
  fi
  echo "[stack] stopping claudebox"
  bash "$REPO_ROOT/deploy/claudebox/stop.sh" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "[stack] starting MCP tools server -> :7878"
GMAIL_MCP_TOOLS_HOST=0.0.0.0 GMAIL_MCP_TOOLS_PORT=7878 \
  python -m gmail_search.agents.mcp_tools_server \
  > "$MCP_LOG" 2>&1 &
echo $! > "$MCP_PID_FILE"

# Wait for MCP server (admin/health endpoint).
for i in {1..20}; do
  if curl -fsS http://localhost:7878/admin/health >/dev/null 2>&1 \
     || curl -fsS http://localhost:7878/mcp >/dev/null 2>&1 \
     || nc -z localhost 7878 2>/dev/null; then
    echo "[stack] MCP server ready"
    break
  fi
  sleep 0.5
done

echo "[stack] starting claudebox"
bash "$REPO_ROOT/deploy/claudebox/start.sh"

# Read the admin token from .env so the runtime adapter can hit the side-channel.
if [ -f "$REPO_ROOT/deploy/claudebox/.env" ]; then
  set -o allexport
  # shellcheck disable=SC1091
  source "$REPO_ROOT/deploy/claudebox/.env"
  set +o allexport
fi

echo "[stack] running comparison"
python scripts/run_deep_compare.py "$@"
