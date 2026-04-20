#!/usr/bin/env bash
# Boot two gmail-search servers side-by-side so `scripts/ab_api_diff.py`
# can fire read-only requests at both and diff the responses.
#
#   port 8090 → DB_BACKEND unset → SQLite (production path)
#   port 8091 → DB_BACKEND=postgres → the migrated PG copy
#
# Usage:
#   ./scripts/run_ab_servers.sh start   # boot both, pid files under /tmp
#   ./scripts/run_ab_servers.sh stop    # SIGTERM both
#   ./scripts/run_ab_servers.sh status  # health check
#
# The port-8090 server is expected to already be running in production
# (data/server.pid). We leave it alone — we only need to start the PG
# twin. This script manages the PG twin at /tmp/gmail_search_pg.pid.

set -euo pipefail

DATA_DIR="/home/ssilver/development/gmail-search/data"
LOG_DIR="/tmp"
PG_PID="/tmp/gmail_search_pg.pid"
PG_LOG="/tmp/gmail_search_pg_server.log"

cmd="${1:-status}"

pg_running() {
  [ -f "$PG_PID" ] && kill -0 "$(cat "$PG_PID")" 2>/dev/null
}

case "$cmd" in
  start)
    if pg_running; then
      echo "pg-backed server already running (pid $(cat "$PG_PID"))"
    else
      DB_BACKEND=postgres \
        DB_DSN="postgresql://gmail_search:gmail_search@127.0.0.1:5544/gmail_search" \
        nohup gmail-search serve --port 8091 --data-dir "$DATA_DIR" \
        >"$PG_LOG" 2>&1 &
      echo $! > "$PG_PID"
      echo "started pg-backed server on :8091 (pid $(cat "$PG_PID"))"
    fi
    echo "sqlite  → :8090 (existing server)"
    echo "postgres → :8091 (pid $(cat "$PG_PID"))  log: $PG_LOG"
    ;;
  stop)
    if pg_running; then
      kill "$(cat "$PG_PID")" || true
      rm -f "$PG_PID"
      echo "stopped pg-backed server"
    else
      echo "pg-backed server not running"
    fi
    ;;
  status)
    for port in 8090 8091; do
      if curl -sf -m 2 "http://127.0.0.1:$port/api/status" >/dev/null 2>&1; then
        echo "  :$port  healthy"
      else
        echo "  :$port  NOT responding"
      fi
    done
    ;;
  *)
    echo "usage: $0 start|stop|status" >&2
    exit 2
    ;;
esac
