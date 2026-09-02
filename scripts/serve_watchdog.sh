#!/usr/bin/env bash
# Watchdog for gmail-search-serve: probe /healthz?ready=1 (which includes the
# search canary) and restart the service after N consecutive failures.
#
# Why: the 2026-07 deploy-skew incident 500'd every search for nine days with
# nobody watching. The canary makes readiness honest; this timer acts on it.
# A restart is the right medicine for the known failure classes (deploy skew,
# wedged engine): it reloads current code and rebuilds the engine.
#
# Guards:
#   * Only acts when the service is ACTIVE-but-unhealthy. If the operator
#     stopped serve on purpose (or systemd is already crash-looping it), the
#     watchdog stays out of the way.
#   * Counter resets on success and after a restart, so post-restart warm-up
#     failures need another full N strikes before acting again.
#
# Env overrides (mostly for testing):
#   WATCHDOG_URL              probe URL (default http://localhost:8090/healthz?ready=1)
#   WATCHDOG_THRESHOLD        strikes (default 3; at 5-min cadence = ~15 min).
#                             Non-numeric/zero values fall back to 3.
#   WATCHDOG_STATE            counter file (keep it in a user-private dir —
#                             contents and path are trusted)
#   WATCHDOG_RESTART_COOLDOWN seconds between auto-restarts (default 3600) —
#                             a persistent, non-restart-recoverable failure
#                             flaps at most once per hour, loudly, instead of
#                             every threshold window forever
#   WATCHDOG_DRY_RUN          1 = log instead of restarting

set -u

URL="${WATCHDOG_URL:-http://localhost:8090/healthz?ready=1}"
THRESHOLD="${WATCHDOG_THRESHOLD:-3}"
case "$THRESHOLD" in ''|*[!0-9]*|0) THRESHOLD=3 ;; esac
COOLDOWN="${WATCHDOG_RESTART_COOLDOWN:-3600}"
case "$COOLDOWN" in ''|*[!0-9]*) COOLDOWN=3600 ;; esac
STATE="${WATCHDOG_STATE:-${XDG_RUNTIME_DIR:-$HOME/.cache}/gmail-search-serve-watchdog.strikes}"
LAST_RESTART="$STATE.lastrestart"
SERVICE="gmail-search-serve.service"

mkdir -p "$(dirname "$STATE")"

if ! systemctl --user is-active --quiet "$SERVICE"; then
    echo "watchdog: $SERVICE not active — leaving it alone (operator stop or systemd handling a crash)"
    rm -f "$STATE"
    exit 0
fi

if curl -sf --max-time 30 "$URL" > /dev/null; then
    rm -f "$STATE"
    exit 0
fi

# Sanitize the persisted counter: digits only, missing/empty -> 0.
prev=$([ -f "$STATE" ] && tr -cd '0-9' < "$STATE" || true)
strikes=$(( ${prev:-0} + 1 ))
echo "$strikes" > "$STATE"
body=$(curl -s --max-time 30 "$URL" 2>&1 | head -c 300)
echo "watchdog: readiness probe FAILED (strike $strikes/$THRESHOLD): $body"

if [ "$strikes" -lt "$THRESHOLD" ]; then
    exit 0
fi
rm -f "$STATE"

now=$(date +%s)
last=$([ -f "$LAST_RESTART" ] && tr -cd '0-9' < "$LAST_RESTART" || true)
if [ -n "$last" ] && [ $(( now - last )) -lt "$COOLDOWN" ]; then
    echo "watchdog: $THRESHOLD consecutive failures but last restart was $(( now - last ))s ago (< ${COOLDOWN}s cooldown) — restart SUPPRESSED; service stays 503 until it recovers or the cooldown passes"
    exit 0
fi

if [ "${WATCHDOG_DRY_RUN:-0}" = "1" ]; then
    echo "watchdog: DRY RUN — would restart $SERVICE now"
    echo "$now" > "$LAST_RESTART"
    exit 0
fi

# Re-check right before acting: the operator may have stopped the service
# during the probe window (up to ~60s of curls above).
if ! systemctl --user is-active --quiet "$SERVICE"; then
    echo "watchdog: $SERVICE went inactive during probing — restart aborted"
    exit 0
fi
echo "watchdog: $THRESHOLD consecutive failures — restarting $SERVICE"
echo "$now" > "$LAST_RESTART"
systemctl --user restart "$SERVICE"
