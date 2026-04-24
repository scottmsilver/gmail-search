#!/usr/bin/env bash
# web-dev-forever.sh — supervise `bun run dev` on :3000 and capture cause of death.
#
# Problem: during long sessions `bun run dev` keeps disappearing. nohup +
# setsid + signal traps didn't help. We never saw an error because Next
# logs were clipped at "[?25h" (VT cursor-reset emitted on graceful
# shutdown) — so whatever killed it was either SIGKILL (untraceable from
# the child) or a signal delivered before Node had a chance to write.
#
# This script:
#   1. Forks `bun run dev` into a new session (setsid) so parent signals
#      don't propagate.
#   2. waits for the child. Bash `wait`'s exit status encodes:
#        0..127   → child exited with that code
#        128+N    → child was terminated by signal N
#      We decode that and append a DEATH line to the log.
#   3. Reads the child's pid-file / /proc/<pid>/status at death time
#      to capture RSS, oom_score, and last 200 lines of stderr.
#   4. Restarts with exponential backoff capped at 30s. If >5 restarts
#      inside 2 minutes, stops and writes a "giving up" line — that's
#      a real bug, not a transient.

set -u

LOG="${WEB_DEV_LOG:-/home/ssilver/development/gmail-search/data/web-dev.log}"
SUPLOG="${WEB_DEV_SUPLOG:-/home/ssilver/development/gmail-search/data/web-dev-supervisor.log}"
WEB_DIR="${WEB_DIR:-/home/ssilver/development/gmail-search/web}"
PORT="${WEB_DEV_PORT:-3000}"

log() {
    # Supervisor-only log: one structured line per event, prefixed with UTC ts.
    printf '%s sup %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >>"$SUPLOG"
}

decode_wait_status() {
    # $1 = `wait` exit code. Returns a human-readable string.
    local rc=$1
    if ((rc >= 128 && rc <= 192)); then
        local sig=$((rc - 128))
        local name
        name=$(kill -l "$sig" 2>/dev/null || echo "?")
        echo "signal=${name}(${sig})"
    else
        echo "exit=${rc}"
    fi
}

capture_proc_snapshot() {
    # Best-effort — the child is already dead by the time we're here, so
    # /proc/<pid>/* may be gone. But if supervise forks fast this can catch
    # /proc in its last moments; on 99% of runs this returns nothing useful.
    local pid=$1
    if [ -r "/proc/$pid/status" ]; then
        log "proc-status: $(awk -F'\t' '/^(State|VmRSS|Threads|oom_score):/ {printf "%s=%s ",$1,$2}' /proc/"$pid"/status | tr -s ' ')"
    fi
}

check_dmesg_for_oom() {
    # Requires dmesg read permission — on Pop!_OS / most recent distros
    # `kernel.dmesg_restrict=1` by default, so a plain user gets nothing.
    # Probe once on startup and warn if we won't be able to see OOMs;
    # otherwise every death silently misses the "Killed process" line
    # the user actually needs.
    local cutoff=$1  # unix ts
    local matches
    matches=$(dmesg -T --since "@${cutoff}" 2>/dev/null | grep -iE "killed process|out of memory|oom-killer" | tail -5)
    [ -n "$matches" ] && log "dmesg-oom: $(echo "$matches" | tr '\n' '|')"
}

check_dmesg_permission_once() {
    if ! dmesg -T 2>/dev/null | head -1 >/dev/null; then
        log "warning: cannot read dmesg (kernel.dmesg_restrict=1?). OOM-kill detection disabled; fix with 'sudo sysctl -w kernel.dmesg_restrict=0' if needed."
    fi
}

# Clean-shutdown hook: supervisor SIGTERM also stops the child (and its
# process group) so there's a way to fully tear down. Without this, a
# SIGTERM to the supervisor would leave bun running in its own session.
shutdown_child_pg_on_signal() {
    local sig=$1
    log "supervisor caught SIG${sig} — tearing down child process group"
    if [ -n "${child:-}" ] && kill -0 "$child" 2>/dev/null; then
        # `-pgid` targets the whole session/group setsid gave the child.
        local pgid
        pgid=$(ps -o pgid= -p "$child" 2>/dev/null | tr -d ' ')
        if [ -n "$pgid" ]; then
            kill -TERM "-${pgid}" 2>/dev/null || true
            sleep 2
            kill -KILL "-${pgid}" 2>/dev/null || true
        else
            kill -TERM "$child" 2>/dev/null || true
        fi
    fi
    exit 0
}

mkdir -p "$(dirname "$LOG")" "$(dirname "$SUPLOG")"
log "starting supervisor pid=$$ web_dir=$WEB_DIR port=$PORT log=$LOG"
check_dmesg_permission_once

# Supervisor-side signal traps. SIGTERM / SIGINT now tear down the
# child process group too — gives operators a way to cleanly stop the
# whole stack. SIGHUP is explicitly IGNORED (we're detached on purpose)
# so terminal disconnects can't kill us.
child=""
trap 'shutdown_child_pg_on_signal TERM' TERM
trap 'shutdown_child_pg_on_signal INT' INT
trap 'log "supervisor caught SIGHUP — ignoring (continuing)"' HUP

restart_count=0
restart_window_start=$(date +%s)
backoff=1

while true; do
    cd "$WEB_DIR" || { log "fatal: cannot cd $WEB_DIR"; exit 1; }

    # Fresh session for the child — its own pgid, no controlling tty, its
    # own stdio handles. Stdout/stderr both land in $LOG so Next's own
    # request log + any Node panic / stderr are in one place.
    # `bun run dev` already carries `-p 3000` from package.json; we do
    # NOT append a second -p here (bun would pass both through and the
    # duplicate confuses Next's arg parser).
    setsid bun run dev >>"$LOG" 2>&1 <&- &
    child=$!
    log "spawned child pid=$child"

    wait "$child"
    rc=$?
    died_at=$(date +%s)
    cause=$(decode_wait_status "$rc")
    log "child $child died rc=$rc ($cause)"
    capture_proc_snapshot "$child"
    check_dmesg_for_oom "$((died_at - 60))"

    # Restart-rate guard so we don't spin on a genuinely broken build.
    now=$(date +%s)
    if ((now - restart_window_start > 120)); then
        restart_window_start=$now
        restart_count=0
    fi
    ((restart_count++))
    if ((restart_count > 5)); then
        log "giving up: $restart_count restarts in under 2 min. Something is broken — check $LOG."
        exit 1
    fi

    log "respawning in ${backoff}s (restart_count=$restart_count)"
    sleep "$backoff"
    # Exponential backoff, cap 30s.
    if ((backoff < 30)); then
        backoff=$((backoff * 2))
        ((backoff > 30)) && backoff=30
    fi
done
