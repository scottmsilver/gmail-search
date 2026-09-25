#!/usr/bin/env bash
# Shared helpers for the /issue-loop scripts. Sourced, never executed.
#
# The landing lock serialises the one thing that cannot run twice at once: a
# full `scripts/test.sh` run (it shares one disposable test Postgres). Landings take it for a whole issue; a deploy takes it
# only around its own suite. Both write who they are and a pid, so a lock left
# behind by a dead process can be told apart from one that is still working.

# Absolute path of the MAIN checkout, even when called from a linked worktree:
# the lock, the ledger and the deploy log are shared state and live there.
loop_main_checkout() {
  local from=${1:-.} common
  common=$(git -C "$from" rev-parse --path-format=absolute --git-common-dir)
  dirname "$common"
}

# owner/repo, never hard-coded. Takes the checkout to ask from, because `gh`
# reads the remote of its own working directory.
loop_repo_slug() { ( cd "${1:-.}" && gh repo view --json nameWithOwner -q .nameWithOwner ); }

loop_state_dir() {
  local dir="$1/.runtime/issue-loop"
  mkdir -p "$dir"
  printf '%s\n' "$dir"
}

loop_now() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# Everything a script is about to do gets announced first.
loop_say() { printf '==> %s\n' "$*"; }
loop_warn() { printf '!!  %s\n' "$*" >&2; }

_loop_lock_dir=""

# Describe the current holder of a lock directory, or nothing if it is free.
loop_lock_holder() {
  local lock="$1"
  [ -d "$lock" ] || return 1
  cat "$lock/owner" 2>/dev/null || printf 'unknown holder\n'
}

# True when the holder line names a pid that is no longer running, on the
# same host that wrote it. A pid only means something on the machine that
# wrote it: the deployer's lock (scripts/deploy) applies the same host
# check, since the two implementations exclude each other over one lock.
loop_lock_is_stale() {
  local holder="$1" host="${2:-$(hostname)}" pid wrote_host
  pid=$(printf '%s\n' "$holder" | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p')
  [ -n "$pid" ] || return 1          # no pid recorded: assume a live agent
  wrote_host=$(printf '%s\n' "$holder" | sed -n 's/.*host=\([^ ]*\).*/\1/p')
  if [ -n "$wrote_host" ] && [ "$wrote_host" != "$host" ]; then
    return 1                          # a pid means nothing on another machine
  fi
  if kill -0 "$pid" 2>/dev/null; then return 1; fi
  return 0
}

# loop_lock_acquire <lock-dir> <label> <refuse|wait:SECONDS>
#
# Bash-side waiters serialise the whole "read the current holder, then act on
# it" step through an advisory lock next to the lock directory itself (never a
# global file, so unrelated locks never serialise against each other). Reading
# a holder and acting on it are two separate filesystem operations; without
# this, a waiter's takeover of what it read as a *stale* lock could still land
# on a *different* directory -- one another waiter had just created and not
# yet finished writing an owner file into -- even after the rename-aside
# takeover below closes the original double-clear (two waiters both deciding
# the same lock is stale and both running `rm -rf`). This mutex makes that
# window impossible between two lib.sh callers, and between lib.sh and
# the deployer's lock (scripts/deploy), whose acquire() takes the same flock on the same
# side file (#555). The mv-aside takeover and the write-then-read-back check
# below stay, as they do on the Node side, for a deploy still running a
# the deployer lock from before #555, which does not take the mutex.
loop_lock_acquire() {
  local lock="$1" label="$2" mode="$3" waited=0 limit=0 holder aside mine moved mutex
  case "$mode" in
    wait:*) limit=${mode#wait:} ;;
  esac
  mutex="$lock.mutex"
  while :; do
    exec 9>"$mutex"
    flock 9
    mine="$label pid=$$ host=$(hostname) $(loop_now)"
    if mkdir "$lock" 2>/dev/null; then
      if printf '%s\n' "$mine" >"$lock/owner" 2>/dev/null &&
         [ "$(cat "$lock/owner" 2>/dev/null)" = "$mine" ]; then
        _loop_lock_dir="$lock"
        flock -u 9; exec 9>&-
        loop_say "landing lock held by $label (pid $$)"
        return 0
      fi
      # A non-bash actor swept this directory away between our mkdir and our
      # write; we do not actually hold it. Go round and retry.
      flock -u 9; exec 9>&-
      continue
    fi
    holder=$(loop_lock_holder "$lock" || printf 'unknown holder\n')
    if loop_lock_is_stale "$holder"; then
      # Renamed out of the way rather than removed in place, matching
      # the deployer's lock (scripts/deploy)'s acquire(): only one rename of a given
      # directory can succeed, so at most one caller ever clears a given
      # stale lock. No other lib.sh or the deployer lock caller can reach this step
      # for the same lock while this one holds $mutex, so it only races a
      # deploy still running a pre-#555 the deployer lock.
      aside="$lock.stale-$$-$(date +%s%N 2>/dev/null || date +%s)"
      if mv "$lock" "$aside" 2>/dev/null; then
        # Confirm what actually got moved is the same stale holder just
        # read, not a non-bash actor's directory that landed on this name in
        # the gap between that read and this rename. If it does not match,
        # this was never stale; give the name back rather than destroying a
        # lock that is not ours to clear.
        moved=$(cat "$aside/owner" 2>/dev/null || printf '')
        if [ "$moved" = "$holder" ]; then
          loop_warn "removing a stale landing lock left by: $holder"
          rm -rf "$aside"
        else
          # This is a best-effort give-back, not a full reconciliation: a
          # non-bash actor's claim landed here between our read and this
          # rename. `-T` forces exact-name (rename) semantics instead of
          # "move into an existing directory" (plain `mv src dst` on an
          # existing dst directory nests src *inside* dst); `-n` refuses
          # instead of clobbering if that actor has since finished writing a
          # new, live claim at "$lock". Either way $aside is not kept: if the
          # restore succeeded there is nothing left to hold, and if it was
          # refused, something else already legitimately owns "$lock" and
          # $aside's contents are superseded, not the only copy of live
          # state. What this cannot do is hand the swept actor back its
          # *specific* interrupted attempt if it has not yet retried. the deployer lock
          # takes $mutex too since #555, so only a deploy still running an
          # older the deployer lock can reach this branch.
          mv -n -T "$aside" "$lock" 2>/dev/null
          rm -rf "$aside"
        fi
      fi
      flock -u 9; exec 9>&-
      continue
    fi
    flock -u 9; exec 9>&-
    if [ "$limit" -eq 0 ]; then
      loop_warn "landing lock is held by: $holder"
      return 1
    fi
    if [ "$waited" -ge "$limit" ]; then
      loop_warn "landing lock still held after ${waited}s by: $holder"
      return 1
    fi
    if [ "$waited" -eq 0 ]; then loop_say "waiting for the landing lock, held by: $holder"; fi
    sleep 10
    waited=$((waited + 10))
  done
}

loop_lock_release() {
  [ -n "$_loop_lock_dir" ] || return 0
  local owner="$_loop_lock_dir"
  _loop_lock_dir=""
  # Only delete a lock this process still owns. A takeover elsewhere could in
  # principle have replaced the directory's contents since this process's own
  # mkdir succeeded; a blind `rm -rf` here would delete whatever is there now,
  # owned or not.
  case "$(loop_lock_holder "$owner" 2>/dev/null || true)" in
    *"pid=$$ "*)
      rm -rf "$owner"
      loop_say "landing lock released"
      ;;
    *)
      loop_warn "not releasing $owner: it no longer names this process (pid $$) as owner"
      ;;
  esac
}

# The marker that tells an owner comment apart from one the loop wrote.
# shellcheck disable=SC2034  # used by the scripts that source this file
LOOP_MARKER='<!-- issue-loop -->'

# --- the release ------------------------------------------------------------
#
# Packaging, qualifying and activating a release is the deployer in
# scripts/deploy/ (repository code with its own tests). The lock, the ledger and
# the say/warn helpers above stay here: the landing scripts use them, and the
# deployer takes the same lock directory by the same protocol so a deploy and a
# landing never run the test suite at once.

# The disposable test database for scripts/test.sh and pytest, kept out of the
# repository: ~/.config/gmail-search/test.env sets GMS_TEST_PG_DSN (and
# GMS_GATEWAY_TEST_DSN). Refuses the live database's port, as conftest does.
LOOP_TEST_ENV="${GMS_TEST_ENV_FILE:-$HOME/.config/gmail-search/test.env}"
loop_test_env() {
  [ -r "$LOOP_TEST_ENV" ] || { loop_warn "no test database config at $LOOP_TEST_ENV (copy deploy/examples/test.env.example)"; return 1; }
  (
    set -a; . "$LOOP_TEST_ENV"; set +a
    case "${GMS_TEST_PG_DSN:-}" in
      "") loop_warn "GMS_TEST_PG_DSN is not set in $LOOP_TEST_ENV"; exit 1 ;;
      *port=5544*|*:5544/*) loop_warn "GMS_TEST_PG_DSN points at the live database port"; exit 1 ;;
    esac
    exec "$@"
  )
}

# The checks that stand in for a build: the locked environment, ruff, and the
# web typecheck and script tests when the change touches web/. Run from the
# tree being checked; web/node_modules is linked from the main checkout.
loop_build() {
  local main
  main=$(loop_main_checkout .)
  uv sync --locked --extra dev && uv run --locked ruff check src/ tests/ || return 1
  if ! git diff --quiet origin/main HEAD -- web/ 2>/dev/null || ! git diff --quiet HEAD -- web/ 2>/dev/null; then
    [ -e web/node_modules ] || ln -s "$main/web/node_modules" web/node_modules
    ( cd web && npx tsc --noEmit -p . && node --import tsx --test scripts/test-*.mjs ) || return 1
  fi
}
