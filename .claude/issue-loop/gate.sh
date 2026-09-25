#!/usr/bin/env bash
# Remove a gate label (loop:proposed / loop:hold) on the loop's behalf, and
# record that the loop did it so watch-issues.sh does not read the removal back
# as the owner approving.
#
# usage: gate.sh <issue> -r <gate-label> [-r <gate-label>] [-a <label>]...
#   gate.sh 460 -r loop:proposed -a loop:working
#
# `gh` authenticates as the owner for every actor here, and the GitHub timeline
# reports `actor=scottmsilver` for the loop's own label edits as well as the
# human's, so the removal cannot be attributed after the fact (#475). The record
# this writes is the only provenance that exists. Every gate removal the loop
# makes — the orchestrator clearing loop:proposed to dispatch, or releasing an
# epic child's loop:hold once the owner has approved it — goes through here
# instead of `gh issue edit`, or the watcher reports it as owner approval.
#
# land.sh and deploy.sh do not need this: they only move loop:working ->
# loop:merged -> loop:deployed, and none of those is a gate label.
#
# -r is restricted to the two gate labels on purpose. Anything else is not what
# the watcher diffs, so routing it through here would record provenance nothing
# reads and imply a guarantee this script does not give; use `gh issue edit`.
#
# The watcher consumes a record when it suppresses a removal, so a record this
# script writes without going on to make that removal would swallow the owner
# clearing the same label on the same issue, up to LOOP_SELF_LABEL_TTL later.
# Two things keep that from happening: the label is checked to be present before
# anything is recorded, and a failed edit takes the record back.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=.claude/issue-loop/lib.sh
. "$here/lib.sh"

main="$(loop_main_checkout "$here")"
dir="$(loop_state_dir "$main")"
self_labels="$dir/self-labels.txt"

usage() { loop_warn "usage: $(basename "$0") <issue> -r loop:proposed|loop:hold [-a <label>]..."; exit 2; }

issue=${1:-}
# Digits only: the issue number is a whitespace-delimited field in the record
# file and a fixed string in the grep that takes a record back.
case "$issue" in
  ''|*[!0-9]*) usage ;;
esac
shift

remove=() args=()
while [ $# -gt 0 ]; do
  case "$1" in
    -r|--remove)
      case "${2:-}" in
        loop:proposed|loop:hold) ;;
        *) loop_warn "-r takes loop:proposed or loop:hold, not '${2:-}': only those two are gates the watcher diffs; use gh issue edit"; exit 2 ;;
      esac
      remove+=("$2"); args+=(--remove-label "$2"); shift 2 ;;
    -a|--add)
      [ -n "${2:-}" ] || { loop_warn "-a needs a label"; exit 2; }
      args+=(--add-label "$2"); shift 2 ;;
    *) loop_warn "unknown argument '$1'"; usage ;;
  esac
done
[ ${#remove[@]} -gt 0 ] || { loop_warn "nothing to remove; this script exists to record a gate removal"; exit 2; }

repo="$(loop_repo_slug "$main")"

# Only record a removal that is actually there to make. A record for a label the
# issue does not carry is a suppression the watcher would spend on the owner.
present=$(gh issue view "$issue" --repo "$repo" --json labels -q '[.labels[].name] | join(" ")')
for g in "${remove[@]}"; do
  case " $present " in
    *" $g "*) ;;
    *) loop_warn "#$issue does not carry $g (labels: ${present:-none}); refusing to record a removal that is not happening"; exit 1 ;;
  esac
done

# watch-issues.sh rewrites this file to consume and prune records, so both sides
# take the same lock around it.
exec 9>"$dir/self-labels.lock"
flock 9

stamp=$(date +%s)
# Recorded BEFORE the edit, never after: the watcher polls every 30s, and a poll
# landing between the edit and the record would read the removal as the owner's.
# The cost of that ordering is a record whose edit then fails, which is what the
# unwind below is for.
for g in "${remove[@]}"; do printf '%s %s %s\n' "$stamp" "$issue" "$g" >>"$self_labels"; done
# Closed, not just unlocked: an open descriptor on the lock file would otherwise
# be inherited by the `gh` below and by everything it spawns.
flock -u 9; exec 9>&-

unwind() {
  local g tmp rc
  exec 9>"$dir/self-labels.lock"
  flock 9
  for g in "${remove[@]}"; do
    tmp=$(mktemp "$dir/.self-labels.XXXXXX")
    # grep exits 1 when it printed nothing, which here means the file is left
    # empty — a legitimate result, not a failure. Only 2 and above are failures,
    # and those must not be allowed to truncate the records.
    set +e
    grep -vxF "$stamp $issue $g" "$self_labels" >"$tmp"
    rc=$?
    set -e
    if [ "$rc" -le 1 ]; then
      mv "$tmp" "$self_labels"
    else
      rm -f "$tmp"
      loop_warn "could not take back the record for #$issue $g; it expires in ${LOOP_SELF_LABEL_TTL:-600}s"
    fi
  done
  flock -u 9
  exec 9>&-
}

loop_say "#$issue: removing ${remove[*]} as the loop, recorded in $(basename "$self_labels")"
if ! gh issue edit "$issue" --repo "$repo" "${args[@]}" >/dev/null; then
  loop_warn "#$issue: gh issue edit failed; taking the provenance record back"
  unwind
  exit 1
fi
