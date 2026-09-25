#!/usr/bin/env bash
# Post or edit an issue comment with the loop marker guaranteed present in the
# body BEFORE the one `gh` call that writes it, so no comment is ever visible
# without the marker for even a moment.
#
# usage: comment.sh <issue> --body-file <path>
#        comment.sh <issue> --body <text>
#        comment.sh edit <comment-id> --body-file <path>
#        comment.sh edit <comment-id> --body <text>
#
# `gh` runs as the owner for every actor on this machine (#475), so a comment
# missing the marker for any window reads as the owner's own words. Every
# script and brief has said "the marker goes in the body as posted" since the
# loop was written, and it has still been violated three times in two days by
# posting unmarked and patching the marker in with a second edit:
#   comment 5750088623 on #457: created 2026-09-20T13:26:19Z, updated ...:26:25Z
#   comment 5762914300 on #460: created 2026-09-21T15:19:55Z, updated ...:32:50Z
# (#483). This script exists so an agent that uses it cannot produce that
# gap, on either path:
#
#   post: the marker is added to the body, if it is not already there, before
#         the single `gh issue comment` call. There is never a second call.
#   edit: refuses outright when the CURRENT (pre-edit) body does not already
#         carry the marker. Editing a marker in is exactly the violation this
#         script exists to close, so that path is not offered — post a new,
#         correctly-marked comment instead. When the current body is already
#         marked, the edit is a content change, not a marker patch, and is
#         allowed; it is recorded in comment-edits.txt (locked and written
#         BEFORE the edit, same as gate.sh's self-labels.txt for the same
#         reason: a reader landing between the write and the edit must not
#         see an edit that has not happened) so
#         check-marker-provenance.sh can tell a sanctioned content edit apart
#         from a comment nobody can vouch for.
#
# It does not, and cannot, stop an agent that ignores the brief and calls `gh
# issue comment` / `gh api -X PATCH` directly — that agent is exactly the one
# this exists for, and no wrapper reaches code that never calls it. That is
# what check-marker-provenance.sh and land.sh's use of it are for (#483).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib.sh
. "$here/lib.sh"

usage() {
  loop_warn "usage: $(basename "$0") <issue> --body-file <path> | --body <text>"
  loop_warn "       $(basename "$0") edit <comment-id> --body-file <path> | --body <text>"
  exit 2
}

[ $# -ge 1 ] || usage

mode=post
target=${1:?}
shift
if [ "$target" = edit ]; then
  mode=edit
  target=${1:-}
  shift || true
  case "$target" in ''|*[!0-9]*) usage ;; esac
else
  case "$target" in ''|*[!0-9]*) usage ;; esac
fi

body_file="" body_text=""
while [ $# -gt 0 ]; do
  case "$1" in
    --body-file) body_file=${2:?--body-file needs a path}; shift 2 ;;
    --body) body_text=${2:?--body needs text}; shift 2 ;;
    *) loop_warn "unknown argument '$1'"; usage ;;
  esac
done
if { [ -n "$body_file" ] && [ -n "$body_text" ]; } || { [ -z "$body_file" ] && [ -z "$body_text" ]; }; then
  usage
fi

main="$(loop_main_checkout "$here")"
repo="$(loop_repo_slug "$main")"
dir="$(loop_state_dir "$main")"
edits="$dir/comment-edits.txt"

new_body=$(mktemp)
trap 'rm -f "$new_body"' EXIT
if [ -n "$body_file" ]; then
  [ -f "$body_file" ] || { loop_warn "no such file: $body_file"; exit 1; }
  cp "$body_file" "$new_body"
else
  printf '%s\n' "$body_text" >"$new_body"
fi

# Marker present anywhere in the body counts as posted-with-the-marker: an
# agent that put it mid-body instead of at the end has not left the
# posted-then-patched gap this exists to close. Appending a second copy would
# just be noise. This mirrors classify_comments() in watch-issues.sh, which
# also treats "contains" as marked.
has_marker() { grep -qF "$LOOP_MARKER" "$1"; }

if [ "$mode" = post ]; then
  has_marker "$new_body" || printf '\n%s\n' "$LOOP_MARKER" >>"$new_body"
  loop_say "#$target: posting comment ($(wc -c <"$new_body" | tr -d ' ') bytes, marker present before post)"
  gh issue comment "$target" --repo "$repo" --body-file "$new_body"
  exit 0
fi

# --- edit --------------------------------------------------------------------

current=$(mktemp)
tmp=""
trap 'rm -f "$new_body" "$current" "$tmp"' EXIT
if ! gh api "repos/$repo/issues/comments/$target" --jq .body </dev/null >"$current" 2>/dev/null; then
  loop_warn "could not fetch comment $target to check its current body"
  exit 1
fi

if ! has_marker "$current"; then
  loop_warn "refusing to edit comment $target: its current body does not carry the loop marker."
  loop_warn "Editing the marker in is the exact bug #483 covers. Post a NEW, correctly-marked"
  loop_warn "comment instead of patching this one; if this comment is not the loop's, it must"
  loop_warn "not be touched at all."
  exit 1
fi

has_marker "$new_body" || printf '\n%s\n' "$LOOP_MARKER" >>"$new_body"

# Recorded BEFORE the edit, never after, for the same reason gate.sh records a
# label removal first: a reader between the write and the edit must see the
# edit as already accounted for, not as an unexplained gap.
stamp=$(date +%s)
record="$stamp $target"
exec 9>"$dir/comment-edits.lock"
flock 9
printf '%s\n' "$record" >>"$edits"
flock -u 9
exec 9>&-

loop_say "comment $target: was already marked; editing content, recording in $(basename "$edits")"
if ! jq -n --rawfile body "$new_body" '{body:$body}' |
    gh api -X PATCH "repos/$repo/issues/comments/$target" --input - >/dev/null; then
  loop_warn "edit of comment $target failed; taking the provenance record back"
  exec 9>"$dir/comment-edits.lock"
  flock 9
  tmp=$(mktemp "$dir/.comment-edits.XXXXXX")
  # grep exits 1 when it printed nothing, which here means the file is left
  # empty -- a legitimate result, not a failure. Only 2 and above are
  # failures, and those must not be allowed to truncate comment-edits.txt (the
  # same class of bug gate.sh's own unwind guards against with `rc -le 1`).
  set +e
  grep -vxF "$record" "$edits" >"$tmp"
  rc=$?
  set -e
  if [ "$rc" -le 1 ]; then
    mv "$tmp" "$edits"
  else
    rm -f "$tmp"
    loop_warn "could not take back the edit record for comment $target; it stays in $(basename "$edits")"
  fi
  tmp=""
  flock -u 9
  exec 9>&-
  exit 1
fi
