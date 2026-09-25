#!/usr/bin/env bash
# Detect a loop marker patched into a comment after it was posted, on one
# issue (#483).
#
# Neither this script nor land.sh, which calls it, tries to decide who posted
# a comment: #475 established that `gh` reports every actor here as the owner,
# so that question cannot be answered from the API at all. This asks a
# narrower question instead, one the API *can* answer: was the marker part of
# the body as first posted, or added later? The owner has never once used
# `<!-- issue-loop -->`, so any comment carrying it is the loop's regardless of
# who GitHub says posted it; the only remaining question is whether it was
# there from the start.
#
# A comment whose created_at and updated_at differ was edited at least once.
# If it now carries the marker and was never edited through comment.sh's edit
# path (comment-edits.txt, written before the edit — see comment.sh), there is
# no record that the marker was present before that edit, so the edit is
# treated as the violation this issue is about: exactly the pattern of
#   5750088623 on #457: created 2026-09-20T13:26:19Z, updated ...:26:25Z (6s)
#   5762914300 on #460: created 2026-09-21T15:19:55Z, updated ...:32:50Z (13m55s)
# comment.sh's edit mode only ever edits a comment that already carries the
# marker (it refuses otherwise), so a comment-edits.txt record is proof the
# edit did not add the marker — the flag below exempts exactly those, and
# nothing else.
#
# What this cannot catch, and does not try to: a comment that never gets the
# marker at all is indistinguishable from a genuine short owner reply by
# anything in the API (#475's finding), so it is not flagged here. That half
# is watch-issues.sh's settle window, not this script's job — see #483's PR
# body for why closing it fully is not possible with what GitHub exposes.
#
# This is scoped to the owner's own account, same as classify_comments() in
# watch-issues.sh (`select(.user.login == $owner)`), and for the same reason:
# this is a public repository, and anyone can comment `<!-- issue-loop -->`
# and then edit their own comment. Without this filter that would fail every
# landing on the issue until a human passed --allow-comment-edit for a
# stranger's comment — an unauthenticated denial of service against land.sh
# (found in review). A comment from another account can never be mistaken for
# the loop's own regardless of what text it contains, so this filter costs
# nothing real: the loop's own comments and the owner's own both still show as
# $owner, which is the whole reason #475 had to solve this differently to
# begin with.
#
# usage: check-marker-provenance.sh <issue> [allowed-comment-id]...
# Any comment id passed after <issue> is exempted from the flag below, same as
# a comment-edits.txt record — land.sh's --allow-comment-edit passes them
# through here. Both are "this edit did not patch the marker in", the
# difference being who is vouching for it: comment.sh records its own edits
# automatically; a raw `gh api PATCH` edit has no such record, and passing its
# id here is a human or agent saying so explicitly, visibly, in the landing
# command (#483).
#
# exit 0: clean. exit 1: at least one violation, described on stdout/stderr.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib.sh
. "$here/lib.sh"

owner=scottmsilver
issue=${1:-}
case "$issue" in ''|*[!0-9]*) loop_warn "usage: $(basename "$0") <issue> [allowed-comment-id]..."; exit 2 ;; esac
shift
allowed=" $* "

main="$(loop_main_checkout "$here")"
repo="$(loop_repo_slug "$main")"
dir="$(loop_state_dir "$main")"
edits="$dir/comment-edits.txt"
touch "$edits"

# --paginate: an issue with more than one page of comments (100) would
# otherwise silently drop anything past the first page from this check (found
# in review) — a real issue's history is not bounded by that count the way
# watch-issues.sh's time-windowed `since` listing is.
comments=$(gh api --paginate "repos/$repo/issues/$issue/comments?per_page=100" </dev/null)

violations=$(printf '%s' "$comments" | jq -r --arg marker "$LOOP_MARKER" --arg owner "$owner" '
  .[]? | select(.user.login == $owner) |
  select((.body // "") | contains($marker)) | select(.created_at != .updated_at) |
  [(.id|tostring), .created_at, .updated_at] | @tsv')

[ -n "$violations" ] || { loop_say "#$issue: no marker-provenance violation found"; exit 0; }

found=0
while IFS=$'\t' read -r id created updated; do
  [ -n "$id" ] || continue
  if awk -v id="$id" '{ if ($2 == id) found=1 } END { exit !found }' "$edits" 2>/dev/null; then
    loop_say "#$issue: comment $id was edited ($created -> $updated) but comment.sh recorded the edit (already marked before it); not a violation"
    continue
  fi
  case "$allowed" in
    *" $id "*)
      loop_say "#$issue: comment $id was edited ($created -> $updated), no comment.sh record, but explicitly allowed on this run; not a violation"
      continue ;;
  esac
  found=1
  loop_warn "#$issue: comment $id carries the loop marker and was edited after posting, with no comment.sh record that it was already marked ($created -> $updated). This is the shape of #483: the marker was likely patched in after the fact."
done <<<"$violations"

[ "$found" -eq 0 ] || exit 1
