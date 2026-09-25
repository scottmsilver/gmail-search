#!/usr/bin/env bash
# Emit one line per owner event for /issue-loop: a new issue the owner wrote,
# a new owner comment, a loop:proposed / loop:hold label removed (the owner
# approving), or the owner approving a landing. State lives in
# .runtime/issue-loop so re-arming does not re-announce.
#
# `gh` authenticates as the owner for every actor on this machine, so nothing in
# the GitHub payload separates a loop agent from the human: 51 consecutive
# labeled/unlabeled events across nine issues all report
# `actor=scottmsilver performed_via_github_app=none`, including the ones the
# loop's own scripts made. Provenance therefore has to come from the loop's own
# records and from waiting long enough to be sure, which is what the rules below
# are (#475).
#
#   1. A comment carrying the loop marker is the loop talking to itself and is
#      dropped. An agent that posts first and patches the marker in afterwards
#      is unmarked for a while, so that check alone is not enough: on #457 the
#      window was 6s, on #460 13m55s, and a poll inside it read the loop's own
#      plan comment as the owner's.
#   2. So an unmarked comment is never classified on sight. It is held in
#      pending-comments.txt and re-fetched by id once a settle window has passed
#      with the body unchanged; if the marker has appeared by then it was the
#      loop's. Editing a comment restarts that window, and the marker edit also
#      moves `updated_at`, which puts the comment back in the next `since`
#      listing where pass 1 recognises it. Nothing is ever dropped for being
#      edited — a comment is only ever held, so the owner fixing a typo is still
#      heard, just no sooner than the window.
#      The window is longer for a long comment, because that is the shape whose
#      marker arrives late. Across 213 comments on the last 60 issues, all 27
#      unmarked owner comments were 177 characters or fewer (median 13, and 24
#      of 27 a single line), while all 186 loop comments were 106 or more
#      (median 2043). Length picks the window and NOTHING else: a comment is
#      never classified by its shape, so a body on the wrong side of the
#      threshold costs latency, never a misclassification. A body that grows
#      past the threshold has its window raised, never lowered.
#   3. A gate label that disappears is only the owner's approval if the loop did
#      not remove it itself. gate.sh records every gate removal the loop makes
#      in self-labels.txt before making it, and those are suppressed here. The
#      orchestrator clearing loop:proposed to dispatch fired `gate-removed`
#      seven times in one day before this. A record is CONSUMED when it
#      suppresses a removal, so the loop clearing a label cannot go on silencing
#      the owner clearing that same label again later.
#   4. A comment's target is confirmed to be an issue, not a pull request. PR
#      conversation comments arrive from this same `/issues/comments` endpoint
#      with `.issue_url` ending in the PR's number, and PR numbers share one
#      sequence with issue numbers — so `land` on PR #480 read as approval of
#      issue #480, which is not a mistake the orchestrator could have noticed.
#      The open-issue list answers this for free most of the time; anything not
#      on it is asked about directly, so an issue closed while its comment was
#      held is still heard rather than quietly dropped.
#
# `land-approved #<n>` is the only line that authorises a commit, so it is the
# strictest. It takes ALL of: the owner's account as author; a target GitHub
# confirms is an issue and not a pull request; a body that, trimmed of ASCII
# whitespace and lowercased, is exactly `land`; no marker when it was first seen;
# that body unchanged for at least the settle window; and no loop marker on a
# fresh re-fetch of that one comment id at the end of it. It is emitted only from
# pass 2, never from the bulk listing. A comment that complies with the marker
# rule cannot satisfy that; neither can one whose marker arrives before the
# window is up. What it finally rests on is that no code path in the loop posts a
# bare `land` — deploy.sh is the only scripted comment and it builds its body
# with $LOOP_MARKER inline — not on the length of any timer.
#
# Failure is resolved towards saying nothing and trying again, never towards
# inventing an owner event: the cursor does not advance unless the comment
# listing actually returned, an issue is not recorded as seen until its body
# fetch gave a definite answer, and a comment stays pending until its re-fetch
# does. The one thing that is NOT retried forever is a comment whose re-fetch
# keeps failing: after LOOP_REFETCH_ATTEMPTS it is dropped, so a deleted comment
# does not sit in the pending set being re-fetched for ever.
#
# Comment bodies are attacker-influenced text from a public repository. They are
# never interpolated into a shell command or a jq program: the marker and the
# owner login go in as jq --arg, and only derived tokens (an id, an issue
# number, two fixed words and a length) cross back into the shell.
set -u
here="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=.claude/issue-loop/lib.sh
. "$here/lib.sh"
# The cursor is shared state and belongs to the main checkout, so a copy of this
# script run from a linked worktree cannot fork it.
dir="$(loop_state_dir "$(loop_main_checkout "$here")")"
seen="$dir/seen.txt"
pending="$dir/pending-comments.txt"
emitted="$dir/emitted-comments.txt"
self_labels="$dir/self-labels.txt"
owner=scottmsilver
settle=${LOOP_SETTLE_SECONDS:-120}
settle_long=${LOOP_SETTLE_LONG_SECONDS:-900}
long_body=${LOOP_LONG_BODY_CHARS:-200}
interval=${LOOP_WATCH_INTERVAL:-30}
self_label_ttl=${LOOP_SELF_LABEL_TTL:-600}
refetch_attempts=${LOOP_REFETCH_ATTEMPTS:-20}
repo=$(gh repo view --json nameWithOwner -q .nameWithOwner)
touch "$seen" "$pending" "$emitted" "$self_labels"
since=$(cat "$dir/since.txt" 2>/dev/null || date -u +%Y-%m-%dT%H:%M:%SZ)

# One tab-separated record per comment the owner's account posted, from a single
# comment object or an array of them on stdin:
#   <id> <issue> marked|bare land|text <body-length>
# The body is reduced to those tokens here and never leaves jq.
classify_comments() {
  jq -r --arg owner "$owner" --arg marker "$LOOP_MARKER" '
    (if type == "array" then .[] else . end)
    | select(.user.login == $owner)
    | (.body // "") as $b
    | [ (.id | tostring),
        # `// ""` because jq dies on split(null), and one bad record would
        # otherwise end classification for the whole batch mid-stream.
        ((.issue_url // "") | split("/") | last),
        (if ($b | contains($marker)) then "marked" else "bare" end),
        # The trim is spelled out rather than left to \s, which in the jq regex
        # engine takes U+00A0 but not U+200B. The one token that authorises a
        # commit should mean exactly what it says — optional ASCII whitespace
        # around the four bytes land — and should not move when the regex engine
        # changes its mind about which Unicode characters are whitespace.
        (if ($b | gsub("^[ \\t\\r\\n\\f]+|[ \\t\\r\\n\\f]+$"; "") | ascii_downcase) == "land"
           then "land" else "text" end),
        ($b | length) ] | @tsv'
}

# How long an unmarked comment of this length has to sit unchanged. Long bodies
# are the shape whose marker is patched in late (13m55s on #460), so they wait
# longer; the owner's register is short, so their answers and their `land` do
# not. This picks a delay, never a verdict.
settle_for() {
  if [ "${1:-0}" -gt "$long_body" ]; then printf '%s\n' "$settle_long"; else printf '%s\n' "$settle"; fi
}

# Emit at most once per comment id, whichever pass gets there first. The old
# script deduplicated with a per-poll `sort -u`, which does nothing across the
# poll boundary that `since` deliberately overlaps.
emit_comment() {
  local id=$1 n=$2 kind=$3
  grep -qxF "$id" "$emitted" && return 0
  printf '%s\n' "$id" >>"$emitted"
  if [ "$(wc -l <"$emitted")" -gt 1000 ]; then
    tail -n 500 "$emitted" >"$emitted.new" && mv "$emitted.new" "$emitted"
  fi
  if [ "$kind" = land ]; then echo "land-approved #$n"; else echo "owner-comment #$n"; fi
}

# pending-comments.txt: <id> <unchanged-since> <window> <body-length> <failures>
pending_field() { awk -v id="$1" -v f="$2" '$1 == id { print $f; exit }' "$pending"; }
pending_remove() {
  awk -v id="$1" '$1 != id' "$pending" >"$pending.new" && mv "$pending.new" "$pending"
}

# Record, or re-record, an unmarked comment. A body that has changed length
# restarts the window, and the window only ever grows: a comment that started
# short and was expanded into a plan is the shape whose marker is still to come,
# and the short timer must not survive that.
pending_note() {
  local id=$1 len=$2 window prev_len prev_window
  window=$(settle_for "$len")
  prev_len=$(pending_field "$id" 4)
  if [ -z "$prev_len" ]; then
    printf '%s %s %s %s 0\n' "$id" "$(date +%s)" "$window" "$len" >>"$pending"
    return 0
  fi
  [ "$prev_len" != "$len" ] || return 0
  prev_window=$(pending_field "$id" 3)
  [ "$window" -gt "${prev_window:-0}" ] || window=$prev_window
  pending_remove "$id"
  printf '%s %s %s %s 0\n' "$id" "$(date +%s)" "$window" "$len" >>"$pending"
}

# A re-fetch that did not answer. Kept for another poll, up to a limit: a
# comment the owner deleted would otherwise be re-fetched for ever.
pending_failed() {
  awk -v id="$1" -v max="$refetch_attempts" '
    $1 != id { print; next }
    { $5 = $5 + 1; if ($5 < max) print }' "$pending" >"$pending.new" && mv "$pending.new" "$pending"
}

# `pull`, `issue`, or empty when GitHub did not say. A pull request and an issue
# share one number sequence here, and both answer on /issues/<n>; only a pull
# request carries a `pull_request` key.
target_kind() {
  local v
  v=$(gh api "repos/$repo/issues/$1" </dev/null 2>/dev/null |
    jq -r 'if has("pull_request") then "pull" else "issue" end' 2>/dev/null) || return 0
  case "$v" in pull|issue) printf '%s\n' "$v" ;; esac
}

# self-labels.txt is written by gate.sh and rewritten here, so both sides take
# this lock around it.
self_lock() { exec 9>"$dir/self-labels.lock"; flock 9; }
self_unlock() { flock -u 9; exec 9>&-; }

# True when gate.sh recorded this exact removal within the TTL: the loop lifted
# the gate itself, so it is not the owner speaking. The record is consumed, so
# one loop removal suppresses exactly one `gate-removed` — the owner clearing
# that same label again inside the TTL is still reported. Expired records are
# pruned in the same pass. A record left behind by a gate.sh that died between
# writing and editing shadows at most one genuine owner removal of that label on
# that issue, and only until it expires.
loop_removed_gate() {
  local n=$1 g=$2 cutoff=$(( $(date +%s) - self_label_ttl )) tmp
  self_lock
  if awk -v n="$n" -v g="$g" -v c="$cutoff" \
      '$1 + 0 >= c && $2 == n && $3 == g { found = 1 } END { exit !found }' "$self_labels" 2>/dev/null
  then
    tmp=$(mktemp "$dir/.self-labels.XXXXXX")
    if [ -n "$tmp" ] && awk -v n="$n" -v g="$g" -v c="$cutoff" '
        $1 + 0 < c { next }
        !used && $2 == n && $3 == g { used = 1; next }
        { print }' "$self_labels" >"$tmp" 2>/dev/null; then
      mv "$tmp" "$self_labels"
    else
      # The record could not be consumed. Still suppress: a spurious
      # `gate-removed` is an unapproved dispatch, while an unconsumed record can
      # at worst swallow one owner removal of this same label on this same issue
      # before the TTL expires. Say so on stderr, which the orchestrator does not
      # parse but a person reading the Monitor will see.
      rm -f "$tmp"
      loop_warn "could not consume the gate record for #$n $g; it may shadow an owner removal for up to ${self_label_ttl}s"
    fi
    self_unlock
    return 0
  fi
  self_unlock
  return 1
}

while true; do
  now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  # An empty list and a failed list are not the same thing: the label work below
  # is a comparison against the previous poll, and treating a failure as "no open
  # issues" would announce the whole board as changed when it came back.
  if list=$(gh issue list --state open --author "$owner" --json number,labels -q '.[] | "\(.number) \([.labels[].name | select(.=="loop:proposed" or .=="loop:hold")] | join(","))"' 2>/dev/null); then
    tracked=" $(awk '{print $1}' <<<"$list" | tr '\n' ' ')"

    # A gate label (loop:proposed / loop:hold) that disappears is the owner
    # approving, unless the loop removed it itself.
    while read -r n gates; do
      [ -n "$n" ] || continue
      prev=$(awk -v n="$n" '$1==n{print $2}' "$dir/gates.txt" 2>/dev/null)
      for g in ${prev//,/ }; do
        [[ ",$gates," == *",$g,"* ]] && continue
        loop_removed_gate "$n" "$g" && continue
        echo "gate-removed #$n $g"
      done
    done <<<"$list"
    # Written on every successful list, empty included: a gates.txt left behind
    # from when the board was busy would be diffed against an issue that reopens.
    if [ -n "$list" ]; then printf '%s\n' "$list" >"$dir/gates.txt"; else : >"$dir/gates.txt"; fi

    while read -r n gates; do
      [ -n "$n" ] || continue
      grep -qxF "$n" "$seen" && continue
      # Issues the loop filed carry the marker, and carry loop:proposed: only
      # the loop applies that label, so either one is the loop's own voice. The
      # label is the signal that survives a body whose marker was patched in
      # late, and nothing is lost by waiting — removing it emits `gate-removed`.
      if [[ ",$gates," == *",loop:proposed,"* ]]; then printf '%s\n' "$n" >>"$seen"; continue; fi
      marked=$(gh issue view "$n" --json body </dev/null 2>/dev/null |
        jq -r --arg marker "$LOOP_MARKER" '(.body // "") | contains($marker)' 2>/dev/null) || marked=""
      # An issue is recorded as seen only once the fetch gave a definite answer.
      # The old shape read a failed `gh issue view` as "no marker" and announced
      # the issue anyway — and, having already written seen.txt, never rechecked.
      case "$marked" in
        true)  printf '%s\n' "$n" >>"$seen" ;;
        false) printf '%s\n' "$n" >>"$seen"; echo "new-issue #$n" ;;
        *)     ;;
      esac
    done <<<"$list"
  else
    # Unknown, not empty. Pass 2 falls back to asking GitHub about each comment's
    # target rather than assuming anything about the board.
    tracked=""
  fi

  # Pass 1: everything new since the cursor. Marked comments are dropped, and
  # every unmarked one is held for pass 2 — nothing is classified here, and
  # nothing is discarded for being on an issue this poll happens not to know
  # about. Pass 2 resolves that against GitHub.
  if comments=$(gh api "repos/$repo/issues/comments?since=$since&per_page=100" </dev/null 2>/dev/null) &&
      classified=$(printf '%s' "$comments" | classify_comments 2>/dev/null); then
    while IFS=$'\t' read -r id n mark kind len; do
      [ -n "$id" ] || continue
      if [ "$mark" = marked ]; then pending_remove "$id"; continue; fi
      pending_note "$id" "$len"
    done <<<"$classified"
    # The cursor moves only once the listing has been read all the way through.
    # Advancing it after a failed fetch — or after a classification that died
    # part way down the page — would skip past the owner's comment for good.
    since=$now; echo "$since" >"$dir/since.txt"
  fi

  # Pass 2: a held comment, re-fetched on its own once its window has passed
  # with the body unchanged. This is the only pass that emits anything.
  snapshot=$(cat "$pending")
  while read -r id first window len fails; do
    [ -n "$id" ] || continue
    [ $(( $(date +%s) - first )) -ge "${window:-$settle}" ] || continue
    if ! raw=$(gh api "repos/$repo/issues/comments/$id" </dev/null 2>/dev/null); then
      pending_failed "$id"; continue
    fi
    # A classification that fails is not an answer, so the comment is kept and
    # retried rather than dropped on the floor.
    if ! fresh=$(printf '%s' "$raw" | classify_comments 2>/dev/null); then
      pending_failed "$id"; continue
    fi
    if [ -n "$fresh" ]; then
      IFS=$'\t' read -r _ n mark kind _ <<<"$fresh"
      if [ "$mark" != marked ]; then      # else: the loop patched its marker in
        if [[ "$tracked" == *" $n "* ]]; then
          pending_remove "$id"; emit_comment "$id" "$n" "$kind"; continue
        fi
        # Not in the open-issue list. Either a pull request's conversation —
        # which arrives from this same endpoint with the PR's number, and PR
        # numbers share the issue sequence — or an issue that was closed while
        # the comment was held. Only the first is not an owner event.
        case "$(target_kind "$n")" in
          pull)  ;;                                   # not an issue event
          issue) pending_remove "$id"; emit_comment "$id" "$n" "$kind"; continue ;;
          *)     pending_failed "$id"; continue ;;    # ask again next poll
        esac
      fi
    fi
    pending_remove "$id"
  done <<<"$snapshot"

  # LOOP_WATCH_ONCE / LOOP_WATCH_INTERVAL exist so the fixture in
  # a test can step the poll loop against a
  # scratch state directory instead of racing a 30s sleep. Never set in the loop.
  [ -n "${LOOP_WATCH_ONCE:-}" ] && break
  sleep "$interval"
done
