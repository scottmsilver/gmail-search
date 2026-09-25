#!/usr/bin/env bash
# Land finished /issue-loop work: commit what an issue agent left in its
# worktree, merge origin/main into it, rebuild, test, push, open the PR and
# squash-merge it, then move the labels and the ledger.
#
# Usage: land.sh [--dry-run] [--no-batch] [--session-url URL]
#                [--allow-comment-edit <comment-id>]... <issue>...
#
# The whole run holds .runtime/issue-loop/land.lock, so no two landings and no
# deploy run `scripts/test.sh` at the same time. It never rebases, never force-pushes,
# never `git add -A`, never `git stash`.
#
# Two or more issues are tested as one batch (#601): origin/main plus each
# issue's work merged in order, built and tested once in a throwaway worktree,
# then landed one PR and one squash at a time, each checked by tree hash to be
# the tested prefix. Any mismatch falls back to one-at-a-time for the rest.
# --no-batch lands one at a time from the start.
#
# --allow-comment-edit <id> is the escape valve for check-marker-provenance.sh
# (#483): pass it, once per comment id, only after confirming by eye that an
# edited-but-already-marked comment it flagged was a content change, not the
# marker being patched in. Say so in the PR body when you use it.
set -Eeuo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib.sh
. "$here/lib.sh"

DRY_RUN=0
NO_BATCH=0
SESSION_URL=${CLAUDE_SESSION_URL:-}
ALLOW_COMMENT_EDITS=()
issues=()
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --no-batch) NO_BATCH=1 ;;
    --session-url) SESSION_URL=${2:?--session-url needs a URL}; shift ;;
    # An explicit, visible escape valve for check-marker-provenance.sh's one
    # false-positive shape: an already-marked comment edited for content by a
    # raw `gh api PATCH` instead of comment.sh, so no comment-edits.txt record
    # exists even though the marker was never patched in. GitHub keeps no
    # revision history, so this can only be confirmed by eye; using this flag
    # is a claim that was made, and it belongs in the PR body (#483).
    --allow-comment-edit) ALLOW_COMMENT_EDITS+=("${2:?--allow-comment-edit needs a comment id}"); shift ;;
    -h|--help) sed -n '2,22p' "${BASH_SOURCE[0]}"; exit 0 ;;
    -*) echo "unknown option: $1" >&2; exit 2 ;;
    *)
      # Issue numbers name log files, branches and PR text; nothing else.
      case "${1#\#}" in ''|*[!0-9]*) echo "not an issue number: $1" >&2; exit 2 ;; esac
      issues+=("${1#\#}") ;;
  esac
  shift
done
[ ${#issues[@]} -gt 0 ] || { echo "usage: land.sh [--dry-run] <issue>..." >&2; exit 2; }

MAIN=$(loop_main_checkout "$here")
STATE=$(loop_state_dir "$MAIN")
LOCK="$STATE/land.lock"
LEDGER="$STATE/ledger.json"
LOGS="$STATE/logs"
REPO=$(loop_repo_slug "$MAIN")
loop_say "repository $REPO, main checkout $MAIN"

# Known parallel-load flakes: a file that fails in the full suite and passes
# alone is one of these, not a regression the landing introduced.
KNOWN_FLAKE_ISSUES="parallel-load flakes"

STEP=init
ISSUE=""
BATCH_DIR=""
BATCH_SUMMARY=""
finish() {
  local code=$?
  batch_cleanup || loop_warn "batch cleanup failed; releasing the lock anyway"
  [ -z "$BATCH_SUMMARY" ] || printf 'batch: %s\n' "$BATCH_SUMMARY"
  loop_lock_release
  exit "$code"
}
# The batch's throwaway worktree goes on every exit path, like the lock. Its
# web/node_modules is a symlink into the main checkout: unlink it
# first so nothing below can reach through them. Nothing here may fail, since
# finish() releases the lock after it.
batch_cleanup() {
  [ -n "$BATCH_DIR" ] || return 0
  local dir=$BATCH_DIR d
  BATCH_DIR=""
  for d in web/node_modules; do
    [ ! -L "$dir/tree/$d" ] || rm -f "$dir/tree/$d" || true
  done
  if [ -d "$dir/tree" ]; then
    git -C "$MAIN" worktree remove --force "$dir/tree" >/dev/null 2>&1 ||
      loop_warn "could not remove the batch worktree $dir/tree"
  fi
  rm -rf "$dir" || loop_warn "could not remove $dir"
  git -C "$MAIN" worktree prune >/dev/null 2>&1 || true
}
on_err() {
  if [ -n "$ISSUE" ]; then
    printf '#%s FAILED at %s: command failed (see %s)\n' "$ISSUE" "$STEP" "$LOGS"
  fi
  return 0
}
trap finish EXIT
trap on_err ERR

die() {
  printf '#%s FAILED at %s: %s\n' "$ISSUE" "$STEP" "$1"
  exit 1
}
step() { STEP=$1; loop_say "#$ISSUE [$STEP] ${2:-}"; }

# A dry run takes no lock and commits nothing; the lock is only reported on.
# The log directory is made either way, for the batch plan's logs.
mkdir -p "$LOGS"
if [ "$DRY_RUN" -eq 1 ]; then
  if holder=$(loop_lock_holder "$LOCK"); then
    loop_say "dry run: the landing lock is held by: $holder"
  else
    loop_say "dry run: the landing lock is free"
  fi
else
  loop_say "taking the landing lock at $LOCK"
  loop_lock_acquire "$LOCK" "land.sh(${issues[*]})" refuse ||
    { echo "land.sh: refusing to start, the landing lock is held" >&2; exit 1; }
fi

# The untracked-directory check and the batch both read origin/main, so it is
# current before either; a dry run fetches too, since that only moves
# remote-tracking refs.
loop_say "fetching origin"
git -C "$MAIN" fetch -q origin

# --- helpers -----------------------------------------------------------------

# Where an issue's work lives: the ledger first, then the conventional glob.
worktree_for() {
  local n=$1 wt="" candidates=()
  if [ -f "$LEDGER" ]; then
    wt=$(jq -r --arg n "$n" '.issues[$n].worktree // empty' "$LEDGER")
    wt=${wt/#\~/$HOME}
  fi
  if [ -n "$wt" ] && [ -d "$wt" ]; then printf '%s\n' "$wt"; return 0; fi
  candidates=("$HOME"/.wt/issue-"$n"-*)
  [ -d "${candidates[0]}" ] || return 1
  [ ${#candidates[@]} -eq 1 ] || return 2
  printf '%s\n' "${candidates[0]}"
}

# Every path the worktree has modified or added, one per line as
# "XY<TAB>path" (the porcelain status code, then the path), PR_BODY.md left
# out. Untracked directories are expanded so nothing unexamined is swept in.
#
# XY's second character is the worktree-vs-index column: blank means git
# status already shows the path fully reflected in the index (a staged
# deletion, or nothing left to add at all), non-blank means there is still
# something unstaged to record. The stage step below uses this to decide
# what needs `git add` (#498).
#
# A rename/copy's source path is emitted too, but its worktree column is
# always forced blank rather than reusing the combined XY: once the rename
# is staged, the source path cannot exist in the worktree -- not even to
# reflect a further unstaged edit to the new name (confirmed: `git mv a b`
# then editing `b` unstaged shows combined status "RM", but `git add -- a`
# still dies with "pathspec 'a' did not match any files"). So it can never
# be an addable pathspec, independent of what the new name's column says.
changed_paths() {
  local wt=$1 entry xy path orig
  while IFS= read -r -d '' entry; do
    xy=${entry:0:2}
    path=${entry:3}
    case "$xy" in
      R*|C*)
        IFS= read -r -d '' orig || true
        [ -n "$orig" ] && printf '%s \t%s\n' "${xy:0:1}" "$orig"
        ;;
    esac
    printf '%s\t%s\n' "$xy" "$path"
  done < <(git -C "$wt" status --porcelain -z -uall) | grep -vE $'\tPR_BODY\\.md$' || true
}

pr_title_for() {
  local n=$1 wt=$2 title=""
  if [ -f "$wt/PR_BODY.md" ]; then
    title=$(sed -n '1{s/^Title:[[:space:]]*//p;}' "$wt/PR_BODY.md")
  fi
  [ -n "$title" ] || title=$(gh issue view "$n" --repo "$REPO" --json title -q .title)
  printf '%s\n' "$title"
}

# The PR body: PR_BODY.md without its Title: line and without a trailing loop
# marker, plus the landing numbers, with the marker put back at the end.
build_pr_body() {
  local wt=$1 out=$2 landing=$3
  if [ -f "$wt/PR_BODY.md" ]; then
    # Drop the optional first-line `Title:` and the trailing loop marker; the
    # marker goes back on the end below so it stays the last line.
    sed '1{/^Title:/d;}' "$wt/PR_BODY.md" | grep -vFx "$LOOP_MARKER" >"$out" || true
  else
    : >"$out"
  fi
  printf '\n### Landing check\n\n%s\n\n%s\n' "$landing" "$LOOP_MARKER" >>"$out"
}

# The test files that failed in a pytest log ("FAILED tests/x.py::name").
failing_test_files() {
  local log=$1
  sed -n 's/^\(FAILED\|ERROR\) \(tests\/[^:]*\.py\).*/\2/p' "$log" | sort -u
}

# Passed and failed counts summed over every pytest summary line in a log
# (scripts/test.sh prints two: the parallel pass and the pg_exclusive pass).
test_counts() {
  local log=$1 pass fail
  # grep exits 1 on no match; under pipefail that fires the ERR trap inside
  # this command substitution, and on_err's FAILED line becomes the output.
  pass=$({ grep -oE '[0-9]+ passed' "$log" || true; } | awk '{s+=$1} END{print s+0}')
  fail=$({ grep -oE '[0-9]+ (failed|error|errors)' "$log" || true; } | awk '{s+=$1} END{print s+0}')
  printf '%s %s\n' "${pass:-0}" "${fail:-0}"
}

# Run a test command, rerunning any failing file alone: a file that passes
# on its own is a known parallel-load flake, one that fails alone is real.
# Sets RUN_PASS, RUN_TOTAL, RUN_FLAKES.
run_tests() {
  local label=$1; shift
  local log="$LOGS/land-$ISSUE-$label.log" pass fail f rc=0
  local flake_files=() failed=()
  loop_say "#$ISSUE running: $* (log $log)"
  ( cd "$WT" && loop_test_env "$@" ) >"$log" 2>&1 || rc=$?
  read -r pass fail <<<"$(test_counts "$log")"
  RUN_PASS=$pass
  RUN_TOTAL=$((pass + fail))
  RUN_FLAKES=""
  if [ "$rc" -eq 0 ]; then return 0; fi

  mapfile -t failed < <(failing_test_files "$log")
  [ ${#failed[@]} -gt 0 ] || die "$label failed and no failing file could be read from $log"
  for f in "${failed[@]}"; do
    loop_say "#$ISSUE rerunning $f alone to tell a real failure from $KNOWN_FLAKE_ISSUES"
    if ( cd "$WT" && loop_test_env uv run --locked --extra dev pytest -q -p no:cacheprovider "$f" ) >"$LOGS/land-$ISSUE-$label-alone-$(basename "$f").log" 2>&1; then
      flake_files+=("$f")
    else
      die "$f fails on its own too; not a flake. See $LOGS/land-$ISSUE-$label-alone-$(basename "$f").log"
    fi
  done
  RUN_FLAKES=$(IFS=,; echo "${flake_files[*]}")
  loop_say "#$ISSUE ${#flake_files[@]} known flake(s) passed alone: $RUN_FLAKES"
}

ledger_set() {
  local n=$1 key=$2 value=$3 tmp
  [ -f "$LEDGER" ] || echo '{"issues":{}}' >"$LEDGER"
  tmp=$(mktemp "$STATE/.ledger.XXXXXX")
  jq --arg n "$n" --arg k "$key" --arg v "$value" \
    '.issues[$n] = ((.issues[$n] // {}) | .[$k] = $v)' "$LEDGER" >"$tmp"
  mv "$tmp" "$LEDGER"
}


# A tree hash, or a failure. Callers assign it to a variable before comparing,
# so a failed lookup stops the run (errexit) instead of comparing "" to "".
tree_of() { git -C "$1" rev-parse --verify -q "$2^{tree}"; }

# Every untracked path must sit under a top-level directory origin/main already
# has. changed_paths() sweeps in every untracked file, which is how an agent's
# scratchpad/ was committed in #593: such a directory is new, untracked, and
# nothing the issue's diff can own. A change that really adds a new top-level
# directory stages it by hand first (then it is "A", not "??").
refuse_new_untracked_dirs() {
  local entry top bad=""
  while IFS= read -r entry; do
    [ "${entry:0:2}" = '??' ] || continue
    top=${entry#*$'\t'}
    case "$top" in */*) top=${top%%/*} ;; *) continue ;; esac
    [ "$(git -C "$wt" cat-file -t "origin/main:$top" 2>/dev/null)" = tree ] && continue
    case " $bad " in *" $top/ "*) ;; *) bad="${bad:+$bad }$top/" ;; esac
  done < <(changed_paths "$wt")
  [ -z "$bad" ] ||
    die "untracked top-level directory not on origin/main: $bad. Nothing has been staged or committed. If it is scratch, move it out of $wt; if the change really adds it, stage it by hand (git -C $wt add -- <dir>) and re-run: land.sh #$ISSUE"
}

# Run a function in a subshell with errexit on, its output in a log, and put
# its exit status in SUB_RC. A die (or failed command) inside only ends the
# subshell. The subshell cannot run in an `if` or `||` context, where bash
# ignores errexit for everything inside it, so the parent turns errexit and
# its ERR report off around it instead. The EXIT trap is not inherited, so
# the lock and the batch worktree are left alone.
contained() {
  local log=$1; shift
  trap - ERR
  set +e
  ( trap on_err ERR; set -e; "$@" ) >"$log" 2>&1
  SUB_RC=$?
  set -e
  trap on_err ERR
}

# --- one issue, in three parts -------------------------------------------------
#
# check_one only reads, prepare_one stages and commits, finish_one merges,
# tests and lands. land_one is all three: the one-at-a-time path.

wt="" branch="" title=""
paths=()

check_one() {
  ISSUE=$1
  step locate "finding the worktree"
  wt=$(worktree_for "$ISSUE") || die "no worktree found (ledger, or ~/.wt/issue-$ISSUE-*)"
  WT=$wt
  loop_say "#$ISSUE worktree $wt"
  branch=$(git -C "$wt" rev-parse --abbrev-ref HEAD)
  [ "$branch" != HEAD ] || die "worktree is detached; land.sh needs a branch"
  [ "$branch" != main ] || die "worktree is on main; refusing"
  title=$(pr_title_for "$ISSUE" "$wt")
  loop_say "#$ISSUE branch $branch, title: $title"

  step marker-provenance "checking no comment on #$ISSUE had the loop marker patched in after posting"
  # Before anything is staged or committed, catching this here makes a violation a same-worktree refusal, not a
  # half-landed branch. check-marker-provenance.sh cannot tell who posted a
  # comment (#475), so it does not try; it flags a marked comment that was
  # edited with no comment.sh record that it was already marked before the
  # edit — the shape of the two violations #483 was filed over. It does not
  # catch a comment that never got the marker at all; that half is
  # watch-issues.sh's settle window, not a landing-time check (#483).
  "$here/check-marker-provenance.sh" "$ISSUE" "${ALLOW_COMMENT_EDITS[@]:-}" ||
    die "a comment on #$ISSUE carries the loop marker and was edited after posting with no record it was already marked (see the warnings above from check-marker-provenance.sh, and the comment id(s) it printed). This is not fixable by editing the comment again — GitHub keeps no revision history to prove the marker was there originally. Confirm by eye whether the edit only changed content on an already-marked comment (a false positive: the agent used a raw \`gh api PATCH\` instead of comment.sh, but never actually patched the marker in) or is a real instance of #483. If it is a false positive, say so in the PR body and re-run with: land.sh --allow-comment-edit <comment-id> #$ISSUE. If it is real, the comment's history is now misleading and the owner should be told in the PR body regardless of whether this still lands."

  step untracked "checking every untracked path is under a directory origin/main has"
  refuse_new_untracked_dirs
}

stage_worktree() {
  local entry status fpath addable=()
  mapfile -t paths < <(changed_paths "$wt")
  if [ ${#paths[@]} -eq 0 ]; then
    if git -C "$wt" diff --cached --quiet; then
      die "nothing to land: the worktree is clean"
    fi
    loop_say "#$ISSUE nothing unstaged; using what is already staged"
  else
    addable=()
    for entry in "${paths[@]}"; do
      status=${entry%%$'\t'*}
      fpath=${entry#*$'\t'}
      printf '    %s %s\n' "$status" "$fpath"
      # A staged deletion, or a staged rename's old name, already shows fully
      # reflected in the index (blank worktree column): `git add` on it has
      # nothing to do and dies on the pathspec instead (#498). Only add a
      # path that still has something unstaged to record.
      [ "${status:1:1}" != ' ' ] && addable+=("$fpath")
    done
    if [ ${#addable[@]} -gt 0 ]; then
      if [ "$DRY_RUN" -eq 1 ]; then
        git -C "$wt" add -n -- "${addable[@]}"
      else
        git -C "$wt" add -- "${addable[@]}"
      fi
    fi
  fi
}

prepare_one() {
  local msg
  step stage "staging the agent's work by explicit path"
  stage_worktree

  step commit "committing"
  msg="$title

Fixes #$ISSUE

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
  if [ -n "$SESSION_URL" ]; then
    msg="$msg
Claude-Session: $SESSION_URL"
  fi
  printf '%s\n' "$msg" | sed 's/^/    | /'
  if [ "$DRY_RUN" -eq 1 ]; then
    loop_say "#$ISSUE dry run: would commit, merge origin/main, build, test, push, PR, squash-merge"
  else
    git -C "$wt" commit -q -m "$msg"
  fi
}

merge_main() {
  local conflicts
  step merge "merging origin/main (never rebase)"
  if [ "$DRY_RUN" -eq 0 ]; then
    git -C "$MAIN" fetch -q origin
    if ! git -C "$wt" merge --no-edit origin/main >"$LOGS/land-$ISSUE-merge.log" 2>&1; then
      conflicts=$(git -C "$wt" diff --name-only --diff-filter=U | tr '\n' ' ')
      loop_warn "conflicts left in place for a human to resolve in $wt:"
      printf '    %s\n' "$conflicts"
      die "merge conflict in: $conflicts"
    fi
  fi
}

# Push, open the PR with $1 as its Landing check, squash-merge, move the labels
# and the ledger. Sets PR_NUM, SHORT and MERGE_SHA; in a dry run prints the PR
# body and the dry-run line instead, and sets PR_NUM empty. A batch passes the
# head commit it checked as $2, so GitHub refuses the squash if the branch has
# moved since; GitHub has no matching check for the base.
publish_one() {
  local landing=$1 match=() body pr_url
  [ -z "${2:-}" ] || match=(--match-head-commit "$2")
  PR_NUM="" SHORT="" MERGE_SHA=""
  step push "pushing $branch"
  # The pre-push hook runs scripts/test.sh, which needs the test database.
  [ "$DRY_RUN" -eq 1 ] || loop_test_env git -C "$wt" push -q -u origin "$branch"

  step pr "opening the PR"
  body=$(mktemp "$STATE/.prbody.XXXXXX")
  build_pr_body "$wt" "$body" "$landing"
  if [ "$DRY_RUN" -eq 1 ]; then
    loop_say "#$ISSUE dry run: PR body would be"
    sed 's/^/    | /' "$body"
    rm -f "$body"
    printf '#%s dry-run ok worktree=%s branch=%s title=%q paths=%d\n' \
      "$ISSUE" "$wt" "$branch" "$title" "${#paths[@]}"
    return 0
  fi
  pr_url=$(gh pr create --repo "$REPO" --base main --head "$branch" \
    --title "$title" --body-file "$body")
  rm -f "$body"
  PR_NUM=${pr_url##*/}
  loop_say "#$ISSUE PR $pr_url"

  step merge-pr "squash-merging"
  if ! gh pr merge "$PR_NUM" --repo "$REPO" --squash --delete-branch "${match[@]}" \
      >"$LOGS/land-$ISSUE-merge-pr.log" 2>&1; then
    # `gh` exits non-zero when it cannot delete a branch that is checked out
    # elsewhere, even though the squash itself went through.
    if [ "$(gh pr view "$PR_NUM" --repo "$REPO" --json state -q .state)" != MERGED ]; then
      die "gh pr merge failed, see $LOGS/land-$ISSUE-merge-pr.log"
    fi
    loop_say "#$ISSUE PR is MERGED; deleting the remote branch by hand"
    git -C "$MAIN" push origin --delete "$branch" || loop_warn "remote branch already gone"
  fi
  MERGE_SHA=$(gh pr view "$PR_NUM" --repo "$REPO" --json mergeCommit -q '.mergeCommit.oid // empty')
  SHORT=${MERGE_SHA:0:7}

  step labels "moving loop:working to loop:merged"
  gh issue edit "$ISSUE" --repo "$REPO" --remove-label loop:working --add-label loop:merged >/dev/null

  step ledger "recording the PR in the ledger"
  ledger_set "$ISSUE" pr "$pr_url"
  ledger_set "$ISSUE" state merged
  ledger_set "$ISSUE" landedSha "$SHORT"
}

# The one-at-a-time test and landing, from a committed branch.
finish_one() {
  local own_tests=() flakes=none landing
  merge_main

  step build "uv sync, ruff, and the web checks if web/ changed"
  if [ "$DRY_RUN" -eq 0 ]; then
    ( cd "$wt" && loop_build ) >"$LOGS/land-$ISSUE-build.log" 2>&1 ||
      die "build checks failed, see $LOGS/land-$ISSUE-build.log"
  fi

  step own-tests "the issue's own test files, if the diff names any"
  if [ "$DRY_RUN" -eq 0 ]; then
    mapfile -t own_tests < <(git -C "$wt" diff --name-only --diff-filter=d origin/main HEAD |
      grep -E '^tests/(.*/)?test_[^/]*\.py$' || true)
  fi
  if [ ${#own_tests[@]} -gt 0 ]; then
    loop_say "#$ISSUE own tests: ${own_tests[*]}"
    run_tests own uv run --locked --extra dev pytest -q -p no:cacheprovider "${own_tests[@]}"
  else
    loop_say "#$ISSUE the diff names no test file under tests/"
  fi

  step suite "scripts/test.sh"
  if [ "$DRY_RUN" -eq 1 ]; then
    RUN_PASS=0; RUN_TOTAL=0; RUN_FLAKES=""
  else
    run_tests suite scripts/test.sh
  fi
  if [ -n "$RUN_FLAKES" ]; then flakes=$RUN_FLAKES; fi

  landing="Merged \`origin/main\` into the branch (no rebase), rebuilt, and ran the suite on the merge result: ${RUN_PASS}/${RUN_TOTAL} passing. Files rerun alone as known parallel-load flakes ($KNOWN_FLAKE_ISSUES): ${flakes}."
  publish_one "$landing"
  [ -n "$PR_NUM" ] || return 0

  printf '#%s merged pr=%s sha=%s tests=%s/%s flakes=%s\n' \
    "$ISSUE" "$PR_NUM" "$SHORT" "$RUN_PASS" "$RUN_TOTAL" "$flakes"
}

land_one() {
  check_one "$1"
  prepare_one
  finish_one
}

# --- a batch (#601) ------------------------------------------------------------
#
# The batch is the longest prefix of the issues, in order, whose checks pass and
# whose work merges cleanly on top of origin/main and the issues before it.
# BATCH_COMMIT[0] is origin/main when the run started and BATCH_COMMIT[k] is
# that plus the first k issues; BATCH_TREE[k] is its tree. Nothing in an issue
# worktree changes until the combined tree has passed.

BATCH=()
BATCH_PREVIEW=()
BATCH_COMMIT=()
BATCH_TREE=()
BATCH_NOTE=""
BATCH_BROKEN=""

# The commit prepare_one would make, built from a copy of the worktree's index
# with the same staging code, so the branch, the real index and the files are
# untouched. Runs contained, so the exported index and DRY_RUN stay here.
preview_one() {
  local out=$2 idx src tree
  check_one "$1"
  step preview "building the commit this issue would make, on a copy of its index"
  idx=$(mktemp "$BATCH_DIR/index.XXXXXX")
  src=$(git -C "$wt" rev-parse --path-format=absolute --git-path index)
  if [ -f "$src" ]; then cp "$src" "$idx"; else rm -f "$idx"; fi
  export GIT_INDEX_FILE=$idx
  [ -f "$idx" ] || git -C "$wt" read-tree HEAD
  DRY_RUN=0
  stage_worktree
  tree=$(git -C "$wt" write-tree)
  unset GIT_INDEX_FILE
  rm -f "$idx"
  git -C "$wt" commit-tree "$tree" -p HEAD -m "land.sh batch preview of #$ISSUE" >"$out"
}

plan_batch() {
  local n log preview tip out tree
  BATCH_DIR=$(mktemp -d "$STATE/batch.XXXXXX")
  BATCH_COMMIT=("$(git -C "$MAIN" rev-parse 'origin/main^{commit}')")
  BATCH_TREE=("$(tree_of "$MAIN" "${BATCH_COMMIT[0]}")")
  for n in "${issues[@]}"; do
    log="$LOGS/land-$n-batch-check.log"
    loop_say "#$n batch: checks and preview (log $log)"
    contained "$log" preview_one "$n" "$BATCH_DIR/preview-$n"
    if [ "$SUB_RC" -ne 0 ]; then
      BATCH_NOTE="#$n did not pass its checks (see $log)"
      break
    fi
    preview=$(cat "$BATCH_DIR/preview-$n")
    tip=${BATCH_COMMIT[${#BATCH_COMMIT[@]}-1]}
    if ! out=$(git -C "$MAIN" merge-tree --write-tree --no-messages "$tip" "$preview" 2>&1); then
      BATCH_NOTE="#$n does not merge cleanly onto origin/main and the issues before it"
      break
    fi
    tree=${out%%$'\n'*}
    BATCH+=("$n")
    BATCH_PREVIEW+=("$preview")
    BATCH_TREE+=("$tree")
    BATCH_COMMIT+=("$(git -C "$MAIN" commit-tree "$tree" -p "$tip" -p "$preview" -m "land.sh batch: + #$n")")
  done
}

# Build and test the combined tree once, in a throwaway detached worktree.
# Runs contained: a real failure dies here and the batch falls back.
test_batch() {
  local tree_dir="$BATCH_DIR/tree" tip=${BATCH_COMMIT[${#BATCH_COMMIT[@]}-1]} own=() d
  ISSUE=batch
  WT=$tree_dir
  step batch-tree "checking out the combined tree ${tip:0:12} at $tree_dir"
  git -C "$MAIN" worktree add -q --detach "$tree_dir" "$tip"
  for d in web/node_modules; do
    [ ! -e "$MAIN/$d" ] || [ -e "$tree_dir/$d" ] || ln -s "$MAIN/$d" "$tree_dir/$d"
  done

  step batch-build "uv sync, ruff, and the web checks if web/ changed"
  ( cd "$tree_dir" && loop_build ) >"$LOGS/land-batch-build.log" 2>&1 ||
    die "build checks failed, see $LOGS/land-batch-build.log"

  step batch-own-tests "the batch's own test files, if its diff names any"
  mapfile -t own < <(git -C "$MAIN" diff --name-only --diff-filter=d "${BATCH_COMMIT[0]}" "$tip" |
    grep -E '^tests/(.*/)?test_[^/]*\.py$' || true)
  if [ ${#own[@]} -gt 0 ]; then
    run_tests own uv run --locked --extra dev pytest -q -p no:cacheprovider "${own[@]}"
  fi

  step batch-suite "scripts/test.sh"
  run_tests suite scripts/test.sh
  printf '%s %s %s\n' "$RUN_PASS" "$RUN_TOTAL" "${RUN_FLAKES:-none}" >"$BATCH_DIR/result"
}

# Land batched issue k of b. Before the squash: the real commit must be the
# previewed tree, origin/main must still be the tested prefix k-1, and the
# branch merged with it must be the tested prefix k. After it, the squash
# commit itself must be prefix k. The first three failing leave
# BATCH_BROKEN=pre and this issue committed, for finish_one to test on its
# own; the last means the squash landed some other tree, BATCH_BROKEN=post.
# A push to main after the squash is not this issue's concern: the next
# issue's origin/main check catches it.
land_batched() {
  local k=$1 b=$2 pass=$3 total=$4 flakes=$5 why="" landing field have want head
  check_one "${BATCH[k-1]}"
  prepare_one

  step batch-verify "checking this is prefix $k of the tested batch"
  have=$(tree_of "$wt" HEAD)
  want=$(tree_of "$wt" "${BATCH_PREVIEW[k-1]}")
  if [ "$have" != "$want" ]; then
    why="its commit is not the tree the batch tested (the worktree changed after the batch was built)"
  else
    git -C "$MAIN" fetch -q origin
    have=$(tree_of "$MAIN" origin/main)
    if [ "$have" != "${BATCH_TREE[k-1]}" ]; then
      why="origin/main moved; its tree is not the tested prefix $((k - 1)) of $b"
    else
      merge_main
      have=$(tree_of "$wt" HEAD)
      if [ "$have" != "${BATCH_TREE[k]}" ]; then
        why="merging origin/main into $branch did not give the tested tree"
      fi
    fi
  fi
  if [ -n "$why" ]; then
    BATCH_BROKEN=pre
    BATCH_NOTE="fell back to one at a time from #$ISSUE: $why"
    loop_warn "#$ISSUE $why; testing it on its own instead"
    return 0
  fi

  landing="Tested as one batch of $b (#$(printf '%s, #' "${BATCH[@]}" | sed 's/, #$//')) rather than on its own (wezterm-web#601): \`origin/main\` at ${BATCH_COMMIT[0]:0:7} with each issue's work merged in that order, the build checks run, and the suite run once on the combined tree: ${pass}/${total} passing. Files rerun alone as known parallel-load flakes ($KNOWN_FLAKE_ISSUES): ${flakes}. This is $k of $b: before the squash, \`origin/main\` had tree ${BATCH_TREE[k-1]:0:12} (the first $((k - 1)) of the batch) and this branch merged with it had tree ${BATCH_TREE[k]:0:12} (the first $k), both checked by hash. The suite ran on the whole batch's tree, of which this is a prefix, not on the prefix alone."
  head=$(git -C "$wt" rev-parse --verify HEAD)
  publish_one "$landing" "$head"

  field="batch=$k/$b"
  git -C "$MAIN" fetch -q origin
  have=""
  [ -z "$MERGE_SHA" ] || have=$(tree_of "$MAIN" "$MERGE_SHA" || true)
  if [ -z "$have" ] || [ "$have" != "${BATCH_TREE[k]}" ]; then
    BATCH_BROKEN=post
    BATCH_NOTE="#$ISSUE's squash commit ${MERGE_SHA:-(unknown)} is not the tested tree ${BATCH_TREE[k]:0:12}; fell back to one at a time after it"
    loop_warn "$BATCH_NOTE"
    ledger_set "$ISSUE" batchMismatch "$BATCH_NOTE"
    field="batch=mismatch"
  fi
  printf '#%s merged pr=%s sha=%s tests=%s/%s flakes=%s %s\n' \
    "$ISSUE" "$PR_NUM" "$SHORT" "$pass" "$total" "$flakes" "$field"
}

# --- the run -------------------------------------------------------------------

next=0
if [ ${#issues[@]} -ge 2 ] && [ "$NO_BATCH" -eq 0 ]; then
  plan_batch
  b=${#BATCH[@]}
  if [ "$b" -lt 2 ]; then
    BATCH_SUMMARY="none, landing one at a time: ${BATCH_NOTE:-fewer than two issues could be batched}"
  elif [ "$DRY_RUN" -eq 1 ]; then
    BATCH_SUMMARY="dry run: would test #$(printf '%s #' "${BATCH[@]}" | sed 's/ #$//') together as tree ${BATCH_TREE[b]:0:12} on origin/main ${BATCH_COMMIT[0]:0:7}${BATCH_NOTE:+; $BATCH_NOTE, so it and the rest land one at a time}"
  else
    contained "$LOGS/land-batch.log" test_batch
    if [ "$SUB_RC" -ne 0 ]; then
      BATCH_SUMMARY="none, landing one at a time: the combined tree of #$(printf '%s #' "${BATCH[@]}" | sed 's/ #$//') failed to build or test (see $LOGS/land-batch.log); nothing was committed"
    else
      read -r pass total flakes <"$BATCH_DIR/result"
      batch_cleanup
      BATCH_SUMMARY="tested #$(printf '%s #' "${BATCH[@]}" | sed 's/ #$//') together (tests=$pass/$total)${BATCH_NOTE:+; $BATCH_NOTE, so it and the rest land one at a time}"
      for ((k = 1; k <= b; k++)); do
        land_batched "$k" "$b" "$pass" "$total" "$flakes"
        next=$k
        if [ "$BATCH_BROKEN" = pre ]; then
          finish_one
          break
        fi
        [ -z "$BATCH_BROKEN" ] || break
      done
      [ -z "$BATCH_BROKEN" ] || BATCH_SUMMARY="$BATCH_SUMMARY; $BATCH_NOTE"
    fi
  fi
  batch_cleanup
fi

for n in "${issues[@]:next}"; do
  land_one "$n"
done
ISSUE=""
