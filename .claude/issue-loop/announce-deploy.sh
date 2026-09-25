#!/usr/bin/env bash
# Tell the issue loop that a release went out.
#
# The deployer lives in the repository (scripts/deploy/) and knows nothing about GitHub labels, comment markers or the
# loop's ledger -- deliberately, because those are the loop's concerns and not a
# release's. This is the loop's half: it reads what the deploy recorded and
# comments, moves labels and updates the ledger.
#
# usage: announce-deploy.sh <release>
#        announce-deploy.sh            # the release the last deploy recorded
#
# It reads .runtime/deploy/<release>/state.json, which the deployer wrote
# (scripts/deploy.sh), finds the issues from the `Fixes #<n>` lines in the
# commits it shipped, and refuses a release whose postcheck did not pass: announcing a deployment that
# was not verified is worse than announcing nothing.
set -Eeuo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib.sh
. "$here/lib.sh"

MAIN=$(loop_main_checkout "$here")
STATE=$(loop_state_dir "$MAIN")
LEDGER="$STATE/ledger.json"
REPO=$(loop_repo_slug "$MAIN")

RELEASE=${1:-}
if [ -z "$RELEASE" ]; then
  RELEASE=$(jq -r .release "$MAIN/.runtime/deploy/last.json" 2>/dev/null) ||
    { loop_warn "no release given and no .runtime/deploy/last.json"; exit 2; }
fi
STATE_FILE="$MAIN/.runtime/deploy/$RELEASE/state.json"
[ -f "$STATE_FILE" ] || { loop_warn "no deploy state at $STATE_FILE"; exit 2; }

if [ "$(jq -r '.dryRun // false' "$STATE_FILE")" = true ]; then
  loop_warn "$RELEASE was a dry run: nothing was activated, so there is nothing to announce"
  exit 2
fi
if [ "$(jq -r '.postcheck.result // ""' "$STATE_FILE")" != deployed ]; then
  loop_warn "$RELEASE has no passing postcheck in $STATE_FILE"
  exit 2
fi

TARGET=$(jq -r .target "$STATE_FILE")
RUNNING=$(jq -r '.running // ""' "$STATE_FILE")
SOURCE_SHORT=${TARGET:0:7}
# The issues this release carries: every `Fixes #<n>` that land.sh wrote into a
# commit between the release it replaced and its target.
range=$TARGET
[ -z "$RUNNING" ] || [ "$RUNNING" = null ] || range="$RUNNING..$TARGET"
mapfile -t ISSUES < <(git -C "$MAIN" log --format=%B "$range" |
  sed -n 's/^Fixes #\([0-9][0-9]*\).*/\1/p' | sort -un)
stamp=$(date -Iseconds)
loop_say "announcing $RELEASE ($SOURCE_SHORT) on ${#ISSUES[@]} issues"

for n in "${ISSUES[@]}"; do
  [ -n "$n" ] || continue
  body=$(printf 'Deployed in %s (%s) at %s; postcheck passed.\n\n%s\n' \
    "$RELEASE" "$SOURCE_SHORT" "$stamp" "$LOOP_MARKER")
  loop_say "commenting on #$n"
  # comment.sh, never `gh issue comment` directly: the marker has to be in the
  # body as posted (#483).
  "$here/comment.sh" "$n" --body "$body" >/dev/null ||
    loop_warn "could not comment on #$n"
  gh issue edit "$n" --repo "$REPO" --remove-label loop:merged --add-label loop:deployed >/dev/null ||
    loop_warn "could not move labels on #$n"
  if [ -f "$LEDGER" ]; then
    tmp=$(mktemp "$STATE/.ledger.XXXXXX")
    jq --arg n "$n" --arg r "$RELEASE" \
      '.issues[$n] = ((.issues[$n] // {}) | .state="deployed" | .release=$r)' "$LEDGER" >"$tmp"
    mv "$tmp" "$LEDGER"
  fi
done

printf 'announced %s sha=%s issues=%s\n' "$RELEASE" "$SOURCE_SHORT" "${ISSUES[*]:-none}"
