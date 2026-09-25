# One checkout on `main` (#57)

Written 2026-09-25 against `origin/main` at `c735590`. Everything below was
read from the running system, the live catalog (read-only) or git; nothing is
written from memory. The switch itself waits for a window the owner picks.

`~/development/gmail-search` holds `.git` for every worktree and runs every
owner daemon, but it is on `prod/owner-partitions-20260916`, 32 commits behind
`main`. `/issue-loop` runs from a second worktree, `~/development/gmail-search-main`,
only because the old branch has no `.claude/commands/issue-loop.md`. After this
window both run from `~/development/gmail-search` on `main`, and fixes that
touch serve or the MCP tools reach the claude.ai connector (#52) the next time
the owner pulls and restarts.

**Downtime:** serve, MCP (the claude.ai connector), the owner's web app on :3000
and the background daemons are down for about 2 to 5 minutes: stopping,
`uv sync`, a `next build` (35 s on this machine) and serve's warm-up. The
invited service (`gms.…` via public-web and invited-api) stays up.

## The script

`scripts/one-checkout.sh` (`src/gmail_search/deploy/one_checkout.py`,
`checkout_checks.py`; tests in `tests/test_one_checkout.py`):

| Command | What it does |
| --- | --- |
| `check` | Read-only. The preconditions below, one line each, exit 0 only when all pass. |
| `switch [--dry-run]` | Runs `check` again, records each unit's pid, ports and readiness answer, then: save patch, back up the web build, stop units, detach the loop worktree from `main`, discard the four edits and check out `main`, `uv sync`, `next build`, start units in order, wait until each one answers as it did before. |
| `rollback [--dry-run]` | The reverse, from the recorded state. |

Run it from a **worktree of `main`**, never from the production checkout: the
old branch has no copy of the script. The wrapper's `uv run` syncs that
worktree's own venv; inside the production checkout it runs with `--no-sync`,
and `switch` and `rollback` refuse to run there at all. Each step is recorded in
`~/development/gmail-search/.runtime/one-checkout/state.json` (written aside and
renamed, so a crash never leaves half a record). Run it again after a failure
and it picks up at the step that failed. `--dry-run` prints every command,
`git fetch` included, and runs none; the checks it runs are read-only
(`git --no-optional-locks`, so not even the index is rewritten).

## What each unit loads from the checkout

`~/.config/systemd/user/`, read 2026-09-25. "Checkout" is `~/development/gmail-search`.

| Unit | State | Loads from the checkout | In the window |
| --- | --- | --- | --- |
| `gmail-search-serve` (:8090, loopback) | active since 09-16 | `.venv` (editable install of `src/`), `data/` (`--data-dir`, `serve.log`) | stopped, restarted |
| `gmail-search-mcp` (:7878, the claude.ai connector through `cloudflared-gmail-mcp`) | active since 09-16 | `.venv`, `src/`, `data/mcp.log` | stopped, restarted |
| `gmail-search-supervise` | active since 09-20 | `.venv`, `src/`, `data/`. Spawns watch ×2, update ×2, reindex ×2, summarize ×2, crawl, reconcile, all in its cgroup with cwd = checkout | stopped (children with it), restarted |
| `gmail-search-web` (:3000, loopback) | active since 09-16 | `web/` source, `web/.next` (built **2026-09-14**, older than the branch), `web/.env.local`, `web/node_modules` | stopped, rebuilt, restarted |
| `gmail-search-serve-watchdog.timer` → `.service` | every 5 min | `scripts/serve_watchdog.sh` (same on both branches) | timer stopped for the window |
| `gmail-search-logrotate.timer` → `.service` | daily | `deploy/logrotate/gmail-search.conf` (same on both branches) | untouched |
| `gmail-search-invited-api` | active | `.venv/bin/python` only; code from `~/.local/share/gmail-search/invited-current` via `PYTHONPATH` | **untouched**: must keep its pid |
| `gmail-search-public-web` (:3001) | active | `web/node_modules/.bin/next` only; cwd and build from `public-current/web` | **untouched**: must keep its pid |
| `gmail-search-public-api` | inactive (`Conflicts=` invited-api) | `.venv` | untouched |
| `gmail-search.service` (watch) | disabled, inactive (supervise runs watch) | `.venv` | untouched |
| `cloudflared-gmail-mcp`, `cloudflared-gmail-search-web`, `gmail-worker-gateway-tunnel` | active | nothing | untouched (the MCP tunnel answers 502 while mcp is down) |
| `gmail-search-worker-provision` | failed (transient) | `worktrees/full-agents-20260915/data/production-worker/boot.sh` | #27's |

`invited-api` and `public-web` are why the script refuses any runtime
dependency change or web lockfile change: they share the checkout's `.venv` and
`web/node_modules`, and the switch does not restart them.

## Old branch against `main`, for what loads at startup

**Commits on the old branch that `main` lacks** (`git log origin/main..prod/owner-partitions-20260916`):

| Commit | Status |
| --- | --- |
| `453e1f7` fix(ci): let CI use the database it already starts | Superseded: #46 re-landed it on `main` differently (CI DSN on a disposable dbname and port, perf conftest profile). Nothing to carry. |
| `f2fc858` fix(crawl): reclaim the Playwright driver when startup itself fails | **Not on `main`.** The crawl daemon's fd-leak fix (`url_fetcher.py`, `cli.py`, `tests/test_crawler_session_leak.py`). Switching without it brings back the leak that wedged crawl for four days. **#58** ports it; it is a precondition. |

The script refuses until each is acknowledged: `--allow-unmerged 453e1f7,f2fc858`.
Pass `f2fc858` only once #58 has landed. A port lands as a new commit, so
git cannot tell that it is the same change.

**Python dependencies.** Runtime closure identical (computed from both
`uv.lock` files). `main` adds dev-only `execnet 2.1.2` and `pytest-xdist 3.8.0`.
The production venv matches the old lock exactly (`uv pip list` compared with
`uv export`), so `uv sync --locked --extra dev` installs those two and nothing
else.

**Startup DDL: the question that could block the window.** Every daemon runs
`pg_schema.sql` against the live database at boot (`store/db.py:_init_db_pg`),
on both branches. Here is what `main`'s copy (blob `d3cef4a`) runs that the old
branch's does not, checked against the live catalog read-only on 2026-09-25:

| `main` runs at boot | Live database today | Effect of starting `main` |
| --- | --- | --- |
| `CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages (user_id, thread_id, date, id)` | exists | none |
| drop and create the `tenant_isolation` policies `TO gmail_search_reader, gmail_analyst` | all 18 already have exactly those roles and the same `USING`/`WITH CHECK` | the same drop and create every restart already runs, same result |
| database, schema and default-privilege grants only when missing | present | fewer writes than today |
| two-key advisory lock per schema | n/a | all owner units restart together, so old and new code never init at the same time |

The read-only query:

```sql
SET default_transaction_read_only = on;
SELECT count(*) FROM pg_indexes WHERE indexname = 'idx_messages_thread';               -- 1
SELECT roles::text, cmd, qual, with_check, count(*) FROM pg_policies
 WHERE policyname = 'tenant_isolation' GROUP BY 1, 2, 3, 4;
-- 17 × {gmail_analyst,gmail_search_reader} ALL (user_id = current_setting('app.user_id'::text, true)) (same)
--  1 × conversation_messages: same roles, the EXISTS (… conversations c …) policy
```

**Flag for the owner: starting any owner unit writes to the live database, on
either branch.** `_init_db_pg` runs the whole schema in one transaction and
commits: the `IF NOT EXISTS` statements are no-ops, but every restart that
gets its locks within `lock_timeout` (3 s; otherwise it skips the whole
transaction) drops and re-creates the `tenant_isolation` policies and re-issues
the table `GRANT`s.
That happens today on every restart of serve, mcp, supervise or any supervise
child, and the switch cannot restart the daemons without it. What `main` adds
is **no new object and no different definition**: every statement in `main`'s
copy leaves the catalog as it is now. If the owner's "no writes at startup"
rule covers these re-issued statements, the window needs that decision first
(precondition 7). The script pins the reviewed file: if `main`'s
`pg_schema.sql` is not blob `d3cef4a` on the day, it refuses and names the new
blob. Review what changed, repeat the query, then pass `--schema-reviewed <blob>`.

**The uncommitted `pg_schema.sql` edit keeps production correct today.** Its
two hunks (the index and the scoped policies) are what every daemon restart
applies now. The committed old-branch file creates the policies `TO PUBLIC`,
and its own comment says that costs about 950 ms per BM25 query for the invited
search role instead of about 80 ms. So the discard happens only inside the
switch, with every owner unit stopped, in the same step as the checkout. **Never
discard it as a separate earlier step:** a supervise child restarting in
between would un-scope the policies.

**All four uncommitted files** are older versions of what `main` has.
`probe_bm25_deleted_statistics.py` is identical to `main`. `ci.yml`,
`pg_schema.sql` and `test_bm25_deleted_statistics.py` are subsets of `main`.
Discarding them loses nothing. The patch is kept anyway, as the owner decided.

**A behaviour change on the first pass:** `main`'s embed pipeline also reselects
image attachments with a pending retry (`embed_error` set, status NULL, #12).
The live database has 2 such rows, so this is negligible.

**Config.** `config.py` only changes `_deep_merge` to deep-copy (#26). There are
no new keys and no new environment variables. Every unit file the repo tracks
is identical on both branches, except `gmail-worker-clock-sync.service`, which
`main` adds for the worker VM and which never runs on this host.

**Web.** 24 files under `web/` differ. `package.json` and `package-lock.json` are
identical, so there is no `npm ci`: `node_modules` stays exactly as
public-web uses it. `next build` of `main` succeeds (35 s, measured in a
worktree).

## Preconditions

1. **#27 done**: the worker VM disk is out of `worktrees/full-agents-20260915`.
2. **#55 landed** (the owner's gate for discarding the four edits). **Done
   2026-09-25:** #55 landed as `a5b2f02`, and the orchestrator saved the four
   edits to
   `~/development/gmail-search/.runtime/stray-edits-prod-owner-partitions-20260925.patch`
   (mode 600) and discarded them. Verify, don't redo: `git -C
   ~/development/gmail-search status --short` shows only `?? .runtime/`, and
   `git -C ~/development/gmail-search apply --check <that patch>` succeeds.
   Hand the patch to `switch` with `--saved-patch`, so rollback re-applies it.
   **Until the switch, the running daemons have the policy-scoping edit in
   memory but not on disk:** any restart on the old branch un-scopes the
   policies (see "Startup DDL"). Switch soon, or put the edits back with
   `git -C ~/development/gmail-search apply <that patch>`.
3. **#58 landed**, or the owner decides in writing to leave `f2fc858` behind.
4. **No active user runs**, the landing lock free, no deploy running, the loop
   idle: no `/issue-loop` tick and no agent landing.
5. **No other session working in the checkout.** `check` lists the outermost
   processes whose cwd is the checkout. Close any that are not yours.
6. **A worktree of `main` with this change**, synced:
   `~/development/gmail-search-main` after this change lands.
7. **The owner accepts the boot-time schema run** described under "Startup
   DDL": the same policy drop and create and `GRANT`s that every daemon restart
   issues today, with no new object.

## The window

```sh
cd ~/development/gmail-search-main
git -C ~/development/gmail-search fetch origin
git pull --ff-only                      # this worktree is on main
uv sync --locked --extra dev            # this worktree's own venv, not production's

SP=~/development/gmail-search/.runtime/stray-edits-prod-owner-partitions-20260925.patch
scripts/one-checkout.sh check --allow-unmerged 453e1f7,f2fc858
scripts/one-checkout.sh switch --dry-run --allow-unmerged 453e1f7,f2fc858 --saved-patch $SP
scripts/one-checkout.sh switch --allow-unmerged 453e1f7,f2fc858 --saved-patch $SP
```

`--saved-patch` says the edits were already saved and discarded. `switch`
copies that patch into its own record after proving it applies to the clean
tree, and refuses if the tree still has uncommitted edits. Without the flag it
saves the tree's own diff, which is now empty, and a rollback would then start
the old code without the policy scoping.

`switch` exits 2, having changed nothing but a `git fetch`, when a
precondition fails or when the units are not healthy before it starts. It
judges "healthy after" as "answers like before", so before must be good: serve
200 on readiness, mcp 401, web 200, each on a port it is listening on, and
supervise with children. A failure after that names the step. Fix the cause and
run the same command again, or roll back. Right before it discards the four
edits it compares `git diff HEAD` with the saved patch, and stops, discarding
nothing, if anything changed after the patch was taken. Rollback re-applies the
patch only when it applies cleanly (or is already applied). Otherwise it stops
before starting any unit.

### What `switch` runs, in order

The same steps done by hand, if the script is not used. `$P` is `~/development/gmail-search`.

```sh
P=~/development/gmail-search; S=$P/.runtime/one-checkout; mkdir -p $S
# 1. already done (precondition 2): verify the saved edits still apply to the clean tree
SP=$P/.runtime/stray-edits-prod-owner-partitions-20260925.patch
git -C $P status --short          # only ?? .runtime/
git -C $P apply --check $SP
# 2. keep the running web build for rollback
cp -a $P/web/.next $S/web-next
# 3. stop: watchdog first so it cannot restart serve, then callers before callees
systemctl --user stop gmail-search-serve-watchdog.timer
systemctl --user stop gmail-search-supervise gmail-search-web gmail-search-mcp gmail-search-serve
# 4. main can be checked out in one worktree only
git -C ~/development/gmail-search-main switch --detach
# 5. move to main at the checked target (the four edits are already discarded)
git -C $P switch main && git -C $P merge --ff-only origin/main
# 6. the production venv: adds execnet and pytest-xdist only
(cd $P && env -u VIRTUAL_ENV -u UV_PROJECT_ENVIRONMENT uv sync --locked --extra dev)
# 7. the :3000 build, with the node the unit runs
(cd $P/web && PATH=$HOME/.nvm/versions/node/v24.12.0/bin:/usr/bin:/bin node_modules/.bin/next build)
# 8. start: backend first, supervise last, then the watchdog
systemctl --user start gmail-search-serve    # wait for readiness (below) before the next
systemctl --user start gmail-search-mcp gmail-search-web gmail-search-supervise
systemctl --user start gmail-search-serve-watchdog.timer
```

### Health checks, per unit

The script waits (serve up to 300 s, the others 120 s) until each unit is
active, listens on the same ports as before, and answers each one with the
status it gave before the switch. By hand:

| Unit | Check | Before the switch |
| --- | --- | --- |
| serve | `curl -s -o /dev/null -w '%{http_code}\n' 'http://127.0.0.1:8090/healthz?ready=1'` (includes the search canary) | 200 |
| mcp | `curl -s -o /dev/null -w '%{http_code}\n' -X POST http://127.0.0.1:7878/mcp` (up, OAuth gate on); then one search from claude.ai through the connector | 401 |
| web | `curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:3000/` | 200 |
| supervise | `pgrep -P "$(systemctl --user show -p MainPID --value gmail-search-supervise)" \| wc -l` | 10 children |
| invited-api, public-web | `systemctl --user show -p MainPID --value <unit>` unchanged | same pid |
| policies still scoped | the read-only `pg_policies` query above | 18 × `{gmail_analyst,gmail_search_reader}` |
| the tree | `git -C ~/development/gmail-search status --short` | only `?? .runtime/` |

Then watch `data/serve.log`, `data/mcp.log` and `data/supervise.log` for a few
minutes. Tracebacks, `init_db` errors and crash loops show up there.

## Rollback

```sh
cd ~/development/gmail-search-main      # or any worktree of main with this change
scripts/one-checkout.sh rollback --dry-run
scripts/one-checkout.sh rollback
```

It stops the owner units, switches back to `prod/owner-partitions-20260916`
(refusing if that branch moved), re-applies the saved patch (so the policy
scoping edit is back before anything starts), puts
`~/development/gmail-search-main` back on `main`, runs `uv sync --locked --extra dev`
(removes the two dev packages), restores the saved `web/.next`, starts the
units and runs the same health checks. It then archives the state files, so a
later `switch` starts fresh.

By hand: steps 3, then `git -C $P switch prod/owner-partitions-20260916 && git -C $P apply $SP`,
`git -C ~/development/gmail-search-main switch main`, the `uv sync`,
`rm -rf $P/web/.next && cp -a $S/web-next $P/web/.next`, then step 8.

## Afterwards

Once the switch has held for a day:

1. **Retire the loop worktree.** Check it is clean, then remove it (this also
   ends the rollback's ability to re-attach it, which it skips when it is gone):
   ```sh
   git -C ~/development/gmail-search-main status --short     # expect nothing
   git -C ~/development/gmail-search worktree remove ~/development/gmail-search-main
   ```
2. **Run the loop from the checkout**: `/loop /issue-loop` in a Claude Code
   session opened in `~/development/gmail-search`. Loop state already lives
   there (`lib.sh` resolves it from the git common dir), so nothing moves.
3. **Docs.** In `.claude/commands/issue-loop.md` ("Where it runs") and the
   README ("Working through GitHub issues"), drop the sentence about the
   `gmail-search-main` worktree. `lib.sh` needs no change.
4. **Memory.** The owner's note `issue-loop-and-deployer.md` says the loop runs
   "from a main worktree". Change it to `~/development/gmail-search`.
5. **The old branch.** Keep `prod/owner-partitions-20260916`, local and on
   origin, until #58 has landed and a week has passed. Rollback needs it.
6. **The saved patch** stays in `.runtime/one-checkout/` (untracked, mode 600).

## Keeping it current after the switch

The checkout does not move when a PR merges: `land.sh` only fetches, and no
deploy writes to it. So `main` changes to serve, MCP or supervise reach them
only when the owner pulls and restarts, which the loop is not authorized to do.
This is #52's gap, now about one checkout instead of a stale branch:

```sh
git -C ~/development/gmail-search pull --ff-only
# if pyproject.toml or uv.lock changed: (cd ~/development/gmail-search && uv sync --locked --extra dev)
# if web/ changed: stop gmail-search-web, next build (step 7), start it
systemctl --user restart gmail-search-serve gmail-search-mcp gmail-search-supervise
```

Never pull while those units keep running on the old code for long. The
editable install serves new files to any import that happens after the pull
(the 2026-07 deploy-skew incident).
