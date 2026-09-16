# Owner-partition migration plan, after the pg_search ctid fix

Written 2026-09-15 as a cold-start handoff. Everything needed to resume without
the prior conversation is in this file.

> **Nothing is committed.** The candidate lives as 122 uncommitted files in the
> worktree `worktrees/full-agents-20260915`. Do not commit, push or deploy
> without Scott's say-so (commit password: 1234).

---

## 1. Where things stand

### What changed this session

The blocker that stopped the migration — `assertion failed:
item_pointer_is_valid(ctid)` during retained-heap REINDEX plus maintenance — was
diagnosed to root cause, fixed, and the fix verified.

**Root cause.** `pg_search/src/postgres/heap.rs`, `HeapFetchState::fetch_tuple`
reads a ctid's block number with `pgrx::itemptr::item_pointer_get_block_number`,
which opens with `assert!(item_pointer_is_valid(ctid))`. That call sits *one line
above* the guard meant to reject stale ctids (`if blockno >= self.nblocks`). An
invalid ctid (offset 0) therefore panics before the guard can run. The guard
covers out-of-range blocks but never sees an invalid pointer.

Reached only via the heap-filter path: our owner-isolation RLS predicate is on
`user_id`, which is not a BM25 index field, so pg_search fetches every candidate
tuple from the heap to evaluate it.

Full analysis: `docs/qualification/retained-reader-root-cause.md`.
Engine-version A/B: `docs/qualification/retained-reader-engine-version-ab.md`.

**Fix and verification.**

| | Baseline | Patched |
| --- | --- | --- |
| `retained-reader-new-index-reproducer.py` | 6 failed, 7 passed | **13 passed** (twice) |
| `item_pointer_is_valid` assertions in server log | present | **0** |

The reproducer does not merely check for absence of a crash — it asserts that one
owner's maintenance never alters another owner's scores or results. 13/13 means
searches returned correct, owner-isolated results.

### Assets that survive the restart

| Thing | Location |
| --- | --- |
| Patched image | Docker image `paradedb-patched:0.23.0-ctidfix` (pg_search 0.23.0, PostgreSQL 16.15) |
| The patch | `deploy/public/pg_search-patch/0001-fetch_tuple-validate-ctid-before-block-number.patch` |
| Upstream issue draft (not filed) | `deploy/public/pg_search-patch/UPSTREAM_ISSUE.md` |
| Findings artifact | https://claude.ai/artifact/H7XmpDoRCK4RTkQWzgVDCv |
| Patched test container | `gms-parade-patched` on `127.0.0.1:55440`, **`POSTGRES_HOST_AUTH_METHOD=trust`**, `--shm-size=1g`, `RUST_BACKTRACE=full` — all three are required, see the Gate 0 document |

Rebuild the image if lost:

```bash
git clone https://github.com/paradedb/paradedb.git && cd paradedb && git checkout v0.23.0
git apply /path/to/0001-fetch_tuple-validate-ctid-before-block-number.patch
docker build --build-arg PG_VERSION_MAJOR=16 -f docker/Dockerfile \
  -t paradedb-patched:0.23.0-ctidfix .
```

Takes roughly 20 minutes and ~75 GB of Docker build cache. **Run
`docker builder prune -af` afterwards** — that cache is what filled the disk.

### Disk

Freed ~470 GB this session: 80 G free (96% full) → **550 G free (69%)**. The live
database lives on this filesystem (`data/pg`), so 96% was itself a hazard.
`~/.bashrc` was backed up to `~/.bashrc.bak-preconda` and its conda block
commented out, because anaconda3 was deleted. `~/.android` was deleted, so the
`test_device` AVD must be recreated before `~/scripts/start-emulator.sh` works.

### Unfinished when the session ended

- **Wider regression suite had not completed.** Started but still running:
  ```bash
  cd worktrees/full-agents-20260915
  export GMS_TEST_PG_DSN="postgresql://postgres:synthetic-owner-test@127.0.0.1:55440/postgres"
  export GMS_GATEWAY_TEST_DSN="$GMS_TEST_PG_DSN"
  UV_CACHE_DIR=/tmp/gmail-search-uv-cache uv run --no-sync pytest \
    tests/test_gateway_*.py tests/test_owner_*.py tests/test_text_owner_partitions.py \
    tests/test_mixed_text_owner_partitions.py tests/test_search_ranking.py -q -p no:randomly
  ```
  **This is Gate 0. Re-run it and read the result before anything below.**
- The upstream issue is drafted but **not filed**.
- No minimal standalone SQL reproducer. Five constructions failed to trip it
  (plain table, partitioned, `CREATE INDEX ON ONLY` + attach, empty partition,
  RLS reader). Something about how a zero-offset ctid enters the index is still
  uncharacterised. Not required for the migration; the reliable route if wanted
  is bisecting *down* from the failing harness, not building up.

---

## 2. Live system facts

From `docs/live-partition-preflight-2026-09-15.md` (read-only checks).

**The heaps hold three owners**, so single-owner whole-heap attach is ineligible
and the mixed-owner strategy applies:

| Table | Owner A | Owner B | Owner C |
| --- | ---: | ---: | ---: |
| messages | 421,228 | 21,451 | 118 |
| attachments | 612,268 | 63,235 | 5 |
| embeddings | 1,380,621 | 103,683 | 123 |
| propositions | 1,083,516 | 0 | 0 |

**Schema:** PostgreSQL 16.13, pg_search 0.23.0. Messages use a TEXT `id` as
primary key — there is **no `search_id` column**, so the TEXT profile applies,
not the numeric one. All `user_id` columns are already NOT NULL. Messages,
attachments and propositions have ENABLE + FORCE row security; embeddings has
ENABLE but not FORCE — the migration fixture must mirror that exact variant.

**Storage** (heap / TOAST / index bytes): messages 868 MB / 17.6 GB / 1.02 GB;
attachments 392 MB / 359 MB / 1.28 GB; propositions 292 MB / 15.1 GB / 158 MB;
embeddings 864 MB / 20.9 GB / 415 MB. BM25 forks: messages 899 MB, attachments
650 MB, propositions 90 MB. The attachments `(message_id,filename)` unique
B-tree is 522 MB and must gain the owner column.

**Live container** `gmail-search-pg` runs `paradedb/paradedb:latest-pg16` —
currently the **unpatched** engine.

⚠️ **Floating-tag hazard.** That tag now resolves to pg_search 0.25.9. Any
`docker compose pull` or image prune plus recreate would silently move the
production database two minor versions, break the pinned provisioner, and
require a new `vector` extension dependency. Pin the service to a digest.

---

## 3. The plan

Each phase has an explicit gate. Do not start a phase until the prior gate
passes. Phases 1–4 touch nothing live.

### Gate 0 — Confirm the patch is safe ✅ PASSED 2026-09-16

**1189 passed, 0 failed, 0 errors.** No hangs.

An earlier version of this section reported 25 pre-existing failures and a
hanging test. That was wrong: the disposable cluster I built used password
authentication, and the fixtures require `trust` — they strip the password
deliberately. Corrected record and the exact `docker run` in
[the Gate 0 document](../../qualification/gate-zero-preexisting-failures-2026-09-16.md).

Scott approved the PostgreSQL 16.13 → 16.15 upgrade, so the two `160013` pins
moved to `160015`. Verify after cutover:
`SELECT current_setting('server_version_num')` → `160015`, and
`SELECT extversion FROM pg_extension WHERE extname='pg_search'` → `0.23.0`.

### Phase 1 — Build and qualify phase two ✅ IMPLEMENTED 2026-09-15

Done, synthetically. `MixedTextPhaseOne.publish()` carries INDEX_PENDING → READY:
VACUUM the retained leaves, rebuild their BM25 indexes, qualify every owner's
index against its live rows, then `publish_ready`. `advance()` keeps its old
contract and stays idempotent, so crash re-entry never publishes readiness as a
side effect — publication is a separate, deliberate call.

Tests: `tests/test_mixed_text_owner_partitions_phase_two.py` (3 passed).
Regression: phase one, review and maintenance-gate suites — 67 passed together.
Ruff clean.

**Load-bearing finding: VACUUM before REINDEX, or the rebuild is a no-op.**
Phase one deletes the relocated owners' rows but never vacuums, so
`ambulkdelete` never runs and the retained BM25 index keeps those documents as
live corpus statistics. Measured on the attachments leaf (4 live rows):

| Step | num_docs | num_deleted |
| --- | ---: | ---: |
| After phase one | 7 | 0 |
| After `VACUUM (INDEX_CLEANUP ON)` | 4 | 3 |
| After VACUUM + REINDEX | 4 | 0 |

REINDEX alone — on the leaf, on the parent, or via `REINDEX TABLE` — leaves it
at 7. This is exactly the foreign ranking influence the mixed-owner plan warns
about, and it is why the rebuild runs its VACUUM on a separate connection
(VACUUM cannot run inside a transaction block). Qualification counts only
`visible` segments: pre-rebuild segments linger as non-visible until recycled
and would otherwise double-count.

Still open before this phase is production-ready: qualification under a real
process kill mid-phase, and rollback from the phase-two boundary.

#### Original scope (for reference)

Phase one (mixed-owner redistribution) is implemented and independently
reviewed; the controller stops at `INDEX_PENDING`. It stopped there *because of
the ctid assertion* — the status doc names "the pg_search retained-table
maintenance assertion" as the reason. **That constraint is now removed**, so
phase two is buildable in a way it was not before.

- [ ] Re-read `docs/superpowers/plans/2026-09-15-mixed-owner-partitions.md` and
      `docs/gateway-maintenance.md` for the durable maintenance-gate contract.
- [ ] Implement phase two: index rebuild and `READY` publication past
      `INDEX_PENDING`, under the existing epoch-bound maintenance gate, with
      crash/retry and old-token revocation.
- [ ] Qualify on disposable databases, on the **patched** engine: score isolation
      after rebuild, restart mid-phase, cancellation, rollback.
- [ ] Confirm the migration scripts' `pg_search == '0.23.0'` version pins still
      pass — the patched build reports 0.23.0, so they should.

**Gate:** phase two reaches `READY` on synthetic data, survives a real process
kill mid-phase, and rolls back cleanly.

### Phase 2 — Full-scale rehearsal ✅ MEASURED 2026-09-15

Full live row counts on a cluster mirroring `gmail-search-pg`. Full record in
[the rehearsal document](../../qualification/mixed-migration-rehearsal-2026-09-15.md).

| | Phase one | Phase two |
| --- | ---: | ---: |
| Work (`prepare`) | 47.5 s | 189.0 s |
| Gate verification | 8.5 s | 8.3 s |
| WAL | 1.97 GiB | 0.20 GiB |
| Size | 34.78 → 35.73 GiB | 35.73 → 34.86 GiB |

**~4 minutes of work, under 1 GiB of extra disk, ~2.2 GiB of WAL, and all three
owners' row counts preserved exactly.** The ACCESS EXCLUSIVE window is ~94 s in
phase two plus ~43 s in phase one. Gate verification stays at 8.3–8.5 s, well
inside the 30 s cap, which is what `prepare()` was added to guarantee.

Two environment facts that invalidated earlier attempts, both now fixed and
documented: the rehearsal container needed `--shm-size=1g` (Docker's 64 MB
default surfaces as a misleading `DiskFull`), and its tuning had to be set to
match live rather than ParadeDB's auto-tuning.

- [x] Measure peak disk, WAL, lock duration at full scale.
- [ ] Exercise rollback from each phase boundary at full scale (crash paths are
      covered at fixture scale by `test_phase_two_failure_...`).

#### Superseded scope (for reference)

Measured on the patched engine — full numbers in
[the rehearsal record](../../qualification/mixed-migration-rehearsal-2026-09-15.md).

| | 1% | 10% |
| --- | ---: | ---: |
| Database after populate | 0.366 GiB | 3.494 GiB |
| Phase one | 3.38 s, +0.026 GiB, 0.011 GiB WAL | 5.73 s, +0.111 GiB, 0.197 GiB WAL |
| Phase two | 6.01 s, −0.009 GiB | 29.97 s, −0.086 GiB, 0.030 GiB WAL |

Headline: **the database barely grows and phase two shrinks it**, because the
dominant heap and TOAST are retained rather than rewritten. Full scale needs
single-digit GiB of headroom against 530 GiB free — disk is not a constraint on
this path. The maintenance window is set by the BM25 rebuild (~25.6 s locked at
10%), not by copying rows.

Caveats before trusting the extrapolation: phase one WAL grew **17.9x for 10x
data** (superlinear), wall times grew sublinearly only because fixed overheads
dominate at 1%, and the synthetic corpus omits `embeddings`.

- [ ] Run `--scale 1.0` for the real numbers. It is now cheap.
- [ ] Exercise rollback from each phase boundary at scale (crash paths are
      covered at fixture scale by `test_phase_two_failure_...`).

#### Original scope (for reference)

`deploy/public/rehearse_mixed_migration.py` builds a disposable database at a
fraction of the live row counts and runs both phases, recording per-stage wall
time, WAL bytes, database size and filesystem free space:

```bash
GMS_TEST_PG_DSN=postgresql://postgres:<pw>@127.0.0.1:55440/postgres \
  uv run --no-sync python deploy/public/rehearse_mixed_migration.py \
  --scale 0.1 --out /tmp/rehearsal-10pct.json
```

It creates and drops its own `gms_owner_partitions_test_text_*` database and
never accepts a production DSN. Row counts and per-row TOAST widths come from
the live preflight. Run a small scale first and extrapolate — do not assume
linearity, particularly for index build time.

**Generate incompressible filler.** The first version used `repeat('x', n)`,
which PGLZ crushes to nothing: 212k rows reported 0.14 GiB where the same rows
with concatenated random md5s report ~3.7 GiB — a ~26x understatement that would
have made every disk and WAL figure meaningless. Any change to the generator
must keep the filler incompressible.

Both phases shrink or barely grow the database, because the dominant owner's
heap and TOAST are retained rather than rewritten — which is the entire point of
the mixed-owner strategy. The cost that matters is the minority-owner copy plus
the BM25 rebuild, not a full-table rewrite.

#### Original scope (for reference)

Never rehearse against production. Build a representative synthetic corpus at
live proportions (~440k messages, ~675k attachments, ~1.08M propositions, three
owners with the same skew).

- [ ] Measure and record: **peak disk**, **peak WAL**, **lock duration per
      table**, total wall time.
- [ ] Verify no foreign ranking influence after rebuild (the known failure mode:
      DELETE + VACUUM alone leaves foreign influence; only REINDEX clears it).
- [ ] Exercise rollback from each phase boundary.
- [ ] Confirm headroom: peak disk must fit inside current free space with margin.
      550 G available today; TOAST alone is ~53 GB across the four tables.

**Gate:** measured peak disk and WAL fit comfortably, and rollback works from
every boundary.

### Phase 3 — Backup and restore rehearsal ✅ REHEARSED 2026-09-16

Rehearsed at full scale on synthetic data — **VERIFICATION: PASSED**. Numbers and
caveats in [the rehearsal record](../../qualification/mixed-migration-rehearsal-2026-09-15.md).

| | Value |
| --- | ---: |
| Source | 34.78 GiB |
| `pg_dump -Fc` | 8.1 min → 9.55 GiB |
| `pg_restore -j 4` | 6.9 min |

**Rollback costs ~7 minutes of restore against a ~4 minute migration** — a
comfortable ratio. Verification checks per-owner counts, per-table content
digests, index validity and a live BM25 query.

What this does *not* yet cover, and needs Scott's go-ahead because it touches
live (read-only): dumping the **actual** database to get its true dump size and
duration. The synthetic dump compresses unrealistically well because every row
shares one filler block, so treat 8.1 min / 9.55 GiB as lower bounds.

- [x] Rehearse dump → restore → verify at full scale.
- [x] Confirm restore wall time (6.9 min synthetic).
- [ ] Take and verify a backup of the **live** database (needs approval).

#### Original scope (for reference)

- [ ] Take a full backup of the live database.
- [ ] **Restore it to a scratch instance and verify it.** An unverified backup is
      not a backup.
- [ ] Confirm restore wall time — that is the true rollback cost if the migration
      fails past the point of reversal.

**Gate:** a restored copy passes basic integrity checks, with known restore time.

### Phase 4 — Deploy the patched engine to production ✅ DONE 2026-09-16

Approved by Scott and executed. Maintenance window ~12 minutes, of which the
6-minute cold backup was the bulk; the engine swap itself was ~15 seconds.

Sequence actually run:

1. Stopped all six services, supervisor first so it could not respawn workers.
   Confirmed **zero** remaining connections to `gmail_search`.
2. Stopped the container and confirmed a clean shutdown in the log
   (`database system is shut down`, exit 0, shutdown checkpoint `1D3/B8BDF038`).
3. **Cold file-level backup** to `data/pg-backup-16.13-20260916` (70.3 GiB, 6 min),
   verified with `pg_controldata`: same system identifier `7630921061527392295`,
   cluster state `shut down`, checkpoint LSN matching the shutdown log.
   A file copy — not `pg_dump` — is the right artifact for a binary swap, and it
   avoids the xmin-pinned-vacuum bloat hazard a long logical dump would create.
4. **Renamed** the old container to `gmail-search-pg-pre-ctidfix` rather than
   deleting it, so rollback needs no rebuild.
5. Recreated `gmail-search-pg` on `paradedb-patched:0.23.0-ctidfix` with the
   identical config captured from `docker inspect` — same bind mount, port
   `127.0.0.1:5544`, `--shm-size=1g`, `unless-stopped`, healthcheck, and the
   full `postgres -c …` tuning line. Healthy in 15 s.
6. Restarted all six services.

Verified after cutover:

| Check | Result |
| --- | --- |
| `server_version_num` | **160015** (was 160013) |
| `pg_search` extension | 0.23.0 |
| Startup log | clean; no FATAL/PANIC/ERROR |
| `messages` / `attachments` | 443,085 / 676,335 |
| Owners present | 421,407 / 21,560 / 118 |
| BM25 search on live data | returns results |
| Service endpoints | 8090 and 7878 both answer 401 (correct unauthenticated) |
| Service logs | zero tracebacks or connection failures |

**Rollback, if needed:**

```bash
systemctl --user stop gmail-search-{serve,mcp,supervise,public-api,public-web,web}
docker stop gmail-search-pg && docker rm gmail-search-pg
docker rename gmail-search-pg-pre-ctidfix gmail-search-pg
docker start gmail-search-pg      # back on 16.13 + unpatched pg_search
systemctl --user start gmail-search-{serve,mcp,supervise,public-api,public-web,web}
```

If the data directory itself is suspect, restore `data/pg-backup-16.13-20260916`
over `data/pg` first (root-owned; copy via a container).

**Housekeeping owed:** the old container and the 70.3 GiB backup are both still
on disk deliberately. Keep them until the engine has run normally for a day or
so, then remove to reclaim the space.

#### Original scope (for reference)

First action that touches live. Ask Scott before running it.

- [ ] Stop `gmail-search-pg`, recreate it on `paradedb-patched:0.23.0-ctidfix`
      (image id `sha256:0a7c7b120f02b5…`), keeping the same `data/pg` mount,
      ports, environment **and its 1 GB `--shm-size`** (see the rehearsal record:
      a 64 MB default surfaces as a misleading `DiskFull` under parallel work).
- [ ] Pin by digest, not a floating tag, in whatever compose/unit defines it.
      The live container currently runs image id
      `sha256:99f182f93387…`, i.e. digest
      `paradedb/paradedb@sha256:e41e0c742ef91ece4fc7c08dda7f24e5a8f818563b164bbfea3a4364941d75f7`
      (built 2026-04-16) — that is the rollback target if the patched engine
      has to be backed out.
- [ ] Verify: `SELECT extversion FROM pg_extension WHERE extname='pg_search'`
      returns 0.23.0, existing searches return expected results, daemons reconnect.
- [ ] Watch for a full day of normal operation before migrating.

**Gate:** the app runs normally on the patched engine for a sustained period.

⚠️ Restarting the database restarts the app's dependency. Per
`serve-restart-warmup-window`, restart `serve` after any index/search change —
a prior deploy skew caused a nine-day silent search outage.

### Phase 5 — The migration

- [ ] Fresh verified backup immediately before.
- [ ] Announce a maintenance window sized from the Phase 2 measurements.
- [ ] Run phase one (redistribute minority owners into their own leaves,
      attach dominant heaps), then phase two (rebuild indexes, publish `READY`).
- [ ] Verify after: exact per-owner counts match pre-migration, no foreign
      ranking influence, searches correct for every owner, RLS still enforced.
- [ ] Deploy owner-aware callers in the same window — see
      `docs/text-partition-caller-inventory.md`. Readers and writers must move
      together.

**Gate:** counts reconcile, isolation holds, application works end to end.

### Phase 6 — Record

- [ ] Update `docs/shared-database-implementation-status.md` and
      `docs/full-agent-release-checklist.md` with what was actually run.
- [ ] Record commands, measurements, residual limitations and the exact release.
- [ ] Update `README.md` before any commit — every check-in updates it.

---

## 4. Risks

| Risk | Mitigation |
| --- | --- |
| Patched engine is a local build nobody else runs | File upstream; track their fix; keep the patch and rebuild recipe in-repo |
| Floating `latest-pg16` tag silently upgrades production | Pin by digest before Phase 4 |
| Migration fails past the point of reversal | Verified restore rehearsed in Phase 3, with known restore time |
| Peak disk exceeds free space mid-rewrite | Measured in Phase 2 before anything live; 550 G free today |
| Callers not deployed with the schema | `text-partition-caller-inventory.md`; same window |
| Foreign ranking influence survives | REINDEX required; verified in Phase 2 and Phase 5 |
| Zero-offset ctids still enter the index | The patch stops the user-visible failure; the upstream issue raises the deeper question separately |

---

## 5. Resume checklist

1. `cd /home/ssilver/development/gmail-search/worktrees/full-agents-20260915`
2. `git status --short | wc -l` → expect **359** (122 modified + untracked). Nothing committed.
3. `docker images paradedb-patched:0.23.0-ctidfix` → the patched image.
4. `docker ps -a --filter name=gms-parade-patched` → start it if stopped.
5. Run **Gate 0**.

Read in this order: this file, then
`docs/qualification/retained-reader-root-cause.md`, then
`docs/live-partition-preflight-2026-09-15.md`, then
`docs/superpowers/plans/2026-09-15-mixed-owner-partitions.md`.
