# Mixed-owner migration rehearsal: measured cost

2026-09-15. Synthetic disposable PostgreSQL on the patched engine
(`paradedb-patched:0.23.0-ctidfix`, pg_search 0.23.0 / PostgreSQL 16.15). No
live database, schema or mailbox was touched. Harness:
`deploy/public/rehearse_mixed_migration.py`.

Row counts scale from the live preflight (messages 421,228 / 21,451 / 118;
attachments 612,268 / 63,235 / 5; propositions 1,083,516 / 0 / 0), with per-row
TOAST widths taken from the live storage observations.

## Measured at full scale (live row counts, live-equivalent cluster)

Definitive run: every owner at its live row count, on a cluster configured to
match `gmail-search-pg` (see the mirroring section below). 34.78 GiB populated.

| | Phase one | Phase two |
| --- | ---: | ---: |
| Work (`prepare`) | 47.5 s | 189.0 s |
| Gate verification | 8.5 s | 8.3 s |
| Total | 56.0 s | 197.3 s |
| WAL | 1.97 GiB | 0.20 GiB |
| Database size | 34.78 → 35.73 GiB | 35.73 → 34.86 GiB |
| Peak extra disk | +0.91 GiB | none (recovers) |

**The whole migration is about four minutes of work and needs under 1 GiB of
spare disk.** Row counts after the migration match the input exactly for all
three owners, with no losses. Total WAL is ~2.2 GiB.

**Gate verification is 8.3–8.5 s at full scale**, comfortably inside the 30 s
cap — confirming that moving the slow work into `prepare()` is what makes a
production-size migration possible at all, and that the safety bound did not
have to be relaxed to achieve it.

The maintenance window is driven by phase two's rebuild: `retained_vacuumed` at
90.3 s, `indexes_rebuilt` at 184.7 s — roughly a **94 s ACCESS EXCLUSIVE window**
for the rebuild transaction, on top of phase one's 43 s locked section.

Population itself took 13 minutes (messages 312 s, propositions 472 s), which is
rehearsal setup cost only and has no production analogue.

## Backup and verified restore, full scale

`deploy/public/rehearse_backup_restore.py`, same 34.78 GiB corpus and cluster.
**VERIFICATION: PASSED.**

| | Value |
| --- | ---: |
| Source database | 34.78 GiB |
| `pg_dump -Fc` | 483.3 s (8.1 min), 9.55 GiB |
| `pg_restore -j 4` onto `template0` | 412.8 s (6.9 min) |
| Restored database | 35.54 GiB |

Verification compares source and restored copy on four axes, all of which must
match: per-owner row counts, an md5 content digest per migrated table, index
validity across the whole database, and a live BM25 query — a restored index
must actually answer, not merely exist.

**Rollback costs roughly 7 minutes of restore**, against a ~4 minute migration.
That asymmetry is comfortable: reversing is not dramatically more expensive than
the change itself.

Three things to carry into the live run:

- **The dump size is optimistic.** Every synthetic row shares one filler block,
  so 34.78 GiB compresses to 9.55 GiB. Real mail differs per row and will not
  compress nearly as well, so both dump size and dump/restore time will be
  larger on live. The only honest number comes from dumping the live database,
  which is read-only and safe but needs approval.
- **Restore onto `template0`.** The ParadeDB image preinstalls the `paradedb`
  schema into `template1`, and the dump recreates it, so a default `createdb`
  collides with `schema "paradedb" already exists`. A real disaster-recovery
  restore lands on an empty cluster anyway.
- **Use a per-run dump path.** A fixed path let a still-unwinding run's cleanup
  delete a live run's dump mid-restore. The harness now includes the run suffix.

## Earlier smaller scales (weaker cluster — see mirroring section)

| | 1% scale | 10% scale | ratio (linear = 10.0) |
| --- | ---: | ---: | ---: |
| Database after populate | 0.366 GiB | 3.494 GiB | 9.5 |
| Phase one wall time | 3.38 s | 5.73 s | 1.7 |
| Phase one WAL | 0.011 GiB | 0.197 GiB | 17.9 |
| Phase one size change | +0.026 GiB | +0.111 GiB | 4.3 |
| Phase two wall time | 6.01 s | 29.97 s | 5.0 |
| Phase two WAL | 0.003 GiB | 0.030 GiB | 10.0 |
| Phase two size change | −0.009 GiB | −0.086 GiB | — |

Phase two is dominated by the BM25 rebuild: at 10% scale, `indexes_rebuilt`
lands at 29.46 s of a 29.97 s phase, with the preceding VACUUM at 3.81 s.

## What it means

**The database does not grow materially, and phase two shrinks it.** The
dominant owner's heap and TOAST are retained rather than rewritten, which is the
whole point of the mixed-owner strategy; the only new bytes are the minority
owners' copied rows and the new leaves' indexes. Extrapolated to full scale the
migration needs single-digit GiB of headroom, not the tens of GiB a full-table
rewrite with `search_id` would have required. There is presently 530 GiB free,
so disk is not a constraint on this path.

**The maintenance window is driven by the BM25 rebuild, not by copying.** The
ACCESS EXCLUSIVE window in phase two runs from the vacuum to the commit — about
25.6 s at 10% scale. Phase one's whole locked section is under 6 s at the same
scale.

Linear extrapolation to 100% gives roughly 1 minute for phase one and 5 minutes
for phase two, with about 2 GiB of WAL. Treat those as lower bounds:

- Phase one WAL grew **17.9x for 10x the data**. It is superlinear in this
  range, so budget well above 2 GiB.
- Phase one and phase two wall times grew *sublinearly* here (1.7x and 5.0x)
  only because fixed overheads dominate at 1%. Between 10% and 100% expect
  closer to linear, and index build time may be worse than linear.
- The synthetic corpus omits `embeddings` (20.9 GB of TOAST live). Embeddings
  are not relocated by this step, but they share the filesystem and the WAL
  stream.

A 100% rehearsal is still required before the live run, and it is now cheap to
perform: `--scale 1.0`.

## The rehearsal cluster must mirror live, or the numbers mean nothing

The first full-scale attempt died with
`DiskFull: could not resize shared memory segment ... No space left on device`
while the host had 528 GiB free. The cause was the **container's `/dev/shm`**,
which Docker defaults to 64 MB; PostgreSQL's parallel workers allocate there.

Live is configured correctly and is **not** affected — `gmail-search-pg` runs
with a 1 GB `/dev/shm`. But the disposable rehearsal container had the default,
and its ParadeDB auto-tuning also diverged sharply from live. Rehearsal figures
taken on a mismatched cluster describe the cluster, not the migration.

| Setting | Live | Rehearsal, before | Rehearsal, after |
| --- | --- | --- | --- |
| `/dev/shm` | 1 GB | 64 MB (Docker default) | 1 GB |
| `shared_buffers` | 1 GB | 128 MB | 1 GB |
| `work_mem` | 32 MB | 4 MB | 32 MB |
| `maintenance_work_mem` | 512 MB | 64 MB | 512 MB |
| `max_parallel_workers` | 8 | (auto-tuned) | 8 |

Recreate the rehearsal container to match before trusting any timing:

```bash
docker run -d --name gms-parade-patched -e POSTGRES_PASSWORD=<pw> \
  -e RUST_BACKTRACE=full --shm-size=1g -p 127.0.0.1:55440:5432 \
  paradedb-patched:0.23.0-ctidfix \
  -c shared_buffers=1GB -c work_mem=32MB -c maintenance_work_mem=512MB \
  -c effective_cache_size=4GB -c max_parallel_workers=8 \
  -c max_parallel_workers_per_gather=2 -c max_parallel_maintenance_workers=2 \
  -c max_worker_processes=8 -c max_connections=50 -c max_wal_size=2GB
```

**The 1% and 10% figures above were measured on the weaker pre-fix cluster**, so
they are pessimistic on time and unaffected on size. They are kept because the
size and WAL conclusions still hold; treat their timings as upper bounds.

## Method notes that matter

**Filler must be incompressible.** The first harness used `repeat('x', n)`,
which PGLZ crushes: 212k rows reported 0.14 GiB where the same rows built from
concatenated random md5s report 3.49 GiB. That is a ~26x understatement, and it
would have made every disk and WAL number here meaningless. Any change to the
generator must preserve this property.

**The gate is load-sensitive.** While a rehearsal was running against the same
cluster, one maintenance-gate test failed with `AccessDenied` from the
publication lock and passed cleanly on re-run. `MaintenanceAdmin` uses a 2 s
lock timeout and a 30 s verification timeout; under production load those may
need raising, and a spurious failure there is indistinguishable from a real one
to the caller.

## Reproduce

```bash
GMS_TEST_PG_DSN=postgresql://postgres:<pw>@127.0.0.1:55440/postgres \
  uv run --no-sync python deploy/public/rehearse_mixed_migration.py \
  --scale 0.1 --out /tmp/rehearsal-10pct.json
```

The harness creates and drops its own `gms_owner_partitions_test_text_*`
database and refuses to run without an explicit disposable DSN.
