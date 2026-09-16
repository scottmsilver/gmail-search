# Retained-table BM25 assertion: root cause identified

2026-09-15, supersedes the "cause undiagnosed" line in
[the blocker record](retained-reader-maintenance-blocker.md). Companion to the
[engine-version A/B](retained-reader-engine-version-ab.md). Synthetic disposable
PostgreSQL only; no live database, schema or mailbox was touched.

## Summary

The fault is a statement-ordering defect in `pg_search`, not in our schema,
migration or reader. A guard written specifically to tolerate stale ctids left
behind by VACUUM is placed *after* an asserting call, so the exact condition it
exists to handle panics before the guard can run.

It is reached only through the heap-filter path, which our owner-isolation RLS
predicate forces. It is present in pg_search 0.23.0, 0.25.9 and current upstream
`main`, and appears to be unreported.

## The stack

Captured by recreating the qualification cluster with `RUST_BACKTRACE=full` and
running the existing reproducer unmodified (pg_search 0.23.0, pgrx 0.17.0):

```
11: pgrx::itemptr::item_pointer_get_block_number   pgrx-0.17.0/src/itemptr.rs:24
12: pg_search::postgres::heap::HeapFetchState::fetch_tuple            heap.rs:323
13: HeapFieldFilter::evaluate_expression_inner             heap_field_filter.rs:114
14: HeapFieldFilter::evaluate                               heap_field_filter.rs:95
15: HeapFilterScorer::passes_heap_filters                  heap_field_filter.rs:356
16: HeapFilterScorer::find_first_valid_document            heap_field_filter.rs:336
17: HeapFilterScorer::new                                  heap_field_filter.rs:327
18: <HeapFilterWeight as tantivy::query::weight::Weight>::scorer      :274
19: tantivy::query::boolean_query::boolean_weight::BooleanWeight::per_occur_scorers
```

This is not the visibility-map fast path that the surrounding code review
suggests; it is the heap *filter* path, entered while the scorer is still being
constructed.

## The defect

`pg_search/src/postgres/heap.rs`, unchanged between v0.23.0 and current `main`:

```rust
/// Wrapper around `table_index_fetch_tuple` that guards against stale ctids
/// referencing heap blocks truncated by VACUUM.
///
/// The BM25 `ambulkdelete` correctly removes dead ctids from the index, but only
/// when VACUUM actually runs. Between VACUUM cycles, the index may still contain
/// ctids pointing to pages that a *previous* VACUUM truncated. The normal scan path
/// (top-K) rarely hits these because it fetches few results, but the heap_filter
/// path fetches ALL matching documents, making truncated-block hits likely.
///
/// Returns `false` if the block has been truncated or the tuple is not visible.
pub unsafe fn fetch_tuple(...) -> bool {
    let blockno = pgrx::itemptr::item_pointer_get_block_number(ctid);   // asserts validity
    if blockno >= self.nblocks {                                        // the guard
        return false;
    }
    ...
```

`pgrx::itemptr::item_pointer_get_block_number` opens with
`assert!(item_pointer_is_valid(ctid))`. A stale ctid whose offset is zero is
therefore fatal on the line that reads its block number — the guard on the next
line is unreachable for precisely the inputs the doc comment describes. The
guard covers *out-of-range* blocks but not *invalid* pointers.

The comment also explains the exposure: the heap-filter path "fetches ALL
matching documents", so a scan that touches every candidate row is far more
likely to meet a stale entry than a top-K scan.

## Why only retained heaps

The crash needs two things at once.

1. **Stale ctids in the index.** Retained heaps carry prior delete and VACUUM
   history, so the index holds entries pointing at pages a previous VACUUM
   already truncated. Freshly created partitions have no such history — which is
   exactly the fresh/retained split observed across both engine versions.
2. **The heap-filter path.** Our fixed-reader RLS predicate is on `user_id`,
   which is not among the BM25 index fields, so pg_search evaluates it as a
   `HeapExpr` filter and fetches every candidate tuple from the heap.

Remove either ingredient and the assertion disappears.

## Confirmed by removing the path, twice

Both experiments used the unmodified reproducer and counted server-log
occurrences of the assertion.

| Configuration | Assertions logged | Outcome |
| --- | --- | --- |
| Baseline | present, 6 failed / 7 passed | the blocker |
| `paradedb.enable_filter_pushdown = off` | **0** | `ERROR: Unsupported query shape` |
| `user_id` added to all three BM25 indexes | **0** | our own `Unsupported parent BM25 definition` validator rejects it |

Neither is a usable mitigation as-is — the first replaces the crash with a hard
refusal for these query shapes, and the second trips our audited index
definition in `gateway/partitions.py` — but together they establish the causal
path beyond doubt. Both settings were reverted; the baseline was re-verified at
6 failed / 7 passed afterwards.

## The patch is the cause of the fix — controlled 2026-09-16

The original evidence left one cell untested and filled it by reading source.
That inference is now a measurement. All four rows use the same reproducer, the
same cluster configuration (trust auth, port 55440, `--shm-size=1g`) and the
same Dockerfile:

| Engine | Reproducer | `item_pointer_is_valid` assertions |
| --- | --- | ---: |
| PG 16.13 + pg_search 0.23.0 | 6 failed, 7 passed | present |
| PG 16.15 + pg_search 0.25.9 | 6 failed | present |
| **PG 16.15 + pg_search 0.23.0, unpatched** | **6 failed, 7 passed** | **6** |
| **PG 16.15 + pg_search 0.23.0, patched** | **13 passed** | **0** |

The last two rows differ only by the two-line change in `fetch_tuple`. The
control image is `paradedb-control:0.23.0-unpatched`, built from the same
Dockerfile with the patch reverted (verified absent in the source before
building). PostgreSQL 16.15 alone does not fix it; the patch does.

This matters because the patched engine now serves the live database, and
"the patch fixed it" was previously an argument rather than an observation.

## Upstream status

Current `main` still computes `blockno` before testing the guard. A related but
distinct assertion in the same file was fixed in
`011333b3 fix: prevent heap filters from tripping assert on aggregate scan (#5508)`,
which addressed a `TTS_IS_VIRTUAL` slot-type assert and refactored `fetch_tuple`
into `fetch_eval_slot`; it did not change this ordering. Earlier,
`8efd56f2 fix: prevent ReadBuffer errors from stale ctids after VACUUM truncation (#4338)`
introduced the `nblocks` guard that this ordering defeats.

The upstream fix is small: test the pointer before reading the block number, and
return `false` for an invalid one rather than asserting.

## What this changes for us

The A/B report concluded that heap-retaining migration profiles sit on the
failing side of a split with no engine remedy. That still holds, but the reason
is now specific and the options are concrete rather than open-ended:

1. **Report upstream** with the reproducer. The change is a few lines in a file
   we do not own, and a fixed release would remove the constraint entirely.
2. **Carry a patched extension.** We already pin an exact pg_search version and
   audit its function surface, so building a patched image is closer to existing
   practice than it might appear. It adds a build to maintain.
3. **Eliminate the heap filter** by making the owner predicate index-evaluable —
   the `user_id` experiment above. This needs the audited index definition, the
   provisioner column profile and the partition validator updated together, and
   needs requalifying; it is a schema-profile decision, not a patch.
4. **Avoid retained heaps** by redistributing into fresh partitions, at the
   rewrite cost phase one was designed to avoid.

Options 1 and 3 are independent and can proceed in parallel. None of them is
qualified yet, and the production migration stays disabled.

## Reproducing the stack

Recreate the disposable cluster with backtraces enabled, then run the reproducer
as documented in the A/B report:

```bash
docker run -d --name gms-owner-parade-qualification \
  -e POSTGRES_PASSWORD=<pw> -e RUST_BACKTRACE=full \
  -p 127.0.0.1:55440:5432 paradedb/paradedb:latest-pg16
```

The current qualification container was recreated with `RUST_BACKTRACE=full`
still set, which is worth keeping for any further native diagnosis.
