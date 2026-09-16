# `HeapFetchState::fetch_tuple` asserts on an invalid ctid before its own stale-ctid guard

## What happens

Queries that go through the heap-filter path abort with:

```
ERROR:  assertion failed: item_pointer_is_valid(ctid)
```

SQLSTATE `XX000`. The backend does not crash, but the statement always fails, and
it keeps failing for the same index until the heap history is rebuilt.

## Where

`pg_search/src/postgres/heap.rs`, in `HeapFetchState::fetch_tuple`:

```rust
/// Wrapper around `table_index_fetch_tuple` that guards against stale ctids
/// referencing heap blocks truncated by VACUUM.
/// ...
/// Returns `false` if the block has been truncated or the tuple is not visible.
pub unsafe fn fetch_tuple(...) -> bool {
    let blockno = pgrx::itemptr::item_pointer_get_block_number(ctid);
    if blockno >= self.nblocks {
        return false;
    }
    pg_sys::table_index_fetch_tuple(...)
```

`pgrx::itemptr::item_pointer_get_block_number` opens with
`assert!(item_pointer_is_valid(ctid))`, and in pgrx 0.17.0 `item_pointer_is_valid`
is:

```rust
pub unsafe fn item_pointer_is_valid(ctid: *const pg_sys::ItemPointerData) -> bool {
    if ctid.is_null() { false } else { (*ctid).ip_posid != pg_sys::InvalidOffsetNumber }
}
```

So a stale ctid whose offset number is zero is fatal on the very first line of the
function. The `nblocks` guard on the next line can never see it.

The guard handles a ctid whose block is *out of range*. It does not handle a ctid
that is *invalid*. The function's own doc comment describes tolerating stale ctids
left behind by VACUUM, and notes that this path "fetches ALL matching documents,
making truncated-block hits likely" — so this is exactly the path where such a
pointer is most likely to show up.

## Backtrace

pg_search 0.23.0, pgrx 0.17.0, PostgreSQL 16.13, captured with `RUST_BACKTRACE=full`:

```
11: pgrx::itemptr::item_pointer_get_block_number
          at pgrx-0.17.0/src/itemptr.rs:24:5
12: pg_search::postgres::heap::HeapFetchState::fetch_tuple
          at pg_search/src/postgres/heap.rs:323:23
13: pg_search::query::heap_field_filter::HeapFieldFilter::evaluate_expression_inner
          at pg_search/src/query/heap_field_filter.rs:114:30
14: pg_search::query::heap_field_filter::HeapFieldFilter::evaluate
          at pg_search/src/query/heap_field_filter.rs:95:14
15: pg_search::query::heap_field_filter::HeapFilterScorer::passes_heap_filters
          at pg_search/src/query/heap_field_filter.rs:356:44
16: pg_search::query::heap_field_filter::HeapFilterScorer::find_first_valid_document
          at pg_search/src/query/heap_field_filter.rs:336:51
17: pg_search::query::heap_field_filter::HeapFilterScorer::new
          at pg_search/src/query/heap_field_filter.rs:327:16
18: <pg_search::query::heap_field_filter::HeapFilterWeight as tantivy::query::weight::Weight>::scorer
          at pg_search/src/query/heap_field_filter.rs:274:22
19: tantivy::query::boolean_query::boolean_weight::BooleanWeight<TScoreCombiner>::per_occur_scorers
```

## Versions affected

Reproduced identically on:

- pg_search **0.23.0**, PostgreSQL 16.13 (`paradedb/paradedb:latest-pg16`, built 2026-04-16)
- pg_search **0.25.9**, PostgreSQL 16.15 (`paradedb/paradedb:latest-pg16`, built 2026-09-11)

The ordering is unchanged on `main` as of `85d6ca06`.

## Conditions

Three things together:

1. A BM25 index over heaps that carry prior `DELETE` + `VACUUM` history — in our
   case pre-existing populated tables attached as partitions to a parent that has
   a BM25 index, rather than freshly created partitions. Freshly built partitions
   never fail.
2. A predicate on a column that is **not** a BM25 index field, so it is evaluated
   as a `heap_filter`. Ours comes from a row-level-security policy on `user_id`;
   `EXPLAIN` shows
   `"field_filters":[{"heap_filter":"(user_id = 'alice'::text)"}]`.
3. `INSERT` / `DELETE` / `VACUUM (INDEX_CLEANUP ON)` cycles with searches in
   between.

Two independent ways of removing the heap-filter path each eliminate the
assertion completely (0 occurrences in the server log, where the baseline
reliably produces them):

- `SET paradedb.enable_filter_pushdown = off` — though these queries then fail
  with `ERROR: Unsupported query shape`, so it is not a usable workaround for us.
- Adding the filtered column (`user_id`) to the BM25 index, so the predicate no
  longer needs a heap fetch.

Both are consistent with the heap-filter path being the only route to this
assertion.

I have not yet reduced this to a short standalone SQL script — my attempts to
rebuild the conditions from scratch in plain SQL did not trip it, so some part of
the history that produces a zero-offset ctid is not yet captured. I am happy to
share the full reproducer, or to run any instrumented build against the setup
that does fail reliably.

## Suggested fix

Test the pointer before reading its block number:

```rust
pub unsafe fn fetch_tuple(...) -> bool {
    // A stale ctid can be not merely out-of-range but outright invalid
    // (offset 0). `item_pointer_get_block_number` asserts validity, so it would
    // panic before the truncated-block guard below could reject it.
    if !pgrx::itemptr::item_pointer_is_valid(ctid) {
        return false;
    }
    let blockno = pgrx::itemptr::item_pointer_get_block_number(ctid);
    if blockno >= self.nblocks {
        return false;
    }
    ...
```

This keeps the existing contract — "returns `false` if the block has been
truncated or the tuple is not visible" — and extends it to the invalid-pointer
case the guard was clearly meant to cover.

Worth considering separately: whether a zero ctid should be reaching the scorer
at all. `HeapFilterScorer::passes_heap_filters` already panics with
`"Could not get ctid for doc_id"` when the fast field is absent, so a
zero-but-present value may point at something upstream in segment or fast-field
handling. The ordering fix above stops the user-visible failure either way.

## Related

- #4338 introduced the `nblocks` guard that this ordering defeats.
- #5508 fixed a different assertion in this same function (`TTS_IS_VIRTUAL` on
  the aggregate scan) and refactored `fetch_tuple` into `fetch_eval_slot`,
  leaving this ordering unchanged.
