# Retained shared BM25 indexes after minority-row deletion

## Decision

**Deleting foreign rows and attaching the old heap/index is insufficient for owner-independent ranking on the pinned pg_search 0.23.0 fixture.** Ordinary VACUUM removes those rows from the live document count but leaves their statistical influence on scores and result order. A supported `REINDEX INDEX` of the retained owner leaf restored the exact clean-owner baseline in all three tested index shapes **only when foreign deletion had already committed**. DELETE, attach and REINDEX in one transaction still produced contaminated scores after commit; that atomic migration sequence is rejected by the probe.

A migration that moves only minority owners can still retain the dominant owner's heap and TOAST. It must budget rebuilding every retained BM25 index with possible mixed-owner history, or separately prove another supported statistics-reset operation before exposing ranked search. Current rows belonging to one owner do not establish single-owner index history. Omitting scores from the response does not solve this: result order changed too.

No production rows, catalogs or migrations were read or changed by this probe. Synthetic corpus proportions are deliberately adversarial; they do not estimate actual mailbox statistics or migration costs.

## Pinned experiment

- PostgreSQL 16.13, pg_search 0.23.0.
- Existing synthetic `127.0.0.1:55440` container `gms-owner-parade-qualification`.
- Image ID `sha256:99f182f93387c2226c810763f736ec06154090b37d654a09483ca44d68b5ba6c`; its mutable `latest-pg16` tag is not the pin.
- Every run creates a random `template0` database and generated direct LOGIN reader, then removes only those resources. The probe refuses other DSN host/port/database/user profiles and hostaddr/service/options overrides, and checks the actual connection address.
- Separate clean and mixed indexes contain identical four Alice rows. Mixed indexes additionally index 200 Bob rows before index creation. Table autovacuum is disabled in this tiny fixture so explicit maintenance stages are observable. There is no old transaction retaining Bob's tuples: connections use autocommit.
- The direct non-superuser, non-BYPASSRLS reader has immutable Alice restrictive RLS alongside a permissive policy. Spoofing `app.user_id` to Bob has no effect. PUBLIC extension routine grants are removed, and the reader receives only the four previously qualified search routines. After attachment, direct child access is refused.

The exact canonical BM25 field lists from the qualified profiles are used:

| Shape | BM25 fields | Key |
| --- | --- | --- |
| TEXT messages | id, subject, body_text, from_addr, to_addr | id (text) |
| Numeric messages | search_id, id, subject, body_text, from_addr, to_addr | search_id (bigint) |
| Attachments | id, filename, extracted_text | id (bigint) |

The probe isolates native index/history behavior. Its tables have the indexed fields and owner-qualified probe keys, rather than the complete product's unrelated columns/FKs. It does not qualify migration dependencies or caller compatibility.

Alice's field values are `alpha alpha`, `beta`, `alpha beta filler filler filler`, and twenty repetitions of `filler`. Every Bob document is `alpha foreignonly`. Queries use the actual `@@@` operator and `paradedb.score(key)`, a fixed literal Alice predicate, score-descending/key tie-break ordering, and LIMIT. Both forced custom/generic prepared plans and entirely fresh reader backends are measured. They agree at every original deletion/maintenance/rebuild stage, ruling out a reused connection's cached score as the explanation. The separate post-rebuild churn branch below exposes a different generic-plan failure.

## Observed scores and ranking

Row numbers below correspond to the first three Alice rows (TEXT keys are `owner1` etc.). Results are identical across the three index shapes:

| Stage | Descending result order | Scores in that order |
| --- | --- | --- |
| Clean owner-only control | 3, 1, 2 | 1.5697745; 1.1926777; 1.0674466 |
| Shared index before DELETE | 2, 3, 1 | 5.6072526; 2.8221111; 0.017096052 |
| DELETE Bob; commit | 2, 3, 1 | unchanged shared-index scores |
| ANALYZE | 2, 3, 1 | unchanged shared-index scores |
| VACUUM (ANALYZE, INDEX_CLEANUP ON) | 2, 3, 1 | unchanged shared-index scores |
| Attach existing heap/index as Alice leaf | 2, 3, 1 | unchanged shared-index scores |
| Further explicit leaf VACUUM passes | 2, 3, 1 | unchanged shared-index scores |
| REINDEX retained leaf BM25 | 3, 1, 2 | exact clean-control scores |

At each post-delete stage the direct reader sees exactly four live rows and zero matches for `foreignonly`. After VACUUM, `paradedb.index_info` reports one segment with four live documents and 200 deleted documents. The segment identifier remains unchanged through attach and further VACUUM passes. The rebuilt index has a new segment and zero deleted documents. Executed plans select only the attached Alice leaf: physical owner pruning does not remove that leaf's historical statistical contamination.

The available `paradedb.force_merge(regclass,bigint)` routine rejects execution with SQLSTATE 22000 and the fixed message `force_merge is deprecated, run VACUUM instead`. No unsupported compaction or broad maintenance workaround was attempted. [Current upstream scoring documentation](https://github.com/paradedb/paradedb/blob/main/docs/documentation/sorting/score.mdx) describes VACUUM-based score refresh; the measured pinned-version results govern this migration decision and should not be generalized to other versions.

## Storage identity, rollback, and tiny-fixture costs

Moving/attaching the retained table preserves its heap, TOAST and BM25 OIDs and relfilenodes. The leaf index has ancestry to the new matching partitioned BM25 parent. `REINDEX INDEX` preserves heap/TOAST identities and the BM25 index OID, but changes the BM25 relfilenode and segment contents. Consequently this is a heap-preserving strategy, not an index-file-preserving strategy.

A forced exception inside a transaction immediately after REINDEX restores the original BM25 relfilenode, main-fork size, segment and polluted score results. A subsequent committed REINDEX yields the clean-owner score baseline. Fresh reader backends also verify both rollback and committed results.

One retained report observed:

| Shape | Committed REINDEX plus verification | Cluster WAL interval | Old/new BM25 main-fork bytes |
| --- | ---: | ---: | ---: |
| TEXT messages | 0.0198 s | 824 bytes | 3,063,808 / 3,022,848 |
| Numeric messages | 0.0180 s | 832 bytes | 3,063,808 / 3,022,848 |
| Attachments | 0.0166 s | 824 bytes | 3,063,808 / 3,022,848 |

Each retained live heap is only 8,192 bytes. Times include synchronous maintenance plus score/fresh-connection verification. WAL deltas use cluster-wide `pg_current_wal_insert_lsn` intervals, can include concurrent synthetic activity, and do not measure total index I/O, disk headroom, WAL retention or production amplification. These tiny values are not production cost estimates. A realistic stopped-writer rehearsal must separately measure minority-row moves, retained-index rebuilding, B-tree/FK changes, locks, rollback, peak storage and restore requirements.

## Reproduce and assertions

- Probe: `deploy/public/probe_bm25_deleted_statistics.py`.
- Regression assertions: `tests/test_bm25_deleted_statistics.py`.
- Sanitized report with all scores, index segments, object identities and executed plans: `docs/qualification/bm25-deleted-statistics.json`.

Run with the explicitly approved synthetic `GMS_TEST_PG_DSN`, candidate `PYTHONPATH=src`, and original virtual environment:

```sh
python deploy/public/probe_bm25_deleted_statistics.py
python -m pytest tests/test_bm25_deleted_statistics.py -q
```

The CLI writes `/tmp/gms-deleted-bm25-report.json` by default; `GMS_PROBE_OUTPUT` selects a local report file only. Twelve focused tests passed in 2.82 seconds, including the atomic-deletion checks below. They assert stale scores/order despite deletion/maintenance/attach, exact restoration after rebuilding, preserved heap/TOAST identities, transactional index rollback, fresh-backend agreement, RLS denial and DSN containment. No migration implementation changed.


## Atomic DELETE + attach + REINDEX is not sufficient

The follow-up starts with fresh clean and shared indexes, then performs all of
these steps inside one explicit transaction, with no intermediate commit:

1. Delete every Bob row from the old mixed heap.
2. Move and attach that same heap/BM25 beneath a matching Alice partition parent.
3. REINDEX the retained BM25 leaf.
4. Either commit, or inject an exception to test rollback.

After commit, ordinary SQL reports only Alice's four rows. Nevertheless, the new
BM25 segment reports **204 indexed documents**, and both reused and entirely
fresh direct-owner backends return the exact polluted `2,3,1` ordering and scores.
The same scores appear inside the migration transaction. All three canonical
index field/key shapes reproduce this. Rebuilding inside the transaction did
not exclude that transaction's deleted foreign tuples from BM25 statistics.

Forced rollback restores the original mixed row counts (Alice 4, Bob 200), heap,
TOAST, index OIDs/relfilenodes/main-fork sizes, and original polluted scores. A
**separate post-commit REINDEX**, with no intervening VACUUM in this fixture,
then restores the exact clean-owner result. This isolates transaction visibility
as a material boundary. It does not authorize an intermediate-commit migration,
prove behavior with old active snapshots, or qualify a staged recovery protocol.

The proposed atomic minority-move/delete/retained-index-rebuild plan therefore
requires redesign before implementation. A separate read-blocked staged approach
would need explicit recovery and readiness guarantees and approval; a transaction
that rewrites the retained heap has different cost/identity properties. Neither
alternative is implemented or approved by this experiment.

Reproduce this branch using `python deploy/public/probe_bm25_deleted_statistics.py
--atomic`. Sanitized full results are retained in
`docs/qualification/bm25-atomic-delete-reindex.json`; three parameterized tests
assert the contaminated atomic commit, exact rollback and clean separate rebuild.

## Explicit two-phase feasibility and restart/rollback probe

A separate synthetic branch (`--staged`) performs the exact alternative order:

1. In phase one, DELETE Bob and ATTACH the retained heap/index, then commit.
   **No REINDEX occurs in this transaction.** Alice-only data is committed, while
   the original index remains statistically contaminated.
2. Open a new administrator connection, REINDEX in a new transaction, and inject
   an exception. Rollback leaves Alice-only data and the original dirty index
   files/scores; it does not restore phase-one-deleted Bob rows.
3. Close that connection, open another, retry REINDEX and commit. Custom/generic
   queries and fresh direct-owner connections now exactly match clean scores.

All three index shapes pass. Heap/TOAST/index OIDs stay unchanged throughout;
phase-two commit changes only the BM25 relfilenode/contents among those identities.
The phase-two rollback restores its previous relfilenode. No VACUUM is required
between these phases in this fixture with no old active transactions. Separate
connections test independence from phase-one session state; they do not simulate
a machine crash, durable maintenance journal, writer admission, or fail-open
restart recovery.

Full results: `docs/qualification/bm25-staged-delete-reindex.json`. Fifteen focused
probe tests passed in 3.49 seconds after adding the staged assertions. The staged
order is technically feasible here, but the interval with owner-only rows and
contaminated ranking must remain behind a durable closed maintenance gate. A
reviewed restore/recovery protocol and the separate post-rebuild churn assertion
investigation remain release gates. No production recommendation or migration
implementation follows from this result alone.

## Separate release blocker: generic plans after owner maintenance

The original exploratory `item_pointer_is_valid(ctid)` assertion was preserved
and is repeatable. Following a committed retained-leaf rebuild, insert one new
Alice row, commit, delete it, commit, and VACUUM the Alice leaf. On pinned 0.23.0,
the next forced-generic prepared search raises SQLSTATE `XX000` with message
`assertion failed: item_pointer_is_valid(ctid)`.

Two query-order variants were repeated in three fresh databases each, across
all three index shapes (18 combinations): one queries after INSERT and DELETE;
the other makes no intervening search. Both fail after VACUUM. The observed
variant succeeds after INSERT and after DELETE, so VACUUM is the measured
transition. Subsequent generic retries fail too. A completely fresh connection
with `force_generic_plan` and `prepare_threshold=0` also fails, ruling out only
an old backend's cached statement state. Nonexecuting EXPLAIN plans are retained.

The same reused connection forced to a custom plan, and fresh default connections,
return the clean owner baseline. That comparison does not establish why the
extension asserts or a generally safe workaround. Normal qualification of fresh
partitions had not covered this repeated-maintenance state. Current product
reader settings are unchanged by this probe.

- Reproduce: `--churn observed` or `--churn unobserved`.
- Evidence: `docs/qualification/bm25-post-reindex-churn.json`.
- Six parameterized assertions preserve the pinned failure and custom/fresh
  control results; a future extension/profile correction must deliberately
  replace this failure expectation with a reviewed passing contract.

### Candidate countermeasure experiment only

`--churn custom_cycles` starts with the exact two-phase delete/attach then rebuild
sequence, including phase-two rollback and new-connection retry. After phase-two
commit it opens fresh connections with `prepare_threshold=None`, a REPEATABLE
READ READ ONLY snapshot, a bounded statement timeout, and
`plan_cache_mode=force_custom_plan`. It executes the fixed lexical projection
inside the reader-style byte-bounded `row_to_json` wrapper and named cursor;
a separate ordinary query on that connection cross-checks the result. It does
not instantiate or change the product SearchReader/catalog audit.

For each index shape, three complete own-row INSERT/DELETE/VACUUM cycles produce
no assertion and return the exact initial baseline after VACUUM. At each cycle,
200 Bob-row INSERT, UPDATE, DELETE and Bob-leaf VACUUM leave Alice's results
unchanged. Both bounded cursor output and direct-query cross-checks agree.
Evidence: `docs/qualification/bm25-custom-unprepared-cycles.json`.

This is a candidate mitigation, not a product fix, extension upgrade choice or
performance qualification. Disabling preparation/custom planning must be reviewed
as an explicit runtime profile with realistic query planning cost and actual
service integration tests before reopening the release gate. The complete probe
suite has 24 tests, passed in 8.17 seconds; Ruff passed. No live data, product
reader settings, migrations or deployment changed.

### Later actual-reader result: mitigation not qualified

The subsequent actual `SearchReader` integration reproduced the native assertion
on retained numeric and retained legacy TEXT heaps after index rebuilding and
own-row maintenance, despite custom planning and disabled preparation/parallelism.
Replacing the BM25 indexes with entirely new OIDs did not prevent the recorded
failure. Fresh TEXT leaves passed their tested cases, but were not a valid control
for retained-table history. See the
[retained-reader release blocker](qualification/retained-reader-maintenance-blocker.md)
for scope and evidence provenance. The standalone successes above remain limited
observations; they must not be used as production-readiness evidence.
