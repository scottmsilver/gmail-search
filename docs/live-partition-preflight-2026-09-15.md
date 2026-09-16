# Live partition preflight observations

Read-only checks on 2026-09-15. Queries returned schema definitions, storage
metadata and aggregate counts; no message content or owner identities were
returned. No production DDL, writes, service restart or cutover was performed.
Counts and sizes are observations during ongoing operation, not a frozen
migration snapshot or a capacity estimate.

## Important finding: the heaps already contain several owners

The single-owner whole-heap attachment rehearsal **cannot run directly on the
current live database**. Anonymous labels below consistently represent the same
owners across tables; they are not new database identities.

| Table | Owner A | Owner B | Owner C |
| --- | ---: | ---: | ---: |
| messages | 421,228 | 21,451 | 118 |
| attachments | 612,268 | 63,235 | 5 |
| embeddings | 1,380,621 | 103,683 | 123 |
| propositions | 1,083,516 | 0 | 0 |

The checked owner columns are non-null. A possible lower-copy strategy is to
relocate the minority owners' messages and attachments into their own leaves,
then attach the original heaps for the dominant owner. This is **not yet an
approved migration mechanism**. Row percentages do not establish byte costs.
An independent synthetic probe confirmed that existing mixed-corpus BM25
indexes retain foreign ranking influence after DELETE, ANALYZE, ordinary VACUUM
and partition attachment. REINDEX restored exact clean owner-only results.
Retained indexes therefore require an explicit rebuild in the revised strategy;
its cost and rollback behavior must be qualified.

## Installed schema

- PostgreSQL 16.13 and `pg_search` 0.23.0 match the qualification environment.
- Messages use TEXT `id` as the scalar primary key and five-field BM25 key;
  there is no `search_id` column or message identity sequence.
- Attachments use scalar numeric `id`, unique `(message_id,filename)`, and the
  three-field BM25 index. Propositions use scalar numeric `id` and two-field BM25.
- All eight inspected `user_id` columns are already NOT NULL.
- The legacy message/attachment foreign keys are present. Propositions and
  `prop_processed` have no message foreign key, matching the historical schema;
  the target must validate and add their owner-qualified references.
- The topic reference is `(user_id,topic_id) -> topics(user_id,topic_id)`.
- Messages, attachments, propositions and the inspected metadata tables have
  ENABLE/FORCE row security. Embeddings have ENABLE but not FORCE. The migration
  fixture must represent this exact observed variant; later reader provisioning
  must satisfy its stricter readiness contract before credentials are issued.

## Storage observations

Bytes below separate each main heap, its TOAST storage (large out-of-line values,
including that storage's indexes), and the main table's indexes.

| Table | Main heap bytes | TOAST total bytes | Main-table index bytes |
| --- | ---: | ---: | ---: |
| messages | 868,212,736 | 17,586,126,848 | 1,024,409,600 |
| attachments | 392,077,312 | 359,112,704 | 1,283,719,168 |
| propositions | 292,364,288 | 15,063,547,904 | 158,089,216 |
| embeddings | 864,247,808 | 20,884,455,424 | 415,047,680 |

Observed BM25 main forks: messages 898,801,664 bytes; attachments 649,756,672;
propositions 90,218,496. The attachment `(message_id,filename)` unique B-tree is
522,371,072 bytes and must change to include the owner. These are current stored
sizes, not forecasts of replacement indexes, peak disk, WAL or lock duration.

The container data mount is the repository's `data/pg` on the same filesystem
as the workspace. Available filesystem space at the check was 112,226,983,936
bytes. A migration must budget concurrent writes, WAL, old/new indexes and
rollback/backup space; this observation alone does not establish sufficient
headroom.

## Next gate

Keep the synthetic direct single-owner migration strict and useful as a building
block. Its current-row scan does not prove clean historical index statistics.
Qualify the [separate mixed-owner strategy](superpowers/plans/2026-09-15-mixed-owner-partitions.md)
against the actual source shape,
including foreign keys, all dependent rows, BM25 history, cancellation/rollback
and compatible writers. Never discard another owner's rows to satisfy an attach
constraint, and never attach a mixed heap under one owner's partition bound.
