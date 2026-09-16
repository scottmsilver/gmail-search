# Retained-table search maintenance: release blocked

2026-09-15. Synthetic PostgreSQL 16 / pg_search 0.23.0 only. No live
database changes or real mailbox contents are involved.

The earlier fresh-partition and standalone-query successes do not qualify the
actual retained-table migration. Stronger tests using the real `SearchReader`
found SQLSTATE `XX000`, `assertion failed: item_pointer_is_valid(ctid)`, after
rebuilding indexes and then inserting/deleting synthetic owner rows and running
`VACUUM (INDEX_CLEANUP ON)`.

| Fixture / operation | Recorded result |
| --- | --- |
| Fresh TEXT owner leaves, REINDEX then maintenance | Passed tested message, attachment and fact branches, including restricted message/attachment queries |
| Retained numeric owner leaves, REINDEX then maintenance | Native assertion in unrestricted messages, attachments and facts |
| Actual direct legacy TEXT migration retaining original heaps, REINDEX then maintenance | Same assertion in all three branches; 3 failed in 3.48 s |
| Retained numeric and direct TEXT heaps, replace canonical parent/child BM25 indexes with new OIDs, then maintenance | Same assertion in all six cases; 6 failed in 6.25 s |

The last experiment checked disjoint old/new BM25 index OIDs and unchanged heap
OID/relfilenode/TOAST relation identities. Initial reader snapshots succeeded;
later snapshots after churn failed. Creating entirely new indexes is therefore
not an established workaround. The code used explicit custom planning,
`prepare_threshold=None`, no parallel query workers, and the actual bounded
named-cursor reader. These settings do not fix all retained-table cases.

The TEXT/numeric difference in the initial tests was confounded with fresh versus
retained table history. Do not describe this as an issue isolated to numeric keys,
or conclude that fresh TEXT partitions establish migration safety. The underlying
native-engine cause remains undiagnosed.

The test agent reported the REINDEX results before its turn failed. Root then
inspected its already-produced `/tmp/gms-policy-directtext.log` and
`/tmp/gms-policy-new-index.log`, including the executed setup and failure stages.
Root did not independently rerun these retained-table experiments. The final
experiment's source is preserved as
`retained-reader-new-index-reproducer.py`; it imports the repository's disposable
database fixtures and is a diagnostic, not a passing acceptance test.

The agent's turn ended with a platform cybersecurity-risk flag. It was not a
database-test success or a shell approval rejection. Its task was not restarted
or rephrased; recovery here only inspected existing source and results.

## Consequence

Keep production migration and phase-two readiness disabled. The separately
reviewed maintenance-gate primitive and synthetic phase-one redistribution work
do not resolve this blocker. A replacement query/index or engine approach must
pass the exact retained-table lifecycle before any release claim.
