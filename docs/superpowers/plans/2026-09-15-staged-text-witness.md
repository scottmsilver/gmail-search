# Staged TEXT witness — approved synthetic phase-one slice

Only disposable databases at configured **and actual** loopback peer
127.0.0.1:55440 with names beginning `gms_owner_partitions_test_` are eligible.
Keep the exact PostgreSQL16/pg_search0.23 and direct database-owner administrator
checks. No production override, numeric migration, or live readiness publication.

Use a frozen plan binding store/profile/epoch, operation, explicit dominant owner,
sorted bounded expected owner set, fixed procedure digest and actual database
identity. One private Registry belongs to that store. Every administrator
connection is newly opened, IDLE and autocommit; orchestration owns transactions
and cleanup. Runtime paths/owners/SQL cannot select this plan.

MaintenanceAdmin owns the outer exclusive publication lock. Its trusted verifier
performs phase one and fresh-connection verification outside SQLite transactions;
it does not recursively call a gate transition or publication_guard. Acquire
PostgreSQL migration, sorted owner, catalog, sorted relation and qualified serial
sequence locks in the established order. Keep the current single-owner public
APIs and behavior unchanged while factoring only reusable private mechanics.

Under the first transaction, validate the complete supported source/dependencies,
copy only minority rows into exact private owner children, verify exact copied
contents in PostgreSQL before deleting them from the retained heaps, attach the
dominant leaves, preserve identifiers/heaps/TOAST/serial allocation, and restore
owner-qualified keys/FKs plus parent-only ACL/RLS. Do not REINDEX retained BM25.
Commit the qualified layout with an administrator-only witness. Its immutable
identity/procedure/owner-set and complete retained/parent/child/index inventory
are checked on every resume. Audit the witness schema, columns, defaults,
constraints/indexes, owner/RLS/ACL and dependencies before trusting its data.
Unexpected/missing leaves or indexes refuse; administrator tampering is outside
the security boundary, but unsupported catalog drift is not silently accepted.

After commit, close that connection and verify witness, contents and exact target
catalogs from a fresh connection. Only then return the evidence digest which
allows SQLite INDEX_PENDING. A crash between PostgreSQL commit and SQLite update
leaves MAINTENANCE; resume verifies the witness and advances without copying
again. Source/partition layout without its corresponding witness, contradictory
phase or changed identity/inventory fails closed. Phase-one failure rolls back
the layout and witness together. After its commit only roll-forward or verified
backup restore is available.

The trusted fence dependency is mandatory. It checks an already durable external
maintenance fence and drains its owned inspection resources; exiting its guard
**never re-enables writers or services**. Only a separate matching-release
deployment can lift that fence. No boolean/empty process counter is a fabricated
global worker/provider/session drain proof. Synthetic fixtures own their random
database and explicitly inventory allowed sessions; no worker/provider is run.

Phase two remains disabled in this slice. Retained TEXT heap/index safety after
REINDEX is still under investigation. Later fresh-transaction rebuilding must
commit its witness, then actual fixed TEXT SearchReader qualification must pass
before READY. Churn and independent content controls are permitted only on
disposable synthetic fixtures, and cannot prove arbitrary live-corpus safety.

Tests first: complete three-owner frozen source, content/TOAST/sequence and index
identity preservation, rollback after copy/delete/attach/witness, unknown owners
and witness/catalog drift refusal, idempotent PG-to-SQLite crash-gap resume with
no duplicate copies, gate denial and no phase-two publication. Preserve existing
single-owner migration tests. PostgreSQL and SQLite are not one transaction.
