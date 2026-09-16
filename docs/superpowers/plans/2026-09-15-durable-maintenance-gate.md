# Durable maintenance gate — approved first slice

One mail datastore uses one private controller Registry SQLite file. Processes
pin an immutable `(store_id, schema_profile, release_epoch)` at startup. Missing,
corrupt, mismatched or closed state denies access. Explicit administrator
initialization/upgrading creates the gate; gated runtime opening creates neither
files nor schema. Updated legacy-default Registry instances also deny authority
when the file contains an enabled gate.

The administrator initializes `MAINTENANCE`, then advances the same release
through `INDEX_PENDING` to `READY`. A later maintenance cycle requires a greater
epoch. Immutable release metadata binds the migration identifier, expected owner
set and procedure. Transitions use revision compare-and-swap and exact retry
bindings. Initialization of an existing Registry and maintenance closure each
atomically cancel all active runs, revoke all capabilities and advance writer
fences. Reservations remain available for settlement. Old run identities and
tokens never become live again when the same file reopens.

Registry authorization checks use the current SQLite transaction. Independent
readiness callbacks use read-only SELECTs, avoiding recursive writer locks.
Publication holds a shared private file lock through credential installation;
controller transitions take the exclusive lock before SQLite transactions.
Acquisition is bounded. External verification runs outside SQLite transactions,
then the controller rechecks the revision before committing. Missing verification
denies transitions toward readiness. A digest records trusted verification; it
does not itself establish a PostgreSQL fact.

## Deferred composition and required evidence

No PostgreSQL witness DDL, migration orchestration, legacy caller integration or
deployment action is included. Future phase one must commit an administrator-only
identity-bound witness with redistribution. A fresh connection verifies that
witness and exact catalogs before SQLite advances to INDEX_PENDING. Fresh phase
two rebuilds retained BM25 indexes and updates its witness in the same PostgreSQL
commit. Fresh restricted-reader qualification under fenced writers precedes READY.
Crashes between either PostgreSQL commit and the SQLite update leave access
closed; restart rechecks committed witnesses rather than recopying rows. After
phase-one commit, recovery is roll-forward or verified backup restore.

Before phase one, deployment must disable gate-unaware processes and their
restarts, prevent database reconnection, terminate old sessions, and withhold
credentials. Drain providers, queries, native index leases and response sends;
reconcile every worker namespace against actual inventory. Cleanup failures
remain closed. A process-local counter or persisted boolean is not a global
drain proof. Restart repeats reconciliation. Old binaries cannot be contained by
new Python checks. Restoring older SQLite/PostgreSQL state requires deployment
fencing and a newly configured epoch; the file cannot detect rollback of itself.

## First-slice verification

Tests cover explicit initialization, missing/runtime-no-create behavior, pinned
identity, old-token revocation across reopening, legacy-default denial, atomic
rollback, concurrent admission/closure and publication, stale revisions, failed
verification, read-only callbacks under writer transactions, and cleanup/billing
after closure. PostgreSQL crash-gap and actual deployment fencing tests belong to
the later composition. No first-slice success claims production readiness.
