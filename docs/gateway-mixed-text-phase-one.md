# Synthetic mixed-owner TEXT migration: phase one

`deploy/public/migrate_mixed_text_owner_partitions.py` implements **phase one
only**. It can leave the Registry in INDEX_PENDING. It cannot rebuild BM25,
publish credentials, or mark READY. Actual retained-index reader qualification
is blocked; see the [preserved evidence](qualification/retained-reader-maintenance-blocker.md).

## Fixed scope

Every opened administrator connection must identify both its configured host and
actual peer as `127.0.0.1:55440`, with a database name beginning
`gms_owner_partitions_test_`. It also requires PostgreSQL16/pg_search0.23.0 and a
direct database-owning superuser without role members. There is no live override.
Database name/OID and cluster system identifier are pinned when capturing the
plan and checked on every new connection.

The source is the same frozen legacy TEXT schema accepted by the
[single-owner mechanism](gateway-text-owner-partition-migration.md). The trusted
plan supplies a dominant owner and an exact, sorted set of 2–16 existing owners.
No owner is inferred or rewritten; null, unlisted and orphan row ownership is
refused. Expected owners with no mail receive empty canonical leaves. Existing
IDs remain unchanged; messages do not acquire a numeric search identity.

`MixedTextPlan` is frozen and the controller's `.plan` property cannot be
reassigned. Its binding includes store/profile/epoch, operation, owner set,
dominant owner and database identity. The procedure digest hashes the actual
phase-one module and its single-owner mechanics module, separated by a zero
byte. Editing either invalidates an old binding; this is not a signature or a
complete Python dependency/environment attestation. Runtime code and
administrator-owned configuration remain trusted.

## APIs and ownership

1. `capture_plan(conn, identity=..., migration_id=..., dominant_owner=...,
   expected_owners=...)` captures the explicitly selected disposable database.
2. Initialize the private Registry with `MaintenanceAdmin.initialize_closed`,
   using the plan's identity, operation, owner-set digest and procedure digest.
3. Construct `MixedTextPhaseOne(connection_factory, registry_path, plan,
   fence=trusted_fence)` and call `.advance(snapshot)`.

There is no default fence. The supplied `hold(plan, deadline=...)` context must
inspect an already durable external maintenance fence and own its inspection
cleanup. Exiting it **must never reopen writers or services**. A separate
matching-release deployment owns reopening. Tests use a clearly labeled fence
double with a real disposable-database session check; they do not establish a
deployment-wide drain or reboot fence.

Factories supply newly opened, idle, autocommit connections with bounded
connection establishment. The controller owns their transactions and closure,
rejects reused connection objects/backend IDs and retains at most 128 inspected
connection references per instance. An uncertain connection-close or fence
cleanup acknowledgment poisons that instance; it cannot silently retry. A new
controller is not a substitute for reconciling uncertain external resources.

The outer MaintenanceAdmin transition owns the exclusive publication lock.
Its verifier performs PostgreSQL work outside SQLite transactions and never
reacquires that lock. A repeated INDEX_PENDING inspection acquires the same lock
directly without nesting another transition. Every SQL execution receives the
remaining phase deadline as its statement timeout; checkpoints, transaction
completion, connections and fence exit also check the aggregate 30-second
deadline. Cleanup must finish before any positive acknowledgment. This does not
turn an arbitrary blocking connection factory or fence into a hard-killable
operation, or establish that 30 seconds fits production-scale migration work.

## Phase-one transaction

The migration takes the established migration, sorted owner, catalog and sorted
relation locks. Existing qualified sequence-locking mechanics refresh allocation
state while holding transaction-long serial locks. All source/dependency checks
precede layout changes.

The original three heaps move into the private schema as the dominant owner's
future leaves. Canonical owner-qualified TEXT parents and minority leaves are
created. Each minority copy uses explicit validated columns and original IDs.
Bidirectional `EXCEPT ALL` inside PostgreSQL compares copied contents before
deletion; inserted/deleted counts must match. The dominant heaps are attached
only after their foreign rows have matching destinations. Parent-only grants,
ENABLE/FORCE RLS and owner-qualified keys/FKs are restored. Shared serial
identities and safe allocation advancement use the existing transactional path.

The single-owner module now exposes private reusable preflight/layout mechanics.
Its public API still admits one owner only, copies no mailbox rows and performs
no REINDEX. Existing single-owner tests remain part of the required regression
run.

## Witness and recovery

The fixed administrator-only `gms_migration_control.text_phase_one` table commits
in the same PostgreSQL transaction as the new layout. Its singleton row contains
LAYOUT_COMMITTED, the canonical plan binding, and bounded canonical inventories:

- Original retained heap/TOAST and nonconstraint-index identities/files, owner
  row counts and locked/refreshed serial allocation state.
- Exact target parents/leaves, index definitions/ancestry/files, constraints,
  row counts and sequence identity/definition/allocation state.

Phase one checks retained identities and data counts against their original
inventory. These checks make **no promise** about index identity preservation in
a future phase two. Extra/missing owner leaves or indexes, altered allocation,
or a mismatched stored inventory refuse resume.

Before reading its data, the witness's schema, owner/private grants, columns,
defaults/collation, RLS, constraints, index options/physical shape, triggers,
publications and stored dependencies are audited. No public schema, table or
column grant is added. Administrator tampering is outside the trust boundary;
unsupported catalog drift still fails closed.

After PostgreSQL commits, its connection closes and a fresh connection verifies
the committed witness and exact target. Only its verified digest can advance
SQLite to INDEX_PENDING. An interrupted PostgreSQL-to-SQLite handoff leaves
MAINTENANCE; a new controller recognizes the exact committed witness and verifies
it without copying or deleting rows again. An already partitioned target with no
matching witness is refused. PostgreSQL and SQLite are not one transaction.

Before the phase-one commit, exceptions roll back copies, deletions, layout and
witness together. Afterward, recovery is roll-forward or verified backup restore.
Old runs/tokens remain revoked in the same Registry. No readiness transition is
available through this controller, including after an otherwise successful
phase-one rehearsal.

## Synthetic evidence

The fixture uses three owners with 4/2/1 rows per searchable table, large TOAST
values in both retained and copied mail, NULL extracted text, bytea values and
owner-qualified derived references. Tests compare contents and physical
identities, exercise rollback checkpoints and the committed handoff gap, reject
witness/leaf/index/allocation drift, and cover fresh connection, deadline and
uncertain-cleanup behavior. Fixture mutations and comparisons occur only in
disposable synthetic databases. They do not qualify arbitrary live mail,
storage/WAL budgets, backup restoration, production fencing or retained-index
search after rebuilding.

Root independently reviewed and approved the synthetic phase-one boundary. The
stable combined suite passed **56 tests in 27.31 seconds**, without skips:
30 phase-one tests, 25 unchanged single-owner tests, and one independent actual
subprocess exit after PostgreSQL commit but before SQLite advancement. That
restart test confirms the committed witness/data survive, the Registry remains
MAINTENANCE, and a new controller reaches INDEX_PENDING without repeating any
copy/delete checkpoint. Scoped Ruff and diff whitespace checks passed.

See the [approved plan](superpowers/plans/2026-09-15-staged-text-witness.md).
