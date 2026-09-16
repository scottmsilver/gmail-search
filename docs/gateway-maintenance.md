# Durable controller maintenance gate

The first implementation is a controller metadata gate for **one mail datastore
per Registry SQLite file**. It does not perform PostgreSQL migrations, stop live
services, publish credentials, or establish that a mailbox is ready for release.

## Configuration and APIs

`ReleaseIdentity(store_id, profile, release_epoch)` is frozen. The profile is an
exact `PartitionSchemaProfile`; the epoch is a positive, increasing integer.
Processes pin this configuration at startup. Guests receive none of it.

| API | Contract |
| --- | --- |
| `MaintenanceAdmin.initialize_closed(identity, migration_id=..., owner_set_digest=..., procedure_digest=...)` | Explicit administrator initialization of a new file or upgrade of an existing Registry; returns MAINTENANCE. |
| `begin_maintenance(identity, ..., expected_revision=...)` | Closes a READY release and installs a greater epoch for the same store. |
| `record_index_pending(snapshot)` | Requires matching MAINTENANCE plus trusted external verification. |
| `publish_ready(snapshot)` | Requires matching INDEX_PENDING plus trusted external verification. |
| `status()` | Reads the current identity, state, revision and evidence digests. |
| `GateReader(path, identity).require_ready()` | Read-only, exact identity/readiness check; missing or malformed state denies. |
| `GateReader.publication_guard()` | Shared publication lock followed by a readiness check; installer owns the guard through external work and cleanup. |
| `Registry(..., release_identity=identity)` | Runtime opening without file creation or schema initialization; run authority checks use the current SQLite transaction. |

Administrative transitions hold an exclusive publication lock. Exact matching
operation retries at the immediately resulting state are idempotent. Changed
migration/owner/procedure bindings, stale revisions, skipped phases and epoch
reuse fail. Immutable release rows retain historical bindings; the active gate
has one current revision and state. Digests contain operational evidence only,
never mailbox data or credentials.

`MaintenanceAdmin` defaults to **no verifier**, which denies both transitions
toward readiness. A separately trusted verifier receives
`(snapshot, target_state, deadline=monotonic_deadline)` and returns a SHA-256
evidence digest after checking actual external state. It executes outside any
SQLite transaction, under the controller publication lock. The controller
rechecks the whole snapshot before committing and rejects late or invalid
results. The verifier must bound and drain its own work; synchronous Python
cannot forcibly stop an arbitrary blocking callback. A digest alone proves
nothing about PostgreSQL. No production verifier is included.

## Revocation and durability

Initialization of an existing Registry and each maintenance closure atomically:

- Cancel every active run and revoke every capability in the file.
- Clear conversation writers and increment their fences.
- Preserve budgets, outstanding reservations and worker resource bindings.

New gated runs carry their release epoch. Authorization, capability issuance,
heartbeat and provider reservation check the current gate in the same SQLite
transaction as their existing operation. Reopening the same file READY cannot
revive old tokens or old request-key runs. Updated Registry instances opened
without `release_identity` also deny run authority once that file has a gate;
the default argument cannot bypass an enabled gate. Provider settlement,
cancellation and worker reconciliation remain available while closed.

The supported storage profile is a private local directory (0700), regular file
(0600), SQLite DELETE rollback journal, and FULL synchronous transactions.
Administrator initialization fsyncs the file and parent directory entries for
the database and publication lock, including retries after failed creation.
Gated readers open with `mode=ro`; existing runtime authority checks reject
journal-mode drift. SQLite lock waits and publication-lock acquisition are
bounded at two seconds. Acquisition failure is not a maintenance acknowledgment.
External publication duration and cleanup still need the caller's own bounds.

Readers never recursively acquire a SQLite writer transaction. This includes a
readiness callback called by Registry's owner-activation callback while Registry
already holds its writer transaction. Runtime Registry construction performs no
DDL. Other components' existing initialization, including worker-table setup,
has not been refactored into a deployment initializer in this slice.

## Required composition before migration

The publication guard is a primitive, not yet wired into `AdmissionProvisioner`
or a credential vault. A future installer must hold it from readiness checking
through PostgreSQL role commit, vault publication and cleanup. Successful
maintenance closure then follows prior publishers. Credentials must also carry
the release identity and require readiness whenever resolved. Already issued
database credentials and sessions need explicit deployment fencing.

Gate-unaware old binaries must be stopped with restart paths disabled. Prevent
their database reconnection and terminate existing sessions; changing NOLOGIN
alone does not stop existing connections. Drain providers, database queries,
native index leases, response sends and every worker namespace using actual
cleanup/inventory acknowledgments. Gate closure alone is not a drain proof.
Failed stops retain their resource bindings. Restart repeats reconciliation;
process-local counters and a persisted 'drained' flag are insufficient.

The approved staged migration still needs administrator-only PostgreSQL witness
records committed with each phase, fresh-connection validation, fresh-transaction
BM25 rebuilding, and restricted-reader qualification before READY. SQLite stays
closed in cross-database commit gaps. After phase-one commit, recovery is explicit
roll-forward or verified backup restore. No distributed atomicity is claimed.
Restoring an earlier database/controller file requires external fencing and a
new configured epoch: a file cannot detect rollback of its own entire history.
Private ancestors, administrator code and filesystem semantics remain trusted.

## Qualification

Synthetic SQLite tests exercise old-token revocation after reopening, atomic
failure and abrupt-exit recovery, publisher/admission races, idempotent retries,
stale or forged bindings, verifier failure/CAS/deadline, creation durability,
journal drift, read-only opening and failed worker-stop retention. Existing
capability, worker, provider and embedding/reranker lifecycle tests are included
in the regression run. No real mailbox, provider call or deployment is involved.

The implementation handoff passed 186 combined tests. Root independently
reviewed this first-slice boundary and reran maintenance, independent crash,
capability and worker tests: **80 passed in 1.55 seconds**, without skips.

See the [approved first-slice plan](superpowers/plans/2026-09-15-durable-maintenance-gate.md)
and [caller release proposal](superpowers/plans/2026-09-15-text-partition-caller-release.md).
