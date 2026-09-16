# Trusted TEXT caller and schema release — proposed, unimplemented

**Status:** Caller and fresh-installer design only. The separate Registry gate's
bulk revocation and restart behavior are reviewed; actual process/session fencing,
PostgreSQL witnesses and caller composition remain incomplete. This document does
not establish release readiness.

**Goal:** Release explicitly selected owner-partitioned TEXT storage without changing existing legacy-default APIs or exposing administrator/database controls to guests.

## Configuration and connection contract

Create `store/profiles.py` with separate legacy compatibility mode (unchanged default) and an immutable `PartitionedStoreConfig(schema_profile, store_id, release_epoch, readiness_gate, dsn)`. Do not equate historical numeric nonpartitioned compatibility with the qualified numeric partition profile. Bind the trusted configuration once at process startup; changed profile/epoch requires a restart. Neither schema inspection nor a failed SQL statement selects a profile.

Preserve `get_connection(db_path)`, `_PgConnWrapper` mapping/positional rows, cursor behavior, commit/rollback and caller-owned transaction/savepoint semantics. Add optional internal configuration plumbing without changing the roughly 60 legacy call sites. Every qualified connection verifies its configured catalog before returning, carries the immutable profile, and closes on failure with a fixed `StoreProfileUnavailable` error. Checks on already-owned connections must preserve transaction ownership and settings. Never weaken canonical public/private-schema checks for old isolated-schema tests; those tests retain legacy mode.

## Durable release readiness

The first Registry maintenance-gate primitive is now implemented and independently
reviewed; see [its scope and remaining composition](../../gateway-maintenance.md).
This does not authorize a caller to infer readiness from catalogs alone. The
staged PostgreSQL witness and retained-table search qualification remain pending.

Matching catalog shape does not prove historical BM25 corpus purity. The same-transaction DELETE/ATTACH/REINDEX probe failed even after commit, so the earlier atomic mixed-owner migration plan is ineligible.

A separate administrator-owned controller record binds `(store_id, schema_profile, release_epoch)` to `MAINTENANCE`, `INDEX_PENDING`, or `READY`. Missing/corrupt/unreadable records and epoch mismatches fail closed. Before redistribution, durably close readiness, revoke old capabilities/credentials as required, drain active readers/writers, and prevent new starts. Stage-one committed redistribution remains `INDEX_PENDING`. Fresh-transaction index rebuilding, exact clean controls and full catalog/reader qualification are prerequisites for READY.

Old processes keep their old pinned epoch and cannot resume on a new release. Old tokens must be durably revoked before READY can reopen; merely wrapping owner activation with a boolean gate would otherwise revive them. Bulk revocation ordering, enforcement against existing SQL sessions, controller crash recovery, stage-two retry and explicit process restart remain required design gates. No readiness marker pretends to prove corpus purity by itself.

Legacy `init_db`, `get_connection` and expensive search/extraction entry points check readiness. Public capability authorization and reader/index admission consume the same trusted gate through controller composition. Guests cannot select a gate/profile/epoch or obtain additional mail database grants.

## Caller changes

- `store/queries.py`: choose only the fixed configured message key; use qualified logical parents and safely quoted immutable owner literals on message and attachment BM25 branches. Bind candidate IDs/query/limits. Qualified errors propagate rather than becoming successful empty/partial lexical scores.
- `search/engine.py`: qualify before ScaNN construction and before every search embedding/index operation. `search_threads` currently embeds before its preliminary connection: move qualification earlier, close the reader before provider calls, and preserve scoring/filter behavior.
- `propositions.py`: qualify at backfill, extraction and embedding entry points outside broad per-message catch blocks. Qualified `ensure_table` verifies and performs no DDL. Select the configured message BM25 key in backfill.
- Require explicit trusted owner selection for qualified mailbox operations; legacy bootstrap conveniences remain limited to compatibility mode. A public model/capability cannot select a database owner.
- `store/db.py`, `server.py`, `agents/analyst.py`, `agents/mcp_tools_server.py`: render matching schema examples/performance hints from the selected profile. Public QueryGateway tools retain their restricted compiler and grants; do not advertise unavailable BM25 functions through that tool.

## Fresh schema and admission

Build a dedicated fresh TEXT schema/installer, separate from historical `pg_schema.sql`. It accepts an empty application database only, creates canonical TEXT partition parents and owner-qualified dependents, and never invokes the numeric prerequisite or creates historical shared analyst roles/password grants. Existing databases use verification-only qualified `init_db`; no startup migration/index rebuild or silent lock-timeout success.

Add explicit profile/readiness parameters to `AdmissionProvisioner`. TEXT admission creates the owner, private leaves and fixed SQL/search/application roles atomically, then publishes a complete credential bundle to the trusted vault. Preserve the existing legacy two-credential callback. Profile/epoch metadata accompanies the bundle; a partial credential installation denies admission.

Application writer grants remain unchanged: that role handles application state, not mailbox ingestion. Trusted ingestion uses qualified owner-aware connections and existing composite writes. Public gateways use fixed services/SearchReader; legacy administrator/private callers remain a separate trust boundary.

## Proposed bounded implementation sequence

1. `store/profiles.py`, `store/db.py`, startup wiring in `cli.py`/`server.py`, reviewed readiness backend/composition: tests for immutable selection, wrong profile/epoch, missing gate, connection cleanup and transaction preservation.
2. `store/queries.py`, `search/engine.py`, `propositions.py`: actual numeric/TEXT fixed-key queries, candidate filters/owner collisions, profile errors escaping broad catches, and no provider/index/extraction work before qualification. Incorporate the separately qualified custom-plan/unprepared pg_search runtime policy; historical generic probes do not establish maintenance safety.
3. Dedicated fresh installer/schema and admission credential bundle: fresh install, existing-database refusal, exact grants, both-owner CRUD/FKs and atomic publication tests.
4. Profile-generated documentation/examples and unchanged legacy-default tests.
5. Actual migrated-schema ingestion/summary/proposition/attachment/deletion qualification and crash/restart tests after stage-one commit. Every qualified caller remains closed until index qualification completes.

## Connection review constraints

The existing compatibility wrapper accepts both mapping and positional row
access; catalog inspectors expect tuple rows. Qualification must use an explicit
tuple-row cursor or a separate owned inspection connection, without replacing
the caller's row factory or committing its transaction. Returning a qualified
new connection must leave it idle in the transaction mode existing callers expect.
Any setup failure closes it; DSNs must be excluded from configuration repr and
fixed public errors. Qualified mode must never fall back to `_pg_dsn()` or the
default environment DSN after failure.

Existing partition administrator helpers require database-owning administrator
authority. Do not silently grant that authority to a new ingestion or application
role merely to reuse a helper. Keep initial legacy/private administrator caller
support explicit; qualify a separate read-only catalog inspector before admitting
other role types. The application writer remains limited to application state.

Bind configuration once before serving or starting background jobs, and deny
rebindings in the same process. A matching gate check at connection creation is
not a lock on later queries: deployment fencing must terminate old sessions and
prevent reconnects before migration. Until those callers and fencing are composed,
the new mode remains unavailable to the live process.

No production migration, service restart, mailbox export, provider call, public grant expansion or automatic full-copy fallback is authorized. Storage/index cost and restore/cutover approval remain separate.
