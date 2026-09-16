# Owner-partitioned Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve native BM25 ranking while preventing another mailbox's contents from changing a user's search results or scores.

**Architecture:** One PostgreSQL database, logical mail tables partitioned by LIST(user_id), and dedicated physical partitions and BM25 indexes for each owner. Parent-only grants and immutable owner RLS remain mandatory. Every admitted owner needs all three searchable relations provisioned; there is no shared default partition.

**Tech Stack:** Python, psycopg, PostgreSQL 16.13, pg_search 0.23.0, pytest.

---

## Authority, evidence, and execution boundaries

The user explicitly selected “partition by user.” This implements the search portion of `docs/superpowers/specs/2026-09-15-full-agent-isolation.md` and supplements `docs/superpowers/plans/2026-09-15-shared-database-full-agents.md`. Qualification evidence: `docs/gateway-ranked-search-qualification.md`; reproducible probe: `deploy/public/qualify_ranked_search.py`.

Work only in `/home/ssilver/development/gmail-search/worktrees/full-agents-20260915`. No live source changes, production DDL, service restart, commit, or deployment until the original release gates pass. Rehearsals use disposable fixture databases on localhost:55440 only. Do not run dependency synchronization against the shared virtualenv.

A global BM25 index changes Alice's ranking when only Bob inserts mail. Qualified physical LIST partitions preserve scores. Parent BM25 indexes are required; child-only indexes do not support the parent query correctly. Physical partitioning complements RLS and the closed SQL compiler; it does not establish complete public application security by itself.

## Task 1: Lock down the schema and migration contract

**Files:** create `docs/owner-partition-migration.md`; inspect `src/gmail_search/store/pg_schema.sql`, `deploy/public/migrate_owner_keys.py`, `deploy/public/migrate_derived_owner_keys.py`, and `src/gmail_search/propositions.py`.

- [ ] Enumerate messages, attachments, propositions, their indexes, incoming/outgoing foreign keys, sequences, policies, privileges, triggers, views and dependent application queries.
- [ ] Specify exact accepted source schemas, target constraints, and an atomic conversion strategy. Reject unknown dependencies instead of dropping them with CASCADE.
- [ ] Resolve global numeric identity uniqueness explicitly: PostgreSQL partitioned unique constraints require user_id. Preserve existing uniqueness enforcement with a reviewed mechanism, or document and qualify a deliberate owner-qualified identity contract across every caller before any conversion. A shared sequence alone is not a uniqueness constraint.
- [ ] Specify bounded hashed partition names, exact single-owner bounds, ownership, no direct reader ACLs, FORCE RLS, and parent-index attachment verification.
- [ ] Document stopped-writer cutover, lock budget, preview-only behavior, rollback after a failed transaction, backup restore after committed conversion, and measured temporary heap/index/WAL headroom. Existing public readers must be drained before changing relation identities.

## Task 2: Implement a synthetic-only conversion rehearsal

**Files:** create `deploy/public/migrate_owner_partitions.py` and `tests/test_owner_partitions.py`.

- [ ] Write failing tests against a disposable exact-version ParadeDB database covering two owners with colliding Gmail IDs and all dependent foreign keys.
- [ ] Run `PYTHONPATH=src /home/ssilver/development/gmail-search/.venv/bin/python -m pytest -q tests/test_owner_partitions.py` with the explicit synthetic fixture DSN; confirm a missing implementation fails.
- [ ] Implement a read-only catalog/data preview and transactionally locked conversion for the reviewed source schema. Initially refuse application outside the disposable fixture environment. Quote identifiers through psycopg.sql and bind owner values.
- [ ] Preserve rows, defaults, identities, constraints, indexes, RLS and intended ACLs. Recheck preflight under locks. Rebind inbound foreign keys explicitly; preserve sequence advancement; refuse unsupported dependency types.
- [ ] Test repeat invocation, injected failure rollback, unknown schema refusal, identity collision enforcement, orphan rejection, owner deletion, insertion after migration, and complete row equality including bytea/text/nulls.
- [ ] Record rehearsal sizes, elapsed time and WAL; do not extrapolate a production capacity claim from tiny fixtures.

## Task 3: Provision and verify each owner's partitions

**Files:** create `src/gmail_search/gateway/partitions.py`, `tests/test_gateway_partitions.py`; modify `src/gmail_search/gateway/admission_provision.py`, and if needed `src/gmail_search/gateway/provision.py` / `database.py`.

- [ ] Write failing tests for provisioning all three partitions, idempotency, concurrent provisioning, foreign/forged partition rejection, missing index rejection and transactional rollback.
- [ ] Implement administrator-only `provision_owner_partitions(conn, owner_id)` and `verify_owner_partitions(conn, owner_id)` under an owner advisory lock. Owner must exist. Names derive from a fixed digest, never an arbitrary identifier from a worker.
- [ ] Inspect actual partition bounds and catalog index ancestry, not names alone. Reject default or mixed-owner partitions, unexpected descendants, writable reader ACLs, and direct reader access to any child.
- [ ] Make partition readiness a prerequisite to publishing invited-account reader/writer credentials. Keep old unpartitioned fixtures explicitly separate; do not silently admit a public searchable owner without partitions.
- [ ] Test direct immutable reader login against parent and child: own rows visible, foreign rows absent, direct child access denied, owner-session-variable spoofing ineffective. Preserve ANALYTICAL_SCHEMA unchanged.

## Task 4: Qualify native search across the entire target schema

**Files:** extend `deploy/public/qualify_ranked_search.py`, `tests/test_owner_partitions.py`; update `docs/gateway-ranked-search-qualification.md` and `docs/gateway-search-integration-plan.md`.

- [ ] Test messages, attachments and propositions independently through partitioned parents with native scoring, LIMIT, count, and colliding identifiers.
- [ ] Record Alice scores and ordering; insert/update/delete only Bob data; assert exact Alice invariance and zero foreign-term results. Include prepared custom/generic plans and repeated auto-prepared queries.
- [ ] Verify EXPLAIN pruning selects Alice's partition, with owner enforcement before LIMIT. Deny all direct child access and exercise intentionally permissive legacy policies alongside restrictive owner policies.
- [ ] Measure representative partition-count planning and provisioning overhead using synthetic small owners; state measured scale and limits. Qualify exact extension grants for the future internal retrieval reader without granting search to arbitrary guest SQL.

## Task 5: Integrate schema lifecycle and recovery gates

**Files:** modify `src/gmail_search/store/pg_schema.sql`, `src/gmail_search/propositions.py`, applicable owner migration compatibility checks, and associated owner ingestion/schema tests only after Tasks 1–4 are reviewed; update `docs/shared-database-implementation-status.md`.

- [ ] Ensure fresh database initialization and existing-database startup cannot silently bypass provisioning, recreate global indexes, or mutate a live unqualified schema. Move partition creation to explicit administrator provisioning.
- [ ] Run existing owner-key, derived-owner, ingestion, reader, writer and admission tests against the candidate. Run the gateway regression suite using its separate synthetic fixture DSN.
- [ ] Obtain independent security/schema review and resolve findings with regression tests.
- [ ] Produce the production preview, capacity estimate, backup/restore rehearsal and stopped-writer cutover checklist. If live measurement is required, perform read-only checks without mail contents or secrets in output.
- [ ] Keep deployment blocked on measured space/WAL/restore feasibility, the complete gateway search implementation, runtime qualification, and two-account browser/consent isolation from the parent plan. Report concrete remaining gates without claiming the public release is complete.

## Review and execution

One independent reviewer checks this plan against the isolation spec before implementation. The schema/migration specialist owns Tasks 1–2; the primary agent owns provisioning integration after that contract is fixed. Independent review follows the migration and provisioning slices. Existing worker-runtime qualification can continue because it shares no partition implementation files. The user has already authorized continuing implementation and appropriate-model delegation; no further execution-choice question is needed.

### Reviewed implementation decisions

The plan reviewer approved the plan without blockers. The deliberate numeric
identity contract is now `(user_id,search_id)` for messages and `(user_id,id)`
for attachments/propositions. Existing values and shared sequences remain;
global uniqueness is no longer claimed. All caller and FK compatibility gates
still apply. Children use the private `gms_mail_partitions` schema and names
`gms_m_`, `gms_a_`, or `gms_p_` plus the first 48 hex SHA-256 owner-ID digits.
Names alone never establish ownership.

The first migration profile attaches existing single-owner heaps and refuses
populated mixed-owner heaps. This avoids a mailbox heap copy, but constraint
index rebuilds, identity/default handling, and WAL/capacity still require proof.
The crawler's ID-based row mutations now use owner-qualified identities;
four reproduced failures pass, with 94 crawler regressions passing. Existing
reader partition behavior passed seven PostgreSQL checks. Naming helpers passed
eight tests. These results do not qualify production conversion.

### Candidate qualification progress

- [x] Independent plan review completed.
- [x] Owner-qualified identity and private child-schema contract recorded.
- [x] Administrator provisioning, catalog inspection and bounded locking implemented.
- [x] Mandatory partition readiness integrated into invitation admission.
- [x] Three-branch native ranking invariance and custom/generic pruning tested.
- [x] Complete canonical schema attach, schema reapply, new-owner CRUD and foreign keys tested.
- [x] Empty-partition retirement implemented with nonempty/dependency refusal and rollback.
- [x] Small-scale setup overhead measured through 25 empty owners.
- [x] Final migration dependency/security review and broad regression evidence recorded (639 broad, 64 existing ownership, 36 migration/lifecycle tests; overlapping suites are not summed).
- [ ] Production-size prerequisite migration, peak WAL/index space and backup restore qualified.
- [ ] Complete public search/runtime/broker/browser release gates from the parent plan.

The partition attach preserves already-qualified heaps. The earlier prerequisite
that introduces numeric search_id may still rewrite existing production message
storage and rebuild its index. Retaining legacy text BM25 keys within owner
partitions is being investigated separately; it is not part of the currently
qualified numeric-key profile.
