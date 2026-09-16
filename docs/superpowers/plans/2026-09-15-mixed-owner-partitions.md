# Mixed-owner TEXT partition migration plan

## Decision and evidence

Keep one PostgreSQL database and partition messages, attachments and propositions
by `user_id`. Retain TEXT message IDs and owner-qualified keys. Embeddings and
the existing derived metadata tables keep their shared storage and acquire the
required owner-qualified foreign keys; they do not need copying for this step.

The live preflight found three owners, so a whole-table single-owner attach is
ineligible. See [the read-only report](../../live-partition-preflight-2026-09-15.md).
Preserving the dominant owner's heaps while relocating minority rows can avoid
copying the largest mailbox. Row counts alone do not establish its byte cost.

The independent synthetic mixed-history probe found that DELETE, ANALYZE and
ordinary VACUUM do not eliminate foreign BM25 score influence. A leaf REINDEX
restored the clean owner-only scores. Therefore **rebuild retained BM25 indexes**
after redistribution; do not infer historical index purity from current rows or
promise an index-free migration. This still avoids adding `search_id` or
rewriting the dominant message heap and its large TOAST storage.

## Implementation boundary

First implement and review a synthetic-only rehearsal using the qualified
PostgreSQL16/pg_search0.23.0 fixture. Keep the actual-address, port and disposable
database guard. No live apply mode, startup migration, service restart, provider
call or mailbox export is part of that implementation.

Accept only the exact legacy source variants independently qualified by the
direct TEXT migration. The trusted caller supplies an explicit dominant owner
and expected owner set. Validate every observed owner against that set and
existing users; refuse drift, null/orphan ownership, unknown dependencies,
partial layouts, unsupported indexes or mismatched permissions. Do not silently
choose a different owner or fall back to a full copy.

## Staged maintenance sequence to qualify

The original single-transaction proposal failed qualification: minority DELETE,
attachment and REINDEX in the same transaction left foreign-influenced scores
after commit, including on fresh backends. It must not be implemented as a safe
atomic cutover. The following staged replacement is under review and requires
new evidence before implementation approval.

Access remains closed throughout all stages. A durable maintenance/readiness
record must survive process failure and reboot, prevent admission and credential
publication, and make startup refuse an incomplete migration. Catalog shape
alone is insufficient proof that an index has a clean owner-only history.

1. Acquire migration/owner/catalog locks in the established order, then bounded
   exclusive locks on every affected table and referenced identity/topic table.
   After validating canonical serial settings, acquire their transaction-held
   locks using supported `ALTER SEQUENCE ... CACHE 1`, then refresh allocation
   state under those locks. Do not use `LOCK TABLE` on a sequence.
   Writers/readers must be drained by the later release
   procedure. Run the complete source catalog, dependency and data preflight.
2. Inventory and remove only the affected, known foreign keys. Preserve the
   underlying row data, existing numeric IDs, sequence state and unaffected
   constraints. Promote dependent metadata keys to include the owner.
3. Move the three original heaps into the private partition schema and create
   their qualified logical parents. For each other expected owner, create a
   private leaf and copy only that owner's rows with explicit canonical columns.
   Do not regenerate identifiers or reinterpret stored content.
4. Verify copied row counts and ownership under the same transaction and locks,
   then remove those rows from the retained original heaps. Refuse unexpected
   triggers/rules/cascades before this operation; no row is discarded without
   its corresponding destination copy. Qualify content preservation explicitly
   in synthetic tests, including large TOAST values and colliding identifiers.
5. Attach every leaf under its exact owner bound and install owner-qualified
   keys/indexes. Build new leaves' BM25 indexes from their owner-only rows.
   Retained BM25 indexes remain unqualified at this stage; no search credentials
   or traffic may be released. Preserve dominant heaps/TOAST and existing IDs.
6. Rehome attachment/proposition sequences and advance them transactionally;
   do not add a message sequence. Recreate the exact owner-qualified foreign
   keys once all destination leaves are attached. Preserve parent ACL/RLS and
   remove direct leaf grants; qualify stricter reader readiness separately.
7. Validate the structural target, all owner bounds, keys, references and private
   leaves as administrator. Commit phase 1 with durable state recording that
   index rebuilding is still required. Any failure before this commit rolls
   back copying, deletion, key changes, sequences and catalog moves together.
8. In a **fresh transaction after phase 1 commits**, explicitly REINDEX retained
   BM25 leaf indexes. Qualify fresh-backend score equivalence to clean synthetic
   controls for this exact sequence. A phase-2 rollback/failure must retain the
   closed maintenance state, preserve owner-isolated rows and allow a validated
   retry. Do not reopen using the stale indexes.
9. After rebuilding commits, verify target catalogs and actual restricted-reader
   operations while client credentials remain withheld. Only the complete,
   reviewed procedure may mark readiness and permit the matching application
   release to publish credentials. Resume must check recorded identity/state
   against actual catalogs, not trust a phase label alone.

Refactor shared migration mechanics only where this preserves the independently
reviewed numeric and single-owner behavior. Do not duplicate a large loosely
checked migration or call the numeric key prerequisite. A production rerun must
recognize the exact qualified phase and safely resume or fail closed. After the
first commit, recovery is an explicit roll-forward or verified backup restore;
it is no longer a single-transaction rollback to the original layout.

## Required synthetic evidence

- Exact live-observed legacy shape with three owners and skewed row counts.
- All three owners retain their exact message/attachment/proposition contents
  and derived references; IDs can subsequently collide safely across owners.
- Dominant heap/TOAST relation identities stay unchanged; minority storage is
  separate. Record which index identities/files change and why.
- Rank/score equality to fresh owner-only controls after historically foreign
  rows are relocated, including restricted candidates under the explicitly
  qualified runtime query profile. Pinned generic plans fail after owner
  maintenance and are not an acceptable production path; custom/unprepared
  planning is undergoing actual-reader qualification.
- Phase-1 fault injection after copy, delete, attach and FK restoration restores
  original data, indexes, keys, ACLs and sequence state. No orphaned copies.
- Phase-2 rollback/restart leaves access closed; retry rebuilds clean indexes
  without copying or losing the already-redistributed rows.
- A restart/reboot or premature provision attempt cannot bypass durable
  maintenance state; no runtime credentials are published for stale indexes.
- Wrong owner set, unknown dependencies, mismatched owners, direct leaf grants,
  index drift and cancellation cannot publish a successful migration.
- Restricted SQL and fixed search readers expose only the bound owner's rows.
- Existing numeric and single-owner migration suites remain passing.

## Production gates remain separate

Measure representative minority copy bytes, replacement B-trees/BM25, WAL, peak
disk and lock duration; verify a backup restore and rehearse failure recovery.
Investigate the separately observed synthetic pg_search maintenance assertion
before qualifying repeated maintenance on the retained-index path.
Release the matching TEXT-aware callers, ingestion and credentials together.
The current 112 GB available-space observation is not a migration budget.
Public invitation/authentication and full runtime qualification must also pass
before enabling multi-user public access.
