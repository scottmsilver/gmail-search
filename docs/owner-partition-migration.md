# Owner-partition migration contract

2026-09-15. The user selected partitioning by owner. This document fixes the
contract for the synthetic conversion rehearsal; it does not authorize production
DDL. The narrow synthetic profile below has a reproducible attach rehearsal; it
does not qualify production scale, source variants, or the entire preceding
identity migration. The root implementation agent
approved the owner-qualified numeric identities and single-owner attach profile
below. Native ranking evidence is in
[`gateway-ranked-search-qualification.md`](gateway-ranked-search-qualification.md).

## 1. Identity contract

Keep existing values and model-facing scalar IDs. Their authoritative identity
is the authenticated owner together with the ID, including numeric IDs:

| Parent | Required target keys | BM25 key |
| --- | --- | --- |
| `public.messages` | PK `(user_id,id)`; UNIQUE `(user_id,search_id)` | `search_id` |
| `public.attachments` | PK `(user_id,id)`; UNIQUE `(user_id,message_id,filename)`; UNIQUE `(user_id,message_id,id)` | `id` |
| `public.propositions` | PK `(user_id,id)` | `id` |

All three parents use `PARTITION BY LIST(user_id)`. No ID is renumbered. Existing
shared allocation sequences continue generating normal new IDs. **A shared
sequence is not a global uniqueness constraint.** Explicit equal numeric IDs in
different owner partitions are permitted under this contract; duplicate IDs
within an owner must fail. Every caller, cache, join and reference must carry the
owner explicitly or execute through a verified immutable owner login.

This is a deliberate change from global `UNIQUE(messages.search_id)` and the
global primary keys on attachments/propositions. PostgreSQL partitioned unique
keys must include the partition key; this plan does not simulate global uniqueness
with a privileged trigger, an identity ledger, or additional mailbox storage.
[PostgreSQL partition constraints](https://www.postgresql.org/docs/16/ddl-partitioning.html#DDL-PARTITIONING-DECLARATIVE-LIMITATIONS)

The analytical schema and tool argument types stay unchanged. A browser or
capability request already has an owner; a numeric argument selects a row within
that owner. No request may supply a different owner or child relation name.

## 2. Accepted source and dependency inventory

Initial apply accepts only the qualified post-owner-key, post-derived-key schema:

- Ordinary, permanent `public.messages`, `attachments`, and `propositions` heaps;
  no existing inheritance/partitions, foreign tables, materialized views, or
  partially converted mixture. The already-qualified target is an idempotent
  no-op only after target verification.
- Nonnull owner keys and the exact source identity constraints described by
  `migrate_owner_keys.py:TARGET_KEYS` and the completed
  `migrate_derived_owner_keys.py` target. Messages has bigint BY DEFAULT identity
  `search_id`; attachment/proposition IDs have their existing bigint sequence
  defaults. Unqualified legacy Gmail-ID keys are refused.
- All populated source heaps contain the same explicitly supplied existing
  `users.id`. Empty heaps may be attached to that specified owner. Null/orphan
  owners, orphan message references and populated multiple-owner heaps are refused.
- The expected canonical columns/defaults/types from `store/pg_schema.sql`,
  including additive attachment status/crawl fields and message
  `crawl_blocked_reason`, are preserved. Unknown generated/domain/custom-type
  columns, arbitrary default functions, or unsupported source variants require
  a separate reviewed profile rather than a generic DDL replay.
- Catalog inventory must cover every incoming and outgoing FK, unique/check
  constraint, index, sequence/default dependency, RLS policy, table/column ACL,
  comment, trigger, rule, view, publication and other dependency. Unknown
  dependencies cause preview/apply refusal. No `DROP ... CASCADE` is permitted.

The current repository's incoming FKs are:

| Referencing relation / columns | Referenced relation / columns | ON DELETE |
| --- | --- | --- |
| attachments `(user_id,message_id)` | messages `(user_id,id)` | NO ACTION |
| embeddings `(user_id,message_id)` | messages `(user_id,id)` | NO ACTION |
| message_topics `(user_id,message_id)` | messages `(user_id,id)` | NO ACTION |
| message_summaries `(user_id,message_id)` | messages `(user_id,id)` | NO ACTION |
| summary_failures `(user_id,message_id)` | messages `(user_id,id)` | CASCADE |
| propositions `(user_id,message_id)` | messages `(user_id,id)` | CASCADE |
| prop_processed `(user_id,message_id)` | messages `(user_id,id)` | CASCADE |
| embeddings `(user_id,message_id,attachment_id)` | attachments `(user_id,message_id,id)` | NO ACTION |

No incoming FK to propositions is defined. Messages and attachments also have
`user_id -> users(id) ON DELETE CASCADE`. Propositions reaches its owner through
its owner-qualified message FK; current canonical DDL has no separate users FK.
The derived migration tolerates an existing validated users FK; a rehearsal
profile must either preserve that explicitly supported variant or refuse it.
Other nonparticipating relations and their owner/topic FKs remain in place.

Every incoming FK must be rebound to the **new parent OID**, not just retain a
matching printed table name. Preserve its column order, update/delete action,
match mode, deferrability and validation semantics. Unsupported variants are
refused. Rebinding validated FKs can scan referencing tables even though their
heaps are not copied; embeddings must be included in this cost estimate.

### Existing indexes and sequences

Preserve the configured BM25 definitions, not merely their key names:

- `messages_bm25_idx`: `(search_id,id,subject,body_text,from_addr,to_addr)`.
- `attachments_bm25_idx`: `(id,filename,extracted_text)`.
- `props_bm25_idx`: `(id,text)`.

The initial implementation accepts only default operator classes, canonical
column collations, and exactly the existing `key_field` option. Custom tokenizer
options, expression/include indexes, noncanonical predicates, and heap reloptions
are refused. Supported tablespaces and index comments are retained. The source must have one valid, ready, ordinary BM25
index per relation; unsupported expressions, partial BM25 definitions or index
variants require refusal. Parent/child BM25 attachment must be proved on the
pinned 0.23.0 image, since a second BM25 index on a leaf is forbidden.

Existing B-tree indexes include message history/date/inbox indexes; attachment
unfetched/crawl-lane/owner indexes; proposition owner/message indexes; and the
source identity indexes. Compatible indexes can attach to matching parent index
metadata. Changed composite keys can require new/rebuilt B-trees. Their cost is
not a mailbox heap copy, but still requires measured temporary space and WAL.

Snapshot sequence OIDs, owner dependencies, schema/name, defaults, identity mode,
last value/is_called, min/max/increment/cache/cycle, and external references.
Normal parent INSERT must continue allocating past existing IDs. Serial defaults
can retain the same sequence while ownership is moved to the parent. Identity
sequence reassociation is a distinct PostgreSQL operation: if preserving BY
DEFAULT identity requires transactional sequence recreation, preserve its public
name, parameters and safe advancement, refuse unknown external dependencies, and
prove rollback. Do not claim sequence OID preservation until the rehearsal proves
it. The qualified profile accepts only canonical public allocator names and
default positive bigint parameters (start/increment/min/cache 1, max signed
bigint, no cycle). Nonowner sequence grants and external catalog dependencies
are refused. Old serial objects use transactional `ALTER SEQUENCE RESTART` to
advance beyond both reserved sequence position and stored maximum; rollback
restores their original state. `setval` is used only on the newly created
identity sequence, which rollback discards. Sequence comments are preserved.

## 3. Parent and child layout

Canonical helper contract is `gateway.partitions.PARTITION_SCHEMA` and
`partition_name(table,owner_id)`:

```text
schema: gms_mail_partitions
digest: sha256(owner_id.encode('utf-8')).hexdigest()[:48]
messages child:     gms_m_<digest>
attachments child:  gms_a_<digest>
propositions child: gms_p_<digest>
```

Names are 54 ASCII bytes. A name/hash is only a lookup hint: verify exact parent,
schema, trusted object owner, relation kind, single literal LIST bound equal to
the full owner ID, and any canonical binding metadata. Reject collisions,
unrelated preexisting objects, default bounds, multiple values, null bounds,
subpartitions and unexpected descendants. The dedicated schema belongs to the
trusted database owner, with no PUBLIC or runtime role privileges.

Both parent and children must have ENABLE/FORCE RLS. Copy the reviewed source
policies to the new parent, including restrictive immutable-owner reader policies;
retain compatible child checks/policies as defense in depth. Preserve intended
table and column privileges on the **parent only**. Remove nonowner direct child
privileges, including column ACLs and inherited PUBLIC exposure, and do not grant
runtime roles USAGE on the private child schema. Readers/writers own no child.
Role memberships and default privileges cannot create an indirect exception.

Parent index names remain the canonical public names above. Each leaf BM25 index
must be valid/ready, have the exact expected definition, and have catalog ancestry
to the correct partitioned parent index. Name matching alone is insufficient.
One logical parent query must prune to the authenticated owner's leaf and score
against its index. Child-only indexes without a parent index failed qualification.

The existing `provision.py` already accepts parent `relkind='p'` and audits direct
child ACLs across schemas. Root's preliminary tests demonstrate parent-only
analytical reader grants; this migration must preserve that boundary, not loosen
`ANALYTICAL_SCHEMA` to accommodate children.

## 4. Single-owner attach conversion

PostgreSQL cannot change an ordinary heap into a partitioned parent in place. It
can attach that heap under a newly created parent. The intended fast path retains
the mailbox heap and its TOAST storage while changing the logical parent OID.
[PostgreSQL attaching existing tables](https://www.postgresql.org/docs/16/ddl-partitioning.html#DDL-PARTITIONING-DECLARATIVE-MAINTENANCE)

Initial apply is confined to a uniquely named disposable fixture DB on
`127.0.0.1:55440`, using a direct administrator connection. Preview performs
read-only catalog/data validation and reports planned work; it never executes
DDL. Apply requires explicit existing owner identity and repeats preview under
locks before any mutation.

Within one transaction:

1. Acquire a migration advisory lock and deterministic relation locks covering
   the three heaps, users, and all known incoming-FK relations. Use 2-second lock
   timeout and 60-second per-statement timeout in the synthetic profile. Recheck
   identity, dependencies and one-owner data under these locks.
2. Snapshot supported definitions/ACLs/policies/defaults and collect heap, TOAST,
   index and sequence OIDs plus sizes. Add/validate an exact owner CHECK where
   needed so ATTACH can avoid a duplicate bound-validation scan.
3. Drop only explicitly enumerated FKs that must change referenced relation OID.
   Move/rename each original heap and its indexes into the private schema using
   the canonical child name. Build an empty public partitioned parent with the
   preserved column/default contract and target owner-qualified keys.
4. Reassociate sequences/identity defaults by the qualified mechanism. Attach
   original heaps under their literal owner bound. Attach/reuse compatible index
   objects and create only required changed-key indexes. Restore canonical
   public partitioned BM25 indexes and verify index ancestry.
5. Restore intended parent ownership, policies, table/column grants, comments and
   known outgoing FKs. Rebind and validate incoming FKs against new parents;
   no dependent table may continue referring only to the bootstrap leaf.
6. Verify complete row equality, heap/TOAST preservation where promised, keys,
   defaults, safe ID allocation, owner bounds, RLS/ACLs, FK targets and BM25 plans.
   Commit only after all checks pass. An injected failure must restore the
   original names, rows, dependencies, policies, ACLs and working indexes.

The ordering above is exercised by failing-then-passing synthetic tests. Parent
`CREATE TABLE ... LIKE ... INCLUDING IDENTITY` creates a new public message
allocator; the old leaf identity is removed after the new allocator is safely
advanced. Serial allocators retain their OIDs and are reassociated with parents. An unavailable reuse operation is a
refusal or explicit measured index operation, never a hidden full-table copy.

**Populated multiple-owner source heaps are refused in the first apply profile.**
Their rows cannot all be attached to one dedicated-owner leaf. A later measured
redistribution profile would need a separately reviewed rewrite/capacity plan.
There is no mixed-owner default partition and no temporary shared-ranked fallback.

## 5. Caller and lifecycle compatibility gates

- `store/queries.py:upsert_message` already conflicts on `(user_id,id)`;
  attachment insertion conflicts on `(user_id,message_id,filename)` and preserves
  `RETURNING id`. Proposition inserts use owner/message identity.
- `propositions.py:_bm25_candidates`, `find_facts` and `_cluster_near_duplicates`
  scope numeric IDs to one user. Existing engine hydration joins owner/message
  identities. These must be exercised with deliberately equal numeric IDs.
- Browser attachment routes constrain ID plus authenticated owner. The gateway
  attachment source's scalar-ID query runs through its immutable owner reader;
  retain that binding and test colliding IDs.
- Crawler ID-only bulk mutation was a concrete incompatibility. Root owns the
  `url_fetcher.py` and `queries.py` corrections and collision regressions; pending
  stubs must retain owner identity through grouping, representative selection,
  locked updates and deletion. Deliberate global URL/host housekeeping is a
  separate existing semantic, not permission to treat numeric IDs as global.
- Maps/caches/sets across multiple owners must use `(owner,id)`, not only the
  scalar ID. Owner-only collections may retain scalar keys. Audit scripts and
  administrative jobs as well as HTTP callers before a release.
- Existing owner-key/derived migrations currently reject partitioned relations.
  Their compatibility checks and tests need an explicit already-partitioned
  target branch; do not rerun their ordinary-table conversion against this target.
- `pg_schema.sql` and `propositions.py:ensure_table/ensure_bm25_index` can create
  absent tables/indexes at startup. Task 5 must prevent runtime startup from
  recreating an unpartitioned table or bypassing administrator provisioning.
- New-owner provisioning must create and verify all three leaves/indexes before
  reader/writer credentials or searchable admission are published. Source rows
  belonging to the existing owner do not make a different owner ready.

Before cutover, stop writers and drain public readers, long transactions,
prepared statements and search/index caches referring to the old parent OIDs.
After commit, reload through the new parents and qualify native search on all
three lexical branches. No runtime receives private child routing configuration.

## 6. Recovery, measurements and implementation gates

The rehearsal must measure elapsed time, heap/TOAST/index bytes and WAL changes;
record actual index/sequence OID reuse. It must test full text/bytea/null row
equality, insert/upsert/delete after conversion, incoming FK enforcement, owner
deletion, duplicate numeric IDs across owners, same-owner duplicate refusal,
repeat invocation, lock refusal, unknown-dependency refusal and transactional
failure rollback. Small fixtures do not predict production downtime or headroom.

Before any production apply: run a read-only production preview,
measure validation/index/WAL/temp space on representative scale, rehearse backup
restore, and pin the compatible caller/provisioning release. A committed
conversion is not automatically reversible after owners begin writing colliding
IDs; recovery then uses the qualified backup/recovery procedure, not an ad hoc
rename or a `CASCADE` rollback script. No production CLI apply path is enabled by
this contract.


## 7. Implemented rehearsal and evidence

Files:

- `deploy/public/migrate_owner_partitions.py`: read-only snapshot preview and
  synthetic-only transactional apply. Apply requires both logical and actual
  connection address `127.0.0.1`, port `55440`, and a database named
  `gms_owner_partitions_test_*`; there is no production apply switch.
- `tests/test_owner_partitions.py`: disposable fixture, source refusal, physical
  reuse, owner-qualified identity, FK, ACL, allocator, rollback and lock tests.
- `tests/test_owner_partition_lifecycle.py`: complete repository `pg_schema.sql`
  fixture, schema reapply, administrator second-owner provisioning, colliding
  Gmail/numeric IDs, lexical search, parent CRUD and FK/delete lifecycle.

Preview runs in `REPEATABLE READ READ ONLY`, returns row counts, relation/index
OIDs and file identities/sizes, and allocator metadata. It never returns mailbox
text. Already-converted preview calls the shared snapshot inspector; apply
calls the shared lock-protected verifier. Preview is an inventory, not an
admission decision. Apply acquires the same owner/catalog advisory locks as
provisioning before relation locks. Lock waits are capped at 2 seconds; each SQL
statement is capped at 60 seconds (these are not a single wall-clock operation
deadline). Writers and readers with old prepared/OID state must still be drained.

The source profile accepts only named canonical columns, their exact built-in
types/nullability/defaults, default deterministic text collations, canonical
allocation and identity constraints, and the complete enumerated FK inventory.
Known additive columns may be absent on older source heaps. RLS expressions are
limited to the canonical `app.user_id` comparison or a literal owner comparison;
arbitrary expressions/subqueries are refused. Table/column grants transfer to
parents; private leaf grants are removed and audited. Nonowner grant options,
unsafe default ACLs, triggers, rules, publications, extended statistics, security
labels, unsupported indexes and extension membership are refused.

The catalog dependency whitelist also refuses stored SQL-body functions,
external policies and row-type/array consumers that would otherwise remain
bound to the moved bootstrap heap. PostgreSQL does not record every dependency
hidden in dynamic SQL or textual function bodies; caller audit and stopped
trusted administrative writers remain required. The script is not a generic
program rewriter.

### Measured tiny fixture (2026-09-15)

Pinned server/image are recorded in `gateway-ranked-search-qualification.md`:
PostgreSQL 16.13, pg_search 0.23.0. A three-row rehearsal (one row per source)
measured:

| Measurement | Observed value |
| --- | ---: |
| Source heap bytes, total | 24,576 |
| Source TOAST bytes, total (mostly empty metadata) | 24,576 |
| Source BM25 bytes, total | 9,068,544 |
| Attach transaction elapsed | 0.042 seconds |
| Cluster WAL position delta during attach | 242,752 bytes |

All three original heap OIDs/file numbers and all three original BM25 index
OIDs/file numbers survived. A separate random large-text/bytea fixture verifies
TOAST heap/file identities and complete values survive; NULL values and comments
are also checked. Compatible proposition B-tree reuse is checked. Public parent
OIDs change; changed numeric uniqueness B-trees are rebuilt. The message identity
sequence is recreated; the two serial sequence OIDs remain. Injected failure
both immediately after rename and after attachment restores original rows,
relation/FK/default/comment/ACL catalogs and allocator state. Lock contention
refuses apply without a partial private schema.

These numbers describe tiny synthetic data only. The WAL position is
cluster-wide and can include other sessions; it is not a per-transaction exact
WAL accounting measure. They establish reuse and rollback feasibility, not
mailbox-scale downtime or storage headroom. Production planning must include
constraint-validation scans, changed B-trees, partition/index metadata growth,
backups and independent restore rehearsal.

Reproduce only against the authorized disposable cluster, using its synthetic
DSN in both environment variables:

```sh
export GMS_RANKED_PROBE_DSN="$DISPOSABLE_PARADESQL_DSN"
export GMS_TEST_PG_DSN="$DISPOSABLE_PARADESQL_DSN"
PYTHONPATH=src /home/ssilver/development/gmail-search/.venv/bin/python -m pytest -q \
  tests/test_owner_partitions.py tests/test_owner_partition_lifecycle.py
# Print fresh small-fixture size/WAL/timing observations:
PYTHONPATH=src /home/ssilver/development/gmail-search/.venv/bin/python -m pytest -q -s \
  tests/test_owner_partitions.py -k rehearsal_reports
```

### Cost boundary and next-profile optimization

**This reuse result starts after the owner-key migration has already created
numeric `messages.search_id` and its BM25 index.** It does not make that
prerequisite identity-column rewrite/index rebuild free. A production heap still
indexed by text `messages.id` cannot enter this profile directly.

Native owner partitions may permit retaining the original text `id` as each
leaf's unique BM25 key, avoiding that prerequisite numeric-key operation. The
extension already supports text keys in the repository's former schema, and
physical leaf separation removes cross-owner key collisions. A separate [text-key probe](gateway-text-key-partition-probe.md) now verifies
parent/child attachment, owner pruning, ranking invariance and prepared queries
on the pinned extension. The lower-cost complete migration profile and caller
changes remain unqualified. Current candidate query paths use `search_id`; no
automatic switch, production conversion or hidden rebuild is implemented here.
