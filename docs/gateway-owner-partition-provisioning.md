# Administrator owner-partition provisioning

The approved layout uses physical `LIST(user_id)` partitions for
`public.messages`, `public.attachments`, and `public.propositions`, with native
BM25 indexes on each parent and attached leaf. The provisioner implements the
administrator portion of [the owner partition plan](superpowers/plans/2026-09-15-owner-partitioned-search.md).
It does not replace the immutable reader/writer ACL and RLS audits.

## APIs and transaction ownership

`gmail_search.gateway.partitions` exposes:

- `provision_owner_partitions(conn, owner_id)`: create all missing owner leaves
  in one transaction, then verify the completed layout. Existing catalog drift
  is refused rather than repaired. Repeated calls preserve existing object OIDs.
- `verify_owner_partitions(conn, owner_id)`: require all three leaves and verify
  the complete existing mail-partition layout under administrator locks.
- `inspect_owner_partitions(conn, owner_id)`: run the same catalog verification
  inside a caller-owned `REPEATABLE READ READ ONLY` or `SERIALIZABLE READ ONLY`
  snapshot. This preview API takes no explicit admission or DDL locks and must
  not be used as a readiness check when publishing credentials.
- `remove_empty_owner_partitions(conn, owner_id)`: remove a complete, verified,
  empty owner set while the user still exists. It takes the same bounded locks
  plus exclusive locks on all three leaves before checking emptiness. It detaches
  and drops referencing leaves first, without `CASCADE`, preserving parent FKs.
  No leaves is an idempotent no-op; a partial set or any remaining row is refused.
  The caller may then delete the user row in the same transaction.
- `partition_name` and `partition_binding`: shared deterministic naming and
  canonical administrator metadata for migration and provisioning.

The connection must authenticate directly as a superuser that owns the database,
parents, and private schema, with no members of that owning role. No administrator
connection enters a guest or public query handler. Nested caller transactions
retain commit/rollback ownership. Failure rolls back partial DDL and the function's
local settings; successful calls also restore the original settings.

Provisioning and readiness verification acquire a per-owner transaction advisory
lock, then a shared catalog advisory lock and fixed parent locks. The catalog lock
serializes concurrent schema creation and partition DDL across owners. Parent
`SHARE ROW EXCLUSIVE` locks can briefly block ingestion. Lock waits are capped at
two seconds and each statement at thirty seconds; stricter caller timeouts remain
in force. These are per-lock/per-statement limits, not a total operation deadline.

## Exact accepted profile

The initial profile is PostgreSQL 16 with `pg_search` 0.23.0 and
`standard_conforming_strings=on`. It accepts the owner-qualified keys and outgoing
FK actions in [the migration contract](owner-partition-migration.md). PostgreSQL's
automatically generated FK clones are distinguished from the declared parent FKs.

Owner identity comparisons are part of the boundary. `public.users.id` and all
parent/leaf `user_id` columns must be nonnull built-in `text` with the deterministic
`pg_catalog.default` collation. Partition keys must use that same collation and
the default built-in B-tree `text_ops` class. Nondeterministic case-insensitive
collations are refused: they can otherwise equate distinct owner IDs such as
`alice` and `ALICE` even when the partition declaration prints `LIST (user_id)`.

Parent indexes must be the canonical named, valid, ready, live partitioned BM25
indexes with exactly these columns and only the specified `key_field` option:

| Parent index | Columns | Key field |
| --- | --- | --- |
| `messages_bm25_idx` | `search_id,id,subject,body_text,from_addr,to_addr` | `search_id` |
| `attachments_bm25_idx` | `id,filename,extracted_text` | `id` |
| `props_bm25_idx` | `id,text` | `id` |

Expressions, predicates, included columns, nondefault operator classes and extra
BM25 options are refused. Each leaf must have exactly one valid native BM25 index
whose catalog ancestry and definition match its parent. PostgreSQL creates these
leaf indexes through `CREATE TABLE ... PARTITION OF`; no extra global numeric-ID
uniqueness is assumed.

## Child identity and access

Children live only in `gms_mail_partitions`. Their names use the fixed table
prefix plus 48 hex characters of the owner ID's SHA-256 digest. Each comment
contains the exact canonical `partition_binding` value. Neither the name nor
comment alone establishes ownership.

The verifier checks actual catalog ancestry, permanent ordinary leaf relation
kind, object ownership, enabled/forced RLS, canonical owner comparison semantics,
and an exact single-owner bound equal to the complete stable owner ID. Bound text
is compared as data against PostgreSQL's canonical quoted literal; it is never
executed. Quote, backslash and Unicode owner IDs have regression coverage. Default,
mixed-owner, null, subpartition, detached, and unrelated private relations fail
closed, including malformed leaves belonging to another owner.

The private schema and its relations, columns and sequences may have no nonowner
ACL entries. Relevant administrator default privileges must also be owner-only.
Ownership is checked separately from ACL expansion so an empty ACL cannot hide
an untrusted schema owner. Runtime credentials continue to receive only reviewed
parent access; this module creates no runtime role or grant.

## Verification and remaining scope

`tests/test_gateway_partition_provision.py` creates a random database from
`template0` on the explicitly allowed synthetic `127.0.0.1:55440` ParadeDB fixture.
It covers native parent search, idempotency, forged/mixed/default bounds, missing
indexes/keys/FKs, RLS and ACL drift, administrator refusal, owner comparison,
rollback, same/different-owner concurrency, timeout restoration and read-only
preview verification. Empty-owner lifecycle tests cover foreign-owner preservation,
subsequent admission, nonempty/partial/forged refusal and transactional rollback
with working parent FKs. Database and generated runtime-role cleanup are scoped to
the fixture's own objects. Existing naming tests remain unchanged.

Ranking invariance and admission integration have separate tests owned by the
parent implementation. This administrator API does not qualify production
migration, capacity, restore procedures, full search behavior or deployment.
The initial canonical-only profile deliberately refuses schema and index variants
that need a separate reviewed migration profile.

## Owner removal lifecycle

Deleting an owner row alone can cascade mailbox rows while leaving empty physical
partitions behind. The global catalog audit correctly refuses partitions whose
owner no longer exists. Administrator retirement must therefore clear retained
mail and call `remove_empty_owner_partitions` **before** deleting the user row,
inside the same caller transaction. The helper neither deletes mail nor deletes
the owner. Public deletion UI, retention policy, and credential retirement remain
separate lifecycle work.
