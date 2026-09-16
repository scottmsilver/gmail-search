# Direct legacy TEXT-key owner partition rehearsal

This is a **synthetic-only, clean-history, single-owner mechanism**. It is not eligible for the currently observed live database: the live message and attachment heaps contain multiple owners. A current single-owner row scan also does **not** prove historical BM25 corpus purity. The separate mixed-history probe showed that deleting foreign rows, analyzing/vacuuming and attaching the existing index preserves foreign-influenced scores; rebuilding the retained leaf index restores clean scores. Any future mixed-owner migration must budget that explicit rebuild even after moving the minority rows.

## Scope and source

`deploy/public/migrate_text_owner_partitions.py` provides `preview_text_owner_partitions(conn, owner_id=...)` and `migrate_text_owner_partitions(conn, owner_id=...)`. Apply accepts only an idle connection whose host **and actual peer address** are `127.0.0.1`, port `55440`, and whose disposable database name starts `gms_owner_partitions_test_`. It has no production override.

The frozen source is `tests/fixtures/legacy_text_owner_schema.sql`, copied byte-for-byte from `src/gmail_search/store/pg_schema.sql` at commit `9e282ca3067ed6fcdaf6e73232340c9aae7c2ec4`, SHA256 `53fef22e7d33837d749a5baf9ab25532968268857590406c36f767b2396e86fb`. Tests substitute only the two cluster-global role names and the hardcoded database CONNECT grant with unique disposable names. The product schema file is unchanged.

Accepted source contracts:

- Ordinary persistent `messages`, `attachments`, and `propositions` heaps with ENABLE/FORCE RLS, canonical defaults/types/collations and indexes. All present rows must have the explicitly supplied owner. No owner is filled, rewritten or inferred.
- Messages: TEXT PK `id`, no `search_id`, BM25 `(id,subject,body_text,from_addr,to_addr)` with key `id`.
- Attachments: numeric PK `id`, UNIQUE `(message_id,filename)`, canonical numeric serial and BM25 `(id,filename,extracted_text)`.
- Propositions: numeric PK `id`, canonical numeric serial and BM25 `(id,text)`. The checked-in source deliberately has **no** message/owner FK; processed markers likewise have no message FK. These exact absences are accepted only in the source contract, with owner/message correspondence checked before DDL. The target adds composite cascading message FKs.
- Existing mandatory message/attachment/user/topic FKs on the other tables must all be present, valid and nondeferrable. Unrelated inbound FKs, including references to the metadata tables whose primary keys change, are refused. Stored dependencies on replaced constraint/backing-index OIDs are refused too.
- `user_id` may be nullable only on the six columns added nullable in the legacy schema; their explicit previously backfilled NOT NULL variant is also accepted. Any actual null data is refused. Propositions and processed-marker keys are already nonnullable.
- Two explicitly observed embedding RLS states are accepted as source: disabled/unforced from the frozen schema, or enabled/unforced from the hardened legacy release. These flags remain unchanged. Both exact equivalent topic FK column orders are accepted and retained. An already-partitioned target may also have enabled/forced embedding RLS after the separate fixed-reader provisioner runs.

On the three moved heaps, unknown columns/defaults/identity allocators, numeric/intermediate key layouts, custom key/index definitions, sequence ownership/parameters/ACL/dependencies, unsupported policies, rowtype/function/view dependencies, triggers, publications, and private-schema drift are refused. Unmoved dependent tables receive relevant owner/key/type, foreign-key, replaced-constraint/index-dependency and data-integrity checks; unrelated columns/defaults remain unchanged and are not comprehensively audited. Reapplication uses the shared strict TEXT partition inspector plus dependent key/FK/data checks.

## Transaction and preservation

The operation takes the existing shared owner/catalog advisory locks and bounded exclusive locks on all participating heaps, dependent tables, users and topics. It validates the whole source before mutation. It then drops only inventoried FKs that need new logical parents, sets validated owner columns NOT NULL, replaces owner-qualified metadata primary keys, and moves the original three heaps into the private partition schema.

It creates empty TEXT-key logical parents with the fixed owner-qualified keys and matching parent index definitions, then attaches the original heaps and their existing BM25/nonconstraint indexes. Messages never acquire an identity column or sequence. After full source preflight, supported `ALTER SEQUENCE ... CACHE 1` statements acquire transaction-held sequence locks (the cache setting was already qualified as 1), then refresh `last_value/is_called` before cutover. Standalone `nextval` calls block until completion; an allocation made between observational preflight and locking is retained on both commit and rollback. Attachment and proposition serial sequences keep their original OIDs; ownership moves to the new parent and `ALTER SEQUENCE ... RESTART` advances them transactionally beyond both stored IDs and prior allocated values. Existing RLS policies and grants are copied to the logical parents; guest grants are removed from the private leaves. Known FKs are restored once against owner-qualified parent keys, followed by strict target verification.

Tests compare row values, heap/TOAST/BM25 OIDs and relfilenodes, serial identities and rollback state. This demonstrates preservation for the tested clean-history synthetic fixtures. It does not mean zero WAL/internal writes. New composite B-tree indexes, NOT NULL scans, owner checks, FK validation and locking still cost work; production-scale costs are unmeasured here. No numeric prerequisite, mailbox heap copy, or BM25 rebuild occurs in this isolated mechanism.

## Qualification

Tests cover full frozen schema, explicit hardened-source variant, colliding owner CRUD/new owner admission, actual fixed TEXT reader integration, null/mixed owners and orphan facts, missing mandatory/extra derived/external FKs, external stored constraint dependencies, sequence default/dependency/advance/rollback, unknown numeric keys, function/view references, real out-of-line TOAST and parent-only ACLs. Injected failures after rename and after attach restore original storage and keys. A direct missing-column regression protects the owner-column catalog check.

Run with the explicitly configured synthetic fixture:

```sh
GMS_TEST_PG_DSN='postgresql://postgres:synthetic-owner-test@127.0.0.1:55440/postgres' PYTHONPATH=src /home/ssilver/development/gmail-search/.venv/bin/python -m pytest -q tests/test_text_owner_partitions.py
```

**Qualified results:** direct migration has 25 tests. Combined direct, unchanged numeric migration/full-schema lifecycle, TEXT profiles and independent profile regressions passed **82 tests in 26.24 seconds**, with no skips. Parent independently reviewed the module and reran the direct 25 plus three profile regressions: **28 passed in 8.91 seconds**. Ruff passed. Review approved after the missing-column and transaction-held sequence-lock fixes. The compatible caller release remains separate; see `docs/text-partition-caller-inventory.md`. No production query, DDL, mail export, provider call or global schema switch is performed by this rehearsal.
