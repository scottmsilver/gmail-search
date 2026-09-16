# Shared mailbox owner-key migration: synthetic qualification candidate

Do not enable invited mailbox ingestion on the strength of reader RLS alone.
The current writer schema cannot represent two owners with the same Gmail
message ID. Privileged ingestion overwrites the first owner's message content
and attachment path; constrained ingestion would reject the collision instead.
The migration and compatible callers must be reviewed together; schema-only
qualification does not enable production ingestion.

## Required schema change

Keep each mailbox row once in the existing shared tables. No mailbox copies,
new per-user tables, or content duplication are needed.

This describes the permanent data model, not the physical migration footprint.
Adding an identity column to populated `messages` can rewrite the table and its
indexes, creating substantial temporary disk and WAL usage while holding an
exclusive lock. Constraint/index work adds its own scans and storage overhead.
The migration is not metadata-only. Rehearse with representative mailbox size,
measure peak disk/WAL and lock duration, and reserve sufficient headroom and a
maintenance window before any production qualification.

| Table | Required identity and references |
| --- | --- |
| messages | Primary key `(user_id,id)`; retain `id` as Gmail message ID |
| attachments | Keep globally generated `id` primary key; unique `(user_id,message_id,filename)`; FK `(user_id,message_id)` to messages; unique `(user_id,message_id,id)` for dependent owner checks |
| embeddings | Keep global generated `id`; FK `(user_id,message_id)` to messages; FK `(user_id,message_id,attachment_id)` to attachments; ensure referenced attachment also has the same message ID |
| message_summaries | Primary key/FK `(user_id,message_id)` to messages |
| summary_failures | Primary key/FK `(user_id,message_id)` to messages, retaining `ON DELETE CASCADE` |
| message_topics | Primary key `(user_id,message_id,topic_id)`; composite message and topic FKs |
| prop_processed | Already keyed `(user_id,message_id)`; require nonnull owner and matching-parent composite FK after auditing deletion semantics |
| propositions | Keep global generated `id`; validate owner/message consistency and add composite FK only after deciding existing deletion semantics |

`embeddings` needs a three-column attachment identity check (for example unique
`attachments(user_id,message_id,id)` with the matching FK) if cross-message
attachment references must also be rejected. Preserve nullable attachment IDs
for message embeddings.

## Additional code changes required in the same release

- Update all message/attachment conflict targets, including unfetched manifests,
  Drive stubs and URL stubs in `store/queries.py`, not just the primary upserts.
- Pass a trusted explicit owner through summarization and failure storage.
  `summarize.py` currently obtains owner through a scalar `SELECT user_id FROM
  messages WHERE id=...`; duplicate Gmail IDs make that query ambiguous. Update
  summary/failure conflict targets and deletes together.
- Audit every message/attachment/embedding join, lookup, update, existence probe,
  and delete to include owner equality. Known candidates occur in
  `store/db.py`, `embed/pipeline.py`, `search/engine.py`, `server.py`, `cli.py`,
  invitation crawl caching, and summarization. Global generated attachment and
  embedding IDs may remain global, but authorization still needs owner checks.
- Preserve owner-separated filesystem roots for Gmail attachments and derived
  artifacts. `gmail/client.py` writes `attachments_dir/message_id/filename`;
  callers must supply an owner-specific root and must not fall back to a shared
  root when an invitee is selected.
- Resolve BM25 search identity before changing the messages key. The existing
  `messages_bm25_idx` uses `key_field='id'` and search code scores that key. Add a
  globally unique internal search key while retaining owner-scoped Gmail IDs,
  rebuild this index, and update corresponding search/score expressions.
  Qualify the actual deployed ParadeDB version on synthetic colliding IDs.
- Update fresh-install schema and migrations together. Phase23 only promotes
  selected aggregate keys; rerunning it does not fix message ownership keys.

## Migration preflight and execution

1. Stop all ingestion and derived writers; preserve the old release and take a
   verified database backup. Reader connections must not race schema changes.
2. Enumerate actual constraints, indexes, policies, triggers and dependencies
   from catalogs. Refuse unexpected dependencies rather than using CASCADE.
3. Refuse any null owner in participating tables, orphan child, or child whose
   owner differs from its parent. Refuse conflicting composite identities.
   Report counts only; do not silently backfill ownership or discard rows.
4. Record per-owner row counts and content checksums. Preserve existing serial
   attachment/embedding/proposition IDs and sequence state.
5. In one administrator transaction with bounded lock timeouts and a migration
   advisory lock, replace only enumerated constraints. Add composite foreign
   keys and validate them; set required owner columns NOT NULL. Complete the
   tested BM25 key/index transition as part of the maintenance procedure.
6. Verify row counts/content checksums, all constraint validation, existing RLS
   and analytical reader column grants. New internal search keys should not
   automatically enter the agent-facing schema allowlist.
7. Deploy all compatible callers before resuming writers. A failed transaction
   rolls back automatically. After new colliding rows exist, the old global key
   cannot be restored without data loss: code/schema rollback must not claim
   that it can simply reinstate the original primary key.

## Qualification gate

Both owners must independently ingest the same message ID, thread ID, attachment
filename and topic ID, then update, summarize, embed, search, delete and reingest
their own data without altering the other owner's data or files. Verify both
raw privileged writer SQL and fixed-role reader visibility. Test mismatched
child owners, null-owner preflight rejection, interruption rollback, idempotent
reapplication, concurrent writers being blocked, BM25 ranking, and unchanged
row counts/content for a noncolliding existing mailbox.

## Implemented candidate and limits

`migrate_owner_keys.py` implements the six-table schema prerequisite and an
explicit transactional BM25 rebuild. Run `--preview` for a read-only preflight
and proposed BM25 DDL. `--apply --rebuild-bm25` permits the key transition only
on disposable `gms_owner_keys_test_*` databases; the CLI still refuses production
application. Both modes use `OWNER_KEY_MIGRATION_DSN`. Do not put credentials in
shell history or qualification reports.

The rebuilt index uses globally unique `messages.search_id`; `(user_id,id)`
remains the mailbox identity, and Gmail `id` remains indexed. One ordinary BM25
index is accepted, including a renamed index. Its name, indexed columns,
collations/operator classes, all relation options except `key_field`, tablespace
and comment are preserved. Expressions, predicates, included columns, per-column
options, unusual ordering, invalid indexes and multiple BM25 indexes are refused.
A legacy BM25 index requires the explicit rebuild flag. Target-key reruns are
idempotent. A mismatched BM25 key/schema is refused.

Migration requires a direct administrator login with full RLS visibility,
accepts only enumerated legacy/target constraints and uses exclusive table locks,
an advisory lock and bounded lock waits. The index drop, identity backfill,
constraint replacement, rebuild and verification share one transaction. Tests
inject failure after the new BM25 index exists and verify rollback restores the
old key, index configuration and working search. This does **not** make rollback
to global Gmail IDs possible after colliding owner rows have committed.

Fresh `pg_schema.sql` creates the target identities. Reapplying startup schema
to an existing installation does not promote message keys or rebuild an existing
BM25 index. Callers must choose search keys matching their schema during release
transition. The internal search ID must remain outside the agent schema allowlist.

### Recorded synthetic verification (2026-09-15)

`tests/test_owner_key_schema_migration.py`: **43 passed**, PostgreSQL 16.13,
`pg_search` 0.23.0. A disposable 2 CPU / 2 GiB container used the exact same
installed Docker image ID as `gmail-search-pg`:
`sha256:99f182f93387c2226c810763f736ec06154090b37d654a09483ca44d68b5ba6c`.
Only Docker image metadata was read from the existing container; no production
SQL, mail copy or schema changes were performed. Synthetic tests cover fresh
schema/rerun, migrated colliding IDs, score/search/update/delete/reinsert,
configuration preservation, invalid-owner refusal, constraints, lock refusal,
content/ID/RLS/grant preservation and transactional rollback.

Production-scale rewrite time, disk/WAL headroom, real-data dependency inventory,
backup restore and a maintenance window remain unqualified. The six-table
migration intentionally does not add proposition/processed-marker FKs: deletion
semantics for those existing derived tables need separate review.

The 60-second statement timeout applies separately to each SQL statement, not
to the total migration duration; the 2-second lock timeout bounds individual
lock waits. Neither setting establishes a total maintenance-window deadline.

## Derived facts and processed markers

`migrate_derived_owner_keys.py` is an explicit second prerequisite, applied only
after the main message owner keys. It gives `propositions` and `prop_processed`
nonnull `(user_id,message_id)` references to `messages(user_id,id)` with
`ON DELETE CASCADE`. These are regenerable facts/markers: removing a message
must remove its derived facts rather than leave searchable facts whose source
has gone away. Existing attachment, embedding, summary and topic deletion
semantics are unchanged.

The derived migration requires an idle direct administrator connection, ordinary
public tables, the exact supported identities, full RLS visibility, bounded
statement/lock waits, an advisory lock, and locked message/derived tables. It
refuses null ownership, orphan or mismatched parents, unexpected/inbound
references, extra unique indexes, partial FK installation and weakened target
nullability. It never repairs ownership, discards rows, changes policies/grants,
or resets sequences. Both migrations can be reapplied after the derived keys
are installed; the main migration recognizes only these exact validated
composite CASCADE references and never drops them.

Its CLI remains restricted to disposable `gms_owner_keys_test_*` databases using
`DERIVED_OWNER_KEY_MIGRATION_DSN` and `--apply`. No production DDL has been applied.
`tests/test_derived_owner_keys.py` exercises colliding owners, foreign-parent
refusal before embedding, deletion during embedding, owner-specific cascades,
policy/grant preservation and migration reapplication. Writer code locks and
rechecks the selected parent before committing derived rows; its guarded
inserts also prevent mismatched ownership on an unmigrated derived table.
