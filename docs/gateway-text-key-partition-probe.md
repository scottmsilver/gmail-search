# Text-key BM25 owner partitions: synthetic feasibility

**Result: supported in this probe.** On PostgreSQL 16.13 / `pg_search` 0.23.0,
a `LIST(user_id)` messages parent with `PRIMARY KEY(user_id,id)` and native
`BM25(id,subject,body_text) WITH(key_field='id')` supported TEXT Gmail IDs,
including equal IDs in Alice and Bob partitions. A numeric `search_id` was not
needed for native scoring in this shape.

The 2026-09-15 probe ran only in a random `template0` database on the approved
synthetic `127.0.0.1:55440` instance. Its database and generated login were removed.
No current product schema, provisioner, migration profile or callers changed.

## What passed

- Started with an ordinary single-owner heap, TEXT `id` primary key, and existing
  TEXT-key BM25 index. Changed only the B-tree primary key to `(user_id,id)`, then
  attached the heap under a matching partitioned parent and parent BM25 index.
- Preserved the existing heap, TOAST and BM25 index **OIDs and relfilenodes**.
  Heap size stayed 8,192 bytes; BM25 main-fork size stayed 3,022,848 bytes. The
  tiny two-row conversion took 0.0183 seconds. These observations do not establish
  production costs, zero WAL, or absence of every internal index-file write.
- `CREATE TABLE ... PARTITION OF` automatically created Bob's valid, ready BM25
  leaf index. Both leaf indexes had catalog ancestry to the native parent index.
- Alice and Bob both stored `shared1` and `shared2`. A direct unprivileged Alice
  login used a literal restrictive RLS policy alongside a permissive legacy
  policy. Changing `app.user_id` to Bob did not change access. Direct child access
  was denied; PUBLIC privileges on all 499 extension routines were removed and
  only the four previously qualified search routines granted to the reader.
- `paradedb.score(id)` returned TEXT-key results. Alice's exact baseline was
  `shared1: 0.8025914`, then `shared2: 0.60996956`. Scores, ordering, count and
  LIMIT results stayed identical after Bob inserted, updated and deleted 200
  additional rows. Bob-only terms returned no Alice result.
- Both `force_custom_plan` and `force_generic_plan` passed with automatically
  prepared statements. Executed JSON plans selected only Alice's leaf.

The fixed query deliberately relied on immutable RLS without an explicit owner
predicate, matching the earlier ranking qualification:

```sql
SELECT id, paradedb.score(id)
FROM public.messages
WHERE id OPERATOR(pg_catalog.@@@) $1
ORDER BY paradedb.score(id) DESC, id
LIMIT $2;
```

## Implication for the next migration profile

A separately reviewed legacy TEXT-key attach profile could potentially avoid
adding/populating a numeric identity and replacing the existing BM25 index solely
to change its key. The existing numeric-key profile remains unchanged and valid.

The alternative still needs production-schema inventory, exact source/target
BM25 field/options compatibility, owner-qualified key changes, inbound FK and
caller review, stopped-writer conversion/rollback tests, and measured B-tree,
validation, WAL and restore headroom. This probe used only messages with a simple
users FK and three indexed fields. It is not a production migration rehearsal or
proof that every existing messages/index variant can attach unchanged.

## Reproducible evidence

Retained assets under
`worktrees/full-agent-assets-20260915/text-key-partition-probe/` in the original
workspace contain `probe.py` and `report.json` (full executed plans and assertions).
The script requires `GMS_TEST_PG_DSN` and refuses nonapproved host/port/database/user
settings before creating its own random database. It does not print passwords.

- Probe SHA256: `47e78bce6b46ac04a7aae5dc2a071b2ce92c6453fe548f9bdd6c02874d1490e0`
- Report SHA256: `4aea5fe3010cb66ebc281bf0ba1d233c88cc84d49fe4ceda76fd30b52b30bc9d`
