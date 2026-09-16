# Retained-table BM25 assertion: engine-version A/B result

2026-09-15, addendum to [the blocker record](retained-reader-maintenance-blocker.md).
Synthetic disposable PostgreSQL only. No live database, mailbox contents, schema
or production container was changed by this work.

## Question

The blocker record leaves the native cause undiagnosed. The cheapest candidate
explanation was that the pinned engine is simply old and the fault is fixed
upstream. The local `paradedb/paradedb:latest-pg16` image was built 2026-04-16
and carries PostgreSQL 16.13 / pg_search 0.23.0; the tag's current content is
PostgreSQL 16.15 / pg_search 0.25.9, two minor releases newer.

## Result: the upgrade does not fix it

`docs/qualification/retained-reader-new-index-reproducer.py` was run unmodified
against both engines.

| Engine | Reproducer outcome | Server log |
| --- | --- | --- |
| PostgreSQL 16.13 / pg_search 0.23.0 | 6 failed, 7 passed | `assertion failed: item_pointer_is_valid(ctid)` |
| PostgreSQL 16.15 / pg_search 0.25.9 | 6 failed, 3 passed (4 deselected) | identical assertion |

The same six branches fail on both engines: unrestricted messages, attachments
and facts, for the retained numeric profile and the direct TEXT migration.
The passing cases on both engines are the fresh-partition fixtures.

**The determinant is fresh versus retained heaps, not the engine version and not
TEXT versus numeric keys.** An engine upgrade is not an available remedy, and the
blocker's consequence stands unchanged.

## Consequence for the migration shape

Every migration profile that preserves an existing populated heap and rebuilds
its BM25 index — including the mixed-owner phase-one design, whose purpose is to
avoid rewrite cost by retaining dominant heaps — sits on the failing side of this
split on two engine generations. Only redistribution into freshly created
partitions has passed. Pricing the fresh-leaf copy (the rewrite cost that phase
one was designed to avoid) is now the load-bearing open question, and the
reproducer is clean enough to submit upstream to ParadeDB.

## Two costs of adopting 0.25.9, neither of which buys a fix

1. `pg_search` 0.25.9 declares `requires = 'vector'` in its control file. A
   database created from `template0` fails `CREATE EXTENSION pg_search` with
   `required extension "vector" is not installed`. `store/pg_schema.sql` and
   `tests/fixtures/legacy_text_owner_schema.sql` would each need an explicit
   `CREATE EXTENSION IF NOT EXISTS vector` before `pg_search`.
2. The fixed-reader provisioner fails closed on 0.25.9: `provision_search_reader.py`
   rejects it at `Unexpected effective function privilege`, because the extension
   exposes executable functions outside the audited allowlist. This is the
   provisioner behaving correctly. Adopting the newer engine would require
   re-auditing the pg_search function surface and reissuing the allowlist — a
   security review, not a version bump.

To obtain the table above, the four `0.23.0` version pins and the four
version-surface audit assertions were bypassed **for the diagnostic only**, and
reverted immediately; the four touched files were restored byte-for-byte from
backup and verified by checksum. No bypass is proposed for any release.

## Operational hazard found while testing

`gmail-search-pg` runs the floating tag `paradedb/paradedb:latest-pg16`. Pulling
that tag now yields 0.25.9, so any `docker compose pull`, image prune plus
recreate, or host rebuild would silently move the database two minor versions,
break the pinned provisioner and require the new `vector` dependency. Consider
pinning the service to the digest actually qualified:

```
paradedb/paradedb@sha256:e41e0c742ef91ece4fc7c08dda7f24e5a8f818563b164bbfea3a4364941d75f7
```

The local tag was repointed back to that digest after testing. The running
`gmail-search-pg` container was never stopped, recreated or otherwise touched.

## Reproduce

```bash
export GMS_TEST_PG_DSN="postgresql://postgres:<pw>@127.0.0.1:55440/postgres"
export GMS_GATEWAY_TEST_DSN="$GMS_TEST_PG_DSN"
cp docs/qualification/retained-reader-new-index-reproducer.py tests/test_zz_repro_tmp.py
UV_CACHE_DIR=/tmp/gmail-search-uv-cache uv run --no-sync pytest tests/test_zz_repro_tmp.py -q -p no:randomly
rm tests/test_zz_repro_tmp.py
```

The reproducer imports sibling fixtures, so it must sit in `tests/` to run; it is
a diagnostic and is deliberately not a collected acceptance test. The disposable
qualification cluster is expected on port 55440 — the fixture in
`tests/test_text_owner_partitions.py` asserts that exact host and port. It was
recreated during this work (the previous one had been started with `--rm` and was
removed when stopped); it holds no test databases between runs, and the rebuilt
cluster reproduces the baseline identically.
