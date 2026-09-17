# The migration window

Written 2026-09-16. Everything here is verified against the live database or
tested on the disposable cluster; nothing is written from memory.

**Search is offline for the whole window.** Budget 45 minutes, of which roughly
4 are the migration itself and ~94 seconds are exclusive locking.

## What makes this runnable now

| | Evidence |
| --- | --- |
| The mechanism works | Rehearsed at full scale, `docs/qualification/mixed-migration-rehearsal-2026-09-15.md` |
| A driver exists | `deploy/public/migrate_production_mixed.py`, tested end to end against a database it did not create (`tests/test_production_migration_driver.py`) |
| The app survives the result | `tests/test_app_against_migrated_schema.py` — writer, readers, partition routing, owner isolation |
| Production is structurally ready | `--report`: 3 owners, 421,554 rows stay, 21,874 move, 0 orphans |
| The engine is right | PostgreSQL 16.15 + patched pg_search 0.23.0. Phase two **requires** the patch |

## The coupling that decides the order

Production has exactly one unique index, `messages_pkey` on `(id)`. The branch
upserts `ON CONFLICT(user_id, id)`. **Neither half works without the other**, so
the migration and the code deploy happen inside the same window, migration
first. Deploying the branch early breaks every write; migrating without it
breaks every write the other way.

## Take the backup as a data-directory copy, not a dump

The rehearsed procedure used `pg_dump -Fc` — 483 s, 9.55 GiB, restore 6.9 min —
and that is a fine backup. But `GMS_MIGRATION_BACKUP` can only *verify* a data
directory: it reads `global/pg_control` at offset 0 and refuses a backup whose
cluster identity does not match the target. Against a dump file it falls back to
"exists, non-empty, recent", which is a declaration rather than a check.

Since the writers are stopped anyway, take the cold copy and get the real check.

## Sequence

Each step says how to know it worked, and what to do if it did not.

### 1. Confirm the report is clean

```sh
uv run python deploy/public/migrate_production_mixed.py \
  --dsn "$PROD_DSN" --report
```

Expect `READY`. Anything else stops the window — nothing has changed yet.

### 2. Stop every writer

```sh
systemctl --user stop gmail-search-supervise gmail-search-serve \
  gmail-search-mcp gmail-search-public-api gmail-search-public-web
```

Leave `gmail-search-web` (Next.js) running or stop it too; it holds no database
connection. Confirm nothing is left attached:

```sh
docker exec gmail-search-pg psql -U gmail_search -d gmail_search -tAc \
  "SELECT count(*) FROM pg_stat_activity WHERE datname='gmail_search' AND pid<>pg_backend_pid()"
```

Must be `0`. The fence enforces this again at apply time and will refuse
otherwise, naming the connection — but finding it here is cheaper.

### 3. Cold backup

The data directory is a bind mount at `data/pg`; the database is 57 GB and the
filesystem had 438 GB free on 2026-09-16, so the copy fits — but check, because
`data/pg-backup-16.13-20260916` is already occupying 70 GB of it.

```sh
df -h /home/ssilver/development/gmail-search/data        # need > 60 GB free
BACKUP=data/pg-backup-16.15-$(date +%Y%m%d-%H%M)
docker stop gmail-search-pg
sudo cp -a data/pg "$BACKUP"
docker start gmail-search-pg
```

Verify it is a copy of *this* cluster before relying on it:

```sh
docker exec gmail-search-pg psql -U gmail_search -d gmail_search -tAc \
  "SELECT system_identifier FROM pg_control_system()"
sudo python3 -c "import struct;print(struct.unpack('<Q',open('$BACKUP/global/pg_control','rb').read(8))[0])"
```

The two numbers must match. They did on 2026-09-16: `7630921061527392295`.

### 4. Mark the target

```sh
NONCE=$(openssl rand -hex 16)
docker exec gmail-search-pg psql -U gmail_search -d gmail_search -c \
  "COMMENT ON DATABASE gmail_search IS 'gms-migration-target:$NONCE'"
```

This is what stops a token read off a restore admitting production: a copy taken
before this write does not carry it. Generate it fresh; do not reuse.

### 5. Apply

```sh
export GMS_MIGRATION_TARGET=production
export GMS_MIGRATION_BACKUP=data/pg-backup-16.15-<stamp>
export GMS_MIGRATION_CONFIRM="gmail_search:7630921061527392295:$NONCE"

uv run python deploy/public/migrate_production_mixed.py \
  --dsn "$PROD_DSN" --apply \
  --registry ~/.config/gmail-search/migration-registry.sqlite \
  --store-id gms-prod --migration-id owner-partitions-1 \
  --receipt data/migration-receipt-owner-partitions-1.json
```

Expect `phase_one -> INDEX_PENDING` then `phase_two -> READY`.

**If phase one fails**, nothing was published; the layout commit is idempotent
and crash re-entry never publishes readiness. **If phase two fails**, the layout
is committed but indexes are not rebuilt — search is wrong until it completes,
so re-run rather than rolling back. **If anything is unrecoverable**, restore
the step-3 copy: stop the container, replace the data directory, start it.

### 6. Deploy the matching code

In the **main checkout**, not the worktree:

```sh
git fetch && git checkout codex/full-agents-worktree-20260916
uv sync --extra dev
```

This is the half that makes writes work again. Without it the daemons still say
`ON CONFLICT(id)`.

### 7. Declare the new shape

`GMS_SCHEMA_PROFILE=text-partitioned-v1` in **all three** Python env files —
they are read by different units and a missed one refuses every connection in
that daemon:

- `~/.config/gmail-search/multi-tenant.env` — serve, supervise
- `~/.config/gmail-search/mcp.env` — mcp
- `~/.config/gmail-search/public-api.env` — public-api

`public-web.env` is Next.js and needs nothing.

### 8. Start and verify

```sh
systemctl --user start gmail-search-supervise gmail-search-serve \
  gmail-search-mcp gmail-search-public-api gmail-search-public-web
```

Then, in order of what fails first if something is wrong:

1. Every unit `active running`.
2. `curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8090/docs` → 200.
3. A search through the UI returns results. **This is the real check** — the
   profile binding would have refused the connection outright if the declaration
   were wrong, so a 200 plus results means shape, code and config agree.
4. Ingestion writes: watch `messages` count climb, or force a sync.

## What will look different afterwards

**Relevance ranking shifts slightly.** BM25 scores are computed per partition,
so each owner's corpus is its own statistical universe. Measured on the
disposable cluster: the same document scored 0.182 unpartitioned and 0.288
partitioned. Ordering within an owner is unaffected in the cases tested, but
absolute scores are not comparable across the migration. This is correct
behaviour, not a regression — worth knowing before concluding search got worse.

## What this window does not do

It does not install the invited service, provision per-owner roles, or enrol the
worker. It makes the database and the application owner-partitioned, which is
the foundation those need. `docs/superpowers/plans/2026-09-16-multi-user-rollout-plan.md`
has the rest.
