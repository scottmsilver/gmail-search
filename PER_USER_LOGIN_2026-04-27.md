# Per-user multi-tenant login — plan (REVISED v2)

**Date**: 2026-04-27
**Author**: Claude (Opus 4.7) with Scott
**Status**: revised after hostile review caught 11 concrete bugs in v1; awaiting go

## Why this needed a v2

v1 (committed earlier today) had 11 real bugs caught by code-reviewer:
- ScaNN "shared + post-hoc filter" was *broken*, not just degraded — recall hits zero for users whose embeddings live in unsearched leaves of the partition tree.
- The MCP server's `register_session` taking `user_id` does *nothing* unless every `_*_impl` is rewritten to enforce it.
- `ALTER TABLE messages ADD COLUMN user_id REFERENCES users(id)` on 410k rows + a ParadeDB BM25 index is an outage, not "minutes."
- `gmail/auth.py` calls `InstalledAppFlow.run_local_server()` which opens a browser ON THE HOST. "Per-user" is a complete rewrite, not a path change.
- Gmail API quota is per-Cloud-Project; concurrent multi-user sync needs a project-level token bucket, not per-user backoff.
- Single 410k-row UPDATE for migration = full table rewrite + MVCC bloat; needs chunking.
- NextAuth JWE → Python verification has known footguns; commit to Bearer JWT (HS256) up front.
- Fernet has no AAD support → ciphertext can be copied between user_id rows; need AES-GCM with `user_id` bound as additional authenticated data.
- Per-user budget needs `UPDATE … RETURNING + check`, not SELECT-then-UPDATE.
- No feature flag infrastructure exists; pick env-var convention now or drop the claim.
- `user_id` on both `conversations` AND `agent_sessions` is redundant; pick one ownership model.

Plus several second-order findings (per-user `data/spell_dictionary.txt`, the `gmail_analyst` Postgres role being global, etc.) — folded into the cross-cutting section.

## Decisions (locked in)

1. **Sign-up policy**: invite-only.
2. **Existing 410k-message corpus**: migrated to Scott's account on first login.
3. **ScaNN strategy**: **per-user indexes** (the v1 "shared + filter" is broken — see review for why).
4. **OAuth refresh-token storage**: encrypted at rest with `USER_TOKEN_KEK` env-supplied key, **AES-256-GCM with `user_id` as AAD** (NOT Fernet — see review #9).
5. **Auth model**: **silver-oauth broker** (../silver-oauth) handles Google
   OAuth for the whole homelab; FastAPI verifies the broker's HS256 handoff
   JWT (60-second TTL) and issues its own session cookie. **Pivoted from
   NextAuth on 2026-04-28** after recognizing the broker already solves the
   per-app Google OAuth client problem we were re-inventing. Eliminates
   `next-auth` dep, `AUTH_GOOGLE_ID`/`AUTH_GOOGLE_SECRET`, and the
   `INTERNAL_SERVICE_TOKEN` chicken-and-egg fix entirely.
6. **Feature flag convention**: `GMAIL_MULTI_TENANT=1` env var. Every new code path that branches on tenancy gates on this. Default 0 until Phase 4 ships.

## Concept

- `users(id PK, google_sub UNIQUE, email UNIQUE, ...)` — root.
- Existing tables get `user_id TEXT REFERENCES users(id) ON DELETE CASCADE`.
- `invited_emails(email PK, ...)` — invite-only gate; empty = nobody can log in.
- Per-user OAuth Gmail tokens encrypted at rest in `user_gmail_tokens`.
- One ScaNN index per user, lazy-loaded with kernel page-cache eviction.
- All sync jobs (sync, summarize, watch, reconcile) round-robin across users via a project-quota-aware supervisor.

## Phase 0 — Empirical validation (gates everything, ½ day)

Before any code commits, validate the assumptions v1 got wrong:

### 0a. NextAuth + FastAPI Bearer JWT round-trip
- Spin up NextAuth-with-Google in `web/`, sign in with Scott's account.
- Have NextAuth issue a Bearer JWT (HS256, signed with `NEXTAUTH_SECRET`).
- FastAPI middleware verifies JWT via `python-jose` (HS256 is well-supported).
- Confirm session cookie + Bearer JWT both work for their respective uses.
- ⚠️ Do NOT bet on JWE — fall to JWT immediately if the cookie path fails.

### 0b. AES-256-GCM with AAD round-trip
- `cryptography.hazmat.primitives.ciphers.aead.AESGCM` — wrap a Gmail refresh_token, store in PG BYTEA, decrypt + bind `user_id` as AAD.
- Confirm: copying ciphertext from User A's row into User B's row → decrypt FAILS (AAD mismatch). This is the property Fernet cannot give us.
- Test KEK rotation: re-wrap with new key, confirm old ciphertext + new key fails, new ciphertext + new key succeeds.

### 0c. ScaNN per-user load latency benchmark
- Write a script that splits the existing 565k-vector index into N synthetic shards (~50k vectors each), saves each to a per-user dir, then measures `load_searcher` time + RSS delta for each.
- Confirms our 1-2s per-shard estimate.
- Confirms the kernel page cache eviction story (run two loads, drop caches, re-load, time).

### 0d. ALTER TABLE on a clone
- `pg_dump` the prod DB to a clone, run the proposed `ALTER TABLE messages ADD COLUMN user_id NULL` (no FK, no DEFAULT). Time it.
- Then run the chunked backfill loop. Time it.
- Then `ALTER TABLE … ADD CONSTRAINT … REFERENCES users(id) NOT VALID; ALTER TABLE … VALIDATE CONSTRAINT …;`. Time it.
- Goal: confirm the staged migration takes a known wall clock so we can plan the downtime window honestly.

**Output**: a one-paragraph note in this file recording each measurement, branch decisions, and any gotchas. THEN proceed to Phase 1.

## Phase 1 — Auth wall (1-2 days)

Goal: only invited Google accounts can hit any UI route or API endpoint. Existing data still single-pool but no public access.

### 1a. Schema additions (migration)
```sql
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    google_sub TEXT NOT NULL UNIQUE,
    email TEXT NOT NULL UNIQUE,
    name TEXT,
    avatar_url TEXT,
    invited_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_login_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS invited_emails (
    email TEXT PRIMARY KEY,
    invited_by TEXT REFERENCES users(id) ON DELETE SET NULL,
    invited_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    note TEXT
);
```

### 1b. NextAuth config (web/)
- Install `next-auth@5`.
- Configure Google provider with our client_id/secret (already exist for Gmail OAuth — separate scope).
- `signIn` callback hits `POST /api/auth/check-invite` on the FastAPI backend; rejects if email not in `invited_emails`.
- On accept: NextAuth creates session cookie AND issues an HS256 Bearer JWT (claims: `sub=google_sub`, `email`, `iat`, `exp`).
- All API calls from the Next.js side carry `Authorization: Bearer <jwt>` automatically via a fetch wrapper.

### 1c. FastAPI middleware
- New module `src/gmail_search/auth/session.py`:
  ```python
  def require_user(request: Request) -> User:
      """FastAPI dependency. Reads Bearer JWT, verifies HS256 with
      NEXTAUTH_SECRET, returns the User row. 401 on missing/invalid."""
  ```
- Every existing FastAPI route gets `user: User = Depends(require_user)` added.
- For Phase 1, the user's identity is captured but NOT yet used to filter data — that's Phase 2/3.

### 1d. CLI
- `gmail-search invite <email> [--note "..."]` — seeds `invited_emails`.
- `gmail-search list-users` — for ops.

### 1e. Bootstrap
- Run `gmail-search invite scottmsilver@gmail.com` once.
- Sign in via Google → creates the first `users` row.

### 1f. Test
- Anonymous request → 401.
- Non-invited Google sign-in → 403.
- Invited sign-in → session cookie + JWT issued.
- API calls with valid JWT → 200; missing/wrong JWT → 401.

## Phase 2 — Scope conversations + sessions (2-3 days)

Goal: each user owns their own chat threads + deep-mode sessions. Corpus still shared (Phase 3 fixes that). One ownership model: ownership lives on `conversations`; everything else joins through it (review #12).

### 2a. Schema migration (chunked + staged)
```sql
-- v1 said "ALTER TABLE conversations ADD COLUMN user_id ..." in one shot.
-- Real plan: ADD NULL → backfill → ADD FK with NOT VALID → VALIDATE.

ALTER TABLE conversations ADD COLUMN user_id TEXT;
-- Backfill (single-row table; no chunking needed for 53 rows):
UPDATE conversations SET user_id = '<scott-user-id>' WHERE user_id IS NULL;
ALTER TABLE conversations ALTER COLUMN user_id SET NOT NULL;
ALTER TABLE conversations
    ADD CONSTRAINT fk_conversations_user
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE NOT VALID;
ALTER TABLE conversations VALIDATE CONSTRAINT fk_conversations_user;
CREATE INDEX idx_conversations_user ON conversations (user_id, updated_at DESC);
```

`agent_sessions.conversation_id` already FKs to `conversations` (review #12). We do NOT add `user_id` to `agent_sessions` — joins through `conversations` for ownership. Same for `agent_events` / `agent_artifacts` (transitively).

### 2b. Code changes
- Every `/api/conversations/*` and `/api/agent/analyze` endpoint adds `WHERE c.user_id = $current_user.id` (joined explicitly when reading `agent_sessions` or `agent_events`).
- The MCP server's `register_session` takes `user_id`. **CRITICAL** (review #2): each `_*_impl` (`search_emails_batch`, `query_emails_batch`, `sql_query_batch`, `get_thread_batch`, `get_attachment_batch`) must:
  1. Accept a `user_id` arg from the wrapper.
  2. Forward `user_id` to the FastAPI backend on every request (e.g. as a header).
  3. The FastAPI backend's `/api/sql`, `/api/search`, etc. enforce ownership via the user_id (Phase 3 wiring), NOT trust the header alone.
- For Phase 2 specifically (where corpus is still shared): the MCP impls take user_id but don't filter on it yet. The plumbing lands now so Phase 3 only adds the WHERE clauses.

### 2c. Test
- Two-user integration test: User A creates a conversation → User B GETs `/api/conversations/<a's conv id>` → 404.
- User A's deep-mode turn → User B can't read its `agent_events`.
- The per-conversation Claude session UUID mapping is already keyed by `conversation_id` — automatic isolation post-Phase-2.

## Phase 3 — Full corpus isolation (8-12 days, biggest chunk)

Goal: every Gmail-corpus table is per-user. Per-user ScaNN indexes. Per-user OAuth tokens. Per-user sync.

### 3a. Schema migration (the dangerous one)

**This is the migration the v1 plan got dangerously wrong.** Real plan:

#### Step 1: Add NULLABLE columns to every corpus table (no FK, no DEFAULT)
```sql
-- Each ALTER acquires AccessExclusiveLock briefly (PG 11+: O(1) for ADD COLUMN
-- with no DEFAULT — just rewrites the catalog, not the table). Should be
-- sub-second per table even on 410k rows.
ALTER TABLE messages         ADD COLUMN user_id TEXT;
ALTER TABLE sync_state       ADD COLUMN user_id TEXT;
ALTER TABLE costs            ADD COLUMN user_id TEXT;
ALTER TABLE term_aliases     ADD COLUMN user_id TEXT;
ALTER TABLE contact_frequency ADD COLUMN user_id TEXT;
ALTER TABLE topics           ADD COLUMN user_id TEXT;
ALTER TABLE thread_summary   ADD COLUMN user_id TEXT;
ALTER TABLE message_summaries ADD COLUMN user_id TEXT;
ALTER TABLE summary_failures ADD COLUMN user_id TEXT;
ALTER TABLE model_battles    ADD COLUMN user_id TEXT;
-- query_cache is special: composite PK (query_text, model). Adding user_id
-- changes the PK semantics. Plan for query_cache:
--   1. Add user_id NULL.
--   2. Backfill to scott.
--   3. DROP existing PK, ADD PRIMARY KEY (query_text, model, user_id).
--      This rewrites the table — single-shot may be acceptable if small.
ALTER TABLE query_cache      ADD COLUMN user_id TEXT;
-- embeddings denormalization (review #5): add user_id directly so ScaNN
-- candidate filter can WHERE on it without joining messages. 565k rows;
-- backfill via chunked UPDATE.
ALTER TABLE embeddings       ADD COLUMN user_id TEXT;
```

#### Step 2: Chunked backfill (review #4)
```python
# scripts/migrate_to_multi_tenant.py — runs against a backed-up DB.
# Does NOT use a single UPDATE — that's a full table rewrite + MVCC bloat
# + holds locks for many minutes on 410k rows.
def chunked_backfill(table, user_id, chunk=5000):
    while True:
        n = conn.execute(f"""
            UPDATE {table} SET user_id = %s
            WHERE id IN (
                SELECT id FROM {table} WHERE user_id IS NULL LIMIT {chunk}
            )
        """, (user_id,)).rowcount
        conn.commit()
        if n == 0: break
        log.info(f"{table}: tagged {n} rows, total so far ...")
```

#### Step 3: SET NOT NULL + add FK NOT VALID + VALIDATE separately
```sql
-- After backfill confirms 0 NULLs:
ALTER TABLE messages ALTER COLUMN user_id SET NOT NULL;
ALTER TABLE messages
    ADD CONSTRAINT fk_messages_user
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE NOT VALID;
ALTER TABLE messages VALIDATE CONSTRAINT fk_messages_user;
-- VALIDATE is a SHARE lock — concurrent reads + writes OK, blocks DDL only.
```

#### Step 4: Indexes for per-user query patterns
```sql
CREATE INDEX CONCURRENTLY idx_messages_user_date ON messages (user_id, date DESC);
CREATE INDEX CONCURRENTLY idx_embeddings_user ON embeddings (user_id);
-- ... per query pattern audit.
```

#### Step 5: ParadeDB BM25 index — partition or filter?
**Concern (review #3)**: `messages_bm25_idx` is a `pg_search` BM25 index. Adding a `user_id` column doesn't break the index, but our BM25 queries (`WHERE id @@@ '...'`) need to add `AND user_id = $1`. ParadeDB supports compound predicates; verify in Phase 0d that this works without a recall hit, or partition the index per user (more work, real isolation).

### 3b. Per-user OAuth tokens

```sql
CREATE TABLE IF NOT EXISTS user_gmail_tokens (
    user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    token_ciphertext BYTEA NOT NULL,         -- AES-256-GCM ciphertext
    nonce BYTEA NOT NULL,                    -- 12-byte GCM nonce
    aad TEXT NOT NULL,                       -- = user_id (binds ciphertext to row)
    scope TEXT NOT NULL,
    encryption_key_version INT NOT NULL DEFAULT 1,
    granted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

**Module `src/gmail_search/auth/token_vault.py`**:
- `wrap(plaintext: bytes, *, user_id: str, key_version: int = LATEST) -> tuple[bytes, bytes]` → `(ciphertext, nonce)`.
- `unwrap(ciphertext: bytes, nonce: bytes, *, user_id: str, key_version: int) -> bytes`.
- Uses `cryptography.hazmat.primitives.ciphers.aead.AESGCM` with `user_id` as additional_authenticated_data. Copying ciphertext between rows fails decryption (review #9).
- KEK lookup table by version: `_KEKS = {1: bytes_from_env("USER_TOKEN_KEK_V1"), 2: ...}`. Rotation = add new env var, set new version as LATEST, re-wrap on next auth.

**Replace `src/gmail_search/gmail/auth.py`** (review #6):
- Current code calls `InstalledAppFlow.run_local_server(port=0)` — opens a browser ON THE HOST. Useless for web flow.
- Rewrite as web OAuth: `/api/auth/connect-gmail` initiates auth code flow (returns Google's authorization URL), `/api/auth/connect-gmail/callback` exchanges code → tokens → wraps + stores in `user_gmail_tokens`.
- The CLI flow stays for dev (`gmail-search auth --user <email>` does the local-server flow + writes to DB instead of file).

### 3c. Sync scheduler (review #7 — Gmail quota is per-Cloud-Project)

```python
# src/gmail_search/sync/supervisor.py
class SyncSupervisor:
    """One process that fans out per-user sync jobs across all users.

    The Gmail API quota is per-Cloud-Project (250 quota units/sec for
    messages.get, etc.). N concurrent users hammering the API would
    thundering-herd into 429s. Use a project-level token bucket that
    every per-user sync acquires from before each Gmail API call.
    Per-user backoff (existing in client.py) handles transient errors;
    the bucket handles steady-state quota."""
    def __init__(self, max_concurrent_users=3, project_quota_per_sec=200):
        self.bucket = TokenBucket(rate=project_quota_per_sec)
        self.semaphore = asyncio.Semaphore(max_concurrent_users)
    async def run_forever(self):
        while True:
            for user in users_due_for_sync():
                async with self.semaphore:
                    await sync_one_user(user, bucket=self.bucket)
```

### 3d. Per-user ScaNN indexes

#### Storage
```
data/
  users/
    <user_id>/
      scann_index__<timestamp>_<hash>/      # same shape as today's per-corpus
      scann_index_pointer (DB row, scoped by user_id)
```

#### Schema
```sql
ALTER TABLE scann_index_pointer ADD COLUMN user_id TEXT;
-- Need to drop the CHECK(id = 1) single-row constraint and replace
-- with UNIQUE(user_id) so each user has one active pointer.
```

#### Loading strategy (the one Scott asked about)
- **Eager load all users' indexes at boot** via `scann_ops_pybind.load_searcher`.
- **Rely on the kernel page cache for cold-user eviction** — the .npy files (dataset.npy, hashed_dataset.npy) are mmap-friendly via natural file-cache behavior. ScaNN's pybind doesn't pass `mmap_mode='r'` to `np.load` today, but the kernel still caches reads; under memory pressure cold pages are evicted; first query after eviction takes ~5-10s for a 1GB user index then warm again.
- **Per-user prewarm**: optional — call `searcher.search(zeros(dim), final_num_neighbors=1)` on each user's index at boot to force the page-cache fill. Trade-off: longer boot, predictable first-query latency.
- **Future**: if RAM pressure becomes acute, add a manual mmap'd loader (one-day project) that explicitly mmap's the .npy files and feeds them to ScaNN.

#### Search code refactor
- `src/gmail_search/search/engine.py`: SearchEngine becomes per-user. Process holds a `dict[user_id -> SearchEngine]`. Cold users: lazy-load on first query.
- `_detect_user_email` (review additional finding): currently reads "most frequent sender" globally; becomes per-user (each user's own corpus).
- `_user_in_participants` ranking: per-user `user_emails`.
- Spell dictionary `data/spell_dictionary.txt`: per-user (mined from each user's corpus during their sync). Stored at `data/users/<user_id>/spell_dictionary.txt`.

### 3e. MCP tool authorization (review #2 — the confused-deputy bug)

```python
# mcp_tools_server.py — register_session signature change:
def register_session(session_id, *, user_id, evidence_records, db_dsn, ...):
    ...
    _SESSIONS[session_id] = SessionContext(
        user_id=user_id,    # NEW
        evidence_records=evidence_records,
        ...
    )

# Every _tool_*_batch wrapper extracts user_id from the SessionContext
# and threads it into the impl:
async def _tool_sql_query_batch(session_id, queries):
    ctx = _get_session(session_id)
    response = await _sql_query_batch_impl(queries, user_id=ctx.user_id)
    ...

# tools.py impls take user_id as a kwarg and forward to the FastAPI backend
# as a header:
async def sql_query_batch(queries, *, user_id):
    return await _post("/api/sql_batch", json={"queries": queries},
                       headers={"X-User-Id": user_id})
```

**The FastAPI backend MUST NOT trust `X-User-Id` blindly** — it's set by the MCP server, but the MCP server is in the same trust domain. Real auth happens at FastAPI's edge (Bearer JWT from NextAuth). For MCP-originated requests: a **service token** scheme — the MCP server has its own bearer token; FastAPI verifies it AND honors `X-User-Id` only when the service token is present. Otherwise → 401.

This is the biggest failure mode in v1. Without it, any leaked MCP admin token = full corpus access for all tenants.

### 3f. Per-user budget (review #11 — atomic increment)

```sql
CREATE TABLE IF NOT EXISTS user_budgets (
    user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    monthly_usd_cap NUMERIC(10,2) NOT NULL DEFAULT 50.00,
    current_month_usd NUMERIC(10,4) NOT NULL DEFAULT 0,
    month_start DATE NOT NULL DEFAULT (DATE_TRUNC('month', NOW()))::DATE
);
```

Atomic increment:
```sql
UPDATE user_budgets
   SET current_month_usd = current_month_usd + $1
 WHERE user_id = $2
RETURNING current_month_usd, monthly_usd_cap;
-- Caller checks RETURNING > cap and refunds + 429s the user.
```

The race window between two concurrent UPDATEs is closed by row-level locking (each UPDATE takes a row lock). v1's "SELECT then UPDATE" would have overshot under any concurrency.

### 3g. Postgres `gmail_analyst` role (additional review finding)

Currently `gmail_analyst` has `SELECT ON messages` — global, no row filter. The Analyst sandbox SQL would leak across users.

**Fix**: switch to per-user PG roles (`gmail_analyst_<user_id>`) with a row-level security (RLS) policy `USING (user_id = current_setting('app.user_id'))`. The sandbox sets `app.user_id` via `SET LOCAL app.user_id = '<user_id>'` on each connection. Standard PG RLS pattern, well-supported.

## Phase 4 — Onboarding + UX (5-7 days)

Goal: a brand-new invited user can land, sign in, connect Gmail, watch sync, use the app — all from the UI.

(Substantively unchanged from v1 — the bugs were all in 1/2/3.)

**UI screens** (new):
- `/login` — Google sign-in CTA.
- `/onboarding/connect-gmail` — Gmail OAuth trigger.
- `/onboarding/initial-sync` — progress page (polls `GET /api/users/me/sync-status`).
- `/settings` — connected Gmail account, monthly cost meter, delete-account button.
- Sidebar avatar + email dropdown.

**Backend support**:
- `GET /api/users/me`, `GET /api/users/me/sync-status`, `DELETE /api/users/me` (cascade-deletes everything).

## Cross-cutting changes (folded in from review)

- **CLI**: `gmail-search invite`, `list-users`, `delete-user`, `rotate-token-kek`.
- **Tests**: new `tests/multi_tenant/` directory with paired-user isolation tests for every endpoint.
- **Logging**: structured logs with `user_id` field on every line.
- **Migration safety**: every `ALTER TABLE` migration runs in chunks per the staged playbook in §3a; full `pg_dump` snapshot before each phase rolls; roll-forward only.
- **Feature flag**: `GMAIL_MULTI_TENANT=1` (default 0). Every new code path branches on it. Flip to 1 atomically when Phase 4 ships.

## Risks left on the table (revised honest list)

1. **Gmail Cloud Project quota** — even with the project-level token bucket, the 250 quota units/sec project cap means N parallel users is bottlenecked by the cap, not by per-user concurrency. At 5 users syncing concurrently, each gets 50 units/sec — slow. Solution path: each user supplies their own Cloud Project credentials (real multi-tenant) OR one shared project with a higher quota tier. Out of scope for now; document so we don't pretend it's free.
2. **Initial sync cost** — first user sync = ~$5 of embedding API calls. Need budget enforcement (3f) BEFORE letting any new user start their sync.
3. **Per-user ScaNN RAM under N invited users** — the kernel page cache handles this gracefully up to maybe 20 users; beyond that we either need RAM growth or active eviction.
4. **NextAuth Bearer JWT exposes the JWT to JavaScript** (HTTP-only cookies are safer) — accept the trade-off for FastAPI integration simplicity. Mitigation: short-lived JWTs (15 min) + refresh.
5. **`USER_TOKEN_KEK` lifecycle** — env-supplied is convenient; if the host is compromised the KEK leaks and every user's Gmail is open. Real KMS = follow-up.
6. **Schema migrations on a live DB are NOT online** — accept downtime windows for each phase. Communicate in advance.
7. **The shared `gmail_analyst` PG role** — Phase 3g RLS migration is non-trivial; if RLS breaks subtly the Analyst could leak across users. Test the RLS policy with a paired-user fixture before shipping.
8. **MCP service-token model** — adds a new auth path. If the service token is shared across all per-user MCP processes, it's a flat-trust system (any MCP server can claim any user_id). Real fix = per-user MCP processes (heavy) OR signed user_id JWTs from FastAPI to MCP. Phase 3 ships with the simpler shared-service-token; harden in a follow-up.
9. **Per-user Postgres roles** for the Analyst RLS = N × user_id roles. PG handles thousands fine but it's noise in `\du`. Acceptable.

## Effort estimate (revised)

| Phase | Effort | Risk | Notes |
|-------|--------|------|-------|
| 0: empirical validation (NextAuth+JWT, AES-GCM+AAD, ScaNN per-user load, ALTER on clone) | ½ day | medium — gates everything | Don't skip ANY of the 4 sub-tasks |
| 1: auth wall | 1-2 days | low | Lock down public access |
| 2: scope conversations + sessions | 2-3 days | medium | Schema migration + MCP plumbing without enforcement yet |
| 3a-3d: corpus tables migration + per-user OAuth + per-user ScaNN + sync scheduler | 8-12 days | **high** | Biggest single chunk; staged migration is critical |
| 3e-3g: MCP authorization + budget + RLS for analyst | 3-5 days | high | The auth model is non-obvious |
| 4: onboarding UX | 5-7 days | medium | Real frontend work |

**Total: 3-5 weeks of focused work.** Plan for 6 with unknowns from Phase 0.

## Suggested ordering

1. **Phase 0 first** (½ day) — gates everything. ALL FOUR sub-tasks.
2. **Phase 1** (1-2 days) — locks down public access. Behind `GMAIL_MULTI_TENANT=1` flag.
3. **Phase 2** (2-3 days) — scope conversations only. Two-user smoke test.
4. **Phase 3a** (corpus schema migration with staged ALTER) — backup snapshot + downtime window.
5. **Phase 3b** (per-user OAuth tokens + new web flow).
6. **Phase 3c** (sync scheduler with project-level token bucket).
7. **Phase 3d** (per-user ScaNN indexes).
8. **Phase 3e** (MCP authorization rewrite — biggest correctness risk).
9. **Phase 3f** (per-user budget).
10. **Phase 3g** (RLS for `gmail_analyst`).
11. **Phase 4** (onboarding UX).

## Open questions to validate before/during

1. **Cloud Project credentials** per user OR shared? Drives the quota story.
2. **Email notifications** on sync complete? Postmark / SES / none?
3. **Account deletion**: hard delete OK, or do we need a soft-delete + 30-day undo window for compliance?
4. **Audit log retention** — how long do we keep per-user activity logs?
5. **Cost cap policy** — what's the default `monthly_usd_cap`? $50? $100? Per-tier?
6. **What happens to a user's data when they're un-invited?** — currently `invited_emails` deletion has no cascade; the user account still works. Probably want UNINVITE semantics.

---

## Investigation notes

### Phase 0c — ScaNN per-user load benchmark (2026-04-28, complete)

Script: `scripts/bench_per_user_scann_load.py`. Each of the V3 index's
13 shards (~43k vectors @ 768d) was treated as a synthetic user; cold
loads forced via `posix_fadvise(POSIX_FADV_DONTNEED)` per file (no
sudo, no global cache flush).

Results vs plan targets:

| metric                       | measured        | target    | margin  |
|------------------------------|-----------------|-----------|---------|
| avg cold load                | **185 ms**      | <2000 ms  | ~10×    |
| avg warm load                | **113 ms**      | —         | —       |
| avg first query post-evict   | **0.6 ms**      | <5000 ms  | ~8000×  |
| avg warm query               | **0.6 ms**      | —         | —       |
| avg RSS per user             | **129 MiB**     | —         | —       |
| eager-load-all 13 indexes    | **2.31 s, 1.8 GiB**  | —    | —       |

**Verdict**: per-user ScaNN strategy is viable as designed.

**Design implications** (refines §3d):
- Lazy-on-first-query is fine. We do NOT need eager-load-at-boot —
  185 ms cold load is well under any reasonable user-perceived latency
  threshold for the first search of a session. Eager-load is an option
  for predictability, not a requirement.
- 129 MiB per user × V3 768d means ~7.7 users per GiB resident. On
  a 32 GiB host that's room for ~200 active users before RAM pressure;
  the kernel page cache handles overflow gracefully past that.
- First-query latency post-eviction is essentially free (~0.6 ms),
  so cold-user wake-up is a non-event. The AH codebook + sparsely
  touched dataset rows fit in a handful of pages.
- Confirms V3 (768d) is the right default for multi-tenant. A 3072d
  baseline index would be ~4× larger per user (~520 MiB), shrinking
  the ceiling to ~50 users. V3's compaction wins here as much as it
  wins on disk.

**No design changes required.** The "kernel page cache for cold-user
eviction" story in §3d holds; we keep it but downgrade eager-load
from "do this" to "optional knob."

### Phase 0d — ALTER TABLE timing on a clone (2026-04-28, complete)

Script: `scripts/bench_phase0d_alter_table.sh`. Cloned `messages` (410k
rows, 14 GB) and `embeddings` (565k rows, 8 GB) via `pg_dump … | psql`
into a fresh `gmail_search_clone` DB (no source outage). Ran the staged
migration playbook end-to-end, timed each step.

| step                                | wall-clock | lock                |
|-------------------------------------|-----------:|---------------------|
| pg_dump + restore (clone setup)     |   449.6 s  | none on source      |
| ADD COLUMN NULL (both tables)       |     0.07 s | brief AccessExcl.   |
| chunked backfill messages (410k)    |    34.9 s  | per-chunk row locks |
| chunked backfill embeddings (565k)  |    14.2 s  | per-chunk row locks |
| SET NOT NULL messages               |     1.46 s | AccessExclusive     |
| SET NOT NULL embeddings             |     0.27 s | AccessExclusive     |
| ADD FK NOT VALID (both)             |     0.06 s | brief               |
| VALIDATE CONSTRAINT (both)          |     0.59 s | SHARE — online      |
| CREATE INDEX CONCURRENTLY (both)    |     0.83 s | SHARE — online      |
| **BM25 + `AND user_id = $1`**       |     0.074 s | —                  |
| **BM25 recall vs unfiltered**       |     **3302 == 3302** (exact) | — |

**Verdict**: §3a's playbook works as designed and is *much* faster than
the plan feared. Total writes-blocked time if all AccessExclusive steps
are batched into one maintenance window: **~2 seconds**. Everything else
(backfill, validate, index create) is online.

**Critical answer to §3a Step 5 (BM25 partition vs filter)**: ParadeDB's
`pg_search` BM25 index composes cleanly with a `WHERE user_id = $1`
SQL predicate. Same recall (3302 rows filtered == 3302 unfiltered in
the single-user clone), ~75 ms latency. **No need to partition the BM25
index per user** — adding the predicate is sufficient.

**Plan refinements**:
- Drop the doom-and-gloom around "single 410k-row UPDATE = full table
  rewrite + many-minute lock" — the chunked loop avoids it cleanly,
  total wall clock is under a minute.
- Phase 3a Step 5 simplified: keep the existing `messages_bm25_idx`
  unchanged; per-user is purely a query-time predicate.
- Realistic Phase 3a downtime budget: announce ~5 min maintenance,
  run the migration script, done.



## Review history

### v1 (this morning)
- Wrote initial plan with shared+filter ScaNN, single-shot ALTER, Fernet token wrap, etc.

### Hostile review (this afternoon, by superpowers:code-reviewer)
- Found 11 concrete bugs (see "Why this needed a v2" at top).

### v2 (now)
- Per-user ScaNN with kernel page cache strategy.
- Staged ALTER TABLE playbook (ADD NULL → chunked backfill → SET NOT NULL → ADD FK NOT VALID → VALIDATE).
- AES-256-GCM with `user_id` as AAD (not Fernet).
- NextAuth issues HS256 Bearer JWT (not JWE).
- MCP confused-deputy fix: per-call user_id threading + service-token auth at FastAPI edge.
- One ownership model: ownership lives on `conversations`; everything joins.
- Per-user `gmail_analyst` RLS policy.
- Project-level Gmail quota token bucket in the sync supervisor.
- `GMAIL_MULTI_TENANT=1` env-flag convention.
- `gmail/auth.py` rewrite for web OAuth (not "path change").
- Atomic budget increment via `UPDATE … RETURNING + check`.

Plus the ScaNN strategy answer Scott asked for: **eager-load all users' indexes at boot, rely on kernel page cache for cold-user eviction**, no manual mmap needed for v1.
