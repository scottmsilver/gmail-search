-- Postgres 16 translation of the SQLite SCHEMA defined in
-- src/gmail_search/store/db.py. When executed against an empty Postgres
-- database, this file creates every table + index the app needs.
--
-- Translation conventions (see db.py for the source of truth):
--   * INTEGER PRIMARY KEY AUTOINCREMENT  -> BIGSERIAL PRIMARY KEY
--   * INTEGER PRIMARY KEY CHECK(id = 1)  -> kept as INTEGER PRIMARY KEY CHECK(id = 1)
--   * TEXT ISO-8601 UTC date columns      -> kept as TEXT (string sort == chronological;
--                                           migration to TIMESTAMPTZ is a follow-up)
--   * DEFAULT CURRENT_TIMESTAMP           -> DEFAULT NOW()
--   * JSON-in-TEXT columns (labels, ...)  -> kept as TEXT (migration to JSONB is a follow-up)
--   * REFERENCES ...                      -> kept (PG enforces FKs natively, which is what we want)
--   * FTS5 virtual tables                 -> replaced with pg_search (paradedb) BM25 indexes
--
-- Every CREATE TABLE uses IF NOT EXISTS so this file is idempotent.

-- pg_search (paradedb) gives us real BM25 ranking via Tantivy — the closest
-- thing to SQLite FTS5's bm25() function. Default PG `ts_rank_cd` doesn't
-- match BM25 semantics and the A/B harness caught unacceptable ranking drift
-- (top-10 overlap 0.55, below the 0.8 bar). pg_search preserves that
-- invariant.
CREATE EXTENSION IF NOT EXISTS pg_search;

-- ─────────────────────────────────────────────────────────────────────
-- Core tables
-- ─────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    from_addr TEXT NOT NULL,
    to_addr TEXT NOT NULL,
    subject TEXT NOT NULL DEFAULT '',
    body_text TEXT NOT NULL DEFAULT '',
    body_html TEXT NOT NULL DEFAULT '',
    date TEXT NOT NULL,
    labels TEXT NOT NULL DEFAULT '[]',
    history_id BIGINT NOT NULL DEFAULT 0,
    raw_json TEXT NOT NULL DEFAULT '{}'
);

-- Invitation crawl guard verdict cache. When the content-based guard
-- (src/gmail_search/gmail/invite_guard.py) classifies a message as an
-- actionable invitation, it creates ZERO URL stubs and records WHY here
-- so a re-sync doesn't re-call Gemini or flip-flop. NULL = not gated
-- (the common case); a non-NULL reason means "all links were skipped
-- for this message." Idempotent add for installs provisioned before the
-- guard existed.
ALTER TABLE messages ADD COLUMN IF NOT EXISTS crawl_blocked_reason TEXT;

CREATE TABLE IF NOT EXISTS attachments (
    id BIGSERIAL PRIMARY KEY,
    message_id TEXT NOT NULL REFERENCES messages(id),
    filename TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    size_bytes BIGINT NOT NULL DEFAULT 0,
    extracted_text TEXT,
    image_path TEXT,
    raw_path TEXT,
    -- URL-stub crawl state (fast/slow lane). A failed crawl leaves
    -- extracted_text NULL; these let pending_url_stubs back off and abandon
    -- dead links instead of re-trying them every cycle (head-of-line blocking).
    crawl_attempts INT NOT NULL DEFAULT 0,
    crawl_last_attempt TIMESTAMPTZ,
    UNIQUE (message_id, filename)
);

-- Serves pending_url_stubs' (crawl_attempts ASC, id DESC) fast/slow-lane order.
CREATE INDEX IF NOT EXISTS idx_attachments_crawl_lane
    ON attachments (crawl_attempts, id DESC)
    WHERE mime_type = 'text/html' AND extracted_text IS NULL;

CREATE TABLE IF NOT EXISTS embeddings (
    id BIGSERIAL PRIMARY KEY,
    message_id TEXT NOT NULL REFERENCES messages(id),
    attachment_id BIGINT REFERENCES attachments(id),
    chunk_type TEXT NOT NULL,
    chunk_text TEXT,
    embedding BYTEA NOT NULL,
    model TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS costs (
    id BIGSERIAL PRIMARY KEY,
    timestamp TEXT NOT NULL,
    operation TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens BIGINT NOT NULL DEFAULT 0,
    image_count BIGINT NOT NULL DEFAULT 0,
    output_tokens BIGINT NOT NULL DEFAULT 0,
    estimated_cost_usd DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    message_id TEXT NOT NULL
);

-- Idempotent column add for databases provisioned before `output_tokens`
-- existed. Previously, deep-mode agents packed LLM output tokens into
-- the `image_count` column (which means "images processed" elsewhere),
-- silently mixing two units in analytics. New rows write the real count
-- here; old rows default to 0.
ALTER TABLE costs ADD COLUMN IF NOT EXISTS output_tokens BIGINT NOT NULL DEFAULT 0;

CREATE TABLE IF NOT EXISTS sync_state (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS thread_summary (
    thread_id TEXT PRIMARY KEY,
    subject TEXT NOT NULL DEFAULT '',
    participants TEXT NOT NULL DEFAULT '[]',
    all_from_addrs TEXT NOT NULL DEFAULT '[]',
    all_labels TEXT NOT NULL DEFAULT '[]',
    message_count BIGINT NOT NULL DEFAULT 0,
    date_first TEXT NOT NULL DEFAULT '',
    date_last TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS topics (
    topic_id TEXT PRIMARY KEY,
    parent_id TEXT,
    label TEXT NOT NULL DEFAULT '',
    depth BIGINT NOT NULL DEFAULT 0,
    message_count BIGINT NOT NULL DEFAULT 0,
    top_senders TEXT NOT NULL DEFAULT '[]',
    sample_subjects TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS message_topics (
    message_id TEXT NOT NULL REFERENCES messages(id),
    topic_id TEXT NOT NULL REFERENCES topics(topic_id),
    PRIMARY KEY (message_id, topic_id)
);

CREATE TABLE IF NOT EXISTS contact_frequency (
    email TEXT PRIMARY KEY,
    message_count BIGINT NOT NULL DEFAULT 0,
    score DOUBLE PRECISION NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS query_cache (
    query_text TEXT NOT NULL,
    model TEXT NOT NULL,
    embedding BYTEA NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (query_text, model)
);

CREATE TABLE IF NOT EXISTS term_aliases (
    term TEXT PRIMARY KEY,
    expansions TEXT NOT NULL DEFAULT '[]',
    similarity DOUBLE PRECISION NOT NULL DEFAULT 0.0
);

-- ─────────────────────────────────────────────────────────────────────
-- message_summaries
-- ─────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS message_summaries (
    message_id TEXT PRIMARY KEY REFERENCES messages(id),
    summary TEXT NOT NULL,
    model TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT NOW()
);

-- Durable record of summarization failures so they can be triaged
-- (backend down, context overflow, parse errors, empty model output).
-- A row is written on each failure and deleted when the message
-- eventually gets a successful summary. The running daemon logs a
-- warning regardless; this table lets us answer "what's still
-- broken?" without scraping logs.
CREATE TABLE IF NOT EXISTS summary_failures (
    message_id TEXT PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    error TEXT NOT NULL,
    attempts INT NOT NULL DEFAULT 1,
    first_seen TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_summary_failures_last_seen
    ON summary_failures (last_seen DESC);

-- ─────────────────────────────────────────────────────────────────────
-- Conversations + chat history
-- ─────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT,
    created_at TEXT NOT NULL DEFAULT NOW(),
    updated_at TEXT NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS conversation_messages (
    id BIGSERIAL PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    seq BIGINT NOT NULL,
    role TEXT NOT NULL,
    parts TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT NOW(),
    UNIQUE (conversation_id, seq)
);

-- ─────────────────────────────────────────────────────────────────────
-- Model battles (A/B harness)
-- ─────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS model_battles (
    id BIGSERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    variant_a TEXT NOT NULL,
    variant_b TEXT NOT NULL,
    winner TEXT NOT NULL CHECK (winner IN ('a', 'b', 'tie', 'both_bad')),
    request_id_a TEXT,
    request_id_b TEXT,
    created_at TEXT NOT NULL DEFAULT NOW()
);

-- ─────────────────────────────────────────────────────────────────────
-- job_progress
-- ─────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS job_progress (
    job_id TEXT PRIMARY KEY,
    stage TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'running',
    total BIGINT NOT NULL DEFAULT 0,
    completed BIGINT NOT NULL DEFAULT 0,
    -- `completed` at job start — used to compute rate/ETA for backfill
    -- where completed tracks total corpus size (starts nonzero).
    start_completed BIGINT NOT NULL DEFAULT 0,
    detail TEXT NOT NULL DEFAULT '',
    started_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    -- OS pid of the worker process currently owning this row. Written by
    -- `JobProgress.__init__` (via os.getpid()), used by the supervisor
    -- and /api/jobs/*-stop to signal the right process without needing
    -- a pid file on disk. Nullable so rows predating the column don't
    -- break.
    pid BIGINT
);

-- Idempotent column add for databases provisioned before `pid` existed.
-- Postgres 9.6+ supports `ADD COLUMN IF NOT EXISTS`; our docker-compose
-- stack is on 16, so this is safe.
ALTER TABLE job_progress ADD COLUMN IF NOT EXISTS pid BIGINT;

-- ─────────────────────────────────────────────────────────────────────
-- scann_index_pointer
--
-- One-row pointer that names the currently-active on-disk ScaNN index
-- directory. Builders write a fresh timestamped sibling and flip this
-- row in a single transaction; readers join here to resolve "where is
-- the live index right now?" instead of counting on filesystem rename
-- atomicity. The CHECK(id = 1) single-row idiom carries over from
-- SQLite verbatim — Postgres enforces it the same way.
-- ─────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS scann_index_pointer (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    current_dir TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT NOW()
);

-- ─────────────────────────────────────────────────────────────────────
-- Deep-analysis agent state. See docs/DEEP_ANALYSIS_AGENT.md for the
-- architecture. One `agent_sessions` row per deep-mode turn; events
-- are a time-ordered log of what each sub-agent (planner, retriever,
-- analyst, writer, critic) did; artifacts hold Analyst outputs
-- (plots, CSVs, tables) the Writer cites as [art:<id>].
-- ─────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS agent_sessions (
    id TEXT PRIMARY KEY,
    conversation_id TEXT,
    mode TEXT NOT NULL,
    question TEXT NOT NULL,
    plan JSONB,
    final_answer TEXT,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'running'
);

CREATE TABLE IF NOT EXISTS agent_events (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES agent_sessions(id) ON DELETE CASCADE,
    seq INT NOT NULL,
    agent_name TEXT NOT NULL,
    kind TEXT NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (session_id, seq)
);

CREATE TABLE IF NOT EXISTS agent_artifacts (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES agent_sessions(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    data BYTEA NOT NULL,
    meta JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_events_session_seq
    ON agent_events (session_id, seq);
CREATE INDEX IF NOT EXISTS idx_agent_sessions_conv
    ON agent_sessions (conversation_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_artifacts_session
    ON agent_artifacts (session_id, created_at DESC);

-- ─────────────────────────────────────────────────────────────────────
-- MCP OAuth provider state (clients, auth codes, token pairs, nonces).
--
-- Durable so an mcp-service restart doesn't invalidate claude.ai's
-- tokens (which forced a re-auth every restart). Token/code rows are
-- keyed by SHA-256 of the secret and the JSON value has the secret
-- field blanked — a DB dump holds no replayable bearer material.
-- `PgOAuthStore` also creates this table at construction (the MCP
-- service can restart before serve applies schema updates); the DDL
-- here keeps the schema file the single source of truth.
CREATE TABLE IF NOT EXISTS mcp_oauth_state (
    kind TEXT NOT NULL,
    key TEXT NOT NULL,
    value JSONB NOT NULL,
    expires_at DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (kind, key)
);
CREATE INDEX IF NOT EXISTS mcp_oauth_state_expires_idx
    ON mcp_oauth_state (expires_at) WHERE expires_at IS NOT NULL;

-- ─────────────────────────────────────────────────────────────────────
-- Per-conversation Claude session UUID mapping.
--
-- Each chat conversation pins exactly one Claude Code session UUID so
-- subsequent deep-mode turns can pass `--resume <uuid>` to claudebox
-- and append to the same JSONL transcript. The first turn establishes
-- the UUID by running without `--resume` and capturing the
-- `sessionId` claudebox returns.
--
-- Concurrency: pg_advisory_xact_lock(hashtext(conversation_id)) is
-- taken around the establishment critical section so two simultaneous
-- first-turn requests serialize into one establisher + one resumer.
-- See agents/service.py:_claim_or_establish_claude_session.
-- ─────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS conversation_claude_session (
    conversation_id TEXT PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
    claude_session_uuid TEXT NOT NULL,
    claimed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Read-only role used by the Analyst sandbox. DO block lets us re-run
-- the schema file idempotently without erroring on duplicate-role
-- create. GRANTs are cumulative — re-running them is a no-op.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'gmail_analyst') THEN
        -- Password is generated at role creation; the sandbox executor
        -- reads it from the main server's env + rotates it if needed.
        -- For v1 we use a fixed password matching the dev-only DSN so
        -- the analyst can connect from Docker without extra plumbing.
        CREATE ROLE gmail_analyst LOGIN PASSWORD 'analyst_readonly';
    END IF;
END
$$;

GRANT CONNECT ON DATABASE gmail_search TO gmail_analyst;
GRANT USAGE ON SCHEMA public TO gmail_analyst;
GRANT SELECT ON messages, attachments, message_summaries, thread_summary,
    topics, message_topics, contact_frequency, term_aliases
    TO gmail_analyst;
-- Embeddings exposed for vector-similarity analysis; the vector
-- column itself is opaque to the analyst but useful with sklearn.
GRANT SELECT ON embeddings TO gmail_analyst;
-- Everything else (costs, sync_state, conversations, agent_*, etc.)
-- stays OUT of the analyst's view by omission.
ALTER DEFAULT PRIVILEGES IN SCHEMA public REVOKE ALL ON TABLES FROM gmail_analyst;

-- ─────────────────────────────────────────────────────────────────────
-- FTS (pg_search / paradedb BM25 indexes)
--
-- SQLite had:
--   CREATE VIRTUAL TABLE messages_fts    USING fts5(...)
--   CREATE VIRTUAL TABLE attachments_fts USING fts5(...)
-- In Postgres we build Tantivy-backed BM25 indexes via pg_search. Query
-- sites use `table @@@ 'query'` (BM25 match) and rank by
-- `paradedb.score(id)`, which is a real BM25 score comparable to the
-- `bm25()` function the SQLite FTS5 path uses.
--
-- The `key_field` is the row PK so `paradedb.score(key_field)` can be
-- joined back to the base row.
-- ─────────────────────────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS messages_bm25_idx
    ON messages
    USING bm25 (id, subject, body_text, from_addr, to_addr)
    WITH (key_field = 'id');

CREATE INDEX IF NOT EXISTS attachments_bm25_idx
    ON attachments
    USING bm25 (id, filename, extracted_text)
    WITH (key_field = 'id');

-- ─────────────────────────────────────────────────────────────────────
-- Secondary indexes (translated verbatim from SQLite)
-- ─────────────────────────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_message_topics_topic ON message_topics (topic_id);
CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations (updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_conv_messages_conv ON conversation_messages (conversation_id, seq);
CREATE INDEX IF NOT EXISTS idx_model_battles_created ON model_battles (created_at);
CREATE INDEX IF NOT EXISTS idx_attachments_message_id ON attachments (message_id);

-- Reconciler watermark (`WHERE history_id > $N`) — without this the
-- drift-detector daemon seq-scans all messages every pass.
CREATE INDEX IF NOT EXISTS idx_messages_history_id ON messages (history_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_message_id ON embeddings (message_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_lookup
    ON embeddings (message_id, attachment_id, chunk_type, model);

-- ─────────────────────────────────────────────────────────────────────
-- Multi-tenant identity (Phase 1 of docs/notes/PER_USER_LOGIN_2026-04-27.md)
-- ─────────────────────────────────────────────────────────────────────
-- Tables exist unconditionally so the `gmail-search invite` CLI works
-- against an un-migrated install. Data scoping (user_id columns on
-- messages/embeddings/etc.) is gated behind GMAIL_MULTI_TENANT=1 and
-- lands in Phases 2/3.

CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    -- google_sub is OPTIONAL: the silver-oauth broker holds the only
    -- Google OAuth client and doesn't expose Google's `sub` claim, so
    -- we just leave this NULL. The column lingers from an earlier
    -- NextAuth attempt that owned its own OAuth client and needed sub
    -- to upsert against. Kept NULL-able + UNIQUE so a future migration
    -- to a real-sub world can land without dropping/recreating the
    -- column. Multi-NULL is OK in PG: NULL ≠ NULL for uniqueness.
    google_sub TEXT UNIQUE,
    email TEXT NOT NULL UNIQUE,
    name TEXT,
    avatar_url TEXT,
    invited_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_login_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_users_email ON users (email);
-- Idempotent migration for installs that already created `users` with
-- the original NOT NULL google_sub. Skipped silently on fresh installs
-- (column is already nullable) and on re-runs (same).
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'users' AND column_name = 'google_sub' AND is_nullable = 'NO'
    ) THEN
        ALTER TABLE users ALTER COLUMN google_sub DROP NOT NULL;
    END IF;
END $$;

-- `sync_enabled` opts a user in/out of the centralized supervisor's
-- watch+update+summarize fan-out. Default TRUE so newly-invited users
-- start syncing the moment they sign in. Admin can pause an account
-- (e.g. to stop billing during a trip) without deleting them.
ALTER TABLE users ADD COLUMN IF NOT EXISTS sync_enabled BOOLEAN NOT NULL DEFAULT TRUE;

CREATE TABLE IF NOT EXISTS invited_emails (
    email TEXT PRIMARY KEY,
    invited_by TEXT REFERENCES users(id) ON DELETE SET NULL,
    invited_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    note TEXT
);

-- ─────────────────────────────────────────────────────────────────────
-- Multi-tenant Phase 2 + 3 — per-user data scoping
-- ─────────────────────────────────────────────────────────────────────
-- Adds `user_id TEXT REFERENCES users(id) ON DELETE CASCADE` to every
-- table that holds per-user data. Phased rollout:
--   1. (this file) add columns NULLABLE — old rows are NULL, new rows
--      written by Phase 3 code carry the writer's user_id.
--   2. (scripts/migrate_phase23.py) chunked backfill: assign all
--      existing rows to the bootstrap user (scott).
--   3. (also in migrate_phase23.py) SET NOT NULL on critical hot paths
--      once the backfill is verified — invariant against silently
--      writing NULL user_id from a missed write site.
--   4. (Phase 3 code) every read path adds WHERE user_id = $current.
--
-- Denormalized columns (added directly even though derivable via JOIN)
-- are flagged in the comment — added so the hot-path search filter
-- doesn't have to JOIN the parent for every candidate. The bench in
-- §3a Step 5 confirmed BM25 composes cleanly with `AND user_id = $1`.
--
-- Tables intentionally kept SHARED across users:
--   * query_cache — cache hit = saved embedding cost; the query
--     embedding is opaque (just a vector) and exposes nothing about
--     other users.
--   * pg_ivm_immv (extension), spatial / postgis tables.
--   * job_progress — per-process daemon tracking, not per-user data.
--     If we want per-user job rows later, key by (user_id, job_id).

ALTER TABLE messages          ADD COLUMN IF NOT EXISTS user_id TEXT REFERENCES users(id) ON DELETE CASCADE;
ALTER TABLE embeddings        ADD COLUMN IF NOT EXISTS user_id TEXT REFERENCES users(id) ON DELETE CASCADE;  -- denormalized
ALTER TABLE attachments       ADD COLUMN IF NOT EXISTS user_id TEXT REFERENCES users(id) ON DELETE CASCADE;  -- denormalized
ALTER TABLE costs             ADD COLUMN IF NOT EXISTS user_id TEXT REFERENCES users(id) ON DELETE CASCADE;
ALTER TABLE message_summaries ADD COLUMN IF NOT EXISTS user_id TEXT REFERENCES users(id) ON DELETE CASCADE;  -- denormalized
ALTER TABLE summary_failures  ADD COLUMN IF NOT EXISTS user_id TEXT REFERENCES users(id) ON DELETE CASCADE;  -- denormalized
ALTER TABLE thread_summary    ADD COLUMN IF NOT EXISTS user_id TEXT REFERENCES users(id) ON DELETE CASCADE;
ALTER TABLE topics            ADD COLUMN IF NOT EXISTS user_id TEXT REFERENCES users(id) ON DELETE CASCADE;
ALTER TABLE message_topics    ADD COLUMN IF NOT EXISTS user_id TEXT REFERENCES users(id) ON DELETE CASCADE;  -- denormalized
ALTER TABLE contact_frequency ADD COLUMN IF NOT EXISTS user_id TEXT REFERENCES users(id) ON DELETE CASCADE;
ALTER TABLE term_aliases      ADD COLUMN IF NOT EXISTS user_id TEXT REFERENCES users(id) ON DELETE CASCADE;
ALTER TABLE model_battles     ADD COLUMN IF NOT EXISTS user_id TEXT REFERENCES users(id) ON DELETE CASCADE;
ALTER TABLE conversations     ADD COLUMN IF NOT EXISTS user_id TEXT REFERENCES users(id) ON DELETE CASCADE;
ALTER TABLE agent_sessions    ADD COLUMN IF NOT EXISTS user_id TEXT REFERENCES users(id) ON DELETE CASCADE;  -- denormalized

-- PK changes for tables with existing rows (contact_frequency,
-- term_aliases, topics) are NOT done here — they require user_id NOT
-- NULL, which depends on backfill. See scripts/migrate_phase23.py.
--
-- scann_index_pointer is special: a fresh schema has zero rows in it
-- (the row only gets created on the first reindex), so we CAN promote
-- it to per-user shape inline. This makes test fixtures work without
-- having to also run migrate_phase23.py — the production migration
-- script handles the case where existing rows need backfill.
ALTER TABLE scann_index_pointer ADD COLUMN IF NOT EXISTS user_id TEXT REFERENCES users(id) ON DELETE CASCADE;
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'scann_index_pointer' AND column_name = 'id'
    ) THEN
        -- Only safe if the table is empty; production rows are
        -- backfilled first by migrate_phase23.py before this runs.
        IF (SELECT COUNT(*) FROM scann_index_pointer WHERE user_id IS NULL) = 0 THEN
            ALTER TABLE scann_index_pointer ALTER COLUMN user_id SET NOT NULL;
            ALTER TABLE scann_index_pointer DROP CONSTRAINT IF EXISTS scann_index_pointer_pkey;
            ALTER TABLE scann_index_pointer DROP COLUMN id;
            ALTER TABLE scann_index_pointer ADD PRIMARY KEY (user_id);
        END IF;
    END IF;
END $$;

-- Per-user query indexes — created CONCURRENTLY in the migration
-- script (CREATE INDEX CONCURRENTLY can't run inside a transaction,
-- and pg_schema.sql is applied as one). The ones below are safe to
-- declare here because they're index-creation-on-empty-or-new tables
-- that complete in microseconds.
CREATE INDEX IF NOT EXISTS idx_messages_user_date          ON messages          (user_id, date DESC);
CREATE INDEX IF NOT EXISTS idx_embeddings_user             ON embeddings        (user_id);
CREATE INDEX IF NOT EXISTS idx_attachments_user            ON attachments       (user_id);
CREATE INDEX IF NOT EXISTS idx_costs_user_ts               ON costs             (user_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_thread_summary_user         ON thread_summary    (user_id);
CREATE INDEX IF NOT EXISTS idx_message_summaries_user      ON message_summaries (user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_user_updated  ON conversations    (user_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_sessions_user         ON agent_sessions   (user_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_model_battles_user          ON model_battles    (user_id, created_at DESC);

-- ─────────────────────────────────────────────────────────────────────
-- Phase 3g — Row-Level Security for the /api/sql analyst sandbox
-- ─────────────────────────────────────────────────────────────────────
-- The deep-mode planner can write SQL like `SELECT count(*) FROM
-- messages` with no `WHERE user_id = ...` and silently see other
-- household members' rows. Instruction-only mitigations don't hold:
-- the planner's prompt is static, doesn't fetch our schema preamble,
-- and Gemini is happy to write unscoped SQL when the question is
-- "how many emails total?".
--
-- This block locks it down at the storage layer:
--   1. A non-superuser role `gmail_search_reader` (NOLOGIN — entered
--      via SET LOCAL ROLE inside the /api/sql transaction). It has
--      SELECT only on the LLM-facing tables (the same set documented
--      in TABLE_DOCS / describe_schema_for_llm). Internal tables
--      (query_cache, sync_state, agent_*, scann_*, invited_emails,
--      users, job_progress, embeddings, spatial_ref_sys) are
--      intentionally NOT granted, so `SELECT * FROM query_cache` from
--      the LLM raises "permission denied" rather than leaking.
--   2. ENABLE + FORCE ROW LEVEL SECURITY on every per-user table.
--      A single policy reads `current_setting('app.user_id', true)`
--      — the second arg makes a missing setting return NULL instead
--      of erroring, and NULL fails the equality so no rows are
--      visible. That's the safe default for any code path that
--      forgets to set the variable.
--   3. /api/sql opens its psycopg connection, then inside one
--      transaction does:
--         SET LOCAL ROLE gmail_search_reader;
--         SET LOCAL app.user_id = <active user>;
--         <the LLM's query>;
--      ROLLBACK at end (or COMMIT — the query is read-only). After
--      the transaction the connection's role/setting reset.
--
-- Daemon writes (the bootstrap user, the watch/update/summarize
-- daemons) keep connecting as `gmail_search` (superuser, BYPASSRLS),
-- so RLS doesn't get in their way. The lockdown is specifically for
-- the LLM-driven /api/sql endpoint — the one place a hostile or
-- forgetful prompt can run free-form SELECT.

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'gmail_search_reader') THEN
        CREATE ROLE gmail_search_reader NOLOGIN;
    END IF;
END $$;

GRANT USAGE ON SCHEMA public TO gmail_search_reader;
-- LLM-facing tables only. Keep this list in sync with TABLE_DOCS in
-- src/gmail_search/store/db.py — anything documented for the LLM
-- needs SELECT here; anything in `_INTERNAL_TABLES` must NOT.
GRANT SELECT ON
    messages,
    attachments,
    contact_frequency,
    conversations,
    conversation_messages,
    costs,
    job_progress,
    message_summaries,
    message_topics,
    model_battles,
    summary_failures,
    term_aliases,
    thread_summary,
    topics
TO gmail_search_reader;

-- Per-user RLS for the tables that have a `user_id` column directly.
DO $$
DECLARE
    tbl TEXT;
BEGIN
    FOREACH tbl IN ARRAY ARRAY[
        'messages',
        'attachments',
        'contact_frequency',
        'conversations',
        'costs',
        'message_summaries',
        'message_topics',
        'model_battles',
        'summary_failures',
        'term_aliases',
        'thread_summary',
        'topics'
    ]
    LOOP
        EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', tbl);
        -- FORCE so the policy applies to the table owner too, not just
        -- to other roles. Combined with the reader role's lack of
        -- BYPASSRLS, this is what makes the gate effective.
        EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', tbl);
        EXECUTE format('DROP POLICY IF EXISTS tenant_isolation ON %I', tbl);
        EXECUTE format(
            'CREATE POLICY tenant_isolation ON %I '
            'USING (user_id::text = current_setting(''app.user_id'', true)) '
            'WITH CHECK (user_id::text = current_setting(''app.user_id'', true))',
            tbl
        );
    END LOOP;
END $$;

-- Multi-tenant PK hardening: thread_summary.thread_id and topics.topic_id are
-- only unique within ONE mailbox (Gmail thread ids; per-user sequential topic
-- ids), so the primary key must include user_id or two tenants with a colliding
-- id can't coexist. Idempotent: only swaps a single-column PK, so re-runs and
-- fresh installs (which already land here after user_id is added) are no-ops.
-- Re-keys the message_topics -> topics FK to the composite key.
DO $$
BEGIN
    -- NULL guard: on an un-backfilled install user_id may be NULL; adding it to
    -- the PK would abort the whole schema apply. Skip until backfilled (no-op),
    -- so init_db never fails on a partially-migrated DB.
    IF EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conname = 'thread_summary_pkey' AND conrelid = 'thread_summary'::regclass
           AND array_length(conkey, 1) = 1
    ) AND NOT EXISTS (SELECT 1 FROM thread_summary WHERE user_id IS NULL) THEN
        ALTER TABLE thread_summary DROP CONSTRAINT thread_summary_pkey,
            ADD PRIMARY KEY (thread_id, user_id);
    END IF;

    IF EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conname = 'topics_pkey' AND conrelid = 'topics'::regclass
           AND array_length(conkey, 1) = 1
    ) AND NOT EXISTS (SELECT 1 FROM topics WHERE user_id IS NULL) THEN
        ALTER TABLE message_topics DROP CONSTRAINT IF EXISTS message_topics_topic_id_fkey;
        ALTER TABLE topics DROP CONSTRAINT topics_pkey,
            ADD PRIMARY KEY (topic_id, user_id);
        ALTER TABLE message_topics
            ADD CONSTRAINT message_topics_topic_id_fkey
            FOREIGN KEY (topic_id, user_id) REFERENCES topics(topic_id, user_id);
    END IF;
END $$;

-- conversation_messages doesn't have its own user_id — scope through
-- the parent conversation. Subquery RLS is slower per row but at
-- household scale (2-5 users) it's unmeasurable.
ALTER TABLE conversation_messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE conversation_messages FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS tenant_isolation ON conversation_messages;
CREATE POLICY tenant_isolation ON conversation_messages
    USING (EXISTS (
        SELECT 1 FROM conversations c
         WHERE c.id = conversation_messages.conversation_id
           AND c.user_id::text = current_setting('app.user_id', true)
    ));

-- ─────────────────────────────────────────────────────────────────────
-- Delta vs SQLite
-- ─────────────────────────────────────────────────────────────────────
-- Tables intentionally DROPPED relative to SQLite:
--   * messages_fts, attachments_fts (and all their FTS5 shadow tables:
--     *_data, *_idx, *_docsize, *_config). Replaced by pg_search BM25
--     indexes (messages_bm25_idx, attachments_bm25_idx) on the base
--     tables.
--
-- Tables intentionally KEPT identical (even though a future pass may
-- migrate them) to keep this translation mechanical:
--   * All ISO-8601-as-TEXT date columns (date, *_at, started_at,
--     updated_at, date_first, date_last). String-sort = chronological
--     is what existing INSERT/WHERE sites rely on.
--   * JSON-in-TEXT columns (labels, participants, all_from_addrs,
--     all_labels, top_senders, sample_subjects, expansions, parts,
--     variant_a, variant_b, raw_json). JSONB migration is a follow-up.
--
-- Behavior change we accept:
--   * FTS ranking is BM25 on both sides, but via different implementations:
--     SQLite FTS5 uses its built-in bm25() on shadow tables; Postgres uses
--     pg_search's Tantivy-backed BM25 on in-table indexes. Scores are in
--     the same family but not bit-identical — top-N overlap in the A/B
--     harness is the gate (target ≥ 0.8).
