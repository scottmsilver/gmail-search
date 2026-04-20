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
--   * FTS5 virtual tables                 -> replaced with generated tsvector columns + GIN indexes
--
-- Every CREATE TABLE uses IF NOT EXISTS so this file is idempotent.

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
    raw_json TEXT NOT NULL DEFAULT '{}',
    -- FTS replacement for messages_fts (FTS5 virtual table in SQLite).
    -- Weights mirror the SQLite FTS column order: subject is primary (A),
    -- from/to are secondary (B), body is the bulk content (C).
    fts tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(subject, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(from_addr, '')), 'B') ||
        setweight(to_tsvector('english', coalesce(to_addr, '')), 'B') ||
        setweight(to_tsvector('english', coalesce(body_text, '')), 'C')
    ) STORED
);

CREATE TABLE IF NOT EXISTS attachments (
    id BIGSERIAL PRIMARY KEY,
    message_id TEXT NOT NULL REFERENCES messages(id),
    filename TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    size_bytes BIGINT NOT NULL DEFAULT 0,
    extracted_text TEXT,
    image_path TEXT,
    raw_path TEXT,
    UNIQUE (message_id, filename),
    -- FTS replacement for attachments_fts.
    fts tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(filename, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(extracted_text, '')), 'C')
    ) STORED
);

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
    estimated_cost_usd DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    message_id TEXT NOT NULL
);

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
    updated_at TEXT NOT NULL
);

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
-- FTS (Postgres equivalent of FTS5 virtual tables)
--
-- SQLite had:
--   CREATE VIRTUAL TABLE messages_fts    USING fts5(...)
--   CREATE VIRTUAL TABLE attachments_fts USING fts5(...)
-- In Postgres we put generated tsvector columns directly on the base
-- tables (see `fts` columns above) and index them with GIN. Query sites
-- that used `messages_fts MATCH 'foo'` become
-- `messages.fts @@ plainto_tsquery('english', 'foo')` (or
-- `websearch_to_tsquery`) and rank with ts_rank_cd instead of bm25.
-- ─────────────────────────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_messages_fts ON messages USING GIN (fts);
CREATE INDEX IF NOT EXISTS idx_attachments_fts ON attachments USING GIN (fts);

-- ─────────────────────────────────────────────────────────────────────
-- Secondary indexes (translated verbatim from SQLite)
-- ─────────────────────────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_message_topics_topic ON message_topics (topic_id);
CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations (updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_conv_messages_conv ON conversation_messages (conversation_id, seq);
CREATE INDEX IF NOT EXISTS idx_model_battles_created ON model_battles (created_at);
CREATE INDEX IF NOT EXISTS idx_attachments_message_id ON attachments (message_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_message_id ON embeddings (message_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_lookup
    ON embeddings (message_id, attachment_id, chunk_type, model);

-- ─────────────────────────────────────────────────────────────────────
-- Delta vs SQLite
-- ─────────────────────────────────────────────────────────────────────
-- Tables intentionally DROPPED relative to SQLite:
--   * messages_fts, attachments_fts (and all their FTS5 shadow tables:
--     *_data, *_idx, *_docsize, *_config). Replaced by `messages.fts`
--     and `attachments.fts` generated tsvector columns + GIN indexes.
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
--   * FTS ranking differs: Postgres uses ts_rank_cd, SQLite FTS5 uses
--     bm25. Scores aren't comparable and top-N ordering can shift.
--     The model-battle / A/B harness has an overlap gate that catches
--     unacceptable drift — rely on it when flipping the backend.
