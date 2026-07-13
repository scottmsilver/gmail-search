-- Index the summarize work-selector so it stops seq-scanning + sorting the
-- whole mailbox on every pass.
--
-- The selector (_messages_needing_summary) picks the newest un-summarized
-- INBOX message for a user. Before this index its plan was a Parallel Seq Scan
-- of all ~416K messages + a 22MB Sort (the ORDER BY used computed, unindexable
-- expressions). This partial index on (user_id, date DESC) WHERE INBOX matches
-- the simplified `ORDER BY m.date DESC`, so the planner walks it newest-first
-- and stops at LIMIT.
--
-- Measured (EXPLAIN ANALYZE, ~416K messages):
--   before: 554 ms (Seq Scan + Sort), every pass regardless of pending count
--   after : 0.22 ms when mail is pending (stops at LIMIT); 273 ms when caught up
--
-- CONCURRENTLY = online, no write lock on `messages`. It CANNOT run inside a
-- transaction block, so apply this file on its own (psql autocommits a lone
-- statement). Safe to re-run (IF NOT EXISTS).
--
--   docker exec -i gmail-search-pg psql -U gmail_search -d gmail_search \
--     < src/gmail_search/store/pg_migration_summarize_index.sql

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_messages_summarize_inbox
    ON messages (user_id, date DESC)
    WHERE labels LIKE '%"INBOX"%';
