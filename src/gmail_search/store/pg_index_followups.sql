-- Indexes added 2026-04-20 after pg_stat_statements surfaced slow patterns.
-- Both are safe to run multiple times (IF NOT EXISTS).

-- Supports FK from embeddings.attachment_id → attachments.id so that
-- DELETE FROM attachments is O(log N) instead of a 546k-row seq scan per
-- deleted row. Without this, a 10k-row batched DELETE took ~30min; with
-- it, ~30s. Partial index (attachment_id IS NOT NULL) keeps it small
-- since most embeddings are on messages, not attachments.
CREATE INDEX IF NOT EXISTS idx_embeddings_attachment_id
  ON embeddings (attachment_id) WHERE attachment_id IS NOT NULL;

-- Supports `SELECT ... FROM embeddings WHERE model = $1 ORDER BY id` used
-- by the restricted-vector-search path in search/engine.py. The existing
-- composite idx_embeddings_lookup has model as the 4th column so it can't
-- be used for a WHERE model = $1 lookup.
CREATE INDEX IF NOT EXISTS idx_embeddings_model_id
  ON embeddings (model, id);
