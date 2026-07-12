-- One-time cleanup for the summarize poison-message fix.
--
-- Backstory: the work-selector never excluded summary_failures, so a message
-- whose body carried a lone UTF-8 surrogate crashed on request-encoding
-- (UnicodeEncodeError, "surrogates not allowed") and was re-selected EVERY
-- pass — one message reached attempts = 71,111 at ~31s each, blocking all
-- real summarize progress. The code fix (llm/openrouter.py::_scrub_surrogates
-- + fail-fast on non-retryable errors, and the selector's attempts-cap +
-- exponential backoff in summarize.py) makes these messages summarizable.
--
-- This clears the now-stale encode failures so those messages re-enter the
-- queue and get summarized on the next run. Transient failures (ReadTimeout /
-- 5xx) are left alone — they retry naturally under the new backoff.
--
-- Safe to re-run.
--
--   docker exec -i gmail-search-pg psql -U gmail_search -d gmail_search \
--     < src/gmail_search/store/pg_migration_summary_poison.sql

DELETE FROM summary_failures
 WHERE error LIKE '%surrogate%'
    OR error LIKE '%UnicodeEncodeError%';
