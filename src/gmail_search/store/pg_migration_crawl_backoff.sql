-- One-time migration for the exponential-backoff crawl-lane change that
-- raises _MAX_CRAWL_ATTEMPTS from 4 to 10 (see store/queries.py).
--
-- WHY: widening the cap would otherwise make `pending_url_stubs` re-select
-- every URL that was already abandoned under the OLD cap of 4 (~336K rows on
-- the current DB, many at hundreds of historical attempts). Re-crawling them
-- would re-leak to those origins and re-load the box for links we already
-- proved dead. This bumps every already-abandoned row (crawl_attempts >= old
-- cap) straight to the new cap so it stays permanently dead.
--
-- WHEN: run ONCE, at or before deploying the new code. Safe to re-run
-- (idempotent — the < 10 guard means a second run is a no-op).
--
--   docker exec -i gmail-search-pg psql -U gmail_search -d gmail_search \
--     < src/gmail_search/store/pg_migration_crawl_backoff.sql

UPDATE attachments
   SET crawl_attempts = 10
 WHERE mime_type = 'text/html'
   AND extracted_text IS NULL
   AND filename LIKE 'URL: %'
   AND crawl_attempts >= 4
   AND crawl_attempts < 10;
