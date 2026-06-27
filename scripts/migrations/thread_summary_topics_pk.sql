-- DEFERRED multi-tenant PK hardening for thread_summary + topics.
--
-- Status: NOT applied. The scoping/attribution fix (commit fae5042) already makes
-- the daemons tenant-correct on real data. This PK change only lets two tenants
-- with a COLLIDING per-mailbox id (thread_id / topic_id) coexist — astronomically
-- unlikely with random Gmail ids — so it's deferred to a maintenance window.
--
-- COUPLING WARNING: this DDL is coupled to code. After the PK becomes
-- (thread_id, user_id), `ON CONFLICT(thread_id)` in recompute_thread_summary
-- becomes INVALID. upsert_message runs that upsert in the SAME transaction as the
-- message insert with NO try/except, so a mismatch breaks live ingest. Apply as
-- ONE coordinated deploy:
--   1. Code: recompute_thread_summary `ON CONFLICT(thread_id)` -> `ON CONFLICT(thread_id, user_id)`.
--      (topics has no ON CONFLICT path.)
--   2. Apply this DDL.
--   3. Restart daemons (editable install) so code + constraints line up.
--
-- PRE-CHECKS — all must return 0 before running:
--   SELECT count(*) FROM (SELECT thread_id FROM thread_summary GROUP BY thread_id HAVING count(*)>1) x;
--   SELECT count(*) FROM (SELECT topic_id, user_id FROM topics GROUP BY topic_id, user_id HAVING count(*)>1) y;
--   SELECT count(*) FROM thread_summary WHERE user_id IS NULL;
--   SELECT count(*) FROM topics WHERE user_id IS NULL;
--
-- ALSO VERIFY: no foreign key references the columns being re-keyed
--   (e.g. message_topics(topic_id) -> topics). If one exists, drop + recreate it
--   to reference (topic_id, user_id), or it will block the PK swap. Check with:
--   SELECT conname, conrelid::regclass FROM pg_constraint
--    WHERE confrelid IN ('thread_summary'::regclass, 'topics'::regclass);

BEGIN;
ALTER TABLE thread_summary DROP CONSTRAINT thread_summary_pkey,
  ADD PRIMARY KEY (thread_id, user_id);
ALTER TABLE topics DROP CONSTRAINT topics_pkey,
  ADD PRIMARY KEY (topic_id, user_id);
COMMIT;
