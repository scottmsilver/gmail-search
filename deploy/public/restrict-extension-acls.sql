-- Exact audited extension objects. Existing roles are gmail_search (superuser),
-- gmail_analyst and gmail_search_reader; retain their prior access explicitly.
GRANT CREATE ON SCHEMA paradedb, pdb TO gmail_analyst, gmail_search_reader;
REVOKE CREATE ON SCHEMA paradedb, pdb FROM PUBLIC;
GRANT ALL ON TABLE paradedb._typmod_cache TO gmail_analyst, gmail_search_reader;
REVOKE ALL ON TABLE paradedb._typmod_cache FROM PUBLIC;
GRANT SELECT ON TABLE public.pg_stat_statements, public.pg_stat_statements_info
  TO gmail_analyst, gmail_search_reader;
REVOKE SELECT ON TABLE public.pg_stat_statements, public.pg_stat_statements_info FROM PUBLIC;
GRANT EXECUTE ON FUNCTION paradedb._save_typmod(text[]) TO gmail_analyst, gmail_search_reader;
REVOKE EXECUTE ON FUNCTION paradedb._save_typmod(text[]) FROM PUBLIC;
