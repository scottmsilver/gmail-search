#!/usr/bin/env bash
# Show the current top-10 slowest queries by total exec time, from
# pg_stat_statements. `paradedb/paradedb:latest-pg16` has this preloaded.
#
# Common moves after running this:
#   * See a slow DELETE / SELECT doing a seq scan on a column without
#     an index → add one:   CREATE INDEX CONCURRENTLY ... ;
#   * Reset stats when you want to see what's slow in a fresh window:
#         docker exec gmail-search-pg psql -U gmail_search -d gmail_search \
#             -c "SELECT pg_stat_statements_reset();"
#   * Individual slow queries (>500ms) are also written to the
#     postgres container log:
#         docker logs gmail-search-pg 2>&1 | grep -iE "duration:"
set -euo pipefail

docker exec gmail-search-pg psql -U gmail_search -d gmail_search -c "
SELECT
  round(mean_exec_time::numeric, 1) AS mean_ms,
  calls,
  round((total_exec_time/1000)::numeric, 1) AS total_s,
  round((100.0 * total_exec_time / sum(total_exec_time) OVER ())::numeric, 1) AS pct,
  substring(query, 1, 120) AS q
FROM pg_stat_statements
WHERE query NOT ILIKE '%pg_stat_statements%'
  AND calls >= 3
ORDER BY total_exec_time DESC
LIMIT 15;"
