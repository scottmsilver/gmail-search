#!/usr/bin/env bash
# Phase 0d — staged ALTER TABLE timing on a real-sized clone.
#
# Plan §3a claims the per-table user_id migration can be staged
# (ADD NULL → chunked backfill → SET NOT NULL → ADD FK NOT VALID →
# VALIDATE) with manageable downtime. This script measures it on a
# clone of the prod DB so we know the actual wall clock before touching
# real data.
#
# Tables under test:
#   - messages   (410k rows, 14 GB, has pg_search BM25 index)
#   - embeddings (565k rows,  8 GB, 5 btree indexes)
#
# Output: per-step elapsed seconds + final BM25-with-user-id sanity check.
# On exit (success or failure), the clone DB is dropped.

set -uo pipefail

# Confirmation gate: this script clones two large tables (~22 GB
# transient disk) and creates a new database. The user has to opt in
# explicitly so a stray invocation doesn't blow up dev disk.
if [[ "${ALLOW_PROD_CLONE:-0}" != "1" ]]; then
    cat >&2 <<'EOF'
refusing to run: this script pg_dumps + restores ~22 GB of data
(messages 14 GB + embeddings 8 GB) into a new DB and runs an
ALTER TABLE migration on it. To confirm, re-run with:
    ALLOW_PROD_CLONE=1 ./scripts/bench_phase0d_alter_table.sh
EOF
    exit 1
fi

# Connection info: defaults match the local ParadeDB compose stack;
# override via env to point at a different host / creds. We never
# write the password to a file — pass it via PGPASSWORD inline below.
DB_HOST="${PGHOST:-127.0.0.1}"
DB_PORT="${PGPORT:-5544}"
DB_USER="${PGUSER:-gmail_search}"
DB_PASS="${PGPASSWORD:-gmail_search}"
DB_NAME="${PGDATABASE:-gmail_search}"
CLONE_NAME="${CLONE_NAME:-gmail_search_clone}"

SOURCE_DSN="postgresql://${DB_USER}:${DB_PASS}@${DB_HOST}:${DB_PORT}/${DB_NAME}"
CLONE_DSN="postgresql://${DB_USER}:${DB_PASS}@${DB_HOST}:${DB_PORT}/${CLONE_NAME}"
ADMIN_DSN="postgresql://${DB_USER}:${DB_PASS}@${DB_HOST}:${DB_PORT}/postgres"

# All psql/pg_dump invocations get -v ON_ERROR_STOP=1 so a SQL error
# aborts the script rather than silently continuing.
PSQL_OPTS=(-v ON_ERROR_STOP=1)

log() { printf '\n=== %s — %s\n' "$(date +%T)" "$*"; }
sql_clone() { PGPASSWORD="${DB_PASS}" psql "${PSQL_OPTS[@]}" "${CLONE_DSN}" -At "$@"; }
sql_admin() { PGPASSWORD="${DB_PASS}" psql "${PSQL_OPTS[@]}" "${ADMIN_DSN}" "$@"; }

# Wrap a SQL statement in psql + report wall-clock seconds. Output goes
# to the named results-file (tab-separated: label TAB seconds).
RESULTS=$(mktemp)
time_step() {
    local label=$1
    local sql=$2
    local t0 t1 elapsed
    t0=$(date +%s.%N)
    if ! sql_clone -c "${sql}" > /dev/null; then
        printf '%s\tFAILED\n' "${label}" >> "${RESULTS}"
        return 1
    fi
    t1=$(date +%s.%N)
    elapsed=$(awk "BEGIN {printf \"%.3f\", ${t1} - ${t0}}")
    printf '%s\t%s\n' "${label}" "${elapsed}" >> "${RESULTS}"
    printf '  → %s sec\n' "${elapsed}"
}

cleanup() {
    log "teardown: dropping ${CLONE_NAME}"
    sql_admin -c "DROP DATABASE IF EXISTS ${CLONE_NAME};" >/dev/null 2>&1 || true
    log "results"
    column -t -s $'\t' "${RESULTS}"
    rm -f "${RESULTS}"
}
trap cleanup EXIT

# --- Phase A: clone setup -----------------------------------------------------
log "create empty clone DB + extension"
sql_admin -c "DROP DATABASE IF EXISTS ${CLONE_NAME};" >/dev/null
sql_admin -c "CREATE DATABASE ${CLONE_NAME};" >/dev/null
sql_clone -c "CREATE EXTENSION IF NOT EXISTS pg_search;" >/dev/null

log "pg_dump messages + embeddings → restore (this is the long step)"
t0=$(date +%s.%N)
PGPASSWORD="${DB_PASS}" pg_dump \
    --no-owner --no-privileges --no-publications --no-subscriptions \
    --table=messages --table=embeddings \
    "${SOURCE_DSN}" | PGPASSWORD="${DB_PASS}" psql "${PSQL_OPTS[@]}" "${CLONE_DSN}" > /dev/null
t1=$(date +%s.%N)
elapsed=$(awk "BEGIN {printf \"%.1f\", ${t1} - ${t0}}")
printf 'A_dump_restore\t%s\n' "${elapsed}" >> "${RESULTS}"
printf '  → %s sec\n' "${elapsed}"

log "verify clone rowcounts"
sql_clone -c "SELECT 'messages' AS tbl, count(*) FROM messages UNION ALL SELECT 'embeddings', count(*) FROM embeddings;"

# --- Phase B: staged ALTER playbook ------------------------------------------
log "create users(id) so the FK has a target"
sql_clone -c "
    CREATE TABLE IF NOT EXISTS users (id TEXT PRIMARY KEY);
    INSERT INTO users (id) VALUES ('scott') ON CONFLICT DO NOTHING;
" >/dev/null

# Step 1: ADD COLUMN NULL (no DEFAULT, no FK) — should be O(1) on PG 11+.
log "step 1a: ALTER messages ADD COLUMN user_id NULL"
time_step "B1_add_col_messages"   "ALTER TABLE messages   ADD COLUMN user_id TEXT;"
time_step "B1_add_col_embeddings" "ALTER TABLE embeddings ADD COLUMN user_id TEXT;"

# Step 2: chunked backfill — the dangerous step. Plan claims a single
# UPDATE on 410k rows = full table rewrite + MVCC bloat; this loop with
# 5000-row batches should bound it.
log "step 2: chunked backfill messages (5000 rows / batch)"
t0=$(date +%s.%N)
sql_clone -c "
    DO \$\$
    DECLARE n_updated INT;
    BEGIN
      LOOP
        UPDATE messages SET user_id = 'scott'
         WHERE id IN (SELECT id FROM messages WHERE user_id IS NULL LIMIT 5000);
        GET DIAGNOSTICS n_updated = ROW_COUNT;
        EXIT WHEN n_updated = 0;
      END LOOP;
    END \$\$;" > /dev/null
t1=$(date +%s.%N)
elapsed=$(awk "BEGIN {printf \"%.1f\", ${t1} - ${t0}}")
printf 'B2_backfill_messages\t%s\n' "${elapsed}" >> "${RESULTS}"
printf '  → %s sec\n' "${elapsed}"

log "step 2: chunked backfill embeddings (5000 rows / batch)"
t0=$(date +%s.%N)
sql_clone -c "
    DO \$\$
    DECLARE n_updated INT;
    BEGIN
      LOOP
        UPDATE embeddings SET user_id = 'scott'
         WHERE id IN (SELECT id FROM embeddings WHERE user_id IS NULL LIMIT 5000);
        GET DIAGNOSTICS n_updated = ROW_COUNT;
        EXIT WHEN n_updated = 0;
      END LOOP;
    END \$\$;" > /dev/null
t1=$(date +%s.%N)
elapsed=$(awk "BEGIN {printf \"%.1f\", ${t1} - ${t0}}")
printf 'B2_backfill_embeddings\t%s\n' "${elapsed}" >> "${RESULTS}"
printf '  → %s sec\n' "${elapsed}"

# Step 3: SET NOT NULL — this scans the whole table to verify, blocking
# writes for the duration. This is the second-most-dangerous step.
log "step 3: SET NOT NULL"
time_step "B3_notnull_messages"   "ALTER TABLE messages   ALTER COLUMN user_id SET NOT NULL;"
time_step "B3_notnull_embeddings" "ALTER TABLE embeddings ALTER COLUMN user_id SET NOT NULL;"

# Step 4: ADD FK NOT VALID is fast; VALIDATE CONSTRAINT scans rows but
# only takes a SHARE lock (concurrent reads/writes still allowed).
log "step 4: ADD FK NOT VALID + VALIDATE"
time_step "B4_addfk_messages" \
    "ALTER TABLE messages ADD CONSTRAINT fk_msgs_user FOREIGN KEY (user_id) REFERENCES users(id) NOT VALID;"
time_step "B4_validate_messages" \
    "ALTER TABLE messages VALIDATE CONSTRAINT fk_msgs_user;"
time_step "B4_addfk_embeddings" \
    "ALTER TABLE embeddings ADD CONSTRAINT fk_emb_user FOREIGN KEY (user_id) REFERENCES users(id) NOT VALID;"
time_step "B4_validate_embeddings" \
    "ALTER TABLE embeddings VALIDATE CONSTRAINT fk_emb_user;"

# Step 5: per-user query indexes. CONCURRENTLY can't run inside a
# transaction; psql -c gives us auto-commit so it's fine here.
log "step 5: CREATE INDEX CONCURRENTLY (per-user query path)"
time_step "B5_idx_messages_user_date" \
    "CREATE INDEX CONCURRENTLY idx_messages_user_date ON messages (user_id, date DESC);"
time_step "B5_idx_embeddings_user" \
    "CREATE INDEX CONCURRENTLY idx_embeddings_user ON embeddings (user_id);"

# --- Phase C: BM25 + user_id filter sanity check -----------------------------
log "C: BM25 with AND user_id filter — does pg_search compose with the WHERE?"
t0=$(date +%s.%N)
hit_count=$(sql_clone -t -c "
    SELECT count(*) FROM messages
     WHERE id @@@ 'subject:invoice'
       AND user_id = 'scott';" | tr -d ' ')
t1=$(date +%s.%N)
elapsed=$(awk "BEGIN {printf \"%.3f\", ${t1} - ${t0}}")
printf 'C_bm25_with_user_filter\t%s\n' "${elapsed}" >> "${RESULTS}"
printf '  → %s rows in %s sec\n' "${hit_count}" "${elapsed}"

# Compare against the same BM25 query WITHOUT the user filter — confirm
# the user filter doesn't blow recall (in our single-user clone they
# should match exactly).
hit_count_unfiltered=$(sql_clone -t -c "
    SELECT count(*) FROM messages WHERE id @@@ 'subject:invoice';" | tr -d ' ')
if [[ "${hit_count}" == "${hit_count_unfiltered}" ]]; then
    printf 'C_bm25_recall_match\tOK_%s_eq_%s\n' "${hit_count}" "${hit_count_unfiltered}" >> "${RESULTS}"
    printf '  ✓ filtered (%s) == unfiltered (%s) — BM25 composes cleanly with user_id\n' \
        "${hit_count}" "${hit_count_unfiltered}"
else
    printf 'C_bm25_recall_match\tMISMATCH_%s_vs_%s\n' "${hit_count}" "${hit_count_unfiltered}" >> "${RESULTS}"
    printf '  ✗ filtered (%s) != unfiltered (%s) — recall changed; investigate before plan §3a Step 5\n' \
        "${hit_count}" "${hit_count_unfiltered}"
fi
