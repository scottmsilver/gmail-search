# Gate 0: pre-existing gateway test failures and a hang

2026-09-16. Recorded while qualifying the pg_search ctid patch. **None of these
are caused by that patch**, by the rebuilt rehearsal container, or by the
phase-two migration work — each was reproduced identically on the original
unpatched `paradedb/paradedb:latest-pg16` image with stock settings.

## Why this took so long to surface

Gate 0 never completed on three attempts. It was not slow, it was **wedged**: a
connection sat `idle in transaction` holding
`LOCK TABLE public.messages IN ACCESS EXCLUSIVE MODE` — 33 minutes in one case —
with pytest blocked in `ep_poll`. One of those attempts was killed by the
system for low memory after running for hours, which looked like a resource
problem and was not.

Two test files take that lock deliberately, to prove a gateway query blocks on
it. When the surrounding assertion path misbehaves, the blocker connection is
never closed and the whole suite stops behind it.

## `tests/test_gateway_query_execution.py` — one hang, five failures

Hangs indefinitely at `test_inflight_owner_revocation_closes_connection`; the
run has to be killed. Outcomes before the hang are **byte-for-byte identical**
on the patched and original engines:

| Test | Result |
| --- | --- |
| `test_gateway_whole_mailbox_aggregate_and_bounded_extract` | FAILED |
| `test_gateway_rejects_revoked_unknown_users_and_arbitrary_sql` | PASSED |
| `test_gateway_byte_limit_marks_incomplete` | FAILED |
| `test_gateway_cancellation_terminates_query_and_frees_slot` | FAILED |
| `test_gateway_deadline_and_revocation_cancel_waiting_query` | FAILED |
| `test_oversized_row_is_bounded_before_transport` | FAILED |
| `test_inflight_owner_revocation_closes_connection` | **HANGS** |

## `tests/test_gateway_service.py` — three failures, no hang in isolation

Completes in ~11 s alone, so it is not the wedger, but three tests fail —
again identically on both engines:

- `test_token_selects_only_its_owners_rows`
- `test_inflight_revocation_or_caller_cancellation_closes_database_connection[False]`
- `test_inflight_revocation_or_caller_cancellation_closes_database_connection[True]`

The first is the one to look at: the name asserts owner isolation, which is the
project's core property. The failure surfaces as
`RuntimeError: Analytical query could not be completed` from
`gateway/database.py:204`, which deliberately masks the underlying
`psycopg.Error` so PostgreSQL detail cannot leak. No server-side ERROR is
logged, so the cause is client-side and needs the mask lifted temporarily to
diagnose. **This is a failing test, not a demonstrated isolation breach** — the
query does not complete at all, rather than returning another owner's rows.

## Full per-file sweep result

Running each file separately under a 240 s timeout, dropping leftover databases
between files (`scratchpad/gate0_sweep.sh`), completes where a single invocation
cannot:

**54 files clean, 13 with problems — 1121 passed, 25 failed, 35 errors.**

Every migration-related suite passes:

| Suite | Result |
| --- | --- |
| `test_gateway_maintenance.py` | 33 passed |
| `test_text_owner_partitions.py` | 25 passed |
| `test_mixed_text_owner_partitions.py` | 30 passed |
| `test_mixed_text_owner_partitions_phase_two.py` | 6 passed |
| `test_gateway_search_ranking.py` | 13 passed |
| `test_search_ranking.py` | 5 passed |

The failures cluster in gateway attachment, retrieval, provisioning, service and
HTTP suites — the gateway's own release surface, not the migration's.

## 35 of those errors were the PostgreSQL version pin — RESOLVED 2026-09-16

**Decision: Scott approved the 16.15 upgrade.** The pins were moved from
`160013` to `160015` in `tests/test_gateway_partition_provision.py` (whose
`partition_database` fixture `test_owner_partition_ranking.py` imports) and
`deploy/public/probe_bm25_deleted_statistics.py`. Both suites now pass:
**35 passed**, previously 35 errors.

Sweep totals become **1156 passed, 25 failed, 0 errors**, with the 25 failures
all pre-existing and confined to the gateway's own surface.

Phase 4 therefore deploys PostgreSQL 16.13 → 16.15 alongside the ctid fix, as an
accepted change rather than an accident. The original analysis follows.

### Original analysis

`test_gateway_partition_provision.py` (32 errors) and
`test_owner_partition_ranking.py` (3 errors) all fail the same assertion:

```
assert 160015 == 160013
```

That is a **PostgreSQL server version pin**. Live runs **16.13** (`160013`); the
patched image built from ParadeDB's current Dockerfile ships **16.15**
(`160015`), because the Dockerfile pulls `postgres:16-trixie`, which has moved on
since the 2026-04-16 image.

So the patched engine is not a pure two-line change: **it would also upgrade
PostgreSQL 16.13 → 16.15 on the live database.** That is a routine minor
upgrade, but it is scope the ctid fix does not need, and the project pins the
exact version deliberately. Two options, and this is a decision, not a detail:

1. **Rebuild the patch against PostgreSQL 16.13** by pinning the Dockerfile's
   base image to a 16.13 digest, so the only delta from today's live engine is
   the ctid fix. Matches the project's pinning posture.
2. **Accept 16.15** and update the `160013` pins in
   `tests/test_gateway_partition_provision.py` and
   `deploy/public/probe_bm25_deleted_statistics.py`, treating the minor upgrade
   as intended.

Until one is chosen, Phase 4 should not proceed: deploying the current patched
image silently bundles a PostgreSQL upgrade with the bug fix.

## Consequences

- Gate 0 can only be run meaningfully with
  `--ignore=tests/test_gateway_service.py --ignore=tests/test_gateway_query_execution.py`
  until these are fixed, and any full-suite run should carry a hard `timeout`
  so a wedge costs minutes rather than hours.
- These are pre-existing defects in uncommitted gateway work. They block the
  gateway's own release criteria; they do not block the migration phases, whose
  own suites pass.
- `pytest-timeout` is not installed. Adding it would have turned three lost
  runs into three clear failures.
