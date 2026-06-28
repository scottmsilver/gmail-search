# Search performance gate

A baseline + tolerance regression gate for `/api/search` and
`SearchEngine.search_threads`. Commissioned after a debugging session found a
single search taking 14–58s with **nothing measuring it** — the hidden culprit
was a `thread_summary` bulk-IN fetch doing ~9.8s of work inside an otherwise
reasonable `search_threads`. This gate makes that class of bug impossible to
merge silently: per-stage timing surfaces a slow sub-stage that would
otherwise hide inside a plausible-looking total.

## Files

| File | What it is |
|------|------------|
| `test_search_perf_gate.py` | The pytest gate (4 test areas). |
| `harness.py` | Reusable plumbing: prereq detection, env readers, the monkeypatch `StageTimer`, baseline load/save + tolerance math. No assertions. |
| `conftest.py` | Registers `--update-baseline`; re-points `DB_DSN` at the real corpus (the repo's autouse fixture otherwise isolates every test to an empty schema). |
| `baseline.json` | Committed per-stage / end-to-end / plan baseline. |

## What it measures

1. **Per-stage search timing** (`test_per_stage_timing`, `perf_slow`).
   Profiler-style monkeypatch of every stage of `search_threads`
   (`_embed_query`, `_resolve_candidate_msg_ids`, `scann_search`,
   `_fetch_embedding_rows`, `bm25_search_fts`, `thread_summary_fetch`,
   `bm25_messages_fetch`, `scoring_loop_residual`, `_collapse_repeat_senders`,
   `_llm_rerank`, `_filter_offtopic`, `total`). **engine.py is never edited** —
   stages are wrapped from the outside and restored on exit. The
   `scoring_loop_residual` stage is `total` minus all attributed stages, so the
   inline scoring/blend loop is still accounted for. Stages that don't fire for
   the probe query (e.g. `_llm_rerank` on a clear winner) are simply absent.
2. **End-to-end API latency** (`test_end_to_end_http_latency`, `perf_slow`).
   Drives the **live** `/api/search` over HTTP across four representative
   queries — clear-winner, tightly-clustered (forces `_llm_rerank`),
   BM25/identifier, and date-filtered — at `k=10`. Each query is warmed once
   (populating `query_cache`), then measured, mirroring real warm traffic.
3. **Batch & concurrency** (`test_batch_contract_no_raise`, fast gate).
   Exercises `search_emails_batch` and asserts the **public contract**: it
   returns a dict with one per-item result, isolates a failing item as a
   per-item `{"error": ...}`, and **never raises** (the original bug was one
   slow item's `ReadTimeout` propagating through `asyncio.gather` and nuking the
   whole batch), within a wall-clock budget. Hermetic — no live server needed.
4. **DB query plans** (`test_db_plans_no_seq_scan`, fast gate).
   `EXPLAIN`s the hot bulk-IN queries (`thread_summary` by `thread_id`+`user_id`
   and the BM25-only `messages IN (...)` fetch) and asserts an Index/Bitmap
   scan, **not** a Seq Scan — directly guarding the missing-index / stale-stats
   regression that caused the 9.8s fetch.

## Running it

```bash
# Fast gate (default pytest already excludes perf_slow): DB plans + batch.
pytest tests/perf

# Full gate including the slow, opt-in tests (builds ONE ~10GB ScaNN engine,
# ~100s, and drives live HTTP — needs `gmail-search serve` running):
pytest tests/perf -m perf_slow
```

The default `pytest` invocation is configured (`addopts = -m 'not integration
and not perf_slow'`) to **never** trigger the 100s engine build. Every
DB/serve-dependent test **auto-skips** when its prerequisite is missing
(Postgres unreachable, empty corpus for the test user, or serve not up), so CI
without a DB exits cleanly instead of hard-failing.

## Tolerance policy

A metric **regresses** when

```
measured > max(baseline * 1.5, baseline + 150ms)
```

The dual bound (`+50%` **or** `+150ms`, whichever is larger) keeps tiny noisy
sub-millisecond stages from tripping on relative jitter while still catching a
10ms stage that balloons into seconds. Tunable in `harness.py`
(`REL_TOLERANCE`, `ABS_TOLERANCE_MS`).

## Updating the baseline (blessed regen)

Run the slow gate with `--update-baseline`. It re-measures from a real run,
rewrites `baseline.json`, and clears the `needs_blessed_regen` flag. Then
**commit** the new baseline.

```bash
# Regenerate per-stage numbers (builds the engine):
pytest tests/perf -m perf_slow -k per_stage_timing --update-baseline

# Regenerate end-to-end HTTP numbers (needs serve running):
pytest tests/perf -m perf_slow -k end_to_end_http_latency --update-baseline
```

Bless the baseline **after** a known-good change lands (e.g. the
`thread_summary` index fix), never to paper over a regression you can't explain.

## Environment notes (this repo)

- Run with the repo's python: `/home/ssilver/anaconda3/bin/python -m pytest …`.
- DB DSN + secrets: the gate reads the live `gmail-search serve` process's env
  straight from `/proc/<pid>/environ` (DSN, admin token) so you don't have to
  export them. No URL or secret is hardcoded in committed code — only a
  localhost default.
- Serve HTTP: `http://127.0.0.1:8090` by default (override
  `GMAIL_SEARCH_API_URL`). Test user: `u_bW4Sa8cN0wT9KPwp` (override
  `GMAIL_PERF_USER_ID`). DB override: `GMAIL_PERF_DB_DSN`.
- Connections are always closed (`harness.closing_connection`) — leaking
  exhausts Postgres `max_connections`.

## No engine.py hook

This gate adds **no** code to `engine.py`. All per-stage timing is done by
monkeypatching `SearchEngine` methods (and the inline `thread_summary` /
`messages IN` fetches via a temporary wrap of the PG connection's `execute`)
from inside `harness.StageTimer`, restored on context exit. No opt-in callback
or behavioral change was needed.
