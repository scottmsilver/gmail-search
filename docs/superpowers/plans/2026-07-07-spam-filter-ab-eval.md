# Spam/Bulk Removal A/B Eval — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether hard-removing `CATEGORY_PROMOTIONS` threads from production `search_threads` results improves LLM-judge relevance scores (NDCG@10 / P@10 / hi-capture) vs. today's soft −0.15 penalty.

**Architecture:** One new script `scripts/spam_filter_judge.py`. Per query: one `search_threads(q, top_k=30)` call → condition A = top-10 as-is, condition B = promo-threads dropped then top-10. Pooled blind Gemini judging (helpers imported from `scripts/scann_compaction_judge.py`, which is `__main__`-guarded and import-safe). Grades cached on disk. Pure logic is unit-tested; DB/LLM glue is smoke-tested via `--dry-run` and `--limit`.

**Tech Stack:** Python 3.12 (uv venv), psycopg (via `gmail_search.store.db.get_connection` shim), ScaNN engine via `gmail_search.search.engine.SearchEngine`, Gemini judge via existing `call_judge`.

## Global Constraints

- **No commits or pushes without Scott's explicit go-ahead** (global CLAUDE.md rule) — all commit steps are deferred; work stays uncommitted.
- **No production changes**: `src/gmail_search/` is untouched; new files only.
- Bulk definition (spec): `BULK_LABELS = {"CATEGORY_PROMOTIONS"}` — labels only, no heuristics.
- One retrieval per query (`top_k=30`); A and B derived from the same ranked list.
- Blind pooled judging: judge grades `union(A₁₀, B₁₀)` once per query; shared grade dict for both conditions' metrics.
- Query sets reported separately: curated (21, `scripts/ab_queries.json`) and sampled (30, fixed seed 42 from `/tmp/scann_compaction_eval/eval_queries_n100.json`).
- Judge model/config unchanged from `scann_compaction_judge.call_judge` (Gemini, temperature 0.0).
- Outputs: `scripts/bench_out/spam_filter_ab.json` + grade cache `scripts/bench_out/spam_filter_grade_cache.json`.
- Runtime deps: Postgres on `127.0.0.1:5544` (or `DB_DSN`), ScaNN index under `data/scann_index`, `GEMINI_API_KEY` in env.

---

### Task 1: Pure helper functions (TDD)

**Files:**
- Create: `scripts/spam_filter_judge.py` (module-level: stdlib imports + pure functions only; heavy imports stay lazy inside functions so tests import fast)
- Test: `tests/test_spam_filter_judge.py`

**Interfaces:**
- Produces:
  - `is_bulk_labels(labels: set[str] | list[str]) -> bool`
  - `derive_filtered(ranked_ids: list[str], bulk_ids: set[str], k: int = 10) -> list[str]`
  - `parse_grades_str(text: str, expected_ids: set[str]) -> dict[str, int]`
  - `hi_capture(ranked_ids: list[str], grades: dict[str, int], k: int = 10) -> float | None`
  - Constants: `BULK_LABELS`, `TOP_K = 10`, `FETCH_N = 30`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_spam_filter_judge.py
"""Unit tests for the pure logic in scripts/spam_filter_judge.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from spam_filter_judge import (
    derive_filtered,
    hi_capture,
    is_bulk_labels,
    parse_grades_str,
)


class TestIsBulkLabels:
    def test_promotions_is_bulk(self):
        assert is_bulk_labels({"INBOX", "CATEGORY_PROMOTIONS"}) is True

    def test_social_is_not_bulk(self):
        # Spec: promotions ONLY — social/updates stay in.
        assert is_bulk_labels({"CATEGORY_SOCIAL"}) is False
        assert is_bulk_labels({"CATEGORY_UPDATES"}) is False

    def test_empty_and_personal(self):
        assert is_bulk_labels(set()) is False
        assert is_bulk_labels(["INBOX", "IMPORTANT"]) is False

    def test_accepts_list(self):
        assert is_bulk_labels(["CATEGORY_PROMOTIONS"]) is True


class TestDeriveFiltered:
    def test_removes_bulk_then_truncates(self):
        ranked = [f"t{i}" for i in range(15)]  # t0..t14
        bulk = {"t1", "t3"}
        out = derive_filtered(ranked, bulk, k=10)
        assert out == ["t0", "t2", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11"]
        assert len(out) == 10

    def test_no_bulk_equals_top_k(self):
        ranked = [f"t{i}" for i in range(15)]
        assert derive_filtered(ranked, set(), k=10) == ranked[:10]

    def test_fewer_than_k_survivors(self):
        ranked = ["a", "b", "c"]
        assert derive_filtered(ranked, {"b"}, k=10) == ["a", "c"]

    def test_preserves_order(self):
        ranked = ["z", "a", "m"]
        assert derive_filtered(ranked, set(), k=2) == ["z", "a"]


class TestParseGradesStr:
    def test_parses_hex_thread_ids(self):
        text = "18c2ab34ff 3\n19d0e0aa01 0\n1a1111beef 2"
        expected = {"18c2ab34ff", "19d0e0aa01", "1a1111beef"}
        assert parse_grades_str(text, expected) == {
            "18c2ab34ff": 3,
            "19d0e0aa01": 0,
            "1a1111beef": 2,
        }

    def test_ignores_unexpected_ids_and_junk(self):
        text = "unknown99 3\nnot a grade line\n18c2 5\n18c2 2"
        assert parse_grades_str(text, {"18c2"}) == {"18c2": 2}  # 5 out of range, unknown dropped

    def test_empty(self):
        assert parse_grades_str("", {"x"}) == {}


class TestHiCapture:
    def test_fraction_of_grade3_in_topk(self):
        grades = {"a": 3, "b": 3, "c": 2, "d": 0}
        assert hi_capture(["a", "c", "d"], grades, k=3) == 0.5  # a in, b out

    def test_none_when_no_grade3(self):
        assert hi_capture(["a"], {"a": 2}, k=10) is None

    def test_full_capture(self):
        assert hi_capture(["a", "b"], {"a": 3, "b": 3}, k=10) == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ssilver/development/gmail-search && uv run pytest tests/test_spam_filter_judge.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'spam_filter_judge'`

- [ ] **Step 3: Write the pure-function module skeleton**

```python
#!/usr/bin/env python
# scripts/spam_filter_judge.py
"""Spam/bulk removal A/B eval on production search_threads.

Per query (one retrieval, top_k=FETCH_N):
  A = top-10 as ships today (includes the existing -0.15 promo penalty)
  B = same ranked list with CATEGORY_PROMOTIONS threads removed, then top-10

The Gemini judge grades union(A, B) once, blind to condition; NDCG@10,
P@10 and hi-capture for both lists come from that shared grade dict, so
the ONLY difference between conditions is the hard removal.

Caveat: A already soft-penalizes promos, so this measures the INCREMENTAL
gain of hard removal on top of what ships, not filtering-vs-nothing.

Usage (from repo root; needs PG up + ScaNN index + GEMINI_API_KEY):
    uv run python scripts/spam_filter_judge.py --dry-run --limit 3   # no LLM calls
    uv run python scripts/spam_filter_judge.py --limit 1             # 1 live judge call
    uv run python scripts/spam_filter_judge.py                       # full 51-query run
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

BULK_LABELS = {"CATEGORY_PROMOTIONS"}  # spec: promotions only
TOP_K = 10
FETCH_N = 30  # one retrieval this deep so >=10 usually survive filtering
SAMPLE_SEED = 42
N_SAMPLED = 30
CURATED_PATH = SCRIPTS_DIR / "ab_queries.json"
EVAL_QUERIES_CACHE = Path("/tmp/scann_compaction_eval/eval_queries_n100.json")
OUT_PATH = SCRIPTS_DIR / "bench_out" / "spam_filter_ab.json"
CACHE_PATH = SCRIPTS_DIR / "bench_out" / "spam_filter_grade_cache.json"


# ----------------------------------------------------------------------
# Pure logic (unit-tested; keep this section import-light)
# ----------------------------------------------------------------------

def is_bulk_labels(labels: set[str] | list[str]) -> bool:
    """A thread is bulk iff any of its messages carries a BULK_LABELS label."""
    return bool(BULK_LABELS & set(labels))


def derive_filtered(ranked_ids: list[str], bulk_ids: set[str], k: int = TOP_K) -> list[str]:
    """Condition B: drop bulk threads from the ranked list, then take top-k."""
    return [tid for tid in ranked_ids if tid not in bulk_ids][:k]


def parse_grades_str(text: str, expected_ids: set[str]) -> dict[str, int]:
    """Parse `<thread_id> <grade>` judge lines. Thread ids are Gmail hex
    strings (unlike scann_compaction_judge's int embedding ids, hence this
    string-keyed variant). Later duplicate lines win; out-of-range grades
    and unexpected ids are dropped. Missing ids are handled by callers
    (default 0) so a judge refusal can't inflate scores."""
    out: dict[str, int] = {}
    for line in text.strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        tid = parts[0]
        try:
            grade = int(parts[1])
        except ValueError:
            continue
        if tid in expected_ids and 0 <= grade <= 3:
            out[tid] = grade
    return out


def hi_capture(ranked_ids: list[str], grades: dict[str, int], k: int = TOP_K) -> float | None:
    """Fraction of grade-3 (highly relevant) pool items present in top-k.
    None when the pool has no grade-3 items (skipped in averages)."""
    hi = {tid for tid, g in grades.items() if g == 3}
    if not hi:
        return None
    return len(hi & set(ranked_ids[:k])) / len(hi)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ssilver/development/gmail-search && uv run pytest tests/test_spam_filter_judge.py -v`
Expected: all 12 tests PASS

- [ ] **Step 5: ~~Commit~~ — deferred (no commits until Scott approves; global rule)**

---

### Task 2: Retrieval + label glue with `--dry-run` smoke test

**Files:**
- Modify: `scripts/spam_filter_judge.py` (append below the pure section)

**Interfaces:**
- Consumes: Task 1's constants and pure functions.
- Produces:
  - `load_queries(n_sampled: int = N_SAMPLED, seed: int = SAMPLE_SEED) -> list[dict]` — `[{"query": str, "set": "curated"|"sampled"}]`
  - `build_engine() -> SearchEngine`
  - `fetch_thread_labels(thread_ids: list[str]) -> dict[str, set[str]]`
  - `retrieve(engine, query: str) -> tuple[list[str], list[str], set[str], dict[str, dict]]` — `(A, B, bulk_ids, candidates)`; `candidates[tid]` has keys `id/subject/from_addr/date/snippet` (the exact dict shape `build_judge_prompt` renders)
  - CLI: `--dry-run` (retrieval + filter stats, zero LLM calls), `--limit N`

- [ ] **Step 1: Append the glue code**

```python
# ----------------------------------------------------------------------
# Live glue (DB + engine; exercised via --dry-run, not unit tests)
# ----------------------------------------------------------------------

def load_queries(n_sampled: int = N_SAMPLED, seed: int = SAMPLE_SEED) -> list[dict]:
    """Curated 21 from ab_queries.json + fixed-seed sample of n from the
    compaction-eval query cache. Sets are tagged and reported separately."""
    queries = [
        {"query": q, "set": "curated"}
        for q in json.loads(CURATED_PATH.read_text())["queries"]
    ]
    if EVAL_QUERIES_CACHE.exists():
        texts = json.loads(EVAL_QUERIES_CACHE.read_text())
        rng = random.Random(seed)
        queries += [
            {"query": q, "set": "sampled"}
            for q in rng.sample(texts, min(n_sampled, len(texts)))
        ]
    else:
        print(
            f"WARNING: {EVAL_QUERIES_CACHE} missing — run "
            "scripts/scann_compaction_eval.py to regenerate; "
            "continuing with the curated set only.",
            file=sys.stderr,
        )
    return queries


def build_engine():
    """Same construction as the CLI search command (cli.py:1784)."""
    from gmail_search.config import load_config
    from gmail_search.search.engine import SearchEngine

    data_dir = REPO_ROOT / "data"
    cfg = load_config(config_path=REPO_ROOT / "config.yaml", data_dir=data_dir)
    return SearchEngine(data_dir / "gmail_search.db", data_dir / "scann_index", cfg)


def fetch_thread_labels(thread_ids: list[str]) -> dict[str, set[str]]:
    """thread_summary.all_labels is a JSON array of every label any message
    in the thread carries — a thread is 'promotional' if ANY message is."""
    if not thread_ids:
        return {}
    from gmail_search.store.db import get_connection

    conn = get_connection(None)  # db_path ignored by the PG shim
    try:
        rows = conn.execute(
            "SELECT thread_id, all_labels FROM thread_summary WHERE thread_id = ANY(%s)",
            (thread_ids,),
        ).fetchall()
    finally:
        conn.close()
    out: dict[str, set[str]] = {}
    for r in rows:
        try:
            out[r["thread_id"]] = set(json.loads(r["all_labels"] or "[]"))
        except (TypeError, ValueError):
            out[r["thread_id"]] = set()
    return out


def retrieve(engine, query: str):
    """One production search; both conditions derived from its ranked list."""
    results = engine.search_threads(query, top_k=FETCH_N)
    ranked_ids = [r.thread_id for r in results]
    labels = fetch_thread_labels(ranked_ids)
    bulk_ids = {tid for tid in ranked_ids if is_bulk_labels(labels.get(tid, set()))}
    cond_a = ranked_ids[:TOP_K]
    cond_b = derive_filtered(ranked_ids, bulk_ids)
    candidates: dict[str, dict] = {}
    for r in results:
        m = r.matches[0] if r.matches else None
        candidates[r.thread_id] = {
            "id": r.thread_id,  # judge keys grades on this
            "subject": r.subject,
            "from_addr": m.from_addr if m else "(unknown)",
            "date": m.date if m else "",
            "snippet": m.snippet if m else "",
        }
    return cond_a, cond_b, bulk_ids, candidates
```

- [ ] **Step 2: Add a minimal `main()` with `--dry-run`/`--limit` and the entry guard**

```python
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="retrieval + filter stats only, no LLM calls")
    ap.add_argument("--limit", type=int, default=None, help="only run the first N queries")
    args = ap.parse_args()

    queries = load_queries()
    if args.limit:
        queries = queries[: args.limit]
    print(f"{len(queries)} queries "
          f"({sum(1 for q in queries if q['set'] == 'curated')} curated, "
          f"{sum(1 for q in queries if q['set'] == 'sampled')} sampled)")

    engine = build_engine()
    try:
        for q in queries:
            cond_a, cond_b, bulk_ids, candidates = retrieve(engine, q["query"])
            removed_in_a = sum(1 for tid in cond_a if tid in bulk_ids)
            print(f"[{q['set']:7s}] {q['query'][:50]!r:52s} "
                  f"results={len(candidates):2d} promo={len(bulk_ids):2d} "
                  f"promo_in_top10={removed_in_a} B_len={len(cond_b)}")
            if not args.dry_run:
                raise SystemExit("judging not implemented yet (Task 3)")
    finally:
        engine.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify prerequisites are up**

Run: `cd /home/ssilver/development/gmail-search && docker compose ps --format '{{.Name}} {{.Status}}' && ls data/scann_index/ | head -3`
Expected: a running Postgres container publishing `:5544`, and index files present. If PG is down: `docker compose up -d` first.

- [ ] **Step 4: Dry-run smoke test on 3 queries**

Run: `cd /home/ssilver/development/gmail-search && uv run python scripts/spam_filter_judge.py --dry-run --limit 3`
Expected: three lines like `[curated] 'receipt' results=30 promo=12 promo_in_top10=2 B_len=10`, no traceback. Sanity: `promo` counts nonzero for commerce-ish queries, `B_len == 10` when enough survivors.

- [ ] **Step 5: Re-run unit tests (still pass, module still imports light)**

Run: `uv run pytest tests/test_spam_filter_judge.py -v`
Expected: 12 PASS

- [ ] **Step 6: ~~Commit~~ — deferred (global rule)**

---

### Task 3: Pooled blind judging, cache, metrics, JSON output

**Files:**
- Modify: `scripts/spam_filter_judge.py` (replace the Task-2 `main()`; add judging + metrics)

**Interfaces:**
- Consumes: Task 1/2 functions; `build_judge_prompt`, `call_judge`, `ndcg_at_k`, `precision_at_k` imported lazily from `scann_compaction_judge` (string ids work: `build_judge_prompt` interpolates `c['id']`, and the metric functions are typed `dict[str, int]` already).
- Produces: `scripts/bench_out/spam_filter_ab.json` per spec schema; grade cache keyed `{query: {thread_id: grade}}`; console summary table.

- [ ] **Step 1: Add grade caching + pooled judging**

```python
# ----------------------------------------------------------------------
# Judging (pooled, blind, cached)
# ----------------------------------------------------------------------

def load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=1))


def grade_pool(query: str, pool_ids: list[str], candidates: dict[str, dict],
               cache: dict) -> dict[str, int]:
    """Grade every pooled candidate once, blind to condition. Cached by
    (query, thread_id) so re-runs don't re-burn judge tokens. Ids the
    judge skips default to 0 (same policy as scann_compaction_judge)."""
    cached: dict[str, int] = cache.setdefault(query, {})
    missing = [tid for tid in pool_ids if tid not in cached]
    if missing:
        from scann_compaction_judge import build_judge_prompt, call_judge

        prompt = build_judge_prompt(query, [candidates[tid] for tid in missing])
        raw = call_judge(prompt)
        grades = parse_grades_str(raw, set(missing))
        for tid in missing:
            cached[tid] = grades.get(tid, 0)
        save_cache(cache)  # persist after every judge call — crash-safe
    return {tid: cached[tid] for tid in pool_ids}
```

- [ ] **Step 2: Replace `main()` with the full loop + metrics + output**

```python
def _mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="retrieval + filter stats only, no LLM calls")
    ap.add_argument("--limit", type=int, default=None, help="only run the first N queries")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    queries = load_queries()
    if args.limit:
        queries = queries[: args.limit]
    print(f"{len(queries)} queries "
          f"({sum(1 for q in queries if q['set'] == 'curated')} curated, "
          f"{sum(1 for q in queries if q['set'] == 'sampled')} sampled)")

    from scann_compaction_judge import ndcg_at_k, precision_at_k  # noqa: E402 (heavy import)

    cache = load_cache()
    engine = build_engine()
    records: list[dict] = []
    try:
        for i, q in enumerate(queries, 1):
            query = q["query"]
            cond_a, cond_b, bulk_ids, candidates = retrieve(engine, query)
            promo_in_a = sum(1 for tid in cond_a if tid in bulk_ids)
            print(f"[{i:2d}/{len(queries)}] [{q['set']:7s}] {query[:48]!r:50s} "
                  f"promo_in_top10={promo_in_a}", flush=True)
            if not candidates:
                print("    no results — skipped")
                continue
            if args.dry_run:
                continue

            pool = list(dict.fromkeys(cond_a + cond_b))  # union, order-stable
            t0 = time.time()
            grades = grade_pool(query, pool, candidates, cache)
            records.append({
                "query": query,
                "set": q["set"],
                "ndcg_A": ndcg_at_k(cond_a, grades),
                "ndcg_B": ndcg_at_k(cond_b, grades),
                "p10_A": precision_at_k(cond_a, grades),
                "p10_B": precision_at_k(cond_b, grades),
                "hi_capture_A": hi_capture(cond_a, grades),
                "hi_capture_B": hi_capture(cond_b, grades),
                "n_promos_removed": promo_in_a,
                "n_promos_in_pool": len(bulk_ids),
                "grades": grades,
                "judge_time_s": round(time.time() - t0, 1),
            })
    finally:
        engine.close()

    if args.dry_run or not records:
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"records": records}, indent=1))
    print(f"\nwrote {args.out}")

    # ---- summary ----
    for qset in ("curated", "sampled"):
        rs = [r for r in records if r["set"] == qset]
        if not rs:
            continue
        touched = [r for r in rs if r["n_promos_removed"] > 0]
        print(f"\n=== {qset} (n={len(rs)}, filter changed top-10 on {len(touched)}) ===")
        for metric in ("ndcg", "p10", "hi_capture"):
            a = _mean([r[f"{metric}_A"] for r in rs])
            b = _mean([r[f"{metric}_B"] for r in rs])
            if a is None or b is None:
                continue
            print(f"  {metric:11s}  A={a:.4f}  B={b:.4f}  Δ={b - a:+.4f}")
        wins = sum(1 for r in rs if r["ndcg_B"] > r["ndcg_A"])
        losses = sum(1 for r in rs if r["ndcg_B"] < r["ndcg_A"])
        print(f"  NDCG wins/ties/losses: {wins}/{len(rs) - wins - losses}/{losses}")
```

- [ ] **Step 3: Re-run unit tests**

Run: `uv run pytest tests/test_spam_filter_judge.py -v`
Expected: 12 PASS

- [ ] **Step 4: One live judged query end-to-end**

Run: `cd /home/ssilver/development/gmail-search && uv run python scripts/spam_filter_judge.py --limit 1 --out scripts/bench_out/spam_filter_ab_smoke.json`
Expected: one judged record; `scripts/bench_out/spam_filter_ab_smoke.json` exists with non-degenerate grades (not all 0); cache file created. If `GEMINI_API_KEY` missing, export it first (check `config.local.yaml` / service env for where it lives).

- [ ] **Step 5: Verify cache hit on re-run**

Run: same command again.
Expected: completes in seconds (no new Gemini call — `judge_time_s` ≈ 0), identical metrics.

- [ ] **Step 6: ~~Commit~~ — deferred (global rule)**

---

### Task 4: Full 51-query run + report

**Files:**
- Output: `scripts/bench_out/spam_filter_ab.json`
- Report: claude.ai Artifact (per Scott's deliverables rule) — summary table + per-query win/loss table

- [ ] **Step 1: Ensure the sampled query cache exists**

Run: `ls /tmp/scann_compaction_eval/eval_queries_n100.json`
If missing: `uv run python scripts/scann_compaction_eval.py` (rebuilds the corpus + query cache; takes a while) — or proceed curated-only and note it in the report.

- [ ] **Step 2: Full run**

Run: `cd /home/ssilver/development/gmail-search && uv run python scripts/spam_filter_judge.py 2>&1 | tee /tmp/spam_filter_run.log`
Expected: 51 queries judged (~51 Gemini calls, minutes), JSON written, summary table printed.

- [ ] **Step 3: Sanity-check results**

- `records` count matches query count (minus any zero-result skips, which must be listed).
- Grades non-degenerate (mix of 0–3, not all one value).
- Per-query: on queries with `n_promos_removed == 0`, `ndcg_A == ndcg_B` exactly (A ≡ B) — any difference is a bug.

- [ ] **Step 4: Build the report Artifact**

Summary tables (curated / sampled separately): mean NDCG@10, P@10, hi-capture for A and B with deltas; win/tie/loss counts; per-query table (query, set, promos removed, ΔNDCG, ΔP@10) sorted by ΔNDCG; the incremental-gain caveat; interpretation + recommendation (keep penalty only / add hard filter / investigate hurt queries).

- [ ] **Step 5: Report back to Scott** — results + artifact link + whether anything looks worth shipping. No commits until approved.
