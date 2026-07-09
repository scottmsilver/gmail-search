#!/usr/bin/env python
"""Spam/bulk removal A/B eval on production search_threads.

Per query (one retrieval, top_k=FETCH_N):
  A = top-10 as ships today (includes the existing -0.15 promo penalty)
  B = same ranked list with CATEGORY_PROMOTIONS threads removed, then top-10
  C = same ranked list with List-Unsubscribe-header threads removed, then
      top-10 (the sender-self-declared bulk signal; broader than B)

The Gemini judge grades union(A, B, C) once, blind to condition; NDCG@10,
P@10 and hi-capture for all lists come from that shared grade dict, so
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
OUT_PATH = SCRIPTS_DIR / "bench_out" / "spam_filter_abc.json"
CACHE_PATH = SCRIPTS_DIR / "bench_out" / "spam_filter_grade_cache.json"
SAMPLED_QUERIES_PATH = SCRIPTS_DIR / "bench_out" / "spam_filter_sampled_queries.json"


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


# ----------------------------------------------------------------------
# Live glue (DB + engine; exercised via --dry-run, not unit tests)
# ----------------------------------------------------------------------


def _sample_queries(n_sampled: int, seed: int) -> list[str]:
    """Fixed-seed sample of real past queries. Resolution order:
    1. scripts/bench_out/spam_filter_sampled_queries.json — the persisted
       pick, so the SAME 30 queries are used on every run even as the
       query_cache table grows.
    2. /tmp compaction-eval cache (same pool the ScaNN judge used).
    3. The query_cache PG table directly (what built that /tmp cache).
    Whatever is picked gets persisted to (1)."""
    if SAMPLED_QUERIES_PATH.exists():
        return json.loads(SAMPLED_QUERIES_PATH.read_text())

    if EVAL_QUERIES_CACHE.exists():
        texts = json.loads(EVAL_QUERIES_CACHE.read_text())
    else:
        from gmail_search.store.db import get_connection

        conn = get_connection(None)
        try:
            rows = conn.execute("SELECT DISTINCT query_text FROM query_cache ORDER BY query_text").fetchall()
        finally:
            conn.close()
        texts = [r["query_text"] for r in rows]

    rng = random.Random(seed)
    sampled = rng.sample(texts, min(n_sampled, len(texts)))
    SAMPLED_QUERIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAMPLED_QUERIES_PATH.write_text(json.dumps(sampled, indent=1))
    return sampled


def load_queries(n_sampled: int = N_SAMPLED, seed: int = SAMPLE_SEED) -> list[dict]:
    """Curated 21 from ab_queries.json + fixed-seed sample of n real past
    queries. Sets are tagged and reported separately."""
    queries = [{"query": q, "set": "curated"} for q in json.loads(CURATED_PATH.read_text())["queries"]]
    queries += [{"query": q, "set": "sampled"} for q in _sample_queries(n_sampled, seed)]
    return queries


def build_engine():
    """Engine over the ACTIVE index for the bootstrap user — same
    resolution the server uses (resolve_active_index_dir consults the
    per-user scann_index_pointer row; data/scann_index is only a
    legacy fallback)."""
    from gmail_search.config import load_config
    from gmail_search.index.searcher import resolve_active_index_dir
    from gmail_search.search.engine import SearchEngine

    data_dir = REPO_ROOT / "data"
    cfg = load_config(config_path=REPO_ROOT / "config.yaml", data_dir=data_dir)
    db_path = data_dir / "gmail_search.db"
    index_dir = resolve_active_index_dir(db_path, data_dir / "scann_index")
    return SearchEngine(db_path, index_dir, cfg)


def fetch_thread_labels(thread_ids: list[str], user_id: str) -> dict[str, set[str]]:
    """thread_summary.all_labels is a JSON array of every label any message
    in the thread carries — a thread is 'promotional' if ANY message is.
    Scoped to the engine's user_id: the DB is multi-user, and a colliding
    thread_id from another mailbox must not label ours."""
    if not thread_ids:
        return {}
    from gmail_search.store.db import get_connection

    conn = get_connection(None)  # db_path ignored by the PG shim
    try:
        rows = conn.execute(
            "SELECT thread_id, all_labels FROM thread_summary" " WHERE thread_id = ANY(%s) AND user_id = %s",
            (thread_ids, user_id),
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


def fetch_thread_unsub(thread_ids: list[str], user_id: str) -> set[str]:
    """Threads where ANY message carries a List-Unsubscribe header —
    the bulk-mail signal senders self-declare, catching newsletters
    Gmail files outside CATEGORY_PROMOTIONS. Precise header check via
    the stored raw Gmail payload (not a substring match on the body).
    Scoped to user_id (multi-user DB); the jsonb_typeof guard skips
    rows whose payload.headers isn't an array instead of aborting the
    whole query. (raw_json is written by json.dumps in the crawler, so
    an invalid-JSON ::jsonb cast failure is not reachable in practice;
    if corrupt rows ever appear we WANT the loud crash over silently
    mislabeling condition C.)"""
    if not thread_ids:
        return set()
    from gmail_search.store.db import get_connection

    conn = get_connection(None)
    try:
        rows = conn.execute(
            """SELECT DISTINCT thread_id FROM messages
               WHERE thread_id = ANY(%s)
                 AND user_id = %s
                 AND jsonb_typeof(raw_json::jsonb -> 'payload' -> 'headers') = 'array'
                 AND EXISTS (
                   SELECT 1 FROM jsonb_array_elements(
                       raw_json::jsonb -> 'payload' -> 'headers') h
                   WHERE lower(h ->> 'name') = 'list-unsubscribe')""",
            (thread_ids, user_id),
        ).fetchall()
    finally:
        conn.close()
    return {r["thread_id"] for r in rows}


def retrieve(engine, query: str):
    """One production search; all conditions derived from its ranked list.
    A = top-10 as shipped; B = CATEGORY_PROMOTIONS removed; C = threads
    with a List-Unsubscribe header removed."""
    results = engine.search_threads(query, top_k=FETCH_N)
    ranked_ids = [r.thread_id for r in results]
    labels = fetch_thread_labels(ranked_ids, engine.user_id)
    bulk_ids = {tid for tid in ranked_ids if is_bulk_labels(labels.get(tid, set()))}
    unsub_ids = fetch_thread_unsub(ranked_ids, engine.user_id)
    cond_a = ranked_ids[:TOP_K]
    cond_b = derive_filtered(ranked_ids, bulk_ids)
    cond_c = derive_filtered(ranked_ids, unsub_ids)
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
    return cond_a, cond_b, cond_c, bulk_ids, unsub_ids, candidates


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


def grade_pool(query: str, pool_ids: list[str], candidates: dict[str, dict], cache: dict) -> dict[str, int]:
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


def _mean(xs: list[float | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    return sum(vals) / len(vals) if vals else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="retrieval + filter stats only, no LLM calls")
    ap.add_argument("--limit", type=int, default=None, help="only run the first N queries")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    queries = load_queries()
    if args.limit:
        queries = queries[: args.limit]
    print(
        f"{len(queries)} queries "
        f"({sum(1 for q in queries if q['set'] == 'curated')} curated, "
        f"{sum(1 for q in queries if q['set'] == 'sampled')} sampled)"
    )

    from scann_compaction_judge import ndcg_at_k, precision_at_k  # heavy import, after arg parsing

    cache = load_cache()
    engine = build_engine()
    records: list[dict] = []
    try:
        for i, q in enumerate(queries, 1):
            query = q["query"]
            cond_a, cond_b, cond_c, bulk_ids, unsub_ids, candidates = retrieve(engine, query)
            promo_in_a = sum(1 for tid in cond_a if tid in bulk_ids)
            unsub_in_a = sum(1 for tid in cond_a if tid in unsub_ids)
            print(
                f"[{i:2d}/{len(queries)}] [{q['set']:7s}] {query[:48]!r:50s} "
                f"promo={len(bulk_ids):2d}/{promo_in_a} unsub={len(unsub_ids):2d}/{unsub_in_a}",
                flush=True,
            )
            if not candidates:
                print("    no results — skipped")
                continue
            if args.dry_run:
                continue

            pool = list(dict.fromkeys(cond_a + cond_b + cond_c))  # union, order-stable
            t0 = time.time()
            grades = grade_pool(query, pool, candidates, cache)
            records.append(
                {
                    "query": query,
                    "set": q["set"],
                    "ndcg_A": ndcg_at_k(cond_a, grades),
                    "ndcg_B": ndcg_at_k(cond_b, grades),
                    "ndcg_C": ndcg_at_k(cond_c, grades),
                    "p10_A": precision_at_k(cond_a, grades),
                    "p10_B": precision_at_k(cond_b, grades),
                    "p10_C": precision_at_k(cond_c, grades),
                    "hi_capture_A": hi_capture(cond_a, grades),
                    "hi_capture_B": hi_capture(cond_b, grades),
                    "hi_capture_C": hi_capture(cond_c, grades),
                    "n_promos_removed": promo_in_a,
                    "n_promos_in_pool": len(bulk_ids),
                    "n_unsub_removed": unsub_in_a,
                    "n_unsub_in_pool": len(unsub_ids),
                    "grades": grades,
                    "judge_time_s": round(time.time() - t0, 1),
                }
            )
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
        touched_b = sum(1 for r in rs if r["n_promos_removed"] > 0)
        touched_c = sum(1 for r in rs if r["n_unsub_removed"] > 0)
        print(f"\n=== {qset} (n={len(rs)}; B changed top-10 on {touched_b}, C on {touched_c}) ===")
        for metric in ("ndcg", "p10", "hi_capture"):
            a = _mean([r[f"{metric}_A"] for r in rs])
            b = _mean([r[f"{metric}_B"] for r in rs])
            c = _mean([r[f"{metric}_C"] for r in rs])
            if a is None or b is None or c is None:
                continue
            print(f"  {metric:11s}  A={a:.4f}  B={b:.4f} (Δ{b - a:+.4f})  " f"C={c:.4f} (Δ{c - a:+.4f})")
        for cond in ("B", "C"):
            wins = sum(1 for r in rs if r[f"ndcg_{cond}"] > r["ndcg_A"])
            losses = sum(1 for r in rs if r[f"ndcg_{cond}"] < r["ndcg_A"])
            print(f"  NDCG {cond} wins/ties/losses vs A: {wins}/{len(rs) - wins - losses}/{losses}")


if __name__ == "__main__":
    main()
