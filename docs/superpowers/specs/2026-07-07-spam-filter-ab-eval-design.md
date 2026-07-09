# Spam/Bulk Removal A/B Eval — Design

**Date:** 2026-07-07
**Status:** Approved (design), pending implementation plan

## Question

We already soft-penalize promotional mail in ranking (`LABEL_SCORES["CATEGORY_PROMOTIONS"] = -0.15` in `src/gmail_search/search/engine.py`). Does **hard-removing** promotions from search results improve result quality further, as measured by our LLM relevance judge?

## Experiment

For each query, produce two 10-result lists from production `SearchEngine.search_threads` and have the judge grade both:

- **A (baseline):** top-10 as ships today (includes the existing −0.15 promo penalty).
- **B (filtered):** the same ranked list with any result labeled `CATEGORY_PROMOTIONS` removed, then top-10.

Retrieval runs **once** per query with `top_k=N` (N=30, so ≥10 results survive filtering). A and B are both derived from that single ranked list — no re-ranking — so the only difference between conditions is the drop.

**Caveat (stated in the report):** A already demotes promos via the soft penalty, so this measures the *incremental* gain of hard removal on top of what ships — not filtering vs. no-filtering. No pure "no-penalty" A0 baseline in this run.

## Bulk definition

```python
def is_bulk(labels: set[str]) -> bool:
    return "CATEGORY_PROMOTIONS" in labels
```

Gmail category labels only — no header heuristics, no LLM classifier. Most conservative bucket; uses the per-message label field the scorer already reads. (Social/Updates/Forums are deliberately left in.)

## Query sets (51 queries total)

1. **Curated (21):** `scripts/ab_queries.json` — hand-written realistic queries, multilingual. Reported per-query for interpretability.
2. **Sampled (30):** sampled with a **fixed seed** from the cached `eval_queries_n100.json` set used by the existing compaction judge (regenerate via `scripts/scann_compaction_eval.py` if the cache under `/tmp/scann_compaction_eval` is missing), so the same 30 queries are used on every run. Reported in aggregate for statistical stability.

The two sets are reported **separately**, never pooled into one mean.

## Judging

Reuse the existing harness in `scripts/scann_compaction_judge.py`: `build_judge_prompt`, `call_judge` (Gemini, `temperature=0.0`), `parse_grades`, `ndcg_at_k`, `precision_at_k`, and the 0–3 grading rubric.

**Blind pooled grading:** per query, pool `union(A_top10, B_top10)` and grade each email once. The judge never sees condition membership, and a shared grade dict guarantees an email cannot be scored differently across A and B. Metrics for both lists are computed from that shared dict. Ungraded/missing ids default to grade 0 (existing behavior).

**Metrics per condition:** NDCG@10, Precision@10 (grade ≥ 2), % highly-relevant (grade 3) captured.

**Grade caching:** grades are cached on disk keyed by `(query, embedding_id)` so re-runs and report tweaks don't re-burn judge tokens.

## Code structure

- **New:** `scripts/spam_filter_judge.py` — the experiment harness (retrieval via `SearchEngine.search_threads`, filter, pooled judging, metrics, JSON output).
- **Judge helpers:** import from `scripts/scann_compaction_judge.py` if it is import-safe (`__main__`-guarded); otherwise extract the ~5 pure functions (`build_judge_prompt`, `call_judge`, `parse_grades`, `ndcg_at_k`, `precision_at_k`) into `scripts/judge_lib.py` and have both scripts import it.
- **No production changes:** the filter lives only in the experiment harness. `src/` is untouched.

## Output

1. **Raw JSON:** `scripts/bench_out/spam_filter_ab.json` — per-query records:
   ```json
   {"query": "...", "set": "curated|sampled",
    "ndcg_A": 0.0, "ndcg_B": 0.0, "p10_A": 0.0, "p10_B": 0.0,
    "hi_capture_A": 0.0, "hi_capture_B": 0.0,
    "n_promos_removed": 0, "grades": {"<id>": 0}}
   ```
2. **Report (claude.ai Artifact):** summary table of mean ΔNDCG@10 / ΔP@10 / Δhi-capture for curated and sampled sets separately, plus a per-query win/loss table showing exactly which queries the filter helped or hurt and how many promos were removed per query.

## Runtime & dependencies

- Postgres on `127.0.0.1:5544`, embeddings/index available (same stack as the compaction judge).
- `GEMINI_API_KEY` set.
- ~51 queries × 1 pooled judge call each ≈ 51 Gemini calls per full run (fewer on re-runs thanks to the grade cache).

## Success criteria

- The harness runs end-to-end on all 51 queries and produces the JSON + report.
- Report clearly answers: does hard removal beat the soft penalty (mean ΔNDCG@10 > 0), and on which queries does it hurt (e.g., queries that legitimately target promotional mail)?

## Non-goals

- No change to production ranking or filtering in this experiment.
- No header-heuristic or LLM spam classifier (possible follow-ups if labels-only shows signal).
- No A0 (penalty-disabled) baseline in this run.
