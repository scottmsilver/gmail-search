# ScaNN index compaction — evaluation + recommendations

**Date**: 2026-04-27
**Author**: Claude (Opus 4.7) with Scott
**Status**: complete; awaiting decision on which variant to ship

## TL;DR

The current production ScaNN index is **7.4 GB** for ~565k vectors at 3072 dims. Three measurement rounds against the real corpus (565k embeddings, 50 LLM-judged queries) showed:

- **V3 (MRL 768d, ah=2) is the winner**: **4× smaller** (1.9 GB), **same Precision@10**, **same highly-relevant capture rate**, only **1.8% NDCG drop**. No latency penalty, no extra code complexity.
- **C (manual full-precision rerank) edges baseline on quality** with 50% size reduction, but at 72 ms Python-rerank latency vs ~5 ms.
- **A (PCA reduction)**, **B (bigger reorder pool)** offer essentially no benefit over baseline / V2.
- **score_ah(m=4) without truncation (V1)** is genuinely bad — avoid.

**Recommendation**: ship V3 as the default. Keep C as a "high-quality tier" option for searches that can tolerate the latency.

## Why this work happened

Multi-tenant scoping (see `PER_USER_LOGIN_2026-04-27.md`) raised the question of per-user index storage. We needed to know whether compaction would degrade search quality, and by how much. The first instinct was "just measure recall@K." That turned out to be misleading and required two more measurement rounds with progressively better methodology to get a defensible answer.

## The corpus

Numbers as of 2026-04-27:

- 565k vectors
- 3072 dims (Gemini `gemini-embedding-2-preview`)
- 13 shards × ~580 MB each
- Total **7.4 GB on disk**, ~85% of which (`dataset.npy`) is the full-precision vectors used for reorder

## Round 1 — recall@K against baseline

**Method**: build 6 variant indexes; for 100 queries from `query_cache`, compare each variant's top-K against the baseline's top-K. Recall@K = `|baseline_top_K ∩ variant_top_K| / K`.

**Variants tested**:

| variant | config | size | R@10 vs baseline | R@100 vs baseline |
|---|---|---|---|---|
| V0 baseline | 3072d, ah=2, reorder=100 | 7466 MB | 100% | 100% |
| V1 | 3072d, ah=4 (only AH change) | 7053 MB | 64.8% | 44.2% |
| V2 | MRL 1536d, ah=2 | 3736 MB | 84.1% | 79.0% |
| V3 | MRL 768d, ah=2 | 1871 MB | 77.2% | 72.2% |
| V4 | 1536d, ah=4 | 3530 MB | 63.1% | 46.1% |
| V5 | 768d, ah=4 | 1768 MB | 61.0% | 45.5% |

**What this told us**: every compaction "loses recall." The picture looked grim — we worried 16-30% recall losses meant 16-30% quality losses.

**What was wrong with this**: recall@K against the baseline is *self-referential*. It measures "do you return the same items as today" — NOT "do you return relevant items." A variant returning DIFFERENT but EQUALLY RELEVANT items would score badly here.

## Round 2 — LLM judge with a smaller variant set

**Method**: ask Gemini 2.5 Pro to grade each candidate's relevance to the query (0-3 scale). Compute industry-standard metrics:
- **NDCG@10** — graded relevance with logarithmic position discount
- **Precision@10** — fraction of returned results graded ≥ 2
- **Highly-relevant recovery** — % of grade=3 items captured

30 queries; 4 variants (V0 + the round-2 newcomers A/B/C).

| variant | NDCG@10 | Prec@10 | H.Rel found |
|---|---|---|---|
| V0 baseline | 0.852 | 71.0% | 151/172 (87.8%) |
| A — PCA(1536d) | **0.877** | 71.0% | 151/172 (87.8%) |
| B — MRL 1536d + reorder=500 | 0.850 | 70.7% | 149/172 (86.6%) |
| C — manual rerank | **0.872** | **71.7%** | 149/172 (86.6%) |

**What this told us**: A and C *outperformed* baseline. The recall@K numbers had lied — variants returning "different" items returned **equally good or better** items. Quality was preserved or improved.

**Why we needed round 3**: 30 queries is small. A's NDCG bump might be noise. We hadn't judged the round-1 V1-V5 variants. And we had no LLM-generated query coverage — only what was already in the cache.

## Round 3 — full sweep, 50 queries, 9 variants

**Method**: 50 queries (30 from cache + 20 LLM-generated for diversity). All 9 variants. Gemini 3.1 Pro Preview as judge. Metrics computed against the union-of-all-variants top-10 per query.

**Why subprocess-per-variant**: the round-3 in-process loop OOMed. Loading multiple 7GB ScaNN indexes in one Python process saturated 19 GB free + 15 GB swap. Even one-at-a-time with `del searcher; gc.collect()` didn't release ScaNN's C++ allocations to the OS reliably. Solution: each variant runs in a fresh `subprocess.run` of `scripts/scann_search_one.py`, writes its 50-query top-10 list to JSON, exits. The orchestrator (`scripts/scann_compaction_judge3.py`) then reads all the JSONs and judges in parallel.

### Final results

50 queries, 222 highly-relevant items in the union, sorted by NDCG@10:

| variant | size | NDCG@10 | Prec@10 | H.Rel found | desc |
|---|---|---|---|---|---|
| **C** | 3732 MB | **0.759** | **51.8%** | 142/222 (64%) | MRL 1536d AH + manual rerank vs full 3072d |
| **V0** baseline | 7466 MB | 0.754 | 51.6% | 142/222 (64%) | reference |
| **A** | 7058 MB | 0.753 | 51.4% | 142/222 (64%) | PCA(1536d) ah=2 |
| **V3** | **1871 MB** | **0.740** | **51.6%** | 142/222 (64%) | **MRL 768d ah=2 — 4× smaller** |
| V5 | 1768 MB | 0.731 | 50.2% | 142/222 (64%) | 768d ah=4 |
| V2 | 3736 MB | 0.723 | 49.2% | 137/222 (62%) | MRL 1536d ah=2 |
| B | 3732 MB | 0.723 | 49.2% | 137/222 (62%) | MRL 1536d + reorder=500 |
| V4 | 3530 MB | 0.709 | 45.6% | 130/222 (59%) | 1536d ah=4 |
| V1 | 7053 MB | 0.693 | 45.4% | 129/222 (58%) | 3072d ah=4 (only AH change) |

### Six findings

1. **V3 is the headline win.** MRL 768d, ah=2: 4× smaller than baseline, only **1.8% NDCG drop** (0.740 vs 0.754), **identical Precision@10** (51.6%), **identical highly-relevant recovery** (142/222). Round 1's recall@K said V3 was "27% worse" — graded relevance says it's essentially the same.

2. **C edges baseline on every quality metric.** NDCG 0.759 vs 0.754, Prec@10 51.8% vs 51.6%, same H.Rel recovery. 50% size reduction. Catch: **72 ms python-rerank latency vs ~5 ms** for the others.

3. **A (PCA) is essentially baseline.** Same quality, no size win. Drop from consideration.

4. **B (bigger reorder pool) is identical to V2.** The 5× larger reorder pool buys nothing at this scale. The lossy step isn't the reorder — it's the AH quantization itself.

5. **score_ah(m=4) without truncation (V1) is the worst variant.** -6pp Prec@10, -6pp H.Rel. Confirms round 1's recall numbers were measuring real quality loss for that single change.

6. **Combined ah=4 + truncation (V4/V5) is fine.** V5 is the smallest at 1.8 GB and still gets the same 64% H.Rel recovery as baseline. NDCG drops modestly (0.731 vs 0.754).

## What "manual rerank" means (variant C)

ScaNN's normal pipeline is two-stage:

1. **AH search** — fast/lossy pass using compressed (asymmetric hashed) vectors. Returns top-N candidates.
2. **Reorder** — rerank those top-N candidates using full-precision vectors stored in `dataset.npy`.

Constraint: ScaNN's built-in reorder uses the SAME dim as the indexed vectors. If you build with 1536d truncated vectors, the reorder is also 1536d. You can't natively do "AH search at 1536d, reorder at 3072d" within one builder.

**Manual rerank**: we do the reorder ourselves in Python with the full-dim vectors.

Pseudocode:
```python
# Build phase: index 1536d truncated vectors normally.
searcher = scann.scann_ops_pybind.builder(truncated_vectors, 500, "dot_product")
    .tree(...)
    .score_ah(2, ...)
    .reorder(500)
    .build()

# Query phase:
def search(full_3072d_query):
    truncated_q = full_3072d_query[:1536]
    candidate_ids, _ = searcher.search(truncated_q, final_num_neighbors=500)
    # Manual rerank against FULL precision vectors held in numpy memmap:
    candidate_vecs = corpus_memmap[positions_for(candidate_ids)]  # shape (500, 3072)
    scores = (candidate_vecs / l2norm) @ (full_3072d_query / l2norm)
    top_10_positions = np.argpartition(scores, -10)[-10:]
    return [candidate_ids[p] for p in sorted_by_score(top_10_positions)]
```

The win: AH stage stays small + fast (1536d quantized vectors), but the FINAL top-10 ranking is computed against full-precision vectors → near-baseline accuracy.

The cost: the Python loop (memmap fetch + numpy dot product) is much slower than ScaNN's C++ reorder. **72 ms vs ~5 ms** in our measurements. Could be cut to ~10 ms by writing the rerank in C++ or by using a more efficient numpy/BLAS call pattern (current implementation is the literal-Python version), but that's separate engineering.

## Recommendation

**Default**: ship **V3 (MRL 768d, ah=2)**. 4× smaller, ~baseline quality, no latency penalty, no new code paths.

For your current single-corpus footprint: replace 7.4 GB on disk with 1.9 GB. Free 5.5 GB.

For multi-tenant per-user indexes: a typical mailbox of 50k messages would need ~165 MB at V3 sizing (vs ~660 MB at full 3072d baseline). 50 invited users at V3 sizing = 8 GB total — trivially fits in RAM with the kernel page cache handling cold users.

**Premium tier**: keep **C (manual rerank)** in the codebase as an option. For agent-driven searches where the writer/critic chain consumes results, the +0.005 NDCG and +0.2pp Precision@10 might be worth the 72ms latency. Optional. Not the default.

**Drop**: A, B, V1 (no benefit). V4, V5 are interesting smallest-possible options if 1.8 GB ever matters (it shouldn't given V3's 1.9 GB).

## Operational rollout (if we ship V3)

1. **Code change**: in `src/gmail_search/index/builder.py:_build_scann_from_vectors`, after loading the corpus vectors, slice + normalize to 768d:
   ```python
   if dimensions > 768:
       vectors = vectors[:, :768].astype(np.float32, copy=True)
       norms = np.linalg.norm(vectors, axis=1, keepdims=True)
       norms[norms == 0] = 1.0
       vectors = vectors / norms
   ```
2. **Query side** (`src/gmail_search/search/engine.py`): truncate + normalize the query embedding the same way before calling `searcher.search`.
3. **Re-index**: `gmail-search index --rebuild`. Takes ~16s per the round-1 measurement.
4. **Embeddings table in PG**: stays at 12288 bytes (3072 float32) for now — only the ScaNN index is truncated. The embeddings table is the source of truth and we keep it full-precision. (If we ever want to also shrink the table, that's a separate ~hours-long migration.)
5. **Verify**: re-run the judge eval against the new live index; NDCG should match V3's 0.740.

## Caveats worth naming

- **N=50 queries**, 222 highly-relevant items in the union. Confidence intervals overlap for small NDCG differences (e.g. C vs V0 at 0.759 vs 0.754 — within noise). The bigger jumps (V3 vs V1 at 0.740 vs 0.693) are real.
- **The judge is itself an approximation.** Gemini 3.1 Pro is strong at relevance grading but not infallible. A human eval of a 30-query subset would tighten this further but costs real time.
- **The recall@K we computed in round 1 was wrong as a quality proxy.** Worth remembering for future evals: measure quality (graded relevance, click-through, user feedback), not "do we return the same items as today."
- **Manual rerank latency is Python-bound, not algorithmically required.** A C++ implementation could plausibly bring C's 72ms to single-digit ms. Out of scope for this work.
- **MRL truncation works because Gemini's `gemini-embedding-2-preview` is MRL-trained.** For a non-MRL embedding model, naive prefix truncation would be much worse.
- **Synthetic queries (20 of 50) were Gemini-generated** to cover topics not represented in the cached query log. They may be more "neutral" than real user queries. The 30 cached queries are the realistic distribution.

## Files produced

All artifacts are in `/tmp/scann_compaction_eval/` (kept around per Scott's request):

- `corpus.memmap` — 6.6 GB, 565k × 3072 float32 vectors loaded from PG
- `corpus_ids.json` — embedding IDs aligned with memmap rows
- `eval_queries_n100.{json,npy}` — 100 random query embeddings from `query_cache` (used by rounds 1-2)
- `judge2_queries.json`, `judge2_queries.npy` — 50-query canonical set for round 3 (30 cached + 20 synthetic)
- `index_v0_ah2_3072d/` through `index_v5_ah4_768d/` — round-1 variants
- `index_round2_A/`, `index_round2_B/`, `index_round2_C/` — round-2 variants
- `searches_<variant>.json` (×9) — per-variant top-10 lists for the 50-query set
- `judge3_results.json` — final aggregate metrics
- `judge3_progress.log` — per-query progress log

## Scripts

- `scripts/scann_compaction_eval.py` — round 1 (recall@K against baseline)
- `scripts/scann_compaction_eval2.py` — round 2 (variants A, B, C builds)
- `scripts/scann_compaction_judge.py` — round 2 (Gemini 2.5 Pro judge, 30 queries)
- `scripts/scann_search_one.py` — subprocess worker (one variant's searches)
- `scripts/scann_compaction_judge3.py` — round 3 orchestrator (50 queries, 9 variants)

Total spend: ~$0.50 for synthetic query generation + judging across all 3 rounds. Compute time: ~2 hours wall clock (mostly index builds).
