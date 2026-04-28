"""Round 2 of ScaNN compaction eval — runs ONE variant per process so
we can launch A/B/C in parallel.

Variant names:
  A  PCA(1536) — ScaNN's built-in PCA reduction to 1536 dims.
  B  MRL 1536d + reorder(500) — same as round-1 V2 but bigger reorder pool.
  C  MRL 1536d AH + MANUAL rerank against full 3072d in Python.

Eval methodology matches round 1: recall@K vs the round-1 baseline
(production config, full 3072d, score_ah(2), reorder(100)).

Usage:
    python scripts/scann_compaction_eval2.py [A|B|C]

Outputs:
    /tmp/scann_compaction_eval/round2_<variant>.json
"""

from __future__ import annotations

import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import psycopg
import scann
from psycopg.rows import dict_row

DSN = "postgresql://gmail_search:gmail_search@127.0.0.1:5544/gmail_search"
MODEL = "gemini-embedding-2-preview"
DIMENSIONS = 3072
EVAL_N = 100
WORK_DIR = Path("/tmp/scann_compaction_eval")
TOP_K = 100


def load_corpus_memmap() -> tuple[list[int], np.memmap]:
    """Reuse the corpus.memmap built by round 1 if present; rebuild if
    missing. Bypasses the slow PG read on subsequent runs."""
    memmap_path = WORK_DIR / "corpus.memmap"
    ids_path = WORK_DIR / "corpus_ids.json"
    if memmap_path.exists() and ids_path.exists():
        ids = json.loads(ids_path.read_text())
        n = len(ids)
        vectors = np.memmap(memmap_path, dtype=np.float32, mode="r", shape=(n, DIMENSIONS))
        print(f"reused cached corpus: {n} vectors x {DIMENSIONS} dims")
        return ids, vectors
    # Else stream from PG (round-1 path).
    conn = psycopg.connect(DSN, row_factory=dict_row)
    count = conn.execute("SELECT COUNT(*) AS n FROM embeddings WHERE model = %s", (MODEL,)).fetchone()["n"]
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    vectors = np.memmap(memmap_path, dtype=np.float32, mode="w+", shape=(count, DIMENSIONS))
    ids: list[int] = []
    cur = conn.execute("SELECT id, embedding FROM embeddings WHERE model = %s ORDER BY id", (MODEL,))
    for i, row in enumerate(cur):
        ids.append(row["id"])
        vectors[i] = np.frombuffer(row["embedding"], dtype=np.float32)
    vectors.flush()
    conn.close()
    ids_path.write_text(json.dumps(ids))
    return ids, vectors


def load_eval_queries(n: int) -> list[tuple[str, np.ndarray]]:
    """Reuse the same 100 queries round 1 used so the recall numbers
    are directly comparable. Persisted to disk on first call."""
    cached = WORK_DIR / f"eval_queries_n{n}.json"
    cached_emb = WORK_DIR / f"eval_queries_n{n}.npy"
    if cached.exists() and cached_emb.exists():
        texts = json.loads(cached.read_text())
        embs = np.load(cached_emb)
        return list(zip(texts, embs))
    conn = psycopg.connect(DSN, row_factory=dict_row)
    rows = conn.execute(
        """SELECT query_text, embedding FROM query_cache
           WHERE model = %s AND OCTET_LENGTH(embedding) = %s
           ORDER BY random() LIMIT %s""",
        (MODEL, DIMENSIONS * 4, n),
    ).fetchall()
    conn.close()
    texts = [r["query_text"] for r in rows]
    embs = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    cached.write_text(json.dumps(texts))
    np.save(cached_emb, embs)
    return list(zip(texts, embs))


def truncate_and_norm(vectors: np.ndarray, dim: int | None) -> np.ndarray:
    if dim is None:
        return vectors
    truncated = vectors[:, :dim].astype(np.float32, copy=True)
    norms = np.linalg.norm(truncated, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return truncated / norms


def measure_size_bytes(index_dir: Path) -> int:
    return sum(f.stat().st_size for f in index_dir.rglob("*") if f.is_file())


def recall_at_k(gold: list[int], candidate: list[int], k: int) -> float:
    return len(set(gold[:k]) & set(candidate[:k])) / k


# ── Baseline gold (loaded from round 1 results) ─────────────────────


def load_round1_baseline() -> list[tuple[str, list[int]]]:
    """Round 1's V0 baseline produced top-100 IDs per query. We need
    those to score round-2 variants without rerunning the baseline.
    Round 1's results.json doesn't include the per-query IDs (we
    stripped them) — so we rebuild + run V0 here against the same
    eval-queries cache."""
    baseline_results_path = WORK_DIR / "baseline_v0_results.json"
    if baseline_results_path.exists():
        return json.loads(baseline_results_path.read_text())

    print("=== building baseline V0 (3072d, ah=2, reorder=100) for gold scores ===")
    ids, vectors = load_corpus_memmap()
    queries = load_eval_queries(EVAL_N)

    n = len(ids)
    num_leaves = max(int(math.sqrt(n)), 1)
    builder = scann.scann_ops_pybind.builder(vectors, TOP_K, "dot_product")
    builder = builder.tree(
        num_leaves=num_leaves,
        num_leaves_to_search=min(num_leaves, 100),
        training_sample_size=min(n, 250000),
    )
    builder = builder.score_ah(2, anisotropic_quantization_threshold=0.2)
    builder = builder.reorder(100)
    t0 = time.time()
    searcher = builder.build()
    print(f"baseline built in {time.time() - t0:.1f}s")

    out: list[tuple[str, list[int]]] = []
    for qtext, emb in queries:
        n_emb = np.linalg.norm(emb)
        emb_q = emb if n_emb == 0 else emb / n_emb
        local_indices, _ = searcher.search(emb_q.astype(np.float32), final_num_neighbors=TOP_K)
        out.append((qtext, [ids[i] for i in local_indices]))

    baseline_results_path.write_text(json.dumps(out))
    return out


# ── Variant A: PCA(1536) ────────────────────────────────────────────


def variant_a(ids, vectors, queries):
    """ScaNN's built-in .pca(target_dim) — learns a data-dependent
    projection rather than naive prefix truncation. Hypothesis: better
    recall than MRL at the same compression."""
    n = len(ids)
    num_leaves = max(int(math.sqrt(n)), 1)

    builder = scann.scann_ops_pybind.builder(vectors, TOP_K, "dot_product")
    builder = builder.tree(
        num_leaves=num_leaves,
        num_leaves_to_search=min(num_leaves, 100),
        training_sample_size=min(n, 250000),
    )
    # PCA before AH: shrinks the per-vector storage to `target_dim`
    # and changes the score_ah dimension count.
    # ScaNN's pca() defaults pca_significance_threshold=0.80 (not None);
    # passing reduction_dim alone trips the "both set" error path. Must
    # explicitly clear the threshold to use reduction_dim mode.
    builder = builder.pca(reduction_dim=1536, pca_significance_threshold=None)
    builder = builder.score_ah(2, anisotropic_quantization_threshold=0.2)
    builder = builder.reorder(100)

    t0 = time.time()
    searcher = builder.build()
    build_time = time.time() - t0

    out_dir = WORK_DIR / "index_round2_A"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    searcher.serialize(str(out_dir))

    eval_results = []
    for qtext, emb in queries:
        # Query is full 3072d; ScaNN handles the PCA projection
        # internally because the projection matrix is in the index.
        n_emb = np.linalg.norm(emb)
        emb_q = emb if n_emb == 0 else emb / n_emb
        t0 = time.time()
        local_indices, _ = searcher.search(emb_q.astype(np.float32), final_num_neighbors=TOP_K)
        latency_ms = (time.time() - t0) * 1000
        eval_results.append((qtext, [ids[i] for i in local_indices], latency_ms))

    return out_dir, build_time, eval_results


# ── Variant B: MRL 1536 + reorder(500) ──────────────────────────────


def variant_b(ids, vectors, queries):
    """Same as round-1 V2 (1536d MRL truncation, ah=2) but with a
    bigger reorder pool. Hypothesis: most recall loss is in the AH
    search candidate-selection step; reranking 5x more candidates at
    full precision recovers most of it. Cost: bigger reorder buffer
    in RAM + slightly slower per-query latency."""
    n = len(ids)
    truncated = truncate_and_norm(vectors, 1536)
    num_leaves = max(int(math.sqrt(n)), 1)

    builder = scann.scann_ops_pybind.builder(truncated, TOP_K, "dot_product")
    builder = builder.tree(
        num_leaves=num_leaves,
        num_leaves_to_search=min(num_leaves, 100),
        training_sample_size=min(n, 250000),
    )
    builder = builder.score_ah(2, anisotropic_quantization_threshold=0.2)
    builder = builder.reorder(500)  # 5x baseline

    t0 = time.time()
    searcher = builder.build()
    build_time = time.time() - t0

    out_dir = WORK_DIR / "index_round2_B"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    searcher.serialize(str(out_dir))

    eval_results = []
    for qtext, emb in queries:
        emb_q = emb[:1536].astype(np.float32, copy=True)
        n_emb = np.linalg.norm(emb_q)
        if n_emb > 0:
            emb_q = emb_q / n_emb
        t0 = time.time()
        local_indices, _ = searcher.search(emb_q, final_num_neighbors=TOP_K)
        latency_ms = (time.time() - t0) * 1000
        eval_results.append((qtext, [ids[i] for i in local_indices], latency_ms))

    return out_dir, build_time, eval_results


# ── Variant C: 1536d AH search + manual rerank against full 3072d ──


def variant_c(ids, vectors, queries):
    """Hybrid: build an AH-only index on truncated 1536d vectors,
    fetch top-500 candidates per query, then rerank in Python using
    the FULL 3072d vectors. Tests the hypothesis that the AH stage
    is the lossy step — reordering at full precision should recover
    near-baseline recall.

    Trade-off: small index on disk for AH (1536d), but the full 3072d
    corpus must be in RAM at query time. With per-user mailboxes ~50k
    msgs each, that's ~600 MB per user — acceptable."""
    n = len(ids)
    truncated = truncate_and_norm(vectors, 1536)
    num_leaves = max(int(math.sqrt(n)), 1)

    # Candidate generation index: 1536d AH, large reorder (500) so the
    # per-leaf candidate set is generous before our manual rerank.
    builder = scann.scann_ops_pybind.builder(truncated, 500, "dot_product")
    builder = builder.tree(
        num_leaves=num_leaves,
        num_leaves_to_search=min(num_leaves, 100),
        training_sample_size=min(n, 250000),
    )
    builder = builder.score_ah(2, anisotropic_quantization_threshold=0.2)
    builder = builder.reorder(500)

    t0 = time.time()
    searcher = builder.build()
    build_time = time.time() - t0

    out_dir = WORK_DIR / "index_round2_C"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    searcher.serialize(str(out_dir))

    # Build a numpy id-position lookup so we can index into `vectors`
    # by PG embedding id quickly.
    id_to_pos = {pid: i for i, pid in enumerate(ids)}

    eval_results = []
    for qtext, emb in queries:
        # 1536d query for the AH search stage
        emb_q_t = emb[:1536].astype(np.float32, copy=True)
        n_t = np.linalg.norm(emb_q_t)
        if n_t > 0:
            emb_q_t = emb_q_t / n_t

        t0 = time.time()
        local_indices, _ = searcher.search(emb_q_t, final_num_neighbors=500)
        candidate_ids = [ids[i] for i in local_indices]

        # Manual rerank against full 3072d vectors. Normalize the full
        # query the same way (cosine ~ dot product on normalized vecs).
        emb_q_full = emb.astype(np.float32, copy=True)
        n_full = np.linalg.norm(emb_q_full)
        if n_full > 0:
            emb_q_full = emb_q_full / n_full
        cand_positions = [id_to_pos[c] for c in candidate_ids]
        # Pull and normalize full-dim vectors for the candidates.
        cand_vecs = vectors[cand_positions]  # shape (500, 3072), reads from memmap
        cand_norms = np.linalg.norm(cand_vecs, axis=1, keepdims=True)
        cand_norms[cand_norms == 0] = 1.0
        cand_vecs_n = cand_vecs / cand_norms
        scores = cand_vecs_n @ emb_q_full
        # Top-100 by score
        top_pos = np.argpartition(scores, -TOP_K)[-TOP_K:]
        top_pos_sorted = top_pos[np.argsort(scores[top_pos])[::-1]]
        top_ids = [candidate_ids[p] for p in top_pos_sorted]

        latency_ms = (time.time() - t0) * 1000
        eval_results.append((qtext, top_ids, latency_ms))

    return out_dir, build_time, eval_results


VARIANTS = {"A": variant_a, "B": variant_b, "C": variant_c}


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in VARIANTS:
        print(f"usage: {sys.argv[0]} [{'|'.join(VARIANTS)}]", file=sys.stderr)
        sys.exit(2)
    variant = sys.argv[1]
    fn = VARIANTS[variant]

    print(f"=== variant {variant} starting ===")
    ids, vectors = load_corpus_memmap()
    queries = load_eval_queries(EVAL_N)
    baseline = load_round1_baseline()
    assert [b[0] for b in baseline] == [q[0] for q in queries], "baseline + queries diverged"

    print(f"running variant {variant}...")
    t0 = time.time()
    idx_dir, build_time, eval_results = fn(ids, vectors, queries)
    total_time = time.time() - t0

    size_b = measure_size_bytes(idx_dir)
    avg_latency = sum(r[2] for r in eval_results) / len(eval_results)

    recalls: dict[int, list[float]] = {k: [] for k in [10, 50, 100]}
    for (qt_b, gold_ids), (qt_v, var_ids, _) in zip(baseline, eval_results):
        assert qt_b == qt_v
        for k in [10, 50, 100]:
            recalls[k].append(recall_at_k(gold_ids, var_ids, k))
    avg_recalls = {k: sum(v) / len(v) for k, v in recalls.items()}

    out = {
        "variant": variant,
        "size_mb": size_b / 1024 / 1024,
        "build_time_s": build_time,
        "total_time_s": total_time,
        "avg_latency_ms": avg_latency,
        "recall_at_10": avg_recalls[10],
        "recall_at_50": avg_recalls[50],
        "recall_at_100": avg_recalls[100],
    }
    summary = WORK_DIR / f"round2_{variant}.json"
    summary.write_text(json.dumps(out, indent=2))
    print(f"\n=== variant {variant} DONE ===")
    print(json.dumps(out, indent=2))
    print(f"\nresults written to {summary}")


if __name__ == "__main__":
    main()
