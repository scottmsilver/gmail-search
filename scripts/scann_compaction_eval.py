"""ScaNN compaction evaluation against the production corpus.

Builds N variant indexes and measures recall + size + latency vs the
current baseline. Recall is computed against the baseline's top-K
results — the baseline IS the gold standard (it's what's serving
queries today). So if a variant returns the same items in the same
positions, recall = 100% even if a smaller index would in principle
return better results.

Eval set: random sample of `EVAL_N` real user queries from
`query_cache`. These are queries you've actually issued.

Variants:
  V0 — baseline (production config: score_ah(2), full 3072 dims)
  V1 — score_ah(4): higher AH compression, same dims
  V2 — MRL truncation to 1536 dims
  V3 — MRL truncation to 768 dims
  V4 — combined: 1536 dims + score_ah(4)
  V5 — extreme: 768 dims + score_ah(4)

Output: a results table you can paste into the plan doc.

Cost: zero API calls. Pure offline benchmark on the existing data.
Wall time: ~30-60 minutes for the full sweep (5 indexes × build + eval).
"""

from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import psycopg
import scann
from psycopg.rows import dict_row

DSN = "postgresql://gmail_search:gmail_search@127.0.0.1:5544/gmail_search"
MODEL = "gemini-embedding-2-preview"
DIMENSIONS = 3072
EVAL_N = 100  # number of queries to evaluate
WORK_DIR = Path("/tmp/scann_compaction_eval")
TOP_K = 100  # we measure recall@10, recall@50, recall@100


def load_corpus() -> tuple[list[int], np.memmap]:
    """Stream all embeddings into a disk-backed memmap so peak python
    RSS during build stays bounded."""
    conn = psycopg.connect(DSN, row_factory=dict_row)
    count_row = conn.execute("SELECT COUNT(*) AS n FROM embeddings WHERE model = %s", (MODEL,)).fetchone()
    count = count_row["n"]
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    memmap_path = WORK_DIR / "corpus.memmap"
    print(f"loading {count} corpus embeddings -> {memmap_path}")
    vectors = np.memmap(memmap_path, dtype=np.float32, mode="w+", shape=(count, DIMENSIONS))
    ids: list[int] = []
    cur = conn.execute("SELECT id, embedding FROM embeddings WHERE model = %s ORDER BY id", (MODEL,))
    for i, row in enumerate(cur):
        ids.append(row["id"])
        vectors[i] = np.frombuffer(row["embedding"], dtype=np.float32)
        if (i + 1) % 50000 == 0:
            print(f"  loaded {i + 1}/{count}")
    vectors.flush()
    conn.close()
    return ids, vectors


def load_eval_queries(n: int) -> list[tuple[str, np.ndarray]]:
    conn = psycopg.connect(DSN, row_factory=dict_row)
    rows = conn.execute(
        """SELECT query_text, embedding FROM query_cache
           WHERE model = %s AND OCTET_LENGTH(embedding) = %s
           ORDER BY random() LIMIT %s""",
        (MODEL, DIMENSIONS * 4, n),
    ).fetchall()
    conn.close()
    out = []
    for r in rows:
        emb = np.frombuffer(r["embedding"], dtype=np.float32).copy()
        out.append((r["query_text"], emb))
    return out


def truncate_and_norm(vectors: np.ndarray, dim: int | None) -> np.ndarray:
    if dim is None:
        return vectors
    truncated = vectors[:, :dim].astype(np.float32, copy=True)
    norms = np.linalg.norm(truncated, axis=1, keepdims=True)
    # Avoid divide-by-zero on any pathological zero-vector rows.
    norms[norms == 0] = 1.0
    return truncated / norms


def build_index(
    vectors: np.ndarray,
    ids: list[int],
    *,
    score_ah_m: int,
    truncate_dim: int | None,
    name: str,
) -> tuple[Path, float]:
    n = len(ids)
    truncated = truncate_and_norm(vectors, truncate_dim)
    num_leaves = max(int(math.sqrt(n)), 1)

    builder = scann.scann_ops_pybind.builder(truncated, TOP_K, "dot_product")
    builder = builder.tree(
        num_leaves=num_leaves,
        num_leaves_to_search=min(num_leaves, 100),
        training_sample_size=min(n, 250000),
    )
    builder = builder.score_ah(score_ah_m, anisotropic_quantization_threshold=0.2)
    builder = builder.reorder(100)

    t0 = time.time()
    searcher = builder.build()
    build_time = time.time() - t0

    out_dir = WORK_DIR / f"index_{name}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    searcher.serialize(str(out_dir))
    (out_dir / "ids.json").write_text(json.dumps(ids))
    return out_dir, build_time


def measure_size_bytes(index_dir: Path) -> int:
    return sum(f.stat().st_size for f in index_dir.rglob("*") if f.is_file())


def eval_index(
    index_dir: Path,
    queries: list[tuple[str, np.ndarray]],
    ids: list[int],
    truncate_dim: int | None,
) -> list[tuple[str, list[int], float]]:
    searcher = scann.scann_ops_pybind.load_searcher(str(index_dir))
    out = []
    for qtext, emb in queries:
        if truncate_dim:
            emb_q = emb[:truncate_dim].astype(np.float32, copy=True)
            n = np.linalg.norm(emb_q)
            if n > 0:
                emb_q = emb_q / n
        else:
            n = np.linalg.norm(emb)
            emb_q = emb if n == 0 else emb / n
        t0 = time.time()
        local_indices, _ = searcher.search(emb_q, final_num_neighbors=TOP_K)
        latency_ms = (time.time() - t0) * 1000
        top_ids = [ids[i] for i in local_indices]
        out.append((qtext, top_ids, latency_ms))
    return out


def recall_at_k(gold: list[int], candidate: list[int], k: int) -> float:
    return len(set(gold[:k]) & set(candidate[:k])) / k


def main() -> None:
    print("=" * 70)
    print("ScaNN compaction evaluation")
    print("=" * 70)

    ids, vectors = load_corpus()
    print(f"corpus: {len(ids)} vectors x {vectors.shape[1]} dims")

    queries = load_eval_queries(EVAL_N)
    print(f"eval queries: {len(queries)}")

    variants = [
        ("v0_ah2_3072d", {"score_ah_m": 2, "truncate_dim": None}),
        ("v1_ah4_3072d", {"score_ah_m": 4, "truncate_dim": None}),
        ("v2_ah2_1536d", {"score_ah_m": 2, "truncate_dim": 1536}),
        ("v3_ah2_768d", {"score_ah_m": 2, "truncate_dim": 768}),
        ("v4_ah4_1536d", {"score_ah_m": 4, "truncate_dim": 1536}),
        ("v5_ah4_768d", {"score_ah_m": 4, "truncate_dim": 768}),
    ]

    results: dict[str, dict] = {}
    for name, cfg in variants:
        print(f"\n--- building {name} (score_ah_m={cfg['score_ah_m']}, truncate_dim={cfg['truncate_dim']}) ---")
        try:
            idx_dir, build_time = build_index(vectors, ids, name=name, **cfg)
        except Exception as exc:
            print(f"  BUILD FAILED: {exc}")
            results[name] = {"error": str(exc), "cfg": cfg}
            continue
        size_b = measure_size_bytes(idx_dir)
        print(f"  built in {build_time:.1f}s, size = {size_b / 1024 / 1024:.0f} MB")
        try:
            eval_results = eval_index(idx_dir, queries, ids, cfg["truncate_dim"])
        except Exception as exc:
            print(f"  EVAL FAILED: {exc}")
            results[name] = {"error": str(exc), "cfg": cfg}
            continue
        avg_latency = sum(r[2] for r in eval_results) / len(eval_results)
        print(f"  avg query latency: {avg_latency:.1f} ms")
        results[name] = {
            "cfg": cfg,
            "size_mb": size_b / 1024 / 1024,
            "build_time_s": build_time,
            "avg_latency_ms": avg_latency,
            "eval_results": eval_results,
        }

    # Compute recall vs baseline.
    baseline = results.get("v0_ah2_3072d")
    if not baseline or "eval_results" not in baseline:
        print("baseline failed — cannot compute recall")
        return

    print()
    print("=" * 100)
    print(
        f"{'variant':22s} {'size':>9s} {'shrink':>7s} {'build':>8s} {'lat':>9s}  {'R@10':>6s}  {'R@50':>6s}  {'R@100':>6s}"
    )
    print("-" * 100)
    for name, r in results.items():
        if "error" in r:
            print(f"{name:22s}  ERROR: {r['error'][:60]}")
            continue
        if name == "v0_ah2_3072d":
            print(
                f"{name:22s} {r['size_mb']:>6.0f} MB {'1.00x':>7s} {r['build_time_s']:>6.1f}s {r['avg_latency_ms']:>6.1f} ms  {'100.0%':>6s}  {'100.0%':>6s}  {'100.0%':>6s}  (baseline)"
            )
            continue
        recalls: dict[int, list[float]] = {k: [] for k in [10, 50, 100]}
        for (qt_b, gold_ids, _), (qt_v, var_ids, _) in zip(baseline["eval_results"], r["eval_results"]):
            assert qt_b == qt_v
            for k in [10, 50, 100]:
                recalls[k].append(recall_at_k(gold_ids, var_ids, k))
        avg = {k: sum(v) / len(v) for k, v in recalls.items()}
        shrink = baseline["size_mb"] / r["size_mb"]
        print(
            f"{name:22s} {r['size_mb']:>6.0f} MB {shrink:>6.2f}x {r['build_time_s']:>6.1f}s "
            f"{r['avg_latency_ms']:>6.1f} ms  {avg[10] * 100:>5.1f}%  {avg[50] * 100:>5.1f}%  {avg[100] * 100:>5.1f}%"
        )

    # Persist raw numbers for follow-up.
    summary_path = WORK_DIR / "results.json"
    summary_path.write_text(
        json.dumps(
            {name: {k: v for k, v in r.items() if k != "eval_results"} for name, r in results.items()},
            indent=2,
            default=str,
        )
    )
    print(f"\nraw results: {summary_path}")


if __name__ == "__main__":
    main()
