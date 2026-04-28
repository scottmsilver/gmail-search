"""Run ONE variant's searches over the cached query set, write results
to JSON, exit. Designed to be spawned as a subprocess per variant so
each variant gets a fresh Python process — and process exit guarantees
all ScaNN C++ memory is freed before the next variant loads. The
in-process loop in scann_compaction_judge2.py thrashed swap because
gc.collect() didn't reliably release C++ allocations.

Usage:
    python scripts/scann_search_one.py <variant_name>

Output:
    /tmp/scann_compaction_eval/searches_<variant>.json
    {"variant": ..., "results": [[id, ...], ...]}  # list per query
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import scann

WORK_DIR = Path("/tmp/scann_compaction_eval")
DIMENSIONS = 3072
TOP_K = 10


def make_simple(searcher, truncate_dim):
    def search(emb):
        if truncate_dim:
            eq = emb[:truncate_dim].astype(np.float32, copy=True)
        else:
            eq = emb.astype(np.float32, copy=True)
        n = np.linalg.norm(eq)
        if n > 0:
            eq = eq / n
        local_idx, _ = searcher.search(eq, final_num_neighbors=TOP_K)
        return local_idx.tolist()

    return search


def make_manual_rerank(searcher, ah_dim, ids, vectors, id_to_pos):
    def search(emb):
        eq_t = emb[:ah_dim].astype(np.float32, copy=True)
        n = np.linalg.norm(eq_t)
        if n > 0:
            eq_t = eq_t / n
        local_idx, _ = searcher.search(eq_t, final_num_neighbors=500)
        cand_ids = [ids[i] for i in local_idx]
        eq_full = emb.astype(np.float32, copy=True)
        nf = np.linalg.norm(eq_full)
        if nf > 0:
            eq_full = eq_full / nf
        cand_pos = [id_to_pos[c] for c in cand_ids]
        cand_v = vectors[cand_pos]
        cn = np.linalg.norm(cand_v, axis=1, keepdims=True)
        cn[cn == 0] = 1.0
        cand_v = cand_v / cn
        scores = cand_v @ eq_full
        top_pos = np.argpartition(scores, -TOP_K)[-TOP_K:]
        top_pos_sorted = top_pos[np.argsort(scores[top_pos])[::-1]]
        # Return PG embedding ids (caller maps via the same ids list).
        # We return positions-in-ids-list to match the simple variants'
        # contract; the calling judge will map both to embedding ids.
        # Actually simpler: convert to positions.
        return [ids.index(cand_ids[p]) for p in top_pos_sorted]

    return search


VARIANT_DEFS = {
    "V0": ("index_v0_ah2_3072d", "simple", None),
    "V1": ("index_v1_ah4_3072d", "simple", None),
    "V2": ("index_v2_ah2_1536d", "simple", 1536),
    "V3": ("index_v3_ah2_768d", "simple", 768),
    "V4": ("index_v4_ah4_1536d", "simple", 1536),
    "V5": ("index_v5_ah4_768d", "simple", 768),
    "A": ("index_round2_A", "simple", None),
    "B": ("index_round2_B", "simple", 1536),
    "C": ("index_round2_C", "rerank", 1536),
}


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in VARIANT_DEFS:
        print(f"usage: {sys.argv[0]} {{{ '|'.join(VARIANT_DEFS) }}}", file=sys.stderr)
        sys.exit(2)
    variant = sys.argv[1]
    path_part, mode, dim_arg = VARIANT_DEFS[variant]

    # Load cached corpus + queries (always cached at this point).
    ids = json.loads((WORK_DIR / "corpus_ids.json").read_text())
    n = len(ids)
    vectors = np.memmap(WORK_DIR / "corpus.memmap", dtype=np.float32, mode="r", shape=(n, DIMENSIONS))

    # Combined query set: 30 cached + 20 synthetic.
    cached_texts = []
    cached_embs = []
    if (WORK_DIR / "judge2_queries.json").exists():
        # Reuse the canonical query set across variants.
        q = json.loads((WORK_DIR / "judge2_queries.json").read_text())
        cached_texts = q["texts"]
        cached_embs = np.load(WORK_DIR / "judge2_queries.npy")
    else:
        print(f"ERROR: no canonical query set at {WORK_DIR}/judge2_queries.json", file=sys.stderr)
        print("Run scann_compaction_judge2.py with the new orchestrator path first.", file=sys.stderr)
        sys.exit(1)

    print(f"variant={variant} mode={mode} dim_arg={dim_arg}", flush=True)
    print(f"loading {path_part} ...", flush=True)
    t0 = time.time()
    searcher = scann.scann_ops_pybind.load_searcher(str(WORK_DIR / path_part))
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)

    if mode == "simple":
        search_fn = make_simple(searcher, dim_arg)
    else:
        id_to_pos = {pid: i for i, pid in enumerate(ids)}
        search_fn = make_manual_rerank(searcher, dim_arg, ids, vectors, id_to_pos)

    print(f"running {len(cached_texts)} searches ...", flush=True)
    t0 = time.time()
    results: list[list[int]] = []
    for emb in cached_embs:
        local_positions = search_fn(emb)
        # Map ScaNN local positions to PG embedding ids. local_positions
        # is a list of ints into the `ids` array.
        results.append([ids[p] for p in local_positions])
    print(f"searches done in {time.time() - t0:.1f}s", flush=True)

    out_path = WORK_DIR / f"searches_{variant}.json"
    out_path.write_text(json.dumps({"variant": variant, "results": results}))
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
