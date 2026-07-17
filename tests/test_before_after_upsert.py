"""Before/after correctness gate: growing the tail via UPSERT must return the
same search results as the old REBUILD path.

True head-to-head — both arms start from the SAME base index and consume the
SAME 20 new embeddings; the only difference is how the open tail is extended:
  BEFORE = rebuild the tail from PG (old path, forced by disabling upsert)
  AFTER  = ScaNN native upsert into the frozen tail (new path)

We disable index GC for the experiment so the BEFORE arm can reset the pointer
back to the base and the AFTER arm can branch from the identical base.
"""

import struct
from datetime import datetime

import numpy as np
from gmail_search.index import builder as B
from gmail_search.index.builder import build_index_delta, build_index_sharded
from gmail_search.index.searcher import ScannSearcher
from gmail_search.store.db import get_connection, init_db
from gmail_search.store.models import EmbeddingRecord, Message
from gmail_search.store.queries import insert_embedding, set_active_index_dir, upsert_message

MODEL = "test-model"

# Clustered vectors — representative of real embeddings, where each item has a
# well-separated neighborhood. (Uniform-random high-dim vectors are nearly
# equidistant, so their "top-10" is a near-tie that any ANN pool difference
# reshuffles — an adversarial case that tests the floating-point tie-break, not
# retrieval.) K clusters, each corpus/query point = a center + small noise.
_K = 40
_CENTERS = None


def _centers(dims):
    global _CENTERS
    if _CENTERS is None:
        rng = np.random.default_rng(12345)
        c = rng.standard_normal((_K, dims)).astype(np.float32)
        c /= np.linalg.norm(c, axis=1, keepdims=True)
        _CENTERS = c
    return _CENTERS


def _clustered_vec(dims, seed, cluster):
    rng = np.random.default_rng(seed)
    v = _centers(dims)[cluster] + 0.08 * rng.standard_normal(dims).astype(np.float32)
    v = v / np.linalg.norm(v)
    return v.astype(np.float32)


def _rand_embedding(dims, seed):
    return struct.pack(f"{dims}f", *_clustered_vec(dims, seed, seed % _K).tolist())


def _add(db, start, n, dims):
    conn = get_connection(db)
    for i in range(start, start + n):
        upsert_message(
            conn,
            Message(
                id=f"m{i:05d}",
                thread_id="t1",
                from_addr="a@b.com",
                to_addr="c@d.com",
                subject="Test",
                body_text="Hello",
                body_html="",
                date=datetime(2025, 1, 1),
                labels=[],
                history_id=1,
                raw_json="{}",
            ),
        )
        insert_embedding(
            conn,
            EmbeddingRecord(
                id=None,
                message_id=f"m{i:05d}",
                attachment_id=None,
                chunk_type="message",
                chunk_text="test",
                embedding=_rand_embedding(dims, seed=i),
                model=MODEL,
            ),
        )
    conn.close()


def _compare(rebuild_dir, upsert_dir, dims, *, manual_rerank, n_queries=200, top_k=10):
    r = ScannSearcher(rebuild_dir, dimensions=dims)
    u = ScannSearcher(upsert_dir, dimensions=dims)
    qrng = np.random.default_rng(777)
    exact = top1 = 0
    jac_sum = 0.0
    for i in range(n_queries):
        # Query near a cluster center (a realistic "find my neighborhood" query).
        q = _centers(dims)[i % _K] + 0.08 * qrng.standard_normal(dims).astype(np.float32)
        q = (q / np.linalg.norm(q)).astype(np.float32)
        ri, _ = r.search(q, top_k=top_k, manual_rerank=manual_rerank)
        ui, _ = u.search(q, top_k=top_k, manual_rerank=manual_rerank)
        if ri == ui:
            exact += 1
        if ri and ui and ri[0] == ui[0]:
            top1 += 1
        sr, su = set(ri), set(ui)
        jac_sum += len(sr & su) / len(sr | su) if (sr | su) else 1.0
    return {
        "exact_topk_pct": 100 * exact / n_queries,
        "top1_pct": 100 * top1 / n_queries,
        "mean_jaccard": jac_sum / n_queries,
    }


def test_upsert_matches_rebuild_search_results(tmp_path, capsys):
    dims, shard_size = 32, 150
    kw = dict(manual_rerank=True, ah_dim=16, reorder_pool=60)
    db = tmp_path / "t.db"
    init_db(db)

    # Disable GC so both arms can branch from the same base dir. NOTE: we manage
    # these builder patches by hand (save/restore in finally) instead of the
    # `monkeypatch` fixture — the autouse PG-schema fixture pins DB_DSN via that
    # same monkeypatch instance, so a stray monkeypatch.undo() here would revert
    # search_path back to `public` and the build would see an empty schema.
    def _no_gc_promote(conn, live_name, new_dir, *, user_id=None):
        set_active_index_dir(conn, str(new_dir), user_id=user_id)
        conn.commit()

    orig_promote = B._promote_and_gc
    orig_upsert = B._upsert_tail_shard
    orig_build_one = B._build_one_shard
    B._promote_and_gc = _no_gc_promote
    try:
        # Base: 570 vectors -> shard_0..2 sealed (150 each), shard_3 open (120, tree).
        _add(db, 0, 570, dims)
        idx = tmp_path / "scann"
        base = build_index_sharded(db, idx, model=MODEL, dimensions=dims, shard_size=shard_size, **kw)

        _add(db, 570, 20, dims)  # 120 -> 140, fits in 150

        # BEFORE arm: force the old rebuild path by disabling upsert.
        B._upsert_tail_shard = lambda *a, **k: None
        rebuild_dir = build_index_delta(db, idx, model=MODEL, dimensions=dims, shard_size=shard_size, **kw)
        assert rebuild_dir != base

        # Reset the pointer back to base so the AFTER arm branches identically.
        conn = get_connection(db)
        set_active_index_dir(conn, str(base), user_id=None)
        conn.commit()
        conn.close()

        # AFTER arm: real upsert path. Count shard builds — must be zero.
        B._upsert_tail_shard = orig_upsert
        calls = {"n": 0}

        def _counting_build_one(*a, **k):
            calls["n"] += 1
            return orig_build_one(*a, **k)

        B._build_one_shard = _counting_build_one
        upsert_dir = build_index_delta(db, idx, model=MODEL, dimensions=dims, shard_size=shard_size, **kw)
        assert calls["n"] == 0, "AFTER arm must use upsert (no shard rebuilt)"
        assert upsert_dir not in (base, rebuild_dir)
    finally:
        B._promote_and_gc = orig_promote
        B._upsert_tail_shard = orig_upsert
        B._build_one_shard = orig_build_one

    rr = _compare(rebuild_dir, upsert_dir, dims, manual_rerank=True)
    raw = _compare(rebuild_dir, upsert_dir, dims, manual_rerank=False)

    with capsys.disabled():
        print("\n" + "=" * 66)
        print("BEFORE (rebuild tail)  vs  AFTER (upsert tail) — same 590 vectors")
        print("200 clustered queries, top-10")
        print("-" * 66)
        print(
            f"  manual-rerank ON  (production): exact top-10 = {rr['exact_topk_pct']:5.1f}%   "
            f"top-1 = {rr['top1_pct']:5.1f}%   Jaccard = {rr['mean_jaccard']:.4f}"
        )
        print(
            f"  raw ScaNN stage   (no rerank) : exact top-10 = {raw['exact_topk_pct']:5.1f}%   "
            f"top-1 = {raw['top1_pct']:5.1f}%   Jaccard = {raw['mean_jaccard']:.4f}"
        )
        print("=" * 66)

    # Production path (manual rerank): the exact full-precision rerank recovers
    # the identical top-k regardless of how the tail is partitioned, so upsert
    # and rebuild return the SAME results on realistic (clustered) data.
    assert rr["top1_pct"] == 100.0
    assert rr["exact_topk_pct"] >= 99.0
    assert rr["mean_jaccard"] >= 0.999

    # The raw ScaNN candidate stage DOES differ (upsert keeps the frozen
    # partitioner; rebuild retrains it), which is exactly why the funnel reranks.
    # Asserted only as documentation of the mechanism, not a quality gate.
    assert raw["exact_topk_pct"] <= rr["exact_topk_pct"]
