"""Native ScaNN upsert delta path (build_index_delta fast path).

The old delta rebuilt the whole open tail shard (up to shard_size vectors)
from PG every cycle, retraining k-means each time. The upsert path extends
the tail via ScaNN's `upsert` — tokenize new vectors against the FROZEN
partitioner + AH codebook, no retrain. These tests prove:
  - the fast path is actually taken (no shard is (re)built), and search finds
    both upserted and pre-existing ids;
  - the manual-rerank corpus grows in lockstep so the searcher's id→corpus map
    stays valid;
  - overflow / legacy (non-docid) tails fall back to the always-correct rebuild.

Sizing: the fast path needs a FULL sealed shard plus a single unsealed
docid-mode tail with >= _UPSERT_MIN_TAIL (100) vectors (so it carries a trained
tree). shard_size=150 + 270 seeded vectors gives shard_0=150 (sealed),
shard_1=120 (open, tree).
"""

import json
import random
import struct
from datetime import datetime

import numpy as np
from gmail_search.index import builder as builder_mod
from gmail_search.index.builder import build_index_delta, build_index_sharded, load_index_metadata
from gmail_search.index.searcher import ScannSearcher
from gmail_search.store.db import get_connection, init_db
from gmail_search.store.models import EmbeddingRecord, Message
from gmail_search.store.queries import insert_embedding, upsert_message

MODEL = "test-model"


def _rand_embedding(dims: int, seed: int) -> bytes:
    rng = random.Random(seed)
    return struct.pack(f"{dims}f", *[rng.uniform(-1.0, 1.0) for _ in range(dims)])


def _add(db_path, start: int, n: int, dims: int) -> None:
    conn = get_connection(db_path)
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


def _vectors_by_id(db_path, dims: int) -> dict[int, np.ndarray]:
    """Map embedding id -> its (normalized) vector, for self-query recall checks."""
    conn = get_connection(db_path)
    rows = conn.execute("SELECT id, embedding FROM embeddings WHERE model=%s ORDER BY id", (MODEL,)).fetchall()
    conn.close()
    out = {}
    for r in rows:
        v = np.frombuffer(r["embedding"], dtype=np.float32).astype(np.float32)
        n = np.linalg.norm(v)
        out[r["id"]] = v / n if n else v
    return out


def _self_recall(searcher: ScannSearcher, vecs: dict[int, np.ndarray], ids: list[int], top_k: int = 20) -> float:
    hits = 0
    for eid in ids:
        found, _ = searcher.search(vecs[eid], top_k=top_k)
        if eid in found:
            hits += 1
    return hits / len(ids)


def _spy_build_one_shard(monkeypatch):
    """Count _build_one_shard calls — zero means the upsert fast path was taken
    (it hardlinks sealed shards and upserts the tail; it never builds a shard)."""
    calls = {"n": 0}
    orig = builder_mod._build_one_shard

    def wrapped(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(builder_mod, "_build_one_shard", wrapped)
    return calls


def test_delta_upsert_extends_tail_without_rebuild(tmp_path, monkeypatch):
    dims, shard_size = 16, 150
    db = tmp_path / "t.db"
    init_db(db)
    _add(db, 0, 270, dims)  # shard_0=150 sealed, shard_1=120 open (tree)
    idx = tmp_path / "scann"
    build_index_sharded(db, idx, model=MODEL, dimensions=dims, shard_size=shard_size)

    _add(db, 270, 20, dims)  # 120 -> 140, fits in 150
    calls = _spy_build_one_shard(monkeypatch)
    delta = build_index_delta(db, idx, model=MODEL, dimensions=dims, shard_size=shard_size)

    # Fast path: nothing (re)built — sealed shard hardlinked, tail upserted.
    assert calls["n"] == 0, "expected upsert (no shard build), but _build_one_shard ran"

    manifest = json.loads((delta / "manifest.json").read_text())
    assert manifest["num_shards"] == 2
    tail = manifest["shards"][1]
    assert tail["count"] == 140 and tail["sealed"] is False and tail["docids"] is True

    # All 290 ids present, and both new + old vectors are self-findable.
    assert sorted(load_index_metadata(delta)) == list(range(1, 291))
    s = ScannSearcher(delta, dimensions=dims)
    vecs = _vectors_by_id(db, dims)
    new_ids = list(range(271, 291))
    old_sample = list(range(1, 271, 15))
    assert _self_recall(s, vecs, new_ids) >= 0.9
    assert _self_recall(s, vecs, old_sample) >= 0.9


def test_delta_upsert_manual_rerank_grows_corpus(tmp_path, monkeypatch):
    dims, shard_size, ah = 16, 150, 8
    db = tmp_path / "t.db"
    init_db(db)
    _add(db, 0, 270, dims)
    idx = tmp_path / "scann"
    kw = dict(manual_rerank=True, ah_dim=ah, reorder_pool=100)
    build_index_sharded(db, idx, model=MODEL, dimensions=dims, shard_size=shard_size, **kw)

    _add(db, 270, 20, dims)
    calls = _spy_build_one_shard(monkeypatch)
    delta = build_index_delta(db, idx, model=MODEL, dimensions=dims, shard_size=shard_size, **kw)
    assert calls["n"] == 0, "expected upsert on manual-rerank tail, but a shard was built"

    # Tail corpus grew by exactly the 20 new full-precision rows.
    tail_corpus = delta / "shard_1" / "corpus_full.f32"
    assert tail_corpus.stat().st_size == 140 * dims * 4

    assert sorted(load_index_metadata(delta)) == list(range(1, 291))
    s = ScannSearcher(delta, dimensions=dims)
    vecs = _vectors_by_id(db, dims)
    # Rerank path (manual_rerank enabled) still resolves upserted ids.
    assert _self_recall(s, vecs, list(range(271, 291))) >= 0.9
    assert _self_recall(s, vecs, list(range(1, 271, 15))) >= 0.9


def test_delta_upsert_overflow_falls_back_to_rebuild(tmp_path, monkeypatch):
    dims, shard_size = 16, 150
    db = tmp_path / "t.db"
    init_db(db)
    _add(db, 0, 270, dims)  # tail shard_1 = 120
    idx = tmp_path / "scann"
    build_index_sharded(db, idx, model=MODEL, dimensions=dims, shard_size=shard_size)

    _add(db, 270, 40, dims)  # 120 + 40 = 160 > 150 -> can't upsert in place
    calls = _spy_build_one_shard(monkeypatch)
    delta = build_index_delta(db, idx, model=MODEL, dimensions=dims, shard_size=shard_size)

    # Overflow -> rebuild path seals the tail (150) and spills the rest (10).
    assert calls["n"] >= 1, "overflow should trigger the rebuild loop"
    manifest = json.loads((delta / "manifest.json").read_text())
    counts = [s["count"] for s in manifest["shards"]]
    assert counts == [150, 150, 10]
    assert sorted(load_index_metadata(delta)) == list(range(1, 311))


def test_delta_legacy_tail_rebuilds_then_upserts(tmp_path, monkeypatch):
    """A tail from before docid-mode (no `docids` flag) can't be upserted: the
    first delta REBUILDS it (which stamps docids), and the next delta upserts."""
    dims, shard_size = 16, 150
    db = tmp_path / "t.db"
    init_db(db)
    _add(db, 0, 270, dims)
    idx = tmp_path / "scann"
    live = build_index_sharded(db, idx, model=MODEL, dimensions=dims, shard_size=shard_size)

    # Simulate a legacy tail: strip `docids` from the open shard's manifest meta.
    mpath = live / "manifest.json"
    manifest = json.loads(mpath.read_text())
    manifest["shards"][1].pop("docids", None)
    mpath.write_text(json.dumps(manifest))

    _add(db, 270, 20, dims)
    calls = _spy_build_one_shard(monkeypatch)
    delta1 = build_index_delta(db, idx, model=MODEL, dimensions=dims, shard_size=shard_size)
    assert calls["n"] >= 1, "legacy (non-docid) tail must rebuild, not upsert"
    # Rebuilt tail is now docid-mode.
    m1 = json.loads((delta1 / "manifest.json").read_text())
    assert m1["shards"][1]["docids"] is True

    _add(db, 290, 5, dims)
    calls2 = _spy_build_one_shard(monkeypatch)
    delta2 = build_index_delta(db, idx, model=MODEL, dimensions=dims, shard_size=shard_size)
    assert calls2["n"] == 0, "second delta should upsert the now-docid tail"
    assert sorted(load_index_metadata(delta2)) == list(range(1, 296))
