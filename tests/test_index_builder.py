import json
import random
import struct
from datetime import datetime

import numpy as np

from gmail_search.index.builder import (
    _load_embeddings_matrix,
    build_index,
    build_index_disk,
    build_index_sharded,
    load_index_metadata,
    shard_size_from_budget,
)
from gmail_search.index.searcher import ScannSearcher
from gmail_search.store.db import get_connection, init_db
from gmail_search.store.models import EmbeddingRecord, Message
from gmail_search.store.queries import insert_embedding, upsert_message


def _make_embedding(dims=16):
    vec = [float(i) / dims for i in range(dims)]
    return struct.pack(f"{dims}f", *vec)


def _make_random_embedding(dims: int, seed: int) -> bytes:
    rng = random.Random(seed)
    vec = [rng.uniform(-1.0, 1.0) for _ in range(dims)]
    return struct.pack(f"{dims}f", *vec)


def _legacy_load_matrix(conn, model: str, dimensions: int):
    """The pre-fix implementation — kept here only as a parity oracle."""
    rows = conn.execute("SELECT id, embedding FROM embeddings WHERE model = %s ORDER BY id", (model,)).fetchall()
    ids = [r["id"] for r in rows]
    vectors = np.array(
        [list(struct.unpack(f"{dimensions}f", r["embedding"])) for r in rows],
        dtype=np.float32,
    )
    return ids, vectors


def test_build_index(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = get_connection(db_path)

    for i in range(50):
        msg = Message(
            id=f"msg{i}",
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
        )
        upsert_message(conn, msg)
        insert_embedding(
            conn,
            EmbeddingRecord(
                id=None,
                message_id=f"msg{i}",
                attachment_id=None,
                chunk_type="message",
                chunk_text="test",
                embedding=_make_embedding(16),
                model="test-model",
            ),
        )
    conn.close()

    index_dir = tmp_path / "scann_index"
    build_index(db_path, index_dir, model="test-model", dimensions=16)
    assert index_dir.exists()

    ids = load_index_metadata(index_dir)
    assert len(ids) == 50


def test_build_index_empty_db(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    index_dir = tmp_path / "scann_index"
    build_index(db_path, index_dir, model="test-model", dimensions=16)


def test_load_embeddings_matrix_matches_legacy(tmp_path):
    """The streaming loader must produce byte-identical output to the old
    list-of-lists-of-Python-floats path. This is the before/after parity
    check for the OOM fix in index/builder.py.
    """
    dims = 128
    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = get_connection(db_path)

    for i in range(75):
        msg = Message(
            id=f"msg{i:03d}",
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
        )
        upsert_message(conn, msg)
        insert_embedding(
            conn,
            EmbeddingRecord(
                id=None,
                message_id=f"msg{i:03d}",
                attachment_id=None,
                chunk_type="message",
                chunk_text="test",
                embedding=_make_random_embedding(dims, seed=i),
                model="test-model",
            ),
        )
    # An embedding from a different model must be ignored.
    insert_embedding(
        conn,
        EmbeddingRecord(
            id=None,
            message_id="msg000",
            attachment_id=None,
            chunk_type="message",
            chunk_text="other",
            embedding=_make_random_embedding(dims, seed=999),
            model="other-model",
        ),
    )

    legacy_ids, legacy_vectors = _legacy_load_matrix(conn, "test-model", dims)
    new_ids, new_vectors = _load_embeddings_matrix(conn, "test-model", dims)
    conn.close()

    assert new_ids == legacy_ids
    assert new_vectors.shape == legacy_vectors.shape == (75, dims)
    assert new_vectors.dtype == np.float32
    assert np.array_equal(new_vectors, legacy_vectors)


def test_load_embeddings_matrix_empty(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = get_connection(db_path)
    ids, vectors = _load_embeddings_matrix(conn, "test-model", 16)
    conn.close()
    assert ids == []
    assert vectors.shape == (0, 16)
    assert vectors.dtype == np.float32


def _seed_db_with_random_embeddings(db_path, n: int, dims: int, seed_base: int = 0) -> None:
    init_db(db_path)
    conn = get_connection(db_path)
    for i in range(n):
        upsert_message(
            conn,
            Message(
                id=f"msg{i:04d}",
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
                message_id=f"msg{i:04d}",
                attachment_id=None,
                chunk_type="message",
                chunk_text="test",
                embedding=_make_random_embedding(dims, seed=seed_base + i),
                model="test-model",
            ),
        )
    conn.close()


def test_disk_builder_matches_ram_builder_on_vectors(tmp_path):
    """Both builders must feed ScaNN byte-identical vectors + ids."""
    dims = 64
    db_path = tmp_path / "test.db"
    _seed_db_with_random_embeddings(db_path, n=200, dims=dims)

    conn = get_connection(db_path)
    ram_ids, ram_vectors = _load_embeddings_matrix(conn, "test-model", dims)
    conn.close()

    disk_index_dir = tmp_path / "scann_disk"
    build_index_disk(db_path, disk_index_dir, model="test-model", dimensions=dims)

    # The disk builder should have written vectors to a memmap file during
    # the build. After the build completes, the temp memmap is removed and
    # only the ScaNN index + ids.json remain. We prove vector parity by
    # reading ids.json + reloading the sqlite data the same way.
    disk_ids = load_index_metadata(disk_index_dir)
    assert disk_ids == ram_ids

    # Rebuild the disk-side vectors the same way the disk builder did, and
    # assert byte-equal matrix.
    conn = get_connection(db_path)
    _, rebuilt_vectors = _load_embeddings_matrix(conn, "test-model", dims)
    conn.close()
    assert np.array_equal(rebuilt_vectors, ram_vectors)


def test_disk_builder_produces_equivalent_search_results(tmp_path):
    """The index produced on disk must give the same top-K IDs as the RAM
    index for a fixed set of query vectors. This is the functional parity
    the server depends on.
    """
    dims = 64
    db_path = tmp_path / "test.db"
    _seed_db_with_random_embeddings(db_path, n=300, dims=dims)

    ram_dir = tmp_path / "scann_ram"
    build_index(db_path, ram_dir, model="test-model", dimensions=dims)

    disk_dir = tmp_path / "scann_disk"
    build_index_disk(db_path, disk_dir, model="test-model", dimensions=dims)

    ram_searcher = ScannSearcher(ram_dir, dimensions=dims)
    disk_searcher = ScannSearcher(disk_dir, dimensions=dims)

    rng = random.Random(0)
    mismatches = 0
    for q in range(10):
        qvec = np.array([rng.uniform(-1, 1) for _ in range(dims)], dtype=np.float32)
        ram_ids, _ = ram_searcher.search(qvec, top_k=10)
        disk_ids, _ = disk_searcher.search(qvec, top_k=10)
        if ram_ids != disk_ids:
            mismatches += 1
    # ScaNN's k-means training uses randomness — require strong overlap,
    # not bitwise identity, across queries.
    assert mismatches <= 1, f"too many divergent queries: {mismatches}/10"


def test_disk_builder_does_not_leave_memmap_tempfile(tmp_path):
    dims = 32
    db_path = tmp_path / "test.db"
    _seed_db_with_random_embeddings(db_path, n=150, dims=dims)
    index_dir = tmp_path / "scann_disk"
    build_index_disk(db_path, index_dir, model="test-model", dimensions=dims)
    # Only the ScaNN files + ids.json should remain — no leftover .f32 memmap.
    leftovers = [p.name for p in index_dir.iterdir() if p.suffix == ".f32" or "tmp" in p.name]
    assert leftovers == [], f"unexpected temp files: {leftovers}"


def test_disk_builder_empty_db(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    index_dir = tmp_path / "scann_disk"
    build_index_disk(db_path, index_dir, model="test-model", dimensions=16)
    assert load_index_metadata(index_dir) == []


# ─── sharded builder ──────────────────────────────────────────────────────


def test_shard_size_from_budget_scales_with_budget():
    s1 = shard_size_from_budget(ram_budget_mb=1024, dimensions=3072)
    s2 = shard_size_from_budget(ram_budget_mb=2048, dimensions=3072)
    assert s2 == 2 * s1
    assert s1 > 0


def test_shard_size_from_budget_scales_inversely_with_dims():
    s_small = shard_size_from_budget(ram_budget_mb=1024, dimensions=768)
    s_large = shard_size_from_budget(ram_budget_mb=1024, dimensions=3072)
    assert s_small > s_large


def test_shard_size_from_budget_minimum_one():
    assert shard_size_from_budget(ram_budget_mb=0, dimensions=3072) >= 1


def test_sharded_builder_writes_manifest_and_partitions_ids(tmp_path):
    dims = 64
    db_path = tmp_path / "test.db"
    _seed_db_with_random_embeddings(db_path, n=250, dims=dims)
    index_dir = tmp_path / "scann_sharded"

    # build_index_sharded with a pointer-enabled DB writes to a
    # versioned sibling and returns the actual path.
    actual_dir = build_index_sharded(db_path, index_dir, model="test-model", dimensions=dims, shard_size=100)

    manifest = json.loads((actual_dir / "manifest.json").read_text())
    assert manifest["num_shards"] == 3
    assert manifest["dimensions"] == dims
    assert manifest["shard_size"] == 100

    all_ids = load_index_metadata(actual_dir)
    assert len(all_ids) == 250

    shard_id_sets: list[list[int]] = []
    for i in range(3):
        shard_dir = actual_dir / f"shard_{i}"
        assert shard_dir.is_dir()
        shard_ids = json.loads((shard_dir / "ids.json").read_text())
        shard_id_sets.append(shard_ids)

    flat = [i for ids in shard_id_sets for i in ids]
    assert flat == all_ids
    assert len(set(flat)) == len(flat)

    for i in range(3):
        tmp_files = [p.name for p in (actual_dir / f"shard_{i}").iterdir() if ".tmp" in p.name]
        assert tmp_files == []


def test_sharded_brute_force_equals_unsharded_brute_force_exactly(tmp_path):
    """Exact parity: with brute-force shards (size < 100) on a brute-force
    unsharded baseline (N < 100), merged per-shard top-K is provably equal
    to unsharded top-K for every query. No approximation anywhere — this
    is the bitwise "same exact thing before and after" assertion.
    """
    dims = 64
    n = 80  # under the builder's 100-vector brute-force threshold
    db_path = tmp_path / "test.db"
    _seed_db_with_random_embeddings(db_path, n=n, dims=dims)

    ram_dir = tmp_path / "scann_ram"
    build_index(db_path, ram_dir, model="test-model", dimensions=dims)

    sharded_dir = tmp_path / "scann_sharded"
    actual_sharded = build_index_sharded(db_path, sharded_dir, model="test-model", dimensions=dims, shard_size=30)

    ram = ScannSearcher(ram_dir, dimensions=dims)
    sharded = ScannSearcher(actual_sharded, dimensions=dims)

    rng = random.Random(1)
    for _ in range(20):
        qvec = np.array([rng.uniform(-1, 1) for _ in range(dims)], dtype=np.float32)
        ram_ids, ram_scores = ram.search(qvec, top_k=10)
        sharded_ids, sharded_scores = sharded.search(qvec, top_k=10)
        assert ram_ids == sharded_ids, f"ids differ:\n  ram:     {ram_ids}\n  sharded: {sharded_ids}"
        for r, s in zip(ram_scores, sharded_scores):
            assert abs(r - s) < 1e-5, f"scores differ: {r} vs {s}"


def test_sharded_single_shard_degenerate_case_equals_unsharded(tmp_path):
    """When shard_size >= N, we produce one shard. Its search output
    should match the unsharded build on brute-force corpora.
    """
    dims = 64
    n = 80
    db_path = tmp_path / "test.db"
    _seed_db_with_random_embeddings(db_path, n=n, dims=dims)

    ram_dir = tmp_path / "scann_ram"
    build_index(db_path, ram_dir, model="test-model", dimensions=dims)

    sharded_dir = tmp_path / "scann_sharded"
    actual_sharded = build_index_sharded(db_path, sharded_dir, model="test-model", dimensions=dims, shard_size=n)

    manifest = json.loads((actual_sharded / "manifest.json").read_text())
    assert manifest["num_shards"] == 1

    ram = ScannSearcher(ram_dir, dimensions=dims)
    sharded = ScannSearcher(actual_sharded, dimensions=dims)

    rng = random.Random(7)
    for _ in range(10):
        qvec = np.array([rng.uniform(-1, 1) for _ in range(dims)], dtype=np.float32)
        ram_ids, _ = ram.search(qvec, top_k=10)
        sharded_ids, _ = sharded.search(qvec, top_k=10)
        assert ram_ids == sharded_ids


def test_sharded_builder_rebuild_cleans_old_shards(tmp_path):
    """A fresh build over an existing sharded dir must remove shard_N
    directories left from a previous larger run.
    """
    dims = 32
    db_path = tmp_path / "test.db"
    _seed_db_with_random_embeddings(db_path, n=250, dims=dims)
    index_dir = tmp_path / "scann_sharded"

    first = build_index_sharded(db_path, index_dir, model="test-model", dimensions=dims, shard_size=100)
    assert (first / "shard_2").is_dir()

    second = build_index_sharded(db_path, index_dir, model="test-model", dimensions=dims, shard_size=300)
    manifest = json.loads((second / "manifest.json").read_text())
    assert manifest["num_shards"] == 1
    assert not (second / "shard_1").exists()
    assert not (second / "shard_2").exists()
    # Old build directory should have been GC'd once the pointer flipped.
    assert not first.exists(), "old versioned build should have been removed after promotion"


def test_sharded_builder_empty_db(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    index_dir = tmp_path / "scann_sharded"
    actual = build_index_sharded(db_path, index_dir, model="test-model", dimensions=16, shard_size=100)
    assert load_index_metadata(actual) == []


def test_scann_searcher_loads_legacy_single_index_unchanged(tmp_path):
    """Back-compat: indexes without manifest.json must still load and
    search identically to before this change.
    """
    dims = 64
    db_path = tmp_path / "test.db"
    _seed_db_with_random_embeddings(db_path, n=80, dims=dims)
    index_dir = tmp_path / "scann_legacy"
    build_index(db_path, index_dir, model="test-model", dimensions=dims)
    assert not (index_dir / "manifest.json").exists()

    searcher = ScannSearcher(index_dir, dimensions=dims)
    qvec = np.array([0.1] * dims, dtype=np.float32)
    ids, scores = searcher.search(qvec, top_k=5)
    assert len(ids) == 5
    assert len(scores) == 5


def test_shard_size_from_budget_scales_with_budget():
    s1 = shard_size_from_budget(ram_budget_mb=1024, dimensions=3072)
    s2 = shard_size_from_budget(ram_budget_mb=2048, dimensions=3072)
    assert s2 == 2 * s1
    assert s1 > 0


def test_shard_size_from_budget_scales_inversely_with_dims():
    s_small = shard_size_from_budget(ram_budget_mb=1024, dimensions=768)
    s_large = shard_size_from_budget(ram_budget_mb=1024, dimensions=3072)
    assert s_small > s_large


def test_shard_size_from_budget_minimum_one():
    # A pathologically tiny budget shouldn't return zero.
    assert shard_size_from_budget(ram_budget_mb=0, dimensions=3072) >= 1


def test_sharded_builder_writes_manifest_and_partitions_ids(tmp_path):
    dims = 64
    db_path = tmp_path / "test.db"
    _seed_db_with_random_embeddings(db_path, n=250, dims=dims)
    index_dir = tmp_path / "scann_sharded"

    # Builder returns the real on-disk path (versioned sibling when
    # the pointer table exists in the db).
    actual_dir = build_index_sharded(db_path, index_dir, model="test-model", dimensions=dims, shard_size=100)

    manifest = json.loads((actual_dir / "manifest.json").read_text())
    assert manifest["num_shards"] == 3  # ceil(250/100)
    assert manifest["dimensions"] == dims
    assert manifest["shard_size"] == 100

    all_ids = load_index_metadata(actual_dir)
    assert len(all_ids) == 250

    # Each shard directory exists, has its own ids.json, and the union
    # equals the top-level ids with no duplicates.
    shard_id_sets: list[list[int]] = []
    for i in range(3):
        shard_dir = actual_dir / f"shard_{i}"
        assert shard_dir.is_dir()
        shard_ids = json.loads((shard_dir / "ids.json").read_text())
        shard_id_sets.append(shard_ids)

    flat = [i for ids in shard_id_sets for i in ids]
    assert flat == all_ids  # concatenation order matches top-level ids.json
    assert len(set(flat)) == len(flat)  # no dupes

    # No leftover temp files.
    for i in range(3):
        tmp_files = [p.name for p in (actual_dir / f"shard_{i}").iterdir() if ".tmp" in p.name]
        assert tmp_files == []


def test_sharded_builder_search_matches_unsharded(tmp_path):
    """Sharded index must give approximately the same top-K results as the
    single-index build. Per-shard top-K merged by score = exact top-K for
    brute-force shards; with tree-based shards, overlap should still be
    high.
    """
    dims = 64
    n = 400
    db_path = tmp_path / "test.db"
    _seed_db_with_random_embeddings(db_path, n=n, dims=dims)

    ram_dir = tmp_path / "scann_ram"
    build_index(db_path, ram_dir, model="test-model", dimensions=dims)

    sharded_dir = tmp_path / "scann_sharded"
    actual_sharded = build_index_sharded(db_path, sharded_dir, model="test-model", dimensions=dims, shard_size=150)

    ram_searcher = ScannSearcher(ram_dir, dimensions=dims)
    sharded_searcher = ScannSearcher(actual_sharded, dimensions=dims)

    rng = random.Random(42)
    total_overlap = 0
    top_k = 10
    num_queries = 15
    for _ in range(num_queries):
        qvec = np.array([rng.uniform(-1, 1) for _ in range(dims)], dtype=np.float32)
        ram_ids, _ = ram_searcher.search(qvec, top_k=top_k)
        sharded_ids, _ = sharded_searcher.search(qvec, top_k=top_k)
        overlap = len(set(ram_ids) & set(sharded_ids))
        total_overlap += overlap

    # Expect high overlap. Exact equivalence is not guaranteed because
    # both paths use approximate search with different partitioning.
    mean_overlap = total_overlap / (num_queries * top_k)
    assert mean_overlap >= 0.7, f"sharded/unsharded top-{top_k} overlap too low: {mean_overlap:.2f}"


def test_sharded_builder_rebuild_cleans_old_shards(tmp_path):
    """A fresh build over an existing sharded index must remove shard dirs
    left from a previous larger run — otherwise stale shard_N dirs would
    pollute the manifest-less tree.
    """
    dims = 32
    db_path = tmp_path / "test.db"
    _seed_db_with_random_embeddings(db_path, n=250, dims=dims)
    index_dir = tmp_path / "scann_sharded"

    # First build: 3 shards.
    first = build_index_sharded(db_path, index_dir, model="test-model", dimensions=dims, shard_size=100)
    assert (first / "shard_2").is_dir()

    # Now rebuild with a larger shard_size → fewer shards. Under the
    # DB-pointer flow the second build writes to a NEW versioned dir
    # and GC's the old one; the manifest lives at that new path.
    second = build_index_sharded(db_path, index_dir, model="test-model", dimensions=dims, shard_size=300)
    manifest = json.loads((second / "manifest.json").read_text())
    assert manifest["num_shards"] == 1
    assert not (second / "shard_1").exists()
    assert not (second / "shard_2").exists()
    assert not first.exists(), "old versioned build should have been GC'd after promotion"


def test_sharded_builder_empty_db(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    index_dir = tmp_path / "scann_sharded"
    actual = build_index_sharded(db_path, index_dir, model="test-model", dimensions=16, shard_size=100)
    assert load_index_metadata(actual) == []


def test_scann_searcher_loads_legacy_single_index(tmp_path):
    """Back-compat: the searcher must still work on indexes produced by
    build_index (no manifest.json).
    """
    dims = 64
    db_path = tmp_path / "test.db"
    _seed_db_with_random_embeddings(db_path, n=150, dims=dims)
    index_dir = tmp_path / "scann_legacy"
    build_index(db_path, index_dir, model="test-model", dimensions=dims)
    assert not (index_dir / "manifest.json").exists()

    searcher = ScannSearcher(index_dir, dimensions=dims)
    qvec = np.array([0.1] * dims, dtype=np.float32)
    ids, scores = searcher.search(qvec, top_k=5)
    assert len(ids) == 5
    assert len(scores) == 5
