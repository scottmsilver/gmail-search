import json
import logging
import math
import sqlite3
from pathlib import Path

import numpy as np
import scann

from gmail_search.store.db import get_connection

logger = logging.getLogger(__name__)


# Empirical multiplier from the 2026-04-18 OOM investigation:
# ScaNN's full-build peak (partitioner training + AH codebook + serialize)
# was ~10.9 GiB for 237K × 3072 float32. That's ~4× the raw float32 size
# (237K × 3072 × 4 = 2.8 GiB). We use 4× as the per-vector cost model to
# translate a RAM budget into a shard size.
_SCANN_PEAK_MULT = 4


def _load_embeddings_matrix(conn: sqlite3.Connection, model: str, dimensions: int) -> tuple[list[int], np.ndarray]:
    """Stream (id, embedding) rows into a preallocated float32 matrix.

    The previous `[list(struct.unpack(...)) for r in rows]` intermediate
    allocated ~N × dims Python float objects — ~30 GiB for 237K × 3072
    vectors — which caused per-cycle RSS spikes and glibc heap
    fragmentation in the watch loop. See OOM_INCIDENT_2026-04-18.md.

    Stored blobs are little-endian float32 (struct 'f'), which matches
    numpy's native float32 layout on the target platforms, so
    `np.frombuffer` is a zero-copy reinterpretation.
    """
    count = conn.execute("SELECT COUNT(*) FROM embeddings WHERE model = ?", (model,)).fetchone()[0]

    ids: list[int] = []
    vectors = np.empty((count, dimensions), dtype=np.float32)
    cursor = conn.execute("SELECT id, embedding FROM embeddings WHERE model = ? ORDER BY id", (model,))
    for i, row in enumerate(cursor):
        ids.append(row["id"])
        vectors[i] = np.frombuffer(row["embedding"], dtype=np.float32)
    return ids, vectors


def _write_empty_index(index_dir: Path) -> None:
    logger.warning("No embeddings found. Skipping index build.")
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "ids.json").write_text("[]")


def _build_scann_from_vectors(ids: list[int], vectors: np.ndarray, index_dir: Path) -> None:
    """The ScaNN-specific part of index building. Takes vectors as any
    numpy-array-like (regular array or memmap) and produces a serialized
    ScaNN index at index_dir.
    """
    dimensions = vectors.shape[1]
    logger.info(f"Building ScaNN index with {len(ids)} vectors, {dimensions} dims")

    num_leaves = max(int(math.sqrt(len(ids))), 1)
    num_leaves = min(num_leaves, len(ids))

    builder = scann.scann_ops_pybind.builder(vectors, 10, "dot_product")

    if len(ids) >= 100:
        builder = builder.tree(
            num_leaves=num_leaves,
            num_leaves_to_search=min(num_leaves, 100),
            training_sample_size=min(len(ids), 250000),
        )
        builder = builder.score_ah(2, anisotropic_quantization_threshold=0.2)
        builder = builder.reorder(100)
    else:
        builder = builder.score_brute_force()

    searcher = builder.build()

    index_dir.mkdir(parents=True, exist_ok=True)
    searcher.serialize(str(index_dir))
    (index_dir / "ids.json").write_text(json.dumps(ids))

    logger.info(f"Index built and saved to {index_dir}")


def build_index(db_path: Path, index_dir: Path, model: str, dimensions: int) -> None:
    conn = get_connection(db_path)
    try:
        ids, vectors = _load_embeddings_matrix(conn, model, dimensions)
    finally:
        conn.close()

    if not ids:
        _write_empty_index(index_dir)
        return

    _build_scann_from_vectors(ids, vectors, index_dir)


def _stream_embeddings_to_memmap(
    conn: sqlite3.Connection,
    model: str,
    dimensions: int,
    memmap_path: Path,
) -> tuple[list[int], np.memmap]:
    """Stream embedding blobs into a disk-backed np.memmap — peak Python
    RSS during load is O(one row), not O(corpus).

    Returns (ids, memmap). Caller is responsible for deleting memmap_path
    after ScaNN has finished consuming the array.
    """
    count = conn.execute("SELECT COUNT(*) FROM embeddings WHERE model = ?", (model,)).fetchone()[0]
    if count == 0:
        return [], np.empty((0, dimensions), dtype=np.float32)  # type: ignore[return-value]

    memmap_path.parent.mkdir(parents=True, exist_ok=True)
    vectors = np.memmap(memmap_path, dtype=np.float32, mode="w+", shape=(count, dimensions))

    ids: list[int] = []
    cursor = conn.execute("SELECT id, embedding FROM embeddings WHERE model = ? ORDER BY id", (model,))
    for i, row in enumerate(cursor):
        ids.append(row["id"])
        vectors[i] = np.frombuffer(row["embedding"], dtype=np.float32)
    vectors.flush()
    return ids, vectors


def build_index_disk(
    db_path: Path,
    index_dir: Path,
    model: str,
    dimensions: int,
) -> None:
    """Disk-backed builder: streams embeddings into a np.memmap file and
    passes the memmap to ScaNN. Python-side peak stays near-O(1) during
    the load step; ScaNN still reads all vectors during training, but
    the OS page cache manages residency.

    Output is the same on-disk ScaNN index that the RAM builder produces —
    the server's `ScannSearcher` loads it identically.
    """
    index_dir.mkdir(parents=True, exist_ok=True)
    memmap_path = index_dir / "_vectors.f32.tmp"

    conn = get_connection(db_path)
    try:
        ids, vectors = _stream_embeddings_to_memmap(conn, model, dimensions, memmap_path)
    finally:
        conn.close()

    try:
        if not ids:
            _write_empty_index(index_dir)
            return
        _build_scann_from_vectors(ids, vectors, index_dir)
    finally:
        # Release the memmap handle before unlinking — Linux tolerates
        # unlinking a mapped file, but Windows would refuse.
        del vectors
        memmap_path.unlink(missing_ok=True)


def shard_size_from_budget(ram_budget_mb: int, dimensions: int) -> int:
    """Pure function — translate a RAM budget into vectors per shard.

    Peak RSS of a single-shard build ≈ shard_size × dimensions ×
    _SCANN_PEAK_MULT × 4 bytes. Invert to get shard_size for a target
    peak.

    Returns at least 1 so callers never produce a zero-vector shard.
    """
    per_vector_bytes = dimensions * _SCANN_PEAK_MULT * 4  # float32
    size = (ram_budget_mb * 1024 * 1024) // per_vector_bytes
    return max(1, size)


def _clean_old_shard_dirs(index_dir: Path) -> None:
    """Remove shard_N directories from a previous build — stale shards
    left behind after shrinking the shard count would silently corrupt
    the on-disk layout.
    """
    import shutil  # inline — formatter strips unused top-level imports

    if not index_dir.exists():
        return
    for entry in index_dir.iterdir():
        if entry.is_dir() and entry.name.startswith("shard_"):
            shutil.rmtree(entry)


def _count_embeddings(conn: sqlite3.Connection, model: str) -> int:
    return conn.execute("SELECT COUNT(*) FROM embeddings WHERE model = ?", (model,)).fetchone()[0]


def _build_one_shard(
    rows: list[tuple[int, bytes]],
    dimensions: int,
    shard_dir: Path,
) -> list[int]:
    """Materialize one shard of at-most-shard_size rows into a memmap
    and build a ScaNN index at shard_dir. Returns the shard's id list.
    """
    shard_dir.mkdir(parents=True, exist_ok=True)
    memmap_path = shard_dir / "_vectors.f32.tmp"
    shard_count = len(rows)

    vectors = np.memmap(memmap_path, dtype=np.float32, mode="w+", shape=(shard_count, dimensions))
    ids: list[int] = []
    for i, (rid, blob) in enumerate(rows):
        ids.append(rid)
        vectors[i] = np.frombuffer(blob, dtype=np.float32)
    vectors.flush()

    try:
        _build_scann_from_vectors(ids, vectors, shard_dir)
    finally:
        del vectors
        memmap_path.unlink(missing_ok=True)
    return ids


def build_index_sharded(
    db_path: Path,
    index_dir: Path,
    model: str,
    dimensions: int,
    shard_size: int,
) -> None:
    """Build N ScaNN indexes, each over at most shard_size vectors, so
    peak RSS during the build is bounded by shard_size regardless of
    total corpus size.

    On-disk layout:
        index_dir/manifest.json    {num_shards, dimensions, shard_size}
        index_dir/ids.json         concatenated ids (same contract as
                                   the single-index builder)
        index_dir/shard_0/         ScaNN files + shard's ids.json
        index_dir/shard_1/         ...

    ScannSearcher detects manifest.json and queries shards in parallel
    logic, merging by score.
    """
    if shard_size < 1:
        raise ValueError(f"shard_size must be >= 1, got {shard_size}")

    index_dir.mkdir(parents=True, exist_ok=True)
    _clean_old_shard_dirs(index_dir)

    conn = get_connection(db_path)
    try:
        total = _count_embeddings(conn, model)
        if total == 0:
            _write_empty_index(index_dir)
            # Wipe any stale manifest from a prior non-empty build.
            (index_dir / "manifest.json").unlink(missing_ok=True)
            conn.close()
            return

        cursor = conn.execute("SELECT id, embedding FROM embeddings WHERE model = ? ORDER BY id", (model,))

        all_ids: list[int] = []
        shard_idx = 0
        buffer: list[tuple[int, bytes]] = []
        for row in cursor:
            buffer.append((row["id"], row["embedding"]))
            if len(buffer) >= shard_size:
                all_ids.extend(_build_one_shard(buffer, dimensions, index_dir / f"shard_{shard_idx}"))
                buffer = []
                shard_idx += 1
        if buffer:
            all_ids.extend(_build_one_shard(buffer, dimensions, index_dir / f"shard_{shard_idx}"))
            shard_idx += 1
    finally:
        conn.close()

    num_shards = shard_idx
    (index_dir / "manifest.json").write_text(
        json.dumps({"num_shards": num_shards, "dimensions": dimensions, "shard_size": shard_size})
    )
    (index_dir / "ids.json").write_text(json.dumps(all_ids))
    logger.info(f"Sharded index built: {num_shards} shards, {len(all_ids)} total vectors at {index_dir}")


def load_index_metadata(index_dir: Path) -> list[int]:
    ids_file = index_dir / "ids.json"
    if not ids_file.exists():
        return []
    return json.loads(ids_file.read_text())
