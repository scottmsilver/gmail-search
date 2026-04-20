import json
import logging
import math
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


def _load_embeddings_matrix(conn, model: str, dimensions: int) -> tuple[list[int], np.ndarray]:
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
    conn,
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


def _count_embeddings(conn, model: str) -> int:
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
) -> Path:
    """Build N ScaNN indexes into a fresh timestamped sibling of
    `index_dir` and record it in a DB pointer row so readers always
    resolve to a fully-written build.

    Why a DB pointer instead of filesystem rename: filesystem rename
    atomicity varies (FUSE, cross-filesystem, WSL1) and the writer
    still leaves torn state if a reader opens mid-build. A pointer
    row in the same SQLite DB the reader already talks to flips in
    one UPDATE — all readers see either the old path or the new one.

    Sequence:
        1. Write all shards + manifest + ids.json into
           `{index_dir.parent}/{index_dir.name}__<utc>_<rand>/`.
        2. `set_active_index_dir(conn, new_path)` in one transaction.
        3. Remove any sibling `{index_dir.name}__*` directories that
           aren't the one we just set.

    Returns the path that the builder actually wrote to. Callers that
    want to open the newly-built index can either use the return
    value directly or call `resolve_active_index_dir(db_path, index_dir)`
    to get it via the pointer.

    Back-compat: the pointer row is optional. If the pointer table
    isn't present in the DB (test fixtures, old installs), the build
    falls back to writing in-place at `index_dir` so existing callers
    keep working unchanged.
    """
    if shard_size < 1:
        raise ValueError(f"shard_size must be >= 1, got {shard_size}")

    use_pointer = _pointer_table_exists(db_path)
    target_dir = _pick_versioned_dir(index_dir) if use_pointer else index_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    if not use_pointer:
        # Legacy in-place path for tests / un-migrated installs.
        # The manifest-first unlink + tolerant searcher combo we
        # already have prevents the torn-read 500 in this mode.
        (target_dir / "manifest.json").unlink(missing_ok=True)
        _clean_old_shard_dirs(target_dir)

    conn = get_connection(db_path)
    try:
        total = _count_embeddings(conn, model)
        if total == 0:
            _write_empty_index(target_dir)
            if use_pointer:
                _promote_and_gc(conn, index_dir, target_dir)
            conn.close()
            return target_dir

        cursor = conn.execute("SELECT id, embedding FROM embeddings WHERE model = ? ORDER BY id", (model,))
        all_ids: list[int] = []
        shard_idx = 0
        buffer: list[tuple[int, bytes]] = []
        for row in cursor:
            buffer.append((row["id"], row["embedding"]))
            if len(buffer) >= shard_size:
                all_ids.extend(_build_one_shard(buffer, dimensions, target_dir / f"shard_{shard_idx}"))
                buffer = []
                shard_idx += 1
        if buffer:
            all_ids.extend(_build_one_shard(buffer, dimensions, target_dir / f"shard_{shard_idx}"))
            shard_idx += 1

        num_shards = shard_idx
        (target_dir / "manifest.json").write_text(
            json.dumps({"num_shards": num_shards, "dimensions": dimensions, "shard_size": shard_size})
        )
        (target_dir / "ids.json").write_text(json.dumps(all_ids))

        if use_pointer:
            _promote_and_gc(conn, index_dir, target_dir)
    finally:
        conn.close()

    logger.info(f"Sharded index built at {target_dir} ({shard_idx} shards)")
    return target_dir


def _pick_versioned_dir(index_dir: Path) -> Path:
    """Build a unique sibling path: `{index_dir}__<utc>_<short-hash>`.
    The short hash avoids collisions when two builds land in the same
    second (extract + reindex overlap, etc.).
    """
    import secrets
    import time

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    short = secrets.token_hex(3)
    return index_dir.parent / f"{index_dir.name}__{stamp}_{short}"


def _pointer_table_exists(db_path: Path) -> bool:
    """Check for the scann_index_pointer table without importing the
    queries module (which would create a cycle through store/db.py).

    Backend-agnostic: instead of querying `sqlite_master` (SQLite-only)
    or `information_schema.tables` (differs by backend), we just
    attempt the lookup and catch the "no such table" failure. SQLite
    raises `OperationalError`, psycopg raises `UndefinedTable` — both
    derive from `Exception`. False means "legacy DB, use in-place
    write"; True means "pointer table present, use versioned writes".
    """
    try:
        conn = get_connection(db_path)
    except Exception:
        return False
    try:
        try:
            conn.execute("SELECT 1 FROM scann_index_pointer LIMIT 1").fetchone()
            return True
        except Exception:
            return False
    finally:
        conn.close()


def _promote_and_gc(conn, live_name: Path, new_dir: Path) -> None:
    """Update the DB pointer to the new dir and remove every sibling
    `{live_name.name}__*` directory other than the one we just wrote.
    `live_name` itself (if it exists as a legacy path) is left alone
    so a reader that still opens it directly keeps working until it
    learns about the pointer.
    """
    import shutil

    from gmail_search.store.queries import set_active_index_dir

    set_active_index_dir(conn, str(new_dir))
    conn.commit()

    prefix = f"{live_name.name}__"
    for sibling in live_name.parent.iterdir():
        if not sibling.is_dir() or not sibling.name.startswith(prefix):
            continue
        if sibling == new_dir:
            continue
        try:
            shutil.rmtree(sibling)
        except OSError as e:
            logger.warning("failed to gc old index dir %s: %s", sibling, e)


def load_index_metadata(index_dir: Path) -> list[int]:
    ids_file = index_dir / "ids.json"
    if not ids_file.exists():
        return []
    return json.loads(ids_file.read_text())
