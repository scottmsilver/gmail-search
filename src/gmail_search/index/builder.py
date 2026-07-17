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

# Delta indexing: how much stale (deleted-but-still-indexed) content to
# tolerate before a delta cycle compacts via a full rebuild. Stale vectors are
# harmless in search (filtered by the message JOIN); a full rebuild is costly
# (reloads the whole index), so we only pay it once staleness is material.
_DELTA_COMPACTION_STALE_FRACTION = 0.02  # 2% of the index

# Manual-rerank corpus layout: each shard keeps its full-precision vectors in
# `shard_N/corpus_full.f32` (rows in the shard's ids.json order). Per-shard
# files — rather than one index-level corpus_full.memmap — let delta builds
# hardlink a sealed shard's corpus along with its ScaNN artifacts, so the
# funnel index stays compatible with the cheap 15-min delta reindex path.
CORPUS_SHARD_FILENAME = "corpus_full.f32"


def _load_embeddings_matrix(conn, model: str, dimensions: int) -> tuple[list[int], np.ndarray]:
    """Stream (id, embedding) rows into a preallocated float32 matrix.

    The previous `[list(struct.unpack(...)) for r in rows]` intermediate
    allocated ~N × dims Python float objects — ~30 GiB for 237K × 3072
    vectors — which caused per-cycle RSS spikes and glibc heap
    fragmentation in the watch loop. See docs/notes/OOM_INCIDENT_2026-04-18.md.

    Stored blobs are little-endian float32 (struct 'f'), which matches
    numpy's native float32 layout on the target platforms, so
    `np.frombuffer` is a zero-copy reinterpretation.
    """
    count = conn.execute("SELECT COUNT(*) FROM embeddings WHERE model = %s", (model,)).fetchone()[0]

    ids: list[int] = []
    vectors = np.empty((count, dimensions), dtype=np.float32)
    cursor = conn.execute("SELECT id, embedding FROM embeddings WHERE model = %s ORDER BY id", (model,))
    for i, row in enumerate(cursor):
        ids.append(row["id"])
        vectors[i] = np.frombuffer(row["embedding"], dtype=np.float32)
    return ids, vectors


def _write_empty_index(index_dir: Path) -> None:
    logger.warning("No embeddings found. Skipping index build.")
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "ids.json").write_text("[]")


def _scann_training_threads() -> int:
    """Cap ScaNN's k-means training parallelism so a rebuild can't pin every
    core. ScaNN defaults to one training thread per core (20 on this box);
    with two per-user reindex daemons rebuilding, that's up to 40 threads
    fighting 20 cores — load spiked to ~6x and starved Postgres's
    connection-accept path, so `psycopg.connect()` hit `connection timeout
    expired` and the reindex daemon crash-looped. Default to ~1/3 of the cores
    (min 1) so a rebuild always leaves headroom; override with
    GMS_SCANN_TRAINING_THREADS.
    """
    import os

    override = os.environ.get("GMS_SCANN_TRAINING_THREADS", "").strip()
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    return max(1, (os.cpu_count() or 4) // 3)


def _truncate_normalize(vectors: np.ndarray, ah_dim: int) -> np.ndarray:
    """MRL truncation: slice to ah_dim then re-normalize so cosine
    similarity (computed via dot_product on unit-normalized vectors)
    stays well-defined for the truncated subspace. Shared by the full
    build and the tail-upsert delta path so both feed ScaNN identically."""
    vectors = vectors[:, :ah_dim].astype(np.float32, copy=True)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _build_scann_from_vectors(
    ids: list[int],
    vectors: np.ndarray,
    index_dir: Path,
    *,
    ah_dim: int | None = None,
    reorder_pool: int = 100,
    skip_reorder: bool = False,
    with_docids: bool = False,
) -> None:
    """The ScaNN-specific part of index building. Takes vectors as any
    numpy-array-like (regular array or memmap) and produces a serialized
    ScaNN index at index_dir.

    `ah_dim`: when set < vectors.shape[1], slice + L2-normalize each
    input vector to that dim before passing to ScaNN. Used by the
    manual-rerank index format (see build_index_sharded). The ScaNN
    index ends up at `ah_dim` (smaller, faster), and the caller is
    responsible for keeping the FULL-precision vectors elsewhere
    (per-shard corpus files) for the per-query rerank step.

    `reorder_pool`: passed to ScaNN's `.reorder(N)` — number of AH
    candidates to rerank with full-precision (within the SAME dim
    that the index was built on). Default 100 matches legacy.

    `skip_reorder`: omit ScaNN's reorder stage entirely. Used by
    manual-rerank builds: the reorder stage only improves candidate
    ORDER, and the funnel's exact full-dim rerank re-orders the pool
    anyway — only pool MEMBERSHIP matters, and that's decided by the
    AH scan. The 2026-07-07 ablation over all 1,058 cached queries
    measured identical oracle-top-10 pool coverage with and without
    (98.82% vs 98.86%), while dropping the stage removes dataset.npy
    (~2 GB RAM at 1.35M x 384d) and cuts AH-stage latency ~35%.
    """
    if ah_dim is not None and ah_dim < vectors.shape[1]:
        vectors = _truncate_normalize(vectors, ah_dim)

    dimensions = vectors.shape[1]
    logger.info(f"Building ScaNN index with {len(ids)} vectors, {dimensions} dims, " f"reorder_pool={reorder_pool}")

    num_leaves = max(int(math.sqrt(len(ids))), 1)
    num_leaves = min(num_leaves, len(ids))

    builder = scann.scann_ops_pybind.builder(vectors, 10, "dot_product")
    builder = builder.set_n_training_threads(_scann_training_threads())

    if len(ids) >= 100:
        builder = builder.tree(
            num_leaves=num_leaves,
            num_leaves_to_search=min(num_leaves, 100),
            training_sample_size=min(len(ids), 250000),
        )
        builder = builder.score_ah(2, anisotropic_quantization_threshold=0.2)
        if not skip_reorder:
            builder = builder.reorder(reorder_pool)
    else:
        builder = builder.score_brute_force()

    # `with_docids`: register each vector under docid str(embedding_id).
    # Docids are what make a shard MUTABLE — ScaNN's `upsert` (the delta
    # path's cheap insert, no k-means retrain) requires them. The searcher
    # detects docid-mode shards at query time (string neighbors) and maps
    # int(docid) directly instead of positional ids.json lookup.
    searcher = builder.build(docids=[str(i) for i in ids]) if with_docids else builder.build()

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
    count = conn.execute("SELECT COUNT(*) FROM embeddings WHERE model = %s", (model,)).fetchone()[0]
    if count == 0:
        return [], np.empty((0, dimensions), dtype=np.float32)  # type: ignore[return-value]

    memmap_path.parent.mkdir(parents=True, exist_ok=True)
    vectors = np.memmap(memmap_path, dtype=np.float32, mode="w+", shape=(count, dimensions))

    ids: list[int] = []
    cursor = conn.execute("SELECT id, embedding FROM embeddings WHERE model = %s ORDER BY id", (model,))
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


def _count_embeddings(conn, model: str, *, user_id: str | None = None) -> int:
    if user_id is None:
        return conn.execute("SELECT COUNT(*) FROM embeddings WHERE model = %s", (model,)).fetchone()[0]
    return conn.execute(
        "SELECT COUNT(*) FROM embeddings WHERE model = %s AND user_id = %s", (model, user_id)
    ).fetchone()[0]


def _build_one_shard(
    rows: list[tuple[int, bytes]],
    dimensions: int,
    shard_dir: Path,
    *,
    ah_dim: int | None = None,
    reorder_pool: int = 100,
    write_full_corpus: bool = False,
    with_docids: bool = False,
) -> list[int]:
    """Materialize one shard of at-most-shard_size rows into a memmap
    and build a ScaNN index at shard_dir. Returns the shard's id list.

    `ah_dim` / `reorder_pool` flow through to `_build_scann_from_vectors`
    for the manual-rerank build mode (truncate input + larger reorder pool).

    `write_full_corpus`: keep this shard's full-precision vectors as
    `shard_dir/corpus_full.f32` for the manual-rerank searcher. The temp
    memmap we build from already holds exactly those vectors (truncation
    happens later, inside `_build_scann_from_vectors`), so this is a rename
    instead of a rewrite.
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
        # write_full_corpus == manual-rerank mode: the exact external rerank
        # makes ScaNN's internal reorder stage redundant (see skip_reorder).
        _build_scann_from_vectors(
            ids,
            vectors,
            shard_dir,
            ah_dim=ah_dim,
            reorder_pool=reorder_pool,
            skip_reorder=write_full_corpus,
            with_docids=with_docids,
        )
    except BaseException:
        # Failed shard: never leave a complete-looking corpus file next to
        # missing/partial ScaNN artifacts (and never mask the build error
        # with a rename failure). Just drop the temp memmap and re-raise.
        del vectors
        memmap_path.unlink(missing_ok=True)
        raise
    del vectors
    if write_full_corpus:
        memmap_path.rename(shard_dir / CORPUS_SHARD_FILENAME)
    else:
        memmap_path.unlink(missing_ok=True)
    return ids


def build_index_sharded(
    db_path: Path,
    index_dir: Path,
    model: str,
    dimensions: int,
    shard_size: int,
    *,
    truncate_dim: int | None = None,
    manual_rerank: bool = False,
    ah_dim: int = 1536,
    reorder_pool: int = 100,
    user_id: str | None = None,
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
    if manual_rerank and ah_dim >= dimensions:
        # Manual rerank with ah_dim >= dimensions is a no-op compared
        # to legacy. Disable rather than build a confusing artifact.
        logger.info(
            f"manual_rerank requested but ah_dim={ah_dim} >= dimensions={dimensions}; " "falling back to legacy build"
        )
        manual_rerank = False

    # Effective AH-stage dim. Three modes:
    #   - manual_rerank=True:   ah_dim < dimensions, plus corpus_full
    #     mmap and a per-query rerank step in the searcher (variant C).
    #   - truncate_dim < dimensions, manual_rerank=False: V3-style —
    #     ScaNN built on a truncated subspace, queries also truncated
    #     at search time, but no full-precision rerank step. Cheapest
    #     compaction: ~75% smaller than baseline at ~1.8% NDCG drop.
    #   - everything else (defaults): legacy full-dim baseline.
    if manual_rerank and truncate_dim and truncate_dim < dimensions and truncate_dim != ah_dim:
        # Both knobs set with different values is almost always a config
        # mistake — manual_rerank uses `ah_dim`, not `truncate_dim`, so
        # the latter is silently ignored. Surface it instead.
        logger.warning(
            "manual_rerank=True with truncate_dim=%d (≠ ah_dim=%d) — "
            "truncate_dim is ignored when manual_rerank is on; using ah_dim",
            truncate_dim,
            ah_dim,
        )
    effective_ah_dim = (
        ah_dim if manual_rerank else (truncate_dim if truncate_dim and truncate_dim < dimensions else None)
    )

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
        # Per-user scoping: when `user_id` is given, count + iterate
        # only that user's embeddings. Daemon callers pass the
        # bootstrap user via the pipeline.reindex resolution.
        total = _count_embeddings(conn, model, user_id=user_id)
        if total == 0:
            _write_empty_index(target_dir)
            if use_pointer:
                _promote_and_gc(conn, index_dir, target_dir, user_id=user_id)
            conn.close()
            return target_dir

        all_ids: list[int] = []
        shards_meta: list[dict] = []  # v2 manifest: per-shard id-range keys for delta reuse
        shard_idx = 0
        # Truncation flows through to ScaNN regardless of manual_rerank;
        # only the per-shard full-precision corpus files (which the
        # searcher reranks against at query time) are gated on it.
        shard_kwargs: dict[str, object] = {"reorder_pool": reorder_pool}
        if effective_ah_dim is not None:
            shard_kwargs["ah_dim"] = effective_ah_dim
        shard_kwargs["write_full_corpus"] = manual_rerank
        # docid-mode shards are MUTABLE: the delta path upserts new vectors
        # into the tail shard (frozen-partitioner tokenize + AH-quantize, no
        # k-means retrain). Built into every new shard so the next delta
        # cycle can take the upsert fast path.
        shard_kwargs["with_docids"] = True

        # Stream the corpus ONE SHARD AT A TIME via keyset pagination on
        # (model, id) — backed by idx_embeddings_model_id. This is the
        # whole point of the memory budget: psycopg's default client-side
        # cursor buffers the ENTIRE result set on execute(), so a single
        # "SELECT … ORDER BY id" pulls every embedding (~N × dims × 4
        # bytes) into RAM up front and blows the per-shard budget (this is
        # what was OOM-killing the backfill at ~20 GB). With keyset paging,
        # each page IS a shard: fetch shard_size rows, build, free, repeat —
        # so client-side working set stays ≈ one shard regardless of corpus
        # size, and we just add shards until the corpus is exhausted.
        import resource as _resource  # stdlib; for per-shard peak-RSS logging

        last_id = -1
        while True:
            if user_id is None:
                page = conn.execute(
                    "SELECT id, embedding FROM embeddings WHERE model = %s AND id > %s ORDER BY id LIMIT %s",
                    (model, last_id, shard_size),
                ).fetchall()
            else:
                page = conn.execute(
                    "SELECT id, embedding FROM embeddings WHERE model = %s AND user_id = %s "
                    "AND id > %s ORDER BY id LIMIT %s",
                    (model, user_id, last_id, shard_size),
                ).fetchall()
            if not page:
                break
            last_id = page[-1]["id"]
            rows_buf = [(r["id"], r["embedding"]) for r in page]
            n = len(rows_buf)
            del page
            shard_ids = _build_one_shard(rows_buf, dimensions, target_dir / f"shard_{shard_idx}", **shard_kwargs)
            all_ids.extend(shard_ids)
            shards_meta.append(_shard_meta(shard_ids, shard_idx, shard_size))
            del rows_buf
            peak_mib = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss // 1024
            logger.info(
                "reindex: built shard %d (%d vecs) — process peak RSS %d MiB",
                shard_idx,
                n,
                peak_mib,
            )
            shard_idx += 1

        num_shards = shard_idx
        manifest_payload: dict[str, object] = {
            "num_shards": num_shards,
            "dimensions": dimensions,
            "shard_size": shard_size,
            # v2: per-shard id-range keys + watermark enable delta rebuilds
            # (reuse sealed shards, rebuild only the open tail). Old readers
            # ignore these keys; the v1 range(num_shards) load still works.
            "format_version": 2,
            "shards": shards_meta,
            "watermark": all_ids[-1] if all_ids else 0,
            "reorder_pool": reorder_pool,  # delta reuse-gate: tail must match sealed
        }
        if effective_ah_dim is not None:
            # Tells the searcher to truncate the query to this dim
            # before passing to ScaNN. Independent of manual_rerank
            # — V3 (truncate only) writes this without manual_rerank,
            # variant C writes both.
            manifest_payload["index_dim"] = effective_ah_dim
        if manual_rerank:
            # Searcher reads this block to switch on the rerank step;
            # absence == "truncate query, search ScaNN, return as-is."
            # `corpus_per_shard` marks the delta-compatible layout
            # (shard_N/corpus_full.f32); the searcher also accepts the
            # old single-file `corpus_full_path` layout for indexes
            # built before this change.
            manifest_payload["manual_rerank"] = {
                "ah_dim": effective_ah_dim,
                "reorder_pool": reorder_pool,
                "full_dim": dimensions,
                "corpus_per_shard": True,
                # AH-only shards (no ScaNN reorder stage / dataset.npy) — the
                # exact external rerank makes it redundant. Delta reuse-gate
                # keys on this so pre-ah_only funnel indexes rebuild once.
                "ah_only": True,
            }
        (target_dir / "manifest.json").write_text(json.dumps(manifest_payload))
        (target_dir / "ids.json").write_text(json.dumps(all_ids))

        if use_pointer:
            _promote_and_gc(conn, index_dir, target_dir, user_id=user_id)
    finally:
        conn.close()

    logger.info(f"Sharded index built at {target_dir} ({shard_idx} shards)")
    return target_dir


def _shard_meta(shard_ids: list[int], shard_idx: int, shard_size: int) -> dict:
    """v2-manifest entry for one shard. `key` is a deterministic id
    (first/last/count) so the searcher can tell across reindexes whether a
    shard is unchanged (reuse its loaded ScaNN searcher) or new (load it).
    `sealed` (== full) → immutable → hardlink-reusable; the partial tail is
    the 'open' shard, rebuilt each delta cycle."""
    return {
        "key": f"{shard_ids[0]}_{shard_ids[-1]}_{len(shard_ids)}",
        "dir": f"shard_{shard_idx}",
        "first_id": shard_ids[0],
        "last_id": shard_ids[-1],
        "count": len(shard_ids),
        "sealed": len(shard_ids) == shard_size,
        # Built with ScaNN docids → mutable via upsert (delta fast path).
        # Legacy manifests lack this key; the delta path treats absent as
        # False and rebuilds the tail once, after which it's docid-mode.
        "docids": True,
    }


def _hardlink_shard(src: Path, dst: Path) -> None:
    """Reuse a sealed shard in a new versioned dir: hardlink the big ScaNN
    artifacts (shared inodes → ~0 disk, and they survive the old dir's GC),
    but REWRITE `scann_assets.pbtxt`, which bakes in ABSOLUTE paths to those
    artifacts at serialize time — a plain hardlink would leave it pointing at
    the GC'd source dir ("File too short" on load). Raises OSError on cross-
    filesystem link — caller falls back to a full rebuild."""
    import os  # noqa: PLC0415

    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if not f.is_file():
            continue
        if f.name == "scann_assets.pbtxt":
            (dst / f.name).write_text(f.read_text().replace(str(src), str(dst)))
        else:
            os.link(f, dst / f.name)


def _delta_effective_ah_dim(dimensions, truncate_dim, manual_rerank, ah_dim):
    if manual_rerank and ah_dim >= dimensions:
        manual_rerank = False
    eff = ah_dim if manual_rerank else (truncate_dim if truncate_dim and truncate_dim < dimensions else None)
    return eff, manual_rerank


# Below this many vectors a shard is built with score_brute_force() (no k-means
# tree) — see _build_scann_from_vectors. Rebuilding such a tiny shard is cheap,
# and upsert's whole payoff is skipping the tree retrain, so we only take the
# upsert fast path once the tail actually carries a trained partitioner.
_UPSERT_MIN_TAIL = 100


def _upsert_tail_shard(
    conn,
    model: str,
    dimensions: int,
    shard_size: int,
    *,
    live_dir: Path,
    target_dir: Path,
    tail_meta: dict,
    sealed_count: int,
    watermark: int,
    eff_ah: int | None,
    manual_rerank: bool,
    user_id: str | None,
) -> tuple[list[int], dict] | None:
    """Extend the live tail shard with new embeddings (id > watermark) via
    ScaNN's native `upsert` — tokenize the new vectors against the tail's
    FROZEN k-means partitioner + AH codebook, with NO retrain. This is the
    whole point of the delta path: the old code rebuilt the open tail (up to
    shard_size vectors) from PG every cycle, retraining k-means each time; here
    the per-cycle cost drops to O(new vectors).

    Writes a new tail shard into `target_dir/shard_{sealed_count}`:
      - load the live tail's ScaNN searcher, upsert the new (docid, vector)s,
        serialize to the new dir;
      - ids.json = old tail ids + new ids. Safe because insert-only upsert
        APPENDS internally (never reorders existing docids), and the searcher
        resolves docid-mode shards by `int(docid)`, not ids.json position;
      - corpus_full.f32 (manual-rerank layout) is COPIED forward — not
        hardlinked; we append the new full-precision rows and must not mutate
        the live file — keeping row order == ids.json order so the searcher's
        id→corpus map stays valid.

    Returns (tail_ids, shard_meta) on success. Returns None to tell the caller
    to fall back to the full tail rebuild (which is always correct) when the
    fast path doesn't apply or anything looks off: tail already full, new rows
    would overflow the shard (rebuild seals + spills into fresh shards), a
    missing/mis-sized corpus, or any ScaNN error.
    """
    import shutil  # noqa: PLC0415

    tail_count = int(tail_meta["count"])
    room = shard_size - tail_count
    if room <= 0:
        return None  # tail already full → let the rebuild loop seal it + start fresh

    # Fetch truly-new rows (id > watermark). LIMIT room+1 so we can detect
    # overflow (more new rows than fit) and defer to the rebuild loop, which
    # already knows how to seal the tail and spill the rest into new shards.
    if user_id is None:
        page = conn.execute(
            "SELECT id, embedding FROM embeddings WHERE model=%s AND id > %s ORDER BY id LIMIT %s",
            (model, watermark, room + 1),
        ).fetchall()
    else:
        page = conn.execute(
            "SELECT id, embedding FROM embeddings WHERE model=%s AND user_id=%s AND id > %s ORDER BY id LIMIT %s",
            (model, user_id, watermark, room + 1),
        ).fetchall()
    if not page:
        return None  # no new rows (caller already checked new_max>watermark, but be safe)
    if len(page) > room:
        return None  # would overflow the shard → rebuild seals + spills

    new_ids = [r["id"] for r in page]
    new_full = np.empty((len(page), dimensions), dtype=np.float32)
    for i, r in enumerate(page):
        new_full[i] = np.frombuffer(r["embedding"], dtype=np.float32)

    live_tail_dir = live_dir / tail_meta["dir"]
    target_tail_dir = target_dir / f"shard_{sealed_count}"

    try:
        searcher = scann.scann_ops_pybind.load_searcher(str(live_tail_dir))
        # Integrity gate: the live tail's actual vector count must match what the
        # manifest claims. A mismatch means a torn/corrupt live shard — upserting
        # into it would carry the corruption forward, so bail to a clean rebuild
        # (nothing written yet; no cleanup needed).
        live_size = int(searcher.size())
        if live_size != tail_count:
            logger.warning(
                "delta reindex: live tail size %d != manifest count %d — falling back to rebuild",
                live_size,
                tail_count,
            )
            return None
        # Match exactly what the full build fed ScaNN: truncate+normalize to the
        # AH subspace when the index is truncated, else the raw blob vectors.
        up_vecs = _truncate_normalize(new_full, eff_ah) if eff_ah is not None else new_full
        searcher.upsert([str(i) for i in new_ids], up_vecs)
        target_tail_dir.mkdir(parents=True, exist_ok=True)
        searcher.serialize(str(target_tail_dir))
    except Exception as e:  # noqa: BLE001
        logger.warning("delta reindex: tail upsert failed (%s) — falling back to tail rebuild", e)
        shutil.rmtree(target_tail_dir, ignore_errors=True)
        return None

    old_tail_ids = json.loads((live_tail_dir / "ids.json").read_text())
    tail_ids = old_tail_ids + new_ids
    (target_tail_dir / "ids.json").write_text(json.dumps(tail_ids))

    if manual_rerank:
        src_corpus = live_tail_dir / CORPUS_SHARD_FILENAME
        expected = tail_count * dimensions * 4
        if not src_corpus.is_file() or src_corpus.stat().st_size != expected:
            logger.warning("delta reindex: tail corpus missing/mis-sized — falling back to tail rebuild")
            shutil.rmtree(target_tail_dir, ignore_errors=True)
            return None
        dst_corpus = target_tail_dir / CORPUS_SHARD_FILENAME
        with open(src_corpus, "rb") as fsrc, open(dst_corpus, "wb") as fdst:
            shutil.copyfileobj(fsrc, fdst)
            fdst.write(np.ascontiguousarray(new_full).tobytes())

    count = len(tail_ids)
    meta = {
        "key": f"{tail_ids[0]}_{tail_ids[-1]}_{count}",
        "dir": f"shard_{sealed_count}",
        "first_id": tail_ids[0],
        "last_id": tail_ids[-1],
        "count": count,
        "sealed": count == shard_size,
        "docids": True,
    }
    logger.info(
        "delta reindex: tail upsert +%d vecs (%d→%d), no retrain",
        len(new_ids),
        tail_count,
        count,
    )
    return tail_ids, meta


def _rebuild_open_shards(
    conn,
    model: str,
    dimensions: int,
    shard_size: int,
    *,
    target_dir: Path,
    resume_from: int,
    start_shard_idx: int,
    shard_kwargs: dict,
    all_ids: list[int],
    shards_meta: list[dict],
    user_id: str | None,
) -> int:
    """Rebuild the open tail (+ any spillover shards) from PG rows id >=
    resume_from, appending to `all_ids`/`shards_meta` in place. Returns the next
    (== total) shard index. The always-correct fallback when the upsert fast
    path doesn't apply (legacy/non-docid tail, fragmented tail, or overflow)."""
    shard_idx = start_shard_idx
    last_id = resume_from - 1
    while True:
        if user_id is None:
            page = conn.execute(
                "SELECT id, embedding FROM embeddings WHERE model=%s AND id > %s ORDER BY id LIMIT %s",
                (model, last_id, shard_size),
            ).fetchall()
        else:
            page = conn.execute(
                "SELECT id, embedding FROM embeddings WHERE model=%s AND user_id=%s AND id > %s ORDER BY id LIMIT %s",
                (model, user_id, last_id, shard_size),
            ).fetchall()
        if not page:
            break
        last_id = page[-1]["id"]
        rows_buf = [(r["id"], r["embedding"]) for r in page]
        del page
        shard_ids = _build_one_shard(rows_buf, dimensions, target_dir / f"shard_{shard_idx}", **shard_kwargs)
        all_ids.extend(shard_ids)
        shards_meta.append(_shard_meta(shard_ids, shard_idx, shard_size))
        del rows_buf
        shard_idx += 1
    return shard_idx


def build_index_delta(
    db_path: Path,
    index_dir: Path,
    model: str,
    dimensions: int,
    shard_size: int,
    *,
    truncate_dim: int | None = None,
    manual_rerank: bool = False,
    ah_dim: int = 1536,
    reorder_pool: int = 100,
    user_id: str | None = None,
) -> Path:
    """Delta rebuild: reuse the live index's SEALED shards (hardlink them into
    a new versioned dir) and rebuild only the OPEN tail shard (+ any shard the
    new embeddings fill). Only the delta is read from PG and built — the bulk
    (sealed shards) is untouched, and the searcher reloads only changed shards.

    Falls back to a full `build_index_sharded` when there's no reusable v2
    live index or the build config (subspace/shard_size/rerank mode) changed.
    Manual-rerank indexes ARE delta-compatible as long as the live index uses
    the per-shard corpus layout (shard_N/corpus_full.f32): sealed shards'
    corpus files hardlink along with their ScaNN artifacts, and only the tail
    shard's corpus is rewritten. Pre-per-shard rerank builds (one index-level
    corpus_full.memmap) can't be reused and trigger one full rebuild.
    """
    full_kwargs = dict(
        truncate_dim=truncate_dim,
        manual_rerank=manual_rerank,
        ah_dim=ah_dim,
        reorder_pool=reorder_pool,
        user_id=user_id,
    )

    def _full():
        return build_index_sharded(db_path, index_dir, model, dimensions, shard_size, **full_kwargs)

    eff_ah, manual_rerank = _delta_effective_ah_dim(dimensions, truncate_dim, manual_rerank, ah_dim)
    if not _pointer_table_exists(db_path):
        return _full()

    from gmail_search.index.searcher import resolve_active_index_dir

    live_dir = resolve_active_index_dir(db_path, index_dir, user_id=user_id)
    manifest_path = live_dir / "manifest.json"
    if not manifest_path.exists():
        return _full()
    try:
        live = json.loads(manifest_path.read_text())
    except Exception:
        return _full()
    # Reuse is only valid when the build subspace + sharding are identical.
    live_mr = live.get("manual_rerank")
    if (
        live.get("format_version") != 2
        or not live.get("shards")
        or bool(live_mr) != manual_rerank  # rerank mode flip → rebuild
        # Old single-file corpus layout spans all shards — not hardlinkable.
        or (manual_rerank and not (live_mr or {}).get("corpus_per_shard"))
        # Pre-ah_only funnel shards still carry dataset.npy (~2 GB RAM at
        # scale); rebuild once so the whole index drops the reorder tier.
        or (manual_rerank and not (live_mr or {}).get("ah_only"))
        or live.get("dimensions") != dimensions
        or live.get("shard_size") != shard_size
        or live.get("index_dim") != eff_ah  # None==None when no truncation
        or live.get("reorder_pool") != reorder_pool  # ranking param must match
    ):
        return _full()

    live_shards = live["shards"]
    watermark = int(live.get("watermark", 0))
    # Reuse only the LEADING run of SEALED shards; rebuild everything from the
    # first unsealed shard onward. A delta build can leave SEVERAL trailing
    # unsealed shards (a partial tail PLUS, e.g., a 1-row shard from an
    # embedding inserted mid-build). Resuming from only the LAST one drops the
    # others' vectors → a silent index gap. Resuming from the id after the last
    # reusable sealed shard rebuilds every trailing row and compacts the
    # fragments back into full shards.
    sealed = []
    for s in live_shards:
        if not s.get("sealed"):
            break
        sealed.append(s)
    if not sealed:
        return _full()  # nothing reusable → full build
    resume_from = int(sealed[-1]["last_id"]) + 1

    conn = get_connection(db_path)
    try:
        if user_id is None:
            new_max = conn.execute("SELECT COALESCE(MAX(id),0) FROM embeddings WHERE model=%s", (model,)).fetchone()[0]
            live_count = conn.execute(
                "SELECT count(*) FROM embeddings WHERE model=%s AND id<=%s", (model, watermark)
            ).fetchone()[0]
        else:
            new_max = conn.execute(
                "SELECT COALESCE(MAX(id),0) FROM embeddings WHERE model=%s AND user_id=%s", (model, user_id)
            ).fetchone()[0]
            live_count = conn.execute(
                "SELECT count(*) FROM embeddings WHERE model=%s AND user_id=%s AND id<=%s",
                (model, user_id, watermark),
            ).fetchone()[0]
        # Deletion handling: the index covers ids ≤ watermark. If the DB now
        # holds FEWER embeddings in that range than the index does, rows were
        # deleted (embed --force, message removal) and the stale vectors linger
        # in immutable sealed shards. Stale vectors are HARMLESS in normal
        # search (the message JOIN drops ids with no row), so we TOLERATE a
        # small amount rather than pay a full rebuild — a full rebuild reloads
        # the whole index (memory peak + serve disruption), so triggering it on
        # every routine deletion would be far worse than the staleness. Only
        # compact (full rebuild) once stale exceeds a threshold (% of the
        # index). Routine deletions ride along in the delta until then.
        index_count = sum(int(s.get("count", 0)) for s in live_shards)
        stale = index_count - live_count
        if stale > _DELTA_COMPACTION_STALE_FRACTION * index_count:
            conn.close()
            logger.info(
                "delta reindex: %d stale (>%.0f%% of %d) → full rebuild (compaction)",
                stale,
                _DELTA_COMPACTION_STALE_FRACTION * 100,
                index_count,
            )
            return _full()
        if new_max <= watermark:
            conn.close()
            logger.info("delta reindex: no new embeddings (watermark=%d) — skipping", watermark)
            return live_dir  # no-op: pointer stays put

        target_dir = _pick_versioned_dir(index_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        # 1) Hardlink sealed shards into the new dir (shard_0..shard_{m-1}).
        all_ids: list[int] = []
        shards_meta: list[dict] = []
        for i, s in enumerate(sealed):
            src = live_dir / s["dir"]
            dst = target_dir / f"shard_{i}"
            if manual_rerank:
                cf = src / CORPUS_SHARD_FILENAME
                expected = int(s["count"]) * dimensions * 4
                if not cf.is_file() or cf.stat().st_size != expected:
                    # Torn/partial live dir: a sealed shard with a missing OR
                    # mis-sized corpus file can't serve the rerank step —
                    # hardlinking it forward would make the searcher disable
                    # rerank on the whole index (silent quality downgrade).
                    logger.warning("delta reindex: %s corpus missing/mis-sized — full rebuild", src)
                    conn.close()
                    import shutil  # noqa: PLC0415

                    shutil.rmtree(target_dir, ignore_errors=True)
                    return _full()
            try:
                _hardlink_shard(src, dst)
            except OSError as e:
                logger.warning("delta reindex: hardlink failed (%s) — full rebuild", e)
                conn.close()
                import shutil  # noqa: PLC0415

                shutil.rmtree(target_dir, ignore_errors=True)
                return _full()
            sealed_ids = json.loads((dst / "ids.json").read_text())
            all_ids.extend(sealed_ids)
            meta = dict(s)
            meta["dir"] = f"shard_{i}"
            shards_meta.append(meta)

        # 2) Extend the open tail. Fast path: if the single unsealed tail shard
        # is a docid-mode (mutable) shard with a trained tree, UPSERT the new
        # embeddings into it (no k-means retrain). Otherwise fall through to the
        # rebuild loop, which reads every row from resume_from and rebuilds —
        # correct for legacy tails, multi-fragment tails, and overflow.
        shard_kwargs: dict[str, object] = {"reorder_pool": reorder_pool}
        if eff_ah is not None:
            shard_kwargs["ah_dim"] = eff_ah
        shard_kwargs["write_full_corpus"] = manual_rerank
        shard_kwargs["with_docids"] = True  # rebuilt tail is mutable next cycle

        tail_metas = live_shards[len(sealed) :]
        upsert_result = None
        if len(tail_metas) == 1 and tail_metas[0].get("docids") and int(tail_metas[0]["count"]) >= _UPSERT_MIN_TAIL:
            upsert_result = _upsert_tail_shard(
                conn,
                model,
                dimensions,
                shard_size,
                live_dir=live_dir,
                target_dir=target_dir,
                tail_meta=tail_metas[0],
                sealed_count=len(sealed),
                watermark=watermark,
                eff_ah=eff_ah,
                manual_rerank=manual_rerank,
                user_id=user_id,
            )

        if upsert_result is not None:
            tail_ids, tail_meta_new = upsert_result
            all_ids.extend(tail_ids)
            shards_meta.append(tail_meta_new)
            shard_idx = len(sealed) + 1
            _log_tail_upsert = True
        else:
            _log_tail_upsert = False
            shard_idx = _rebuild_open_shards(
                conn,
                model,
                dimensions,
                shard_size,
                target_dir=target_dir,
                resume_from=resume_from,
                start_shard_idx=len(sealed),
                shard_kwargs=shard_kwargs,
                all_ids=all_ids,
                shards_meta=shards_meta,
                user_id=user_id,
            )

        manifest_payload: dict[str, object] = {
            "num_shards": shard_idx,
            "dimensions": dimensions,
            "shard_size": shard_size,
            "format_version": 2,
            "shards": shards_meta,
            "watermark": all_ids[-1] if all_ids else watermark,
            "reorder_pool": reorder_pool,
        }
        if eff_ah is not None:
            manifest_payload["index_dim"] = eff_ah
        if manual_rerank:
            manifest_payload["manual_rerank"] = {
                "ah_dim": eff_ah,
                "reorder_pool": reorder_pool,
                "full_dim": dimensions,
                "corpus_per_shard": True,
                # AH-only shards (no ScaNN reorder stage / dataset.npy) — the
                # exact external rerank makes it redundant. Delta reuse-gate
                # keys on this so pre-ah_only funnel indexes rebuild once.
                "ah_only": True,
            }
        (target_dir / "manifest.json").write_text(json.dumps(manifest_payload))
        (target_dir / "ids.json").write_text(json.dumps(all_ids))
        _promote_and_gc(conn, index_dir, target_dir, user_id=user_id)
    finally:
        conn.close()

    logger.info(
        "delta reindex: %d sealed shards reused, tail extended via %s, %d shards touched " "(watermark %d→%d) at %s",
        len(sealed),
        "upsert (no retrain)" if _log_tail_upsert else "rebuild",
        shard_idx - len(sealed),
        watermark,
        all_ids[-1] if all_ids else watermark,
        target_dir,
    )
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


def _promote_and_gc(conn, live_name: Path, new_dir: Path, *, user_id: str | None = None) -> None:
    """Update the DB pointer to the new dir and remove every sibling
    `{live_name.name}__*` directory other than the one we just wrote.
    `live_name` itself (if it exists as a legacy path) is left alone
    so a reader that still opens it directly keeps working until it
    learns about the pointer.

    `user_id` flows through to `set_active_index_dir` so the per-user
    pointer row gets flipped — the table is keyed (user_id) post-Phase-3a."""
    import shutil

    from gmail_search.store.queries import set_active_index_dir

    set_active_index_dir(conn, str(new_dir), user_id=user_id)
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
