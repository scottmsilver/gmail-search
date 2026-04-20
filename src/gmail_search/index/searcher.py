import heapq
import json
import logging
from pathlib import Path

import numpy as np
import scann

logger = logging.getLogger(__name__)


def resolve_active_index_dir(db_path: Path, fallback: Path) -> Path:
    """Return the active on-disk ScaNN index directory, consulting the
    `scann_index_pointer` row in SQLite first and falling back to
    `fallback` (typically `data/scann_index`) when the pointer is
    absent or points at a missing path.

    Every search-path caller (server, CLI query, battle mode) should
    route through this instead of hard-coding `data_dir / "scann_index"`
    so a reindex swap is picked up without a process restart.
    """
    from gmail_search.store.db import get_connection

    try:
        conn = get_connection(db_path)
    except Exception:
        return fallback
    try:
        # SELECT against the pointer table directly, catching the
        # "no such table" failure for the case where a caller hands us
        # a DB that predates the pointer schema. This is backend-agnostic:
        # SQLite raises `OperationalError("no such table")`, Postgres
        # raises `UndefinedTable`. Either way we fall through to the
        # default path. Replaces the old `sqlite_master` probe which
        # didn't exist on Postgres.
        try:
            row = conn.execute("SELECT current_dir FROM scann_index_pointer WHERE id = 1").fetchone()
        except Exception:
            return fallback
    finally:
        conn.close()
    if row and row["current_dir"]:
        active = Path(row["current_dir"])
        if active.exists():
            return active
    return fallback


class ScannSearcher:
    """Loads a ScaNN index produced by either build_index (single) or
    build_index_sharded (multi-shard). Sharded indexes query each shard
    for top_k and merge by score.
    """

    def __init__(self, index_dir: Path, dimensions: int):
        self.index_dir = index_dir
        self.dimensions = dimensions

        ids_file = index_dir / "ids.json"
        if not ids_file.exists():
            raise FileNotFoundError(f"Index not found at {index_dir}. Run 'gmail-search reindex' first.")

        self.embedding_ids: list[int] = json.loads(ids_file.read_text())
        # List of (scann_searcher, local_ids) pairs. One entry for an
        # unsharded legacy index; N entries for a sharded one. Empty list
        # = empty index.
        self._shards: list[tuple[object, list[int]]] = []

        manifest_path = index_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            self._load_sharded(manifest)
        else:
            self._load_legacy_single()

    def _load_sharded(self, manifest: dict) -> None:
        # Tolerate missing/partial shards: the sharded builder writes
        # manifest.json LAST, but we still defend against a torn read
        # during a concurrent reindex (old manifest, new half-written
        # shard dirs). Missing shards drop out of the search, which is
        # strictly better than a 500 on the whole query path.
        for i in range(manifest["num_shards"]):
            shard_dir = self.index_dir / f"shard_{i}"
            ids_file = shard_dir / "ids.json"
            if not ids_file.exists():
                logger.warning("shard_%d missing ids.json; skipping — reindex may be in progress", i)
                continue
            try:
                shard_ids = json.loads(ids_file.read_text())
            except Exception as e:
                logger.warning("shard_%d ids.json unreadable (%s); skipping", i, e)
                continue
            if not shard_ids:
                continue
            try:
                searcher = scann.scann_ops_pybind.load_searcher(str(shard_dir))
            except Exception as e:
                logger.warning("shard_%d load_searcher failed (%s); skipping", i, e)
                continue
            self._shards.append((searcher, shard_ids))
        logger.info(
            "Loaded sharded ScaNN index: %d/%d shards, %d total vectors",
            len(self._shards),
            manifest["num_shards"],
            len(self.embedding_ids),
        )

    def _load_legacy_single(self) -> None:
        if not self.embedding_ids:
            logger.warning("Empty index loaded")
            return
        searcher = scann.scann_ops_pybind.load_searcher(str(self.index_dir))
        self._shards.append((searcher, self.embedding_ids))
        logger.info(f"Loaded ScaNN index with {len(self.embedding_ids)} vectors")

    def search(self, query_vector: np.ndarray, top_k: int = 20) -> tuple[list[int], list[float]]:
        if not self._shards:
            return [], []

        if len(self._shards) == 1:
            # Fast path — avoid the merge step for unsharded indexes.
            searcher, ids = self._shards[0]
            neighbors, distances = searcher.search(query_vector, final_num_neighbors=top_k)
            return [ids[i] for i in neighbors], distances.tolist()

        # Sharded: get each shard's top_k, merge by score. For dot_product
        # similarity, higher is better → heapq.nlargest.
        candidates: list[tuple[float, int]] = []
        for searcher, ids in self._shards:
            neighbors, distances = searcher.search(query_vector, final_num_neighbors=top_k)
            for idx, score in zip(neighbors, distances):
                candidates.append((float(score), ids[idx]))
        top = heapq.nlargest(top_k, candidates, key=lambda x: x[0])
        return [eid for _, eid in top], [score for score, _ in top]
