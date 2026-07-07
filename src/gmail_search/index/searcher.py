import heapq
import json
import logging
from pathlib import Path

import numpy as np
import scann

logger = logging.getLogger(__name__)


def resolve_active_index_dir(db_path: Path, fallback: Path, *, user_id: str | None = None) -> Path:
    """Return the active on-disk ScaNN index directory, consulting the
    `scann_index_pointer` row in SQLite first and falling back to
    `fallback` when the pointer is absent or points at a missing path.

    Per-user: Phase 3a moved the pointer table from the legacy single-row
    (CHECK(id=1)) shape to one row per user (PK = user_id). The lookup
    here resolves the bootstrap user when no `user_id` is passed, so
    daemon callers keep working unchanged.

    Every search-path caller (server, CLI query, battle mode) should
    route through this instead of hard-coding `data_dir / "scann_index"`
    so a reindex swap is picked up without a process restart.
    """
    from gmail_search.auth.write_user import resolve_write_user_id
    from gmail_search.store.db import get_connection

    try:
        conn = get_connection(db_path)
    except Exception:
        return fallback
    try:
        # Resolve the pointer's user_id. The `users` table may not exist
        # in test fixtures / un-migrated installs; in that case fall
        # back to the legacy single-row lookup so older callers keep
        # working unchanged.
        try:
            uid = resolve_write_user_id(conn, user_id=user_id)
        except Exception:
            uid = None
        try:
            if uid is None:
                row = conn.execute("SELECT current_dir FROM scann_index_pointer LIMIT 1").fetchone()
            else:
                row = conn.execute("SELECT current_dir FROM scann_index_pointer WHERE user_id = %s", (uid,)).fetchone()
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

    def __init__(self, index_dir: Path, dimensions: int, *, prev: "ScannSearcher | None" = None):
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
        # Stable per-shard keys (parallel to _shards), used by reload to reuse
        # an already-loaded shard's C++ searcher instead of re-loading it.
        self._shard_keys: list[str] = []

        # Manual-rerank state. When the manifest carries the
        # `manual_rerank` block (variant C — see docs/notes/SCANN_COMPACTION_2026-04-27.md),
        # we mmap the full-precision corpus and rerank ScaNN's
        # truncated-AH candidates against it at query time.
        # Two corpus layouts (see _maybe_attach_manual_rerank): legacy
        # single index-level `corpus_full.memmap`, or per-shard
        # `shard_N/corpus_full.f32` files (delta-compatible). Row order
        # in both is ids.json order; `_gather_full` hides the difference.
        self._manual_rerank: dict | None = None
        self._corpus_full: np.memmap | None = None
        self._corpus_files: list[np.memmap] | None = None
        self._corpus_offsets: np.ndarray | None = None  # per-file start rows
        self._id_to_pos: dict[int, int] = {}
        # `index_dim` (when present in the manifest) tells us the
        # actual dimensionality the underlying ScaNN index was built
        # on. Set independently of `manual_rerank` so the V3 build —
        # truncate-only, no rerank step — also gets query truncation.
        # Defaults to `dimensions` for legacy indexes that pre-date
        # this field, which means "no truncation" (no-op).
        self._index_dim: int = dimensions

        manifest_path = index_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            self._load_sharded(manifest, prev=prev)
            # Read index_dim BEFORE the rerank attach: the rerank
            # block also implies truncation, so they share the same
            # query-projection logic.
            mfd = manifest.get("index_dim")
            if isinstance(mfd, int) and mfd > 0:
                self._index_dim = mfd
            self._maybe_attach_manual_rerank(manifest)
        else:
            self._load_legacy_single()

    def close(self) -> None:
        """Release the ScaNN C++ searchers and the corpus mmap. Dropping
        these references frees the (multi-GB) native memory immediately via
        refcounting — no global gc.collect() needed, so the caller can reclaim
        a retired index without stalling the event loop. Idempotent."""
        self._shards = []
        self._corpus_full = None
        self._corpus_files = None
        self._corpus_offsets = None
        self._manual_rerank = None

    def _maybe_attach_manual_rerank(self, manifest: dict) -> None:
        """If the manifest has a `manual_rerank` block, mmap the
        full-precision corpus so .search() can rerank candidates.
        Two layouts: `corpus_per_shard` (shard_N/corpus_full.f32,
        written by delta-compatible builds) or the legacy single
        `corpus_full.memmap`. Best-effort: a missing or unreadable
        file logs a warning and we fall back to plain ScaNN search."""
        cfg = manifest.get("manual_rerank")
        if not cfg:
            return
        full_dim = cfg.get("full_dim")
        if not full_dim:
            logger.warning("manual_rerank manifest block lacks full_dim — falling back to plain ScaNN search")
            return
        try:
            if cfg.get("corpus_per_shard"):
                files: list[np.memmap] = []
                offsets: list[int] = [0]
                for s in manifest.get("shards", []):
                    p = self.index_dir / s["dir"] / "corpus_full.f32"
                    count = int(s["count"])
                    if not p.is_file() or p.stat().st_size != count * full_dim * 4:
                        raise FileNotFoundError(f"corpus shard file missing/mis-sized: {p}")
                    files.append(np.memmap(p, dtype=np.float32, mode="r", shape=(count, full_dim)))
                    offsets.append(offsets[-1] + count)
                if offsets[-1] != len(self.embedding_ids):
                    raise ValueError(f"corpus rows {offsets[-1]} != {len(self.embedding_ids)} indexed ids")
                self._corpus_files = files
                self._corpus_offsets = np.asarray(offsets[:-1], dtype=np.int64)
            else:
                full_path = self.index_dir / cfg.get("corpus_full_path", "corpus_full.memmap")
                n_rows = len(self.embedding_ids)
                if not full_path.is_file() or full_path.stat().st_size != n_rows * full_dim * 4:
                    raise FileNotFoundError(f"corpus_full missing/mis-sized: {full_path}")
                self._corpus_full = np.memmap(full_path, dtype=np.float32, mode="r", shape=(n_rows, full_dim))
            # Position-in-ids-list IS the global row index (the builder
            # writes shards — and their corpus files — in id-order).
            self._id_to_pos = {eid: i for i, eid in enumerate(self.embedding_ids)}
            self._manual_rerank = cfg
            logger.info(
                "manual_rerank enabled: ah_dim=%s reorder_pool=%s full_dim=%s layout=%s",
                cfg.get("ah_dim"),
                cfg.get("reorder_pool"),
                full_dim,
                "per-shard" if self._corpus_files is not None else "single-file",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "manual_rerank manifest present but corpus unusable (%s) — falling back to plain ScaNN search",
                exc,
            )
            self._corpus_full = None
            self._corpus_files = None
            self._corpus_offsets = None
            self._manual_rerank = None

    def _corpus_ready(self) -> bool:
        return self._corpus_full is not None or self._corpus_files is not None

    def _gather_full(self, positions: list[int]) -> np.ndarray:
        """Rows of the full-precision corpus at the given global positions
        (ids.json order), whichever layout is attached."""
        if self._corpus_full is not None:
            return self._corpus_full[positions]
        pos = np.asarray(positions, dtype=np.int64)
        out = np.empty((len(pos), self._corpus_files[0].shape[1]), dtype=np.float32)
        file_idx = np.searchsorted(self._corpus_offsets, pos, side="right") - 1
        for fi in np.unique(file_idx):
            mask = file_idx == fi
            out[mask] = self._corpus_files[fi][pos[mask] - self._corpus_offsets[fi]]
        return out

    def _load_sharded(self, manifest: dict, prev: "ScannSearcher | None" = None) -> None:
        # Tolerate missing/partial shards: the sharded builder writes
        # manifest.json LAST, but we still defend against a torn read
        # during a concurrent reindex (old manifest, new half-written
        # shard dirs). Missing shards drop out of the search, which is
        # strictly better than a 500 on the whole query path.
        #
        # Delta reuse: when a v2 manifest carries per-shard `key`s and `prev`
        # is supplied, reuse prev's already-loaded ScaNN searcher for any
        # unchanged (sealed) shard instead of calling load_searcher again —
        # so a swap loads only the changed (open) shard, not the whole index.
        prev_by_key: dict[str, tuple] = {}
        if prev is not None and getattr(prev, "_shard_keys", None):
            prev_by_key = dict(zip(prev._shard_keys, prev._shards))

        shards_meta = manifest.get("shards")
        if shards_meta:
            entries = [(s["dir"], s.get("key")) for s in shards_meta]
        else:  # v1 manifest — synthesize dir + (later) key from loaded ids
            entries = [(f"shard_{i}", None) for i in range(manifest["num_shards"])]

        reused = 0
        for shard_name, key in entries:
            shard_dir = self.index_dir / shard_name
            ids_file = shard_dir / "ids.json"
            if not ids_file.exists():
                logger.warning("%s missing ids.json; skipping — reindex may be in progress", shard_name)
                continue
            try:
                shard_ids = json.loads(ids_file.read_text())
            except Exception as e:
                logger.warning("%s ids.json unreadable (%s); skipping", shard_name, e)
                continue
            if not shard_ids:
                continue
            if not key:  # v1 or missing — derive a stable key from the ids
                key = f"{shard_ids[0]}_{shard_ids[-1]}_{len(shard_ids)}"
            cached = prev_by_key.get(key)
            if cached is not None:
                searcher = cached[0]  # reuse the loaded C++ searcher (shared ref)
                reused += 1
            else:
                try:
                    searcher = scann.scann_ops_pybind.load_searcher(str(shard_dir))
                except Exception as e:
                    logger.warning("%s load_searcher failed (%s); skipping", shard_name, e)
                    continue
            self._shards.append((searcher, shard_ids))
            self._shard_keys.append(key)
        logger.info(
            "Loaded sharded ScaNN index: %d shards (%d reused), %d total vectors",
            len(self._shards),
            reused,
            len(self.embedding_ids),
        )

    def _load_legacy_single(self) -> None:
        if not self.embedding_ids:
            logger.warning("Empty index loaded")
            return
        searcher = scann.scann_ops_pybind.load_searcher(str(self.index_dir))
        self._shards.append((searcher, self.embedding_ids))
        self._shard_keys.append(f"{self.embedding_ids[0]}_{self.embedding_ids[-1]}_{len(self.embedding_ids)}")
        logger.info(f"Loaded ScaNN index with {len(self.embedding_ids)} vectors")

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 20,
        *,
        manual_rerank: bool | None = None,
    ) -> tuple[list[int], list[float]]:
        """Return (embedding_ids, scores) for the top_k candidates.

        `manual_rerank` (default None = use whatever the index was built
        with):
          - True/None+rerank-ready: AH search at `ah_dim`, then rerank
            against full-precision corpus_full.memmap. Slower (~70 ms)
            but matches baseline NDCG with a smaller AH index.
          - False: pure ScaNN search, no manual rerank — used by the
            engine's restricted/over-fetch path where post-filtering
            dominates and we'd be wasting cycles reranking candidates
            that get filtered out anyway.

        On indexes without a `manual_rerank` manifest block, the value
        is ignored and we return the legacy ScaNN-only result."""
        if not self._shards:
            return [], []

        # When the index was built with manual_rerank, ScaNN only knows
        # the truncated AH subspace (e.g. 1536d). A `manual_rerank=False`
        # caller is asking us to skip the rerank step — but we still
        # have to truncate+normalize the query before handing it to
        # ScaNN, or it raises "Query doesn't match dataset
        # dimensionality". Both branches below get a query that matches
        # the index's actual dim.
        index_query = self._truncate_query_if_needed(query_vector)
        use_rerank = (
            self._manual_rerank is not None
            and self._corpus_ready()
            and (manual_rerank is None or manual_rerank is True)
        )
        if use_rerank:
            return self._search_with_manual_rerank(query_vector, index_query, top_k)
        return self._search_scann_only(index_query, top_k)

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        """Return a float32 unit-norm copy of `vec`. Idempotent on
        already-normalized inputs (norm stays 1.0)."""
        out = vec.astype(np.float32, copy=True)
        n = float(np.linalg.norm(out))
        if n > 0:
            out /= n
        return out

    def _truncate_query_if_needed(self, query_vector: np.ndarray) -> np.ndarray:
        """Project the query into the same subspace ScaNN's index was
        built on, then renormalize. Driven by manifest's `index_dim`:
          - V3 (truncate-only): `index_dim < dimensions`, no rerank.
          - Variant C (manual rerank): `index_dim < dimensions` AND
            manual_rerank block present.
          - Legacy baseline: `index_dim == dimensions` → renormalize only.

        Always normalizes — slicing breaks unit-norm even when the caller
        already normalized, and AH dot-product scores assume unit-norm
        queries. Cheap (~10µs at 3072d) and idempotent."""
        if query_vector.shape[0] < self._index_dim:
            # Caller passed fewer dims than the index expects. We can't
            # invent dims; let ScaNN produce a clear error downstream.
            return query_vector
        if query_vector.shape[0] > self._index_dim:
            query_vector = query_vector[: self._index_dim]
        return self._l2_normalize(query_vector)

    def _search_scann_only(self, query_vector: np.ndarray, top_k: int) -> tuple[list[int], list[float]]:
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

    def _search_with_manual_rerank(
        self,
        query_vector_full: np.ndarray,
        query_vector_truncated: np.ndarray,
        top_k: int,
    ) -> tuple[list[int], list[float]]:
        """Variant-C search: AH stage at ah_dim, manual rerank at full_dim.

        Caller passes BOTH the truncated and full-dim query so we don't
        re-truncate twice (the dispatch in `search()` already did the
        truncation work for the no-rerank branch). The AH stage uses
        `query_vector_truncated`; the manual rerank uses
        `query_vector_full`.

        We over-fetch from ScaNN (per `reorder_pool` in the manifest, or
        a generous default) so the rerank has enough headroom to recover
        items the truncated AH stage would have ranked just below the
        cut. Fewer candidates → faster but worse recall."""
        cfg = self._manual_rerank
        # Use the manifest's reorder_pool as the AH overfetch, capped at
        # the total corpus to avoid asking ScaNN for more than exists.
        rerank_pool = max(int(cfg.get("reorder_pool", 500)), top_k)
        rerank_pool = min(rerank_pool, len(self.embedding_ids))

        q_t = query_vector_truncated

        # AH-stage candidate gather. Asking every shard for the FULL pool
        # is wasteful at high pool sizes (31 shards × 4000 = 124k tuples
        # to merge for a 4000-candidate rerank). The global top-pool is
        # spread across shards, so each shard only needs its expected
        # share (pool/num_shards) times a generous headroom factor —
        # a shard would have to hold >4× its proportional share of the
        # pool before this clips anything.
        cand_ids: list[int] = []
        if len(self._shards) == 1:
            searcher, ids = self._shards[0]
            neighbors, _ = searcher.search(q_t, final_num_neighbors=rerank_pool)
            cand_ids = [ids[i] for i in neighbors]
        else:
            per_shard = max(top_k, -(-rerank_pool // len(self._shards)) * 4)
            per_shard = min(per_shard, rerank_pool)
            # Per-shard id arrays, built once per loaded searcher (the
            # lists in _shards are 40k+ entries; converting per query
            # would dominate the merge).
            if getattr(self, "_shard_ids_np", None) is None or len(self._shard_ids_np) != len(self._shards):
                self._shard_ids_np = [np.asarray(ids, dtype=np.int64) for _, ids in self._shards]
            id_chunks: list[np.ndarray] = []
            score_chunks: list[np.ndarray] = []
            for (searcher, _ids), ids_arr in zip(self._shards, self._shard_ids_np):
                neighbors, distances = searcher.search(q_t, final_num_neighbors=per_shard)
                if len(neighbors) == 0:
                    continue
                id_chunks.append(ids_arr[np.asarray(neighbors, dtype=np.int64)])
                score_chunks.append(np.asarray(distances, dtype=np.float32))
            if id_chunks:
                all_ids = np.concatenate(id_chunks)
                all_scores = np.concatenate(score_chunks)
                kk = min(rerank_pool, all_scores.shape[0])
                sel = np.argpartition(all_scores, -kk)[-kk:]
                cand_ids = [int(i) for i in all_ids[sel]]

        if not cand_ids:
            return [], []

        # Manual rerank: dot-product against full-precision vectors.
        # Pair id↔position in one pass so the score-row index always
        # matches the kept id at the same offset (no implicit ordering
        # coupling). Skip any candidate whose id we somehow can't locate
        # in the memmap (shouldn't happen but defending against torn state).
        kept = [(c, self._id_to_pos[c]) for c in cand_ids if c in self._id_to_pos]
        if not kept:
            return [], []
        kept_cand_ids = [c for c, _ in kept]
        positions = [p for _, p in kept]
        cand_vecs = self._gather_full(positions)  # (P, full_dim)
        cand_norms = np.linalg.norm(cand_vecs, axis=1, keepdims=True)
        cand_norms[cand_norms == 0] = 1.0
        cand_vecs_n = cand_vecs / cand_norms
        q_full = self._l2_normalize(query_vector_full)
        scores = cand_vecs_n @ q_full
        # Top-K by exact score.
        k = min(top_k, scores.shape[0])
        top_pos = np.argpartition(scores, -k)[-k:]
        top_pos_sorted = top_pos[np.argsort(scores[top_pos])[::-1]]
        out_ids = [kept_cand_ids[p] for p in top_pos_sorted]
        out_scores = [float(scores[p]) for p in top_pos_sorted]
        return out_ids, out_scores
