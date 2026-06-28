"""Reindex orchestration — one function, three callers.

Before this module existed, the watch cycle, the update batch, and the
`gmail-search reindex` CLI each inlined their own version of "rebuild
the on-disk surfaces search depends on", with subtly different subsets.
Any new rebuild step had to be added in three places and the watch
path tended to get stuck with a stale subset.

`reindex(...)` is now the single source of truth. `light=True` runs
the minimum needed to make new messages searchable (ScaNN + FTS +
thread summary) and is what the watch loop uses between cycles.
`light=False` additionally refreshes the slower rebuilds (contact
frequency, spell dictionary, topics, term aliases) and clears the
query cache — appropriate after a backfill or an explicit reindex.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from gmail_search.index.builder import build_index_delta, build_index_sharded, shard_size_from_budget
from gmail_search.store.db import (
    clear_query_cache,
    rebuild_contact_frequency,
    rebuild_fts,
    rebuild_spell_dictionary,
    rebuild_term_aliases,
    rebuild_thread_summary,
    rebuild_topics,
)

logger = logging.getLogger(__name__)

# Default cap on the per-shard ScaNN build peak. Matches the default
# in config.yaml; can be overridden via cfg['indexing']['scann_peak_budget_mb'].
# ScaNN's full-corpus build peaks at ~4× raw float32 size — at 400K ×
# 3072 dims that's ~18 GiB peak and has OOM'd this box before. The
# sharded builder keeps each shard's peak under this cap; the searcher
# merges shards at query time with no change to the API.
_DEFAULT_SCANN_BUDGET_MB = 2048


def reindex(
    db_path: Path,
    data_dir: Path,
    cfg: dict[str, Any],
    *,
    light: bool = False,
    user_id: Optional[str] = None,
) -> None:
    """Rebuild the on-disk surfaces that back /api/search.

    light=True  → hot-path rebuilds only (ScaNN + FTS + thread summary).
                  What the watch cycle runs between syncs.
    light=False → adds the slower rebuilds (contact, spell, topics,
                  aliases) and wipes the query embedding cache.
                  What the `reindex` CLI + post-backfill path run.

    Per-user: rebuilds only the given user's surfaces. Daemon callers
    pass `user_id=None` and the bootstrap user is resolved. Phase 3c
    per-user sync iterates and passes each syncing user's id explicitly.
    """
    # Resolve user_id once at the top — every downstream rebuild_*
    # accepts it explicitly so the bootstrap-vs-real-user decision is
    # made in one place, not threaded through each call.
    from gmail_search.auth.write_user import resolve_write_user_id
    from gmail_search.store.db import get_connection as _get_conn

    _conn = _get_conn(db_path)
    try:
        uid = resolve_write_user_id(_conn, user_id=user_id)
    finally:
        _conn.close()

    # `index_dir` is the canonical PREFIX. Per-user index lives under
    # `data/users/<user_id>/scann_index` so two users' indexes never
    # share files. `build_index_sharded` writes a timestamped sibling
    # and flips the DB pointer row; readers resolve through
    # `resolve_active_index_dir` so a mid-reindex query always lands
    # on a fully-written build.
    index_dir = Path(data_dir) / "users" / uid / "scann_index"
    index_dir.parent.mkdir(parents=True, exist_ok=True)
    dimensions = int(cfg["embedding"]["dimensions"])
    budget_mb = int(cfg.get("indexing", {}).get("scann_peak_budget_mb", _DEFAULT_SCANN_BUDGET_MB))
    shard_size = shard_size_from_budget(budget_mb, dimensions)
    # Manual rerank (variant C): build the AH index on a truncated
    # subspace + an index-level corpus_full.memmap; the searcher
    # reranks ScaNN's candidates against full-precision vectors at
    # query time. ~+70 ms per query, ~50% AH-index size, NDCG ≥
    # baseline. See SCANN_COMPACTION_2026-04-27.md. Off by default
    # via cfg; flip indexing.manual_rerank=true to enable. The
    # searcher auto-detects via manifest, so existing indexes keep
    # working when this is on (until the next reindex rebuilds).
    indexing_cfg = cfg.get("indexing", {})
    manual_rerank = bool(indexing_cfg.get("manual_rerank", False))
    ah_dim = int(indexing_cfg.get("manual_rerank_ah_dim", 1536))
    reorder_pool = int(indexing_cfg.get("manual_rerank_reorder_pool", 100))
    # `truncate_dim` is the V3-style knob: if set < dimensions, the
    # AH index is built on a truncated subspace and the searcher
    # truncates queries the same way before searching. No corpus_full,
    # no manual rerank step. Independent of (and weaker than) the
    # manual_rerank flag — both can technically be set, but
    # manual_rerank takes precedence and uses ah_dim instead of
    # truncate_dim. Default = no truncation (legacy baseline).
    truncate_dim_cfg = indexing_cfg.get("truncate_dim")
    truncate_dim = int(truncate_dim_cfg) if truncate_dim_cfg else None
    logger.info(
        "reindex: sharded build with shard_size=%d (budget=%d MiB, dims=%d, "
        "truncate_dim=%s manual_rerank=%s ah_dim=%d reorder_pool=%d)",
        shard_size,
        budget_mb,
        dimensions,
        truncate_dim,
        manual_rerank,
        ah_dim,
        reorder_pool,
    )
    # Delta build on the frequent (light) path: reuse sealed shards, rebuild
    # only the open tail — keeps the reindex memory peak to ~1 shard instead of
    # the whole corpus, and lets serve reload only the changed shard. Full
    # builds (light=False: manual reindex / post-backfill) act as compaction
    # and also refresh spell/aliases. build_index_delta auto-falls-back to a
    # full build on first run / config change / manual_rerank.
    delta = bool(indexing_cfg.get("delta_index", False)) and light
    _build = build_index_delta if delta else build_index_sharded
    written_to = _build(
        db_path=db_path,
        index_dir=index_dir,
        model=cfg["embedding"]["model"],
        dimensions=dimensions,
        shard_size=shard_size,
        truncate_dim=truncate_dim,
        manual_rerank=manual_rerank,
        ah_dim=ah_dim,
        reorder_pool=reorder_pool,
        user_id=uid,
    )
    logger.info("reindex: active ScaNN index for user %s now at %s", uid, written_to)
    rebuild_fts(db_path)
    rebuild_thread_summary(db_path, user_id=uid)

    if light:
        return

    # Heavy navigational rebuilds (light=False only — the manual `reindex`
    # path, NOT the per-cycle watch/backfill loops). Log peak RSS after
    # each so the fixed-budget guarantee is observable end to end.
    import resource as _resource

    def _peak_mib() -> int:
        return _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss // 1024

    rebuild_contact_frequency(db_path, user_id=uid)
    logger.info("reindex: after contact_frequency — peak RSS %d MiB", _peak_mib())
    rebuild_spell_dictionary(db_path, data_dir, user_id=uid)
    logger.info("reindex: after spell_dictionary — peak RSS %d MiB", _peak_mib())
    rebuild_topics(db_path, user_id=uid)
    logger.info("reindex: after topics — peak RSS %d MiB", _peak_mib())
    rebuild_term_aliases(db_path, data_dir=data_dir, user_id=uid)
    logger.info("reindex: after term_aliases — peak RSS %d MiB", _peak_mib())
    # Cached query embeddings can now point at stale term-alias
    # expansions; wipe so the next query re-expands + re-embeds.
    # query_cache is intentionally shared across users (cache hit =
    # saved embedding cost; the query embedding is opaque) so the
    # clear is global, not per-user.
    clear_query_cache(db_path)


def reindex_if_needed(
    db_path: Path,
    data_dir: Path,
    cfg: dict[str, Any],
    *,
    user_id: Optional[str] = None,
    min_new: int = 2000,
    max_age_s: int = 600,
) -> bool:
    """Reindex (light) only when a *quantum* of new embeddings has built up.

    Decouples indexing from the embed loop: the embedder just embeds; this
    decides when the ScaNN/FTS surfaces are stale enough to rebuild. Tracks
    the max embeddings.id folded into the live index in a per-user state file
    and rebuilds when either >= `min_new` new embeddings exist, OR any new
    exist and the index is older than `max_age_s` (freshness floor). Returns
    True if it rebuilt. Cheap to call on a short interval — the COUNT is
    index-served and it no-ops when there's nothing new.
    """
    import json
    import time

    from gmail_search.auth.write_user import resolve_write_user_id
    from gmail_search.store.db import get_connection

    conn = get_connection(db_path)
    try:
        uid = resolve_write_user_id(conn, user_id=user_id)
        cur_max = conn.execute("SELECT COALESCE(MAX(id), 0) FROM embeddings WHERE user_id = %s", (uid,)).fetchone()[0]
        state_path = Path(data_dir) / "users" / uid / "reindex_state.json"
        indexed_max, last_at = 0, 0.0
        if state_path.exists():
            try:
                s = json.loads(state_path.read_text())
                indexed_max = int(s.get("indexed_max_id", 0))
                last_at = float(s.get("last_reindex_at", 0.0))
            except Exception:
                indexed_max, last_at = 0, 0.0
        new = conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE user_id = %s AND id > %s",
            (uid, indexed_max),
        ).fetchone()[0]
    finally:
        conn.close()

    if new <= 0:
        return False
    age = time.time() - last_at
    if new < min_new and age < max_age_s:
        return False  # work exists but below the quantum — wait

    reindex(db_path, data_dir, cfg, light=True, user_id=uid)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"indexed_max_id": cur_max, "last_reindex_at": time.time()}))
    logger.info("reindex_if_needed: rebuilt %s (%d new embeddings, index age %.0fs)", uid, new, age)
    return True
