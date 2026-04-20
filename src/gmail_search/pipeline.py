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
from typing import Any

from gmail_search.index.builder import build_index_sharded, shard_size_from_budget
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


def reindex(db_path: Path, data_dir: Path, cfg: dict[str, Any], *, light: bool = False) -> None:
    """Rebuild the on-disk surfaces that back /api/search.

    light=True  → hot-path rebuilds only (ScaNN + FTS + thread summary).
                  What the watch cycle runs between syncs.
    light=False → adds the slower rebuilds (contact, spell, topics,
                  aliases) and wipes the query embedding cache.
                  What the `reindex` CLI + post-backfill path run.
    """
    # `index_dir` is the canonical PREFIX. `build_index_sharded` writes
    # a timestamped sibling and flips the DB pointer row; readers
    # resolve through `resolve_active_index_dir` so a mid-reindex
    # query always lands on a fully-written build.
    index_dir = Path(data_dir) / "scann_index"
    dimensions = int(cfg["embedding"]["dimensions"])
    budget_mb = int(cfg.get("indexing", {}).get("scann_peak_budget_mb", _DEFAULT_SCANN_BUDGET_MB))
    shard_size = shard_size_from_budget(budget_mb, dimensions)
    logger.info(
        "reindex: sharded build with shard_size=%d (budget=%d MiB, dims=%d)",
        shard_size,
        budget_mb,
        dimensions,
    )
    written_to = build_index_sharded(
        db_path=db_path,
        index_dir=index_dir,
        model=cfg["embedding"]["model"],
        dimensions=dimensions,
        shard_size=shard_size,
    )
    logger.info("reindex: active ScaNN index now at %s", written_to)
    rebuild_fts(db_path)
    rebuild_thread_summary(db_path)

    if light:
        return

    rebuild_contact_frequency(db_path)
    rebuild_spell_dictionary(db_path, data_dir)
    rebuild_topics(db_path)
    rebuild_term_aliases(db_path)
    # Cached query embeddings can now point at stale term-alias
    # expansions; wipe so the next query re-expands + re-embeds.
    clear_query_cache(db_path)
