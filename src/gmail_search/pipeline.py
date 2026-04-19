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

from gmail_search.index.builder import build_index
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


def reindex(db_path: Path, data_dir: Path, cfg: dict[str, Any], *, light: bool = False) -> None:
    """Rebuild the on-disk surfaces that back /api/search.

    light=True  → hot-path rebuilds only (ScaNN + FTS + thread summary).
                  What the watch cycle runs between syncs.
    light=False → adds the slower rebuilds (contact, spell, topics,
                  aliases) and wipes the query embedding cache.
                  What the `reindex` CLI + post-backfill path run.
    """
    index_dir = Path(data_dir) / "scann_index"
    build_index(
        db_path=db_path,
        index_dir=index_dir,
        model=cfg["embedding"]["model"],
        dimensions=cfg["embedding"]["dimensions"],
    )
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
