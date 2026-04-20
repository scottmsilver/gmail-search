"""Tests for the reindex orchestrator.

Three call-sites used to duplicate different subsets of the rebuild
sequence. We now have one function; tests pin which steps run in
`light` vs full mode, using mocks so we don't have to set up a real
corpus + ScaNN build.
"""

from __future__ import annotations

from unittest.mock import patch


def test_reindex_full_runs_every_step(tmp_path):
    """Full reindex = everything downstream of an embedding change:
    ScaNN + FTS + thread_summary + contact_frequency + spell + topics
    + term_aliases + clear_query_cache.
    """
    from unittest.mock import DEFAULT  # inline — formatter strips unused top-level imports

    from gmail_search.pipeline import reindex

    cfg = {"embedding": {"model": "test-model", "dimensions": 16}}
    data_dir = tmp_path
    db_path = tmp_path / "db.sqlite"
    db_path.touch()

    targets = [
        "gmail_search.pipeline.build_index_sharded",
        "gmail_search.pipeline.rebuild_fts",
        "gmail_search.pipeline.rebuild_thread_summary",
        "gmail_search.pipeline.rebuild_contact_frequency",
        "gmail_search.pipeline.rebuild_spell_dictionary",
        "gmail_search.pipeline.rebuild_topics",
        "gmail_search.pipeline.rebuild_term_aliases",
        "gmail_search.pipeline.clear_query_cache",
    ]
    with patch.multiple(
        "gmail_search.pipeline",
        build_index_sharded=DEFAULT,
        rebuild_fts=DEFAULT,
        rebuild_thread_summary=DEFAULT,
        rebuild_contact_frequency=DEFAULT,
        rebuild_spell_dictionary=DEFAULT,
        rebuild_topics=DEFAULT,
        rebuild_term_aliases=DEFAULT,
        clear_query_cache=DEFAULT,
    ) as m:
        reindex(db_path, data_dir, cfg, light=False)
    for t in targets:
        name = t.rsplit(".", 1)[-1]
        assert m[name].called, f"{name} was not called in full reindex"


def test_reindex_light_skips_heavy_rebuilds(tmp_path):
    """The watch loop runs this between cycles — it must not trigger
    the minutes-long spell / topics / aliases rebuilds or the query
    cache wipe.
    """
    from unittest.mock import DEFAULT  # inline — formatter strips unused top-level imports

    from gmail_search.pipeline import reindex

    cfg = {"embedding": {"model": "test-model", "dimensions": 16}}
    data_dir = tmp_path
    db_path = tmp_path / "db.sqlite"
    db_path.touch()

    with patch.multiple(
        "gmail_search.pipeline",
        build_index_sharded=DEFAULT,
        rebuild_fts=DEFAULT,
        rebuild_thread_summary=DEFAULT,
        rebuild_contact_frequency=DEFAULT,
        rebuild_spell_dictionary=DEFAULT,
        rebuild_topics=DEFAULT,
        rebuild_term_aliases=DEFAULT,
        clear_query_cache=DEFAULT,
    ) as m:
        reindex(db_path, data_dir, cfg, light=True)

    # These three must run on every cycle — they're how the searcher
    # sees new messages.
    assert m["build_index_sharded"].called
    assert m["rebuild_fts"].called
    assert m["rebuild_thread_summary"].called

    # Everything else is too slow for the hot path.
    for skipped in (
        "rebuild_contact_frequency",
        "rebuild_spell_dictionary",
        "rebuild_topics",
        "rebuild_term_aliases",
        "clear_query_cache",
    ):
        assert not m[skipped].called, f"{skipped} should be skipped in light mode"
