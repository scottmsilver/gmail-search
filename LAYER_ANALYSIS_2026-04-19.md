# gmail-search layer analysis — 2026-04-19

Snapshot of architecture, layering, and refactor priorities.

## TL;DR

Bones are solid. The domain decomposition (storage → Gmail client → pipeline → search → HTTP) is sensible and the dependency direction is clean — no backward-arrow violations. Main issues:

1. Two god objects (`store/db.py` at 1,129 lines, `cli.py` at 852 lines).
2. SQL filter-building duplicated between `server.py` and `search/engine.py`.
3. `pipeline.py` owns `reindex()` but not the ingest pipeline — so `update` and `watch` each inline their own "download → extract → embed → reindex" block.
4. `JobProgress.start_completed` has an untested semantic contract.

No architectural rot. Mostly "things that grew organically and want extraction."

## Layer diagram

```
┌─────────────────────────────────────────────────────────────┐
│                          PYTHON                              │
│                                                              │
│    CLI commands              FastAPI server                  │
│    (cli.py)                  (server.py)                     │
│         │                         │                          │
│         └──────────┬──────────────┘                          │
│                    ▼                                         │
│      ┌──────────────────────────┐                            │
│      │  orchestration helpers   │                            │
│      │  jobs  locks  reap       │                            │
│      │  pipeline (reindex only) │                            │
│      └─────┬────────────────────┘                            │
│            ▼                                                 │
│      ┌──────────────────┐      ┌──────────────────┐          │
│      │  search/         │      │  embed/          │          │
│      │  engine, parser  │      │  client, pipeline│          │
│      └─────┬────────────┘      └────┬─────────────┘          │
│            │                        │                        │
│            ▼                        ▼                        │
│      ┌──────────────────────────────────┐                    │
│      │  index/   builder, searcher      │                    │
│      └────┬─────────────────────────────┘                    │
│           │                                                  │
│           ▼                                                  │
│      ┌────────────────────────────────────┐                  │
│      │  store/                            │                  │
│      │  db.py + queries.py + models.py    │                  │
│      └────┬───────────────────────────────┘                  │
│           │                                                  │
│           ├── gmail/ (client, auth, parser) → Gmail API      │
│           ├── extract/ (pdf, docx, xlsx, …)  → attachments   │
│           └── embed client → Gemini API                      │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│                          WEB (Next.js)                       │
│                                                              │
│    app/(pages): /, /search, /settings                        │
│        │                                                     │
│        ▼                                                     │
│    components/(feature): SearchView, SettingsView, …         │
│        │                                                     │
│        ▼                                                     │
│    components/ui/ (Button, Card, Input, Label, Progress)     │
│                                                              │
│    app/api/(proxies): /api/search, /api/jobs/*, /api/chat    │
│        │                                                     │
│        ▼                                                     │
│    lib/backend.ts (typed fetch wrappers → Python server)     │
│    lib/config.ts + lib/utils.ts                              │
└─────────────────────────────────────────────────────────────┘
```

## God objects

| file | lines | concerns stuffed in |
|---|---|---|
| `src/gmail_search/store/db.py` | **1,129** | schema + init + WAL/busy_timeout + JobProgress + reap_stale_jobs + rebuild_thread_summary + rebuild_topics (clustering + k-means + label auto-naming) + rebuild_term_aliases (LLM validation + cooccurrence) + rebuild_spell_dictionary + rebuild_contact_frequency + rebuild_fts + clear_query_cache + TABLE_DOCS schema descriptions |
| `src/gmail_search/cli.py` | **852** | auth + download + sync + extract + embed + reindex + update (inlined 220-line pipeline) + watch (inlined cycle) + start/stop + reap + progress + status + cost + serve + search + summarize |
| `src/gmail_search/server.py` | **1,038** | SQL safety gate + SQL timeout runner + SQL WHERE builder + thread summary loader + snippet extractor + topic facet computer + 13+ route handlers (search, query, sql, jobs, threads, chat, conversations, …) |

`server.py` is heavy but cohesive (routes + their plumbing); leave until we actually want a split. `db.py` and `cli.py` are the ones that want breaking up.

## Concrete layering observations

### Clean boundaries (no changes needed)

- `store/db.py` doesn't import anything HTTP, subprocess, or Gmail-related. Good.
- `gmail/client.py` doesn't touch search/index. Good.
- `jobs.py`, `locks.py`, `reap.py`, `pipeline.py` are each single-purpose modules with narrow surface. Good.
- Web frontend: `components/ui/*` primitives only used by feature components; feature components only consumed by pages; pages hit Next proxy routes which forward to Python. Clean stack.
- `ScannSearcher` handles both sharded + legacy layouts transparently — server doesn't need to know which.

### Real duplication

1. **SQL filter building.** `server.py:_build_query_filters` (date/sender/subject/label) vs. `search/engine.py:search_threads` (builds its own date WHERE inline, twice). Both would benefit from a shared `QueryBuilder`.
2. **Per-batch ingest pipeline.** `cli.py update` and `cli.py watch` each inline the "download → extract → embed → reindex" sequence with subtle differences. Same smell as the old `reindex` triplication — one level up.
3. **Progress reporting styles.** CLI uses `progress.update(...)` + `click.echo(...)`. `embed/pipeline.py` uses `logger.info` + tqdm. Not critical but inconsistent.

### Leaky abstractions

- `JobProgress.start_completed` baseline: the contract ("rate = (completed - start_completed) / elapsed") lives only as a CLI comment + docstring. No test pins it. When someone refactors `update` or adds a new long-running job, nothing catches drift.
- `cli.py` manually builds some SQL (e.g. line ~153 `UPDATE attachments SET ...`) instead of going through `store/queries.py`. Not a layering violation per se (CLI can talk to DB), but the query logic belongs in one place.

### Non-issues (flagged by rote, but fine)

- `reap.py` depending on psutil — that's the right abstraction; no need to invent a process-inspection interface.
- `extract/*` modules all doing `except Exception as e: logger.warning(...)` — this is actually the right pattern for "extraction can fail per-file without blocking the batch." No refactor needed.
- `TABLE_DOCS` living in `db.py` — tight coupling by design; schema and its LLM-facing docs should drift together.

## Top 5 refactors (ordered by impact ÷ effort)

### 1. Unified `store/query_builder.py` — cheap + high clarity

Collapse `server.py:_build_query_filters` and `search/engine.py` inline filter logic into one `QueryBuilder` class. One helper, 3 callsites updated, ~100 lines total.

**Why first:** pure deduplication, no semantic change, easy to test, unlocks future "add a new filter param" without touching two modules.

### 2. Split `store/db.py`

Target split:
- `store/schema.py` — `SCHEMA`, `TABLE_DOCS`, `init_db`, `get_connection`, `describe_schema_for_llm`, `assert_table_docs_cover_schema`
- `store/job_progress.py` — `JobProgress`, `reap_stale_jobs`
- `store/index_rebuilders.py` — `rebuild_topics`, `rebuild_term_aliases`, `rebuild_spell_dictionary`, `rebuild_contact_frequency`, `rebuild_fts`, `rebuild_thread_summary`, `clear_query_cache` + their helpers (bisect, cluster_coherence, etc.)

Pure motion, no logic change. `db.py` shrinks from 1,129 → ~300 lines.

### 3. Grow `pipeline.py` to own ingest orchestration

Add `run_ingest_batch(db_path, data_dir, cfg, *, service, max_messages=None) -> IngestReport` that runs download → extract → embed. Both `update` and `watch` call it; the "reindex at end of cycle" stays where it is.

Follows the same pattern that fixed the `reindex` mess. ~80 lines extracted into one place, two CLI callsites shrink to a few lines each.

### 4. Pin the `JobProgress.start_completed` contract with a test

~15 lines in `tests/test_store_db.py`:

```python
def test_job_progress_baseline_preserved_across_updates(tmp_path):
    # after JobProgress(start_completed=N), subsequent update() calls
    # must NOT overwrite start_completed.
```

Cheap, stops quiet drift.

### 5. Split `cli.py` into `src/gmail_search/commands/`

Separate files: `auth.py`, `ingest.py` (download/sync/update), `watch.py`, `admin.py` (reap/progress/status/cost), `search_cli.py`, `serve.py`. `cli.py` becomes a thin dispatcher that imports from `commands/*`.

Bigger change, but makes future command additions localized. Bundle with (3) to keep `update` and `watch` tidy after their pipeline extraction.

## Suggested commit grouping

- **Commit A: SQL builder + pipeline.run_ingest_batch** — (1) and (3) together, both medium-small, one coherent theme ("consolidate the N-places-doing-the-same-thing patterns").
- **Commit B: db.py split** — (2) alone, pure file motion.
- **Commit C: test + cli split** — (4) and (5), "test + reorganize."

Each commit shrinks god objects and eliminates a duplication class. No behavior change in any of them.

## What NOT to touch

- `search/engine.py` ranking weights and LLM reranker — cohesive even though large.
- `gmail/client.py` — the Gmail API interaction is intrinsically complex (batching, retries, history IDs); splitting would just add indirection.
- `extract/*` dispatchers — they're already one-file-per-type. The `except Exception: log warning; continue` pattern is correct for per-file extraction failures.
- Theme system in `globals.css` — the `--bg-*/--fg-*` tokens plus the Tailwind mapping are working; don't churn.

## Test coverage snapshot

| area | coverage |
|---|---|
| `index/*` | good — 19 tests |
| `search/parser` | good — 25 tests |
| `search/engine` | partial — 5 tests, no reranker/off-topic filter tests |
| `store/db` (JobProgress, reap) | partial — 6 tests |
| `gmail/*` | weak — only client safety, no parser/sync tests |
| `embed/*` | partial — pipeline + client smoke |
| `cli.py` | none |
| `server.py` routes | SQL endpoint + schema docs only; no search/jobs/chat route tests |
| `locks.py` | good — 4 tests |
| `reap.py` | good — 7 tests |
| `pipeline.py` | good — 2 tests |
| web | none (test-units.mjs is broken) |

Biggest gaps: CLI commands and web frontend. Neither is trivially testable in this session but worth knowing.
