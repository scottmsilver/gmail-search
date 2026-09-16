# Run facts service

**Goal:** Restore owner-bound hybrid fact retrieval without database copies,
runtime DDL, bootstrap identity, or false claims of exhaustive mailbox coverage.

**Architecture:** `RunFactsService` authorizes `retrieval` / `facts.find`, embeds
the original query through an injected budgeted adapter, and uses one immutable
owner reader snapshot for lexical candidates, streamed vectors and final
hydration. Scalar ranks are bounded; a corpus embedding matrix is never built.

**Tech stack:** Python asyncio, NumPy scalar-vector scoring, existing capability
registry and fixed PostgreSQL search reader. The private HTTP route is optional; no public route is mounted.

## Approved implementation sequence

- [x] Write failing synthetic contract tests in `tests/test_gateway_facts_service.py`.
- [x] Implement fixed arguments, capability lifecycle and immutable owner identity
  in `src/gmail_search/gateway/facts_service.py`.
- [x] Preserve lexical stopwords/singular expansion, cosine scoring, reciprocal
  rank fusion (`1/(60+rank)`), owner boost (`0.02`) and exact-text deduplication.
- [x] Test more than 8,000 facts with a semantic match on the last page and no
  lexical anchor. Stream at most 128 vectors per page and 200,000 scalar records.
- [x] Test malformed vectors/cursors, explicit database/vector/output limits,
  cancellation, revocation and repeated cleanup cancellation.
- [x] Qualify the adapter against the dedicated synthetic partitioned PostgreSQL
  fixture, then request independent review.

## Fixed contract and limits

`find_facts(token, query=..., exhaustive=True, k=200)` accepts 1–1,000 query
characters and `k` from 1–500. Ownership comes from the run capability. Trusted
configuration supplies the owner identity and embedding model/dimensions.
Every operation shares one 30-second deadline. Native/threaded work and provider
cleanup must finish before cancellation is acknowledged. After all owned work
and watcher cleanup finishes, the service performs one fresh authorization check
before returning; it performs no further await after that check. The trusted
embedder must honor the shared deadline and drain its own resources on
cancellation. Owner configuration is copied into a read-only mapping of frozen
identity records; the reader and embedder model and dimensions must agree.

Exhaustive candidate selection means cosine at least 0.5 or a lexical match.
Non-exhaustive selection takes the top 20 semantic and top 20 lexical facts.
Result entries retain `fact`, `message_id`, `thread_id`, `cosine`, `bm25`, and
`owner`. Near-duplicate clustering remains disabled, matching the existing
default. Identical text is deduplicated after ranking; ties use fact IDs.

Coverage distinguishes the extracted fact corpus from mailbox extraction
coverage, which this reader cannot establish. Missing/invalid/wrong-model
vectors, lexical caps, transfer caps and result limits remain explicit. An
incomplete scan with no usable result raises a fixed error rather than returning
an apparently successful empty search. Final hydration uses the same snapshot.

The reader's 128 MiB session budget includes JSON and hex vector encoding.
At 3,072 dimensions it can truncate before 8,000 facts; small-dimensional tests
do not qualify full traversal of a large production corpus. Rehydration shares
that budget and may also require explicit incomplete-search refusal. Output,
including coverage, is bounded to 4 MiB. Atomic fact text is never clipped.

The `coverage` object includes `corpus_count`, `examined`,
`selection_complete`, `mailbox_extraction_complete` (always false here),
`exhaustive_requested`, and sorted `reasons`. Complete selection refers to the
requested candidate algorithm, not all facts ever present in a mailbox. Reasons
are closed constants: vector/model gaps, unavailable facts, row/byte/session
budgets, and service vector/result/response limits. Unrecognized reader metadata
raises a fixed error and is never reflected into coverage. A `result_limit`
reason is added only after finding another distinct fact beyond `k`.

## Verification

The focused suite passes **35 tests**, including two real partitioned
PostgreSQL/ParadeDB owner cases. Alice and Bob share numeric fact ID 1; each
receives only their own fact, and the opposite owner's lexical anchor produces
an empty complete selection. The same injected `DataAdmission` is held by the
reader session and is released afterward. Embeddings are synthetic.

Synthetic regressions cover an 8,001-fact corpus with only the final vector
matching and no lexical anchors; stable ties, owner boost and exact deduplication;
invalid vectors, model mismatch and stale lexical identities; usable hydration
continuation; scan and hydration budget exhaustion; complete empty selections;
4 MiB output including coverage; repeated provider/ranking cancellation; and
revocation during scanning and final watcher cleanup. The cleanup-race and
closed-diagnostic regressions failed before their fixes. Ruff also passes.

```bash
GMS_TEST_PG_DSN="$SYNTHETIC_PARADEDB_DSN" PYTHONPATH=src \
  /home/ssilver/development/gmail-search/.venv/bin/python -m pytest -q \
  tests/test_gateway_facts_service.py
```

Use only the dedicated synthetic fixture DSN. The fixture creates and removes
its own randomly named database and roles. No production mailbox, actual
provider, HTTP route, or full end-to-end agent workflow was exercised here.

## Private transport integration

`POST /v1/find-facts` is installed only by explicit `facts=` injection into the
worker gateway. It authorizes before bounded JSON reads and accepts only query,
exhaustive and k. The guest exposes the single-query `find_facts` contract over
the retrieval capability; the relay permits only the exact route without query
parameters. MCP and CLI retain their existing shared socket, response and batch
budgets. The current VM images predate this tool and require new qualification.

The HTTP helper now drains owned disconnect cleanup before a fresh permission
and absolute deadline check. Four facts/search regressions demonstrate denial
when access is revoked or the deadline expires during that cleanup. This bounds
owned cleanup races; it cannot recall response bytes already transmitted.
