# Ranked search and facts through the run gateway

Status: repository inspection and proposed integration sequence, 2026-09-15.
No search gateway, new reader grants, DDL, provider calls, or database copies
were implemented during this analysis. This plan implements the existing
approved full-agent specification; production qualification remains outstanding.

## Architecture and existing behavior

Keep `gateway/schema.py:ANALYTICAL_SCHEMA` and guest analytical SQL unchanged.
Add typed retrieval operations backed by fixed internal queries and immutable
owner-specific database logins. Reuse existing owner-specific ScaNN indexes on
trusted infrastructure; never build a mailbox/index copy per run or expose
index paths and embedding matrices to the guest.

The relevant existing implementations are:

- `agents/tools.py:search_emails`, `search_emails_batch`, and `find_facts`:
  model-facing arguments, batch envelopes, and citation references.
- `server.py:api_search`, `_format_thread_result`, `_format_thread_ref`:
  response shaping, detail levels, per-thread match caps, and pending-index state.
- `search/parser.py:parse_query`: Gmail-style structured filters and temporal intent.
- `search/engine.py:SearchEngine.search_threads`: vector/BM25 candidate merge,
  thread scoring, repeat-sender collapse, off-topic filtering, and optional reranking.
- `store/queries.py:_sanitize_fts_tokens`, `_build_bm25_query`,
  `_search_fts_postgres`: fixed-field phrase/disjunction searches over messages
  and attachments, phrase boost, and score normalization.
- `propositions.py:_query_terms`, `_singularize`, `find_facts`: fact retrieval,
  cosine scoring, reciprocal-rank fusion, and duplicate handling.
- `index/searcher.py:ScannSearcher`: existing owner-index search and rerank data.

Do not call the existing application endpoints or instantiate `SearchEngine`
unchanged. They open legacy application connections; query embeddings read/write
an unscoped persistent query cache and cost ledger; fact requests invoke table
and index creation; search can fall back to a global spell dictionary. The
search topic helpers also required explicit owner-qualified joins and labels;
that narrow bug fix is separate from implementing this search gateway.

## Typed operations

Proposed fixed HTTP routes, authenticating before bounded body reads:

| Route | Audience / operation | Arguments |
| --- | --- | --- |
| `POST /v1/search` | `retrieval` / `search` | `query`, `date_from?`, `date_to?`, `top_k?`, `detail?`, `max_matches?` |
| `POST /v1/find-facts` | `retrieval` / `facts.find` | `query`, `exhaustive?`, `k?` |

No request accepts owner, session, database role, index path, model, SQL fragment,
upstream URL, or provider configuration. The run capability determines ownership.
The operation names above are proposed additions; they are not mounted today.

Preserve these existing model-facing contracts:

- `search_emails_batch` returns ordered `{results: [{input, result}]}` with
  independent per-item errors. Each search result retains thread IDs/citations,
  scores, matches, and requested detail. Defaults are `top_k=10`,
  `detail="snippet"`, `max_matches=3`; details are refs/snippet/summary/full.
- `find_facts` returns `{facts: [...]}` with `fact`, `message_id`, `thread_id`,
  `cosine`, `bm25`, and `owner`; defaults are `exhaustive=True`, `k=200`.
- Existing endpoint limits are query length 1–1000 characters, up to 100 search
  threads and up to 500 facts. Keep the guest adapter's explicit 20-item batch
  limit and two active requests per instance.
- Preserve `pending_index` for a genuinely unbuilt owner index. A query/index/
  embedding failure is a fixed sanitized error, not an apparently successful
  empty result. Never log private queries, tokens, or database diagnostics.
- Add explicit coverage/truncation metadata where existing code only logs lost
  recall. A limited search sample is not a whole-mailbox count. For `full`
  detail, do not silently relabel clipped bodies as complete originals.

## Internal reader and query contract

Prefer a separate immutable owner-bound retrieval reader using the existing
`gateway/provision.py` and `gateway/database.py` identity/resource patterns.
The analytical reader currently grants only `ANALYTICAL_SCHEMA` columns and
`verify_reader_access` rejects any extras. Do not weaken that existing contract.

Required internal columns beyond the analytical schema:

| Relation | Extra columns |
| --- | --- |
| `messages` | `search_id` |
| `embeddings` | `user_id,id,message_id,attachment_id,chunk_type,chunk_text,embedding,model` |
| `propositions` | `embedding` |

Existing analytical columns supply thread summaries, message summaries, topic
metadata, aliases, contacts, and attachment text. Grant no attachment filesystem
paths, user administration, OAuth data, query-cache tables, sequences, writes,
TEMP/CREATE privileges, or security-definer execution. Use literal owner-bound
permissive and restrictive RLS policies; no privileged login followed by SET ROLE.

Define a closed internal operation set rather than a general raw-query method:

1. Structured candidate IDs: bound sender/recipient/subject/date predicates,
   plus owner-qualified attachment existence.
2. Message lexical candidates: select Gmail ID and
   `paradedb.score(search_id)` using the installed qualified `@@@` operator
   against `messages.search_id` and a bound, fixed-field lexical query.
3. Attachment lexical candidates: select `message_id` and
   `paradedb.score(id)` against attachment IDs, with owner/candidate predicates.
4. Fact lexical candidates: select proposition ID from the owner-scoped fixed
   text query, ordered by score, with a bound candidate limit.
5. Embedding candidate hydration: explicit columns and owner-qualified joins
   from embeddings to messages and attachments. Revalidate every returned ID.
6. Thread/detail hydration: selected columns from thread summaries, messages,
   message summaries, and owner-qualified topic joins.
7. Fact vector hydration: owner count and bounded selected proposition rows.
8. Restricted vector hydration: stream vectors in bounded blocks and retain
   the required top candidates, rather than materializing an unbounded matrix.

Pin the installed `@@@` operator implementation and applicable
`paradedb.score` overload through trusted extension/catalog verification.
Grant schema usage and only required EXECUTE overloads; verify invoker semantics
and extension identity. The repository does not establish the exact installed
function signatures/OIDs. Resolve and qualify them on the pinned disposable
PostgreSQL/pg_search fixture before finalizing grants. Never grant every function
in the extension schema or introduce a security-definer bypass.

Keep existing connection, read-only transaction, statement/lock timeout,
work_mem/temp-file, admission, and cancellation guarantees. Internal vector
queries need explicit block/byte bounds distinct from guest JSON result limits.
Use one absolute operation deadline across all phases, shared owner/global
admission across retrieval and SQL work, and teardown acknowledgement before
releasing admission. The initial profile can use a 30-second total deadline,
existing 10-second statement limit, and a 4 MiB guest JSON response ceiling;
qualify these together with the relay/guest limits before publishing the tools.

## Ranking, embeddings, and owner indexes

Preserve the current algorithm rather than equating keyword search with parity:

- Spell correction and aliases must come only from the authenticated owner's
  trusted resources, with no global-owner fallback.
- Structured filters precede candidate selection. Keep original plus expanded
  lexical passes, message and attachment matches, phrase boost, normalization,
  temporal intent, thread grouping, and the existing thread-score blend.
- Preserve repeat-sender collapse, detail shaping, and off-topic handling.
- Use the existing owner ScaNN index, selected by a trusted owner/generation
  registry. Bind model, dimensions, owner and path, and retain old generations
  until in-flight searches drain. A foreign/stale candidate must be rejected
  by the owner reader before publication.
- Current configuration uses `gemini-embedding-2-preview`, 3072 dimensions and
  `RETRIEVAL_QUERY`. This is a separate provider operation from streamed text
  inference. Add a fixed internal embedding adapter with run budget reservation,
  bounded text, deadline/cancellation and trusted usage settlement; no raw-vector
  or arbitrary embedding endpoint is offered to the guest.
- `SearchEngine._llm_rerank` uses `gemini-3.1-flash-lite-preview` when top scores
  cluster. Full parity needs a qualified fixed internal reranker and budget
  accounting. A deliberately disabled reranker is an explicit profile difference.
- Inject the authenticated owner's trusted identity for fact ownership boosts;
  current `find_facts` calls the bootstrap environment's `owner_string()`.
- Use run/owner-scoped bounded caches if needed; do not reuse global persistent
  query-cache writes or put provider cost writes through a read-only mail login.

## Recall and shared BM25 qualification

Existing code has these limitations that the adapter must not conceal:

- Structured candidate IDs truncate above 100,000 messages.
- Filtered vector searches above the small-corpus threshold use ANN overfetch
  and post-filtering, so recall is approximate.
- `find_facts` loads all fact embeddings only below 8,000 facts. Above that it
  reranks only the BM25 candidate pool (`cap * 4`); without a lexical anchor it
  returns no results even with `exhaustive=True`.
- Index freshness, missing vectors, fact caps and response limits further bound
  coverage. Existing empty/truncated behavior does not prove exhaustive recall.

`messages_bm25_idx` now uses globally unique `search_id`; messages retain
owner-qualified Gmail IDs. Attachment and proposition search keys are already
global numeric IDs. The current index definitions do not include owner fields.
See `deploy/public/OWNER_KEY_MIGRATION.md` and the derived-owner migration for
identity/FK prerequisites; do not change keys or duplicate shared mailbox rows.

RLS row visibility and extension ranking are separate qualification questions.
Verify the pinned extension does not truncate globally before the owner filter
and returns no foreign row metadata, count, snippet, or diagnostic. PostgreSQL
policies ordinarily filter rows before non-leakproof user expressions, but this
is not proof of a particular extension planner's behavior.
[PostgreSQL 16 RLS documentation](https://www.postgresql.org/docs/16/ddl-rowsecurity.html)

BM25 includes corpus-frequency and length statistics. A shared index can
therefore make scores/order depend on other owners' documents: this is a risk
requiring explicit testing, not a claim that the deployed extension leaks rows
or provides owner-local statistics. Omitting raw scores alone would not remove
rank-order dependence.
[ParadeDB's BM25 explanation](https://www.paradedb.com/learn/search-concepts/bm25)

## Implementation sequence and acceptance

1. Add the fixed-query retrieval reader/executor and tests using the previously
   qualified disposable pg_search version. Verify privileges, operator/function
   identity, resource settings, and owner-qualified search plans.
2. Add `RunSearchService` with injected database operations, query embedder,
   owner ScaNN index, and optional reranker. Extract/reuse pure ranking and
   formatting logic from the existing modules; do not import their connection,
   DDL, bootstrap identity, or persistent-cache behavior.
3. Add search/facts output formatting, citations, and explicit coverage metadata.
4. Mount exact routes, extend the closed relay path list, and add guest
   dispatch/MCP registrations using the existing tool names and batch shapes.
5. Qualify actual embedding/rerank transport and the complete CLI workflow before
   describing ranked/hybrid search as enabled or feature-parity complete.

Synthetic tests must cover colliding message/topic IDs, foreign IDs and terms,
foreign-heavy Top-K results, score/order changes after foreign-corpus changes,
owner-scoped aliases/dictionaries, missing/stale indexes, more than 8,000 facts,
pure-semantic facts without lexical anchors, narrow date/sender filters,
malformed lexical input, result/body/candidate caps, cancellation/revocation in
each phase, and budget settlement for failed embedding/rerank operations.

The smallest useful first search implementation is a dependency-injected
`RunSearchService.search` preserving the existing thread-ranking and output
algorithm against synthetic fixed-query/vector/embedding providers, paired
with the real fixed-reader BM25 qualification. Follow it with the facts adapter.
A keyword-only endpoint or structured-filter wrapper is useful separately but
must not be labeled full ranked/hybrid search parity.

## Approved physical partition decision (2026-09-15)

The user selected LIST(user_id) partitions in the shared database to preserve native BM25 while isolating corpus statistics. Implementation and recovery gates are tracked in [the owner partition plan](superpowers/plans/2026-09-15-owner-partitioned-search.md). This decision covers messages, attachments, and propositions; it does not mean the migration or search gateway is deployed.

## Execution update after partition qualification

Physical user partitions, numeric owner-qualified keys, guarded provisioning,
invitation readiness and migration rehearsal now pass independent review. The
function identities and four EXECUTE grants are recorded in
`gateway-ranked-search-qualification.md`; that evidence supersedes the earlier
open signature question above. The current numeric-key profile uses
`(user_id,search_id)` for messages and `(user_id,id)` for attachments/facts.
No global numeric uniqueness is assumed.

Next implementation slice: immutable **internal search reader**. Create
`gateway/search_reader.py`, `gateway/provision_search_reader.py`, and
`gateway/search_queries.py`; test in `tests/test_gateway_search_reader.py` and
`tests/test_gateway_search_queries.py`. Keep analytical credentials, grants and
compiler unchanged. The new role is only available to trusted fixed operations;
neither its DSN nor a raw SQL method is exposed to guests.

- [ ] Define a separate deterministic owner login, credential/registry binding,
  exact columns and extension grants, RLS and privilege audit. Require qualified
  owner partitions before installing this reader. Include owner RLS on embeddings
  through reviewed administrator provisioning, never at query time.
- [ ] Define closed fixed-query operations for structured candidates, three
  lexical branches, embedding/message/thread hydration, owner aliases/contacts,
  and bounded fact vector pages. Return explicit completion metadata for every
  capped selection; never silently truncate an exhaustive claim.
- [ ] Use real colliding-owner fixture rows and direct login tests; reject writes,
  sequence/default grants, raw paths, foreign IDs, role/session spoofing and
  unexpected index/function definitions. Exercise exact custom/generic native
  ranking through this reader.
- [ ] Bound parameters, text, vector bytes and page sizes before database work.
  Drain cancellation and connection cleanup before admission release. Avoid
  returning private database diagnostics.
- [ ] Independently review reader and fixed-query contracts before integrating
  RunSearchService. Then implement semantic/vector/lexical orchestration and
  facts against those interfaces with injected budgeted embedding/reranking.

Owner-index work can proceed independently in `gateway/search_index.py` and
`tests/test_gateway_search_index.py`: explicit trusted owner/generation/model/
dimension bindings, no fallback to bootstrap/global paths, generation retention
until in-flight work drains, bounded candidate results, stale/foreign candidate
revalidation by the owner reader. Public handlers accept no index paths or
vectors. Do not import `resolve_active_index_dir` fallback behavior.

All work remains in the isolated candidate, with synthetic credentials, corpus
and provider responses until the real provider/release gates are satisfied.

### Search implementation evidence (candidate only)

The shared `DataAdmission` now supports SQL and internal search capacity under
one process-local quota. `QueryGateway` retains its lease through repeated
cancellation until backend cleanup finishes. Independent review passed 30
focused tests, including real synthetic PostgreSQL cancellation. The public
composition must inject the same admission object into both readers; these
limits do not provide a cross-process quota.

Existing pure ranking signals, result types, sender collapse and off-topic
filtering moved to `search/ranking.py`, preserving legacy engine reexports.
Independent executable-AST comparison found no behavior change. The combined
legacy search, filter, parser and extraction suite passed 41 tests with the
synthetic ParadeDB fixture, without skips.

`gateway/search_vectors.py` streams up to 128 owner vectors per page, retains
only the best candidates, enforces a 200,000-vector ceiling and reports missing
vectors, model mismatches and database caps as coverage gaps. Review corrected
an initial normalization difference: the existing restricted search uses raw
float32 dot products, despite its cosine docstring. The new helper preserves
that behavior; the existing ANN manual reranker normalizes vectors, so these
paths must not be described as uniformly cosine for non-unit data. Zero or
invalid vectors are omitted with explicit incomplete coverage. Nested query
vectors are rejected before NumPy allocation. Eighteen focused tests pass.

`gateway/search_ranking.py` merges owner-revalidated semantic and lexical rows
using the existing thread blend. Unhydrated index IDs are excluded before score
normalization and recorded as coverage gaps. Missing thread summaries likewise
remain explicit. Winning attachment chunks update their matching filename.
Thirteen focused tests pass; independent review and full service integration
remain pending. The fixed search reader and strict native index adapter are
being qualified separately. No search route or provider adapter is mounted.

Run-service orchestration tests may proceed against injected synthetic readers,
indexes and providers while the concrete reader/index reviews finish. Connecting
the actual reader to this service remains gated on those reviews. The service
will derive ownership only from the retrieval capability, pin the index before
embedding, close its preliminary alias/contact snapshot before provider work,
and use one bounded main snapshot. Trusted `OwnerSearchContext` supplies
canonical owner emails and an optional explicitly owner-bound spell corrector;
absence is an explicit coverage/profile difference, never a global fallback.
Embedding and reranking adapters own budget reservation/settlement and receive
the verified run lease plus the same absolute deadline and revocation check.
No synthetic adapter is mounted in the public application.

### Reviewed run search and actual database integration

The immutable fixed reader now passes independent review and 32 tests. Review
closed whole-provision DDL timeout and grant-option delegation gaps. Exact native
generic plans require an owner literal derived with psycopg.sql.Literal from the
immutable credential; lexical text and other query values remain parameters.
This is an extension planner limitation, not caller-selectable SQL.

`RunSearchService` passed independent review and 26 synthetic orchestration tests.
It preserves the 10,000-candidate overfetch for large filtered ANN, trims unused
rows before thread hydration, batches large lexical unions, and checks reranker
outputs as an exact permutation of supplied threads. It pins an owner generation
before embedding and closes preliminary context snapshots before provider work.
The shared database quota is held through snapshot teardown. Ranked output has
explicit approximate/capped/body coverage, citations and owner-bound topic
facets counted by thread. A disabled reranker or absent owner spell resource is
an explicit profile difference; this is not full production search parity.

The optional private-worker `/v1/search` route authenticates before reading its
bounded body and accepts only the six documented search fields. HTTP and actual
integration passed independent review: 11 ASGI tests and three disposable
partitioned-PostgreSQL/native-ScaNN tests. The latter use both owners with colliding
mail IDs and an intentionally polluted native candidate index whose highest hit
belongs to the other owner. That hit does not influence returned owner results.
A live search snapshot also blocks SQL admission through the same capacity
object. All provider responses in these tests remain synthetic.

Current Google documentation lists the configured preview embedding past its
scheduled shutdown date and identifies `gemini-embedding-2` as its replacement.
The new internal adapter is explicitly for that supported stable model, at 3072
dimensions with the documented retrieval prompt; it does not relabel existing
preview-model vectors. Reviewed official sources do not establish preview/GA
vector-space equivalence. Qualify compatibility and cost before deciding on any
re-embedding. See `gateway-search-embedding.md`; budget/transport cancellation
review is still in progress. No live index, provider configuration or service
has changed.
