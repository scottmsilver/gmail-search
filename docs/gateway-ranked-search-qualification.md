# Ranked search: disposable PostgreSQL qualification

2026-09-15. **The shared BM25 index passed the tested row-isolation and recall
cases, but its scores and result order depend on other owners' documents.**
It must not be described as owner-independent ranking. Two alternatives were
also exercised: native BM25 over physical owner partitions, and document-local
PostgreSQL ranking over Boolean BM25 candidates. The user selected physical owner partitions on 2026-09-15. Product
implementation and migration/capacity qualification are tracked in
[the owner partition plan](superpowers/plans/2026-09-15-owner-partitioned-search.md).

## Reproduction and scope

Probe: [`deploy/public/qualify_ranked_search.py`](../deploy/public/qualify_ranked_search.py).
It requires `GMS_RANKED_PROBE_DSN` for the explicitly approved disposable
`127.0.0.1:55440/postgres` administrator, refuses other host/port/database/user
targets, creates a random database from `template0` and two random login roles,
and removes only those objects in `finally`. It prints synthetic observations,
exact queries, full JSON plans, assertions, and cleanup status. It never imports
the app, reads mailbox data, modifies existing readers, or prints passwords.

With that environment variable already supplied:

```sh
/home/ssilver/development/gmail-search/.venv/bin/python \
  deploy/public/qualify_ranked_search.py > /tmp/gms-ranked-search-report.json
```

Final probe completed successfully, including all assertions and owned-object
cleanup. Existing container and databases were left in place. Pinned environment:

- PostgreSQL **16.13**, Debian package `16.13-1.pgdg13+1`, x86_64.
- `pg_search` **0.23.0**; the probe asserts this version.
- Disposable container `gms-owner-parade-qualification`, published only at
  `127.0.0.1:55440`, 2 CPUs / 2 GiB.
- Image ID `sha256:99f182f93387c2226c810763f736ec06154090b37d654a09483ca44d68b5ba6c`.
  Container tag `paradedb/paradedb:latest-pg16` is not the version pin.

These are small functional experiments, not production latency, capacity,
side-channel, migration, all-query-shapes, or full search-feature certification.
Timing and segment identifiers in regenerated plans can vary.

## Fixture and fixed query

Synthetic `messages` has a unique numeric `search_id`, owner-qualified Gmail
identity `UNIQUE(user_id,id)`, subject/body text, and an ungranted sentinel column.
Alice and Bob reuse Gmail IDs. The shared BM25 index contains only
`(search_id,body_text)`, so ownership is not an indexed field.

RLS is enabled and forced, with a deliberately permissive legacy policy and a
literal restrictive policy for each login, e.g. `USING(user_id='alice')`.
Both readers authenticate directly; `session_user=current_user` is checked.
Neither owns objects or has superuser, create-role, create-database, replication,
or bypass-RLS privileges. No administrator connection followed by `SET ROLE` is
used for a reader query.

The tested ranked statement deliberately omits an explicit owner predicate:

```sql
SELECT search_id,user_id,id,paradedb.score(search_id) AS score
FROM public.messages
WHERE search_id OPERATOR(pg_catalog.@@@) $1
ORDER BY paradedb.score(search_id) DESC LIMIT $2;
```

Production fixed operations should still include their bound owner predicate.
Separate tests confirm explicit owner predicates agree with RLS and a
contradictory owner predicate returns nothing. Equal-score BM25 ties are not
claimed deterministic; the document-local alternative below uses a stable ID
tiebreaker.

## Catalog identity and grants

The exact installed operator is `pg_catalog.@@@(anyelement,text)`, implemented
by **`paradedb.search_with_parse(anyelement,text)`**. The schema of the operator
and its function differ. Catalog dependencies identify this as `pg_search`.
`paradedb.score(anyelement)` is the score projection. Both are invoker functions
and are not marked leakproof.

Inside the disposable database, the probe removes PUBLIC privileges on relevant
schemas, relations, sequences, and all **499 extension routines**, then grants:

```sql
GRANT CONNECT ON DATABASE <probe_database> TO <reader>;
GRANT USAGE ON SCHEMA public,paradedb,pdb TO <reader>;
GRANT SELECT(search_id,user_id,id,subject,body_text)
  ON public.messages TO <reader>;
GRANT EXECUTE ON FUNCTION paradedb.search_with_parse(anyelement,text)
  TO <reader>;
GRANT EXECUTE ON FUNCTION paradedb.score(anyelement) TO <reader>;
GRANT EXECUTE ON FUNCTION
  paradedb.with_index(regclass,paradedb.searchqueryinput) TO <reader>;
GRANT EXECUTE ON FUNCTION
  paradedb.parse_with_field(paradedb.fieldname,text,boolean,boolean)
  TO <reader>;
```

This is a tested sufficient extension grant profile for these fixed statements,
including repeated prepared queries with `plan_cache_mode=force_generic_plan`.
It is not a proposal to run these revocations on an existing application DB.
Built-in PostgreSQL function/type privileges retain their normal defaults.

- Removing `pdb` schema USAGE causes even the probe's ordinary message SELECT
  to fail with `42501` in this installed planner environment.
- Removing score EXECUTE blocks ranked queries.
- Generic/prepared execution requires `with_index` and `parse_with_field`;
  independently removing each produces `42501`. Initial custom plans did not
  expose these requirements.
- **Operator EXECUTE is not an admission control.** After optimizer rewriting,
  Boolean count queries work with no extension routine EXECUTE grants, and
  early custom ranked plans work with score alone. The closed SQL compiler and
  immutable role/RLS boundary remain necessary; denying operator EXECUTE cannot
  substitute for them.
- Ungranted columns, DELETE, SET ROLE postgres, CREATE, TEMP, and sequence access
  fail with `42501`, even after the reader disables its advisory default
  read-only setting. Changing `app.user_id` does not change visibility;
  `row_security=off` makes the query fail.

Keep the existing analytical `ANALYTICAL_SCHEMA` and reader verification intact.
Any ranked-search reader/executor is a separately reviewed internal contract;
the guest never supplies an operator, relation, role, SQL fragment, or index.

## Row visibility and Top-K recall results

Alice owns 12 rows: two alpha/beta documents and ten low-scoring needle documents.
Bob adds 200 shorter, repeated-term needle documents. The administrator's global
top 200 needle results are all Bob, ahead of every Alice match.

| Direct reader test | Observed result |
| --- | --- |
| Alice ordinary SELECT | Exactly her 12 rows |
| Alice foreign-only lexical term | Zero rows |
| Alice needle count | 10 |
| Alice needle LIMIT 1 / 5 / 10 / 20 / 500 | 1 / 5 / 10 / 10 / 10 own rows |
| Alice broad term disjunction covering all documents | Exactly her 12 rows |
| Alice forced generic-plan needle / foreign-only queries | 10 / 0 rows |
| Bob needle count | 200 |
| Alice changes owner GUC to Bob | Foreign-only count remains zero |

The 10 complete matches have exactly the expected search IDs. No foreign row
was observed, and global Top-K did not crowd out these owner results.

Representative plan shape (full plans are emitted by the probe):

```text
Limit
  [Gather Merge, when parallel execution is selected]
    Custom Scan: ParadeDB Base Scan
      Exec Method: TopKScanExecState
      TopK Limit: 5
      Scores: true
      Tantivy Query:
        heap_filter(indexed_query=body_text:needle,
                    field_filters=[user_id = 'alice'::text])
```

The count uses `NormalScanExecState`, `Scores:false`, the same owner heap filter,
and 10 actual rows below the Aggregate. Plans include shared segment statistics;
do not publish raw EXPLAIN or extension diagnostics through the guest endpoint.
Bare `*` against the numeric-key text-parse overload is not a supported match-all
spelling in this probe: it raises `XX000`; the successful broad case is an
explicit fixed-field disjunction. A query error must not become an empty success.

## Confirmed shared-corpus ranking dependence

Alice's documents remain unchanged. Adding Bob's 200 alpha documents changes:

| Alice document | Before Bob corpus | After Bob corpus |
| --- | ---: | ---: |
| `alpha` | 3.6247847 | 0.09288017 |
| `beta filler` | 3.5955374 | 8.201775 |

The query is `body_text:alpha OR body_text:beta`. The first result changes from
alpha to beta solely because another owner added documents. Hiding numeric scores
would preserve this rank-order dependency. This proves that the tested global
BM25 index does not provide owner-independent lexical ranking statistics.

## Alternative A: document-local ranking, no table migration

The probe also executes this fixed shape through Alice's direct reader:

```sql
SELECT search_id,user_id,id,
  pg_catalog.ts_rank_cd(
    pg_catalog.to_tsvector('pg_catalog.english'::regconfig,
                          coalesce(subject,'') || ' ' || body_text),
    pg_catalog.to_tsquery('pg_catalog.english'::regconfig,$1),2
  ) AS lexical_rank
FROM public.messages
WHERE search_id OPERATOR(pg_catalog.@@@) $2
ORDER BY lexical_rank DESC,id ASC LIMIT $3;
```

Bound synthetic inputs are `alpha | beta`,
`body_text:alpha OR body_text:beta`, and 10. Both before and after Bob's corpus
change, Alice receives alpha **0.1**, then beta **0.05**, with identical rows and
scores. Foreign-only queries return nothing. PostgreSQL documents that these
ranking functions do not use global information.
[PostgreSQL ranking documentation](https://www.postgresql.org/docs/16/textsearch-controls.html#TEXTSEARCH-RANKING)

The observed plan is `Limit -> Sort -> ParadeDB NormalScanExecState`, with
`Scores:false` and the owner `heap_filter`. Only Alice's two matching rows are
fetched and delivered to the rank/sort projection. There is no global BM25 Top-K
inside this query. LIMIT is applied after document-local ranking.

This is the **least schema-change candidate** to qualify next. It needs no new
mailbox copy, per-owner index, or persistent generated vector. It can keep the
existing owner ScaNN branch and thread/detail tools while replacing lexical
scoring explicitly. It is not BM25 parity: dictionaries, stemming, stop words,
phrase/proximity behavior, field boosts, normalization, and hybrid thresholds
need a fixed profile and relevance evaluation. The two bound lexical query forms
must be generated consistently from the trusted parser; blindly sending identical
raw syntax to both engines is not equivalent.

Its cost is evaluating text vectors/rank for **all owner lexical candidates**
before the final LIMIT. Preserve statement/operation deadlines, memory/temp-file
limits and admission. A candidate-count/byte preflight can refuse over-budget
work explicitly; silently taking global BM25 top N or arbitrary first N would
change recall. Stored vectors or owner statistics are future optimizations that
require separate storage/write-path review.

Large-body handling is mandatory: the synthetic 150,000-distinct-token case
raises SQLSTATE `54000`, reporting a 2,197,986-byte vector over the 1,048,575-byte
maximum. PostgreSQL also limits individual lexemes to under 2 KiB, positions to
16,383, and positions per lexeme to 256. Therefore on-demand `tsvector` ranking
cannot silently promise complete phrase/proximity coverage for arbitrary long
mail bodies. Choose explicit rejection or a separately specified chunking/ranking
strategy; clipping is a behavior change.
[PostgreSQL text-search limits](https://www.postgresql.org/docs/16/textsearch-limitations.html)

## Alternative B: native BM25 over physical owner partitions

A single partial BM25 index `WHERE user_id='alice'` succeeds and restores Alice's
original isolated-corpus scores. A second partial index for Bob fails with
`a relation may only have one USING bm25 index`. A global-plus-partial combination
fails identically. Thus per-owner partial indexes on the current shared relation
are unavailable on 0.23.0; this is an engine constraint before catalog/planning
cost can even be evaluated. Current official docs also specify one index per
table, although those docs describe a newer release.
[ParadeDB index documentation](https://www.paradedb.com/docs/documentation/indexing/create-index)

A separate synthetic `PARTITION BY LIST(user_id)` table **does** work with one
physical partition per owner and a partitioned parent BM25 index. Child indexes
alone do not make the parent searchable: the query reports no BM25 index until
the parent index is created. Readers get column grants on the parent only;
literal parent RLS prunes Alice's query to `partition_alice`, and direct child
access remains denied.

With 200 Bob rows present, Alice's scores are exactly **3.6247847 / 3.5955374**.
Adding 200 more Bob beta rows and vacuuming leaves them exactly unchanged. Alice's
foreign-only result remains empty and needle count remains 10. The plan uses
only Alice's child BM25 index with `TopKScanExecState` and her owner heap filter.
This is evidence for **native physical partition-local statistics**, not a
tenant-statistics option inside one global index. No such in-index option was
established by this qualification.

This best preserves native lexical BM25 behavior, at a larger migration cost:

- Keep one logical shared table/database and one stored copy of each row in its
  owner partition. It is not a per-run mailbox clone. Migration can temporarily
  require rewrite/WAL/backup headroom and must be measured separately.
- Each owner's indexed documents are stored once across the child indexes,
  plus per-index segment/catalog overhead. Small-owner indexes and large owner
  counts need measured storage, planning, catalog, locking and maintenance costs.
- Parent uniqueness must include the partition key. The existing global
  `search_id` uniqueness contract and all owner-qualified FKs/writer paths need
  explicit review; do not silently remove constraints during conversion.
- One mixed-owner default/hash partition would share BM25 statistics among its
  occupants. Owner provisioning must establish a qualified dedicated partition
  and index before ranked search is enabled; no fallback to a shared ranked index.
- Audit RLS/grants on parent and children, partition routing/pruning, attach and
  detach lifecycle, prepared plans, concurrent writes and deletion. This small
  fixture does not qualify those production operations.
- Apply the decision consistently to message, attachment and proposition lexical
  branches. Isolating message statistics alone leaves other global-ranking inputs.

## Recommendation and next decision

Choose native owner partitions if preserving BM25 ranking behavior warrants a
measured physical migration. Choose the document-local fixed SQL profile if
avoiding that migration is the priority, with explicit relevance and CPU/large-body
qualification before claiming equivalent product usefulness. Both preserve the
shared database and can support the existing tool set; neither completes that
integration by itself. Per-owner partial indexes on the existing relation are
not an available third option on this version.

No search gateway, feature flag, production grant, migration, or ranking change
was applied. The broader adapter/embedding/ScaNN/facts requirements remain in
[`gateway-search-integration-plan.md`](gateway-search-integration-plan.md).

## Internal fixed-reader qualification (2026-09-15)

The new internal reader is separate from analytical SQL credentials and its
closed schema. Its provisioner grants selected search columns, including
embeddings and fact vectors, only to a deterministic direct owner login with
literal permissive **and** restrictive owner policies. The shared catalog audit
runs during provisioning and when opening a runtime snapshot. Embeddings RLS is
installed only by trusted administrator provisioning; queries perform no DDL.

The four function signatures above are additionally pinned to C implementations
from `$libdir/pg_search`, owned by the extension administrator and belonging to
`pg_search`; all are invoker, non-leakproof, parallel-safe ordinary functions:

| Signature name | Return | Volatility | C entry point |
| --- | --- | --- | --- |
| `search_with_parse` | boolean | immutable | `search_with_parse_wrapper` |
| `score` | real | stable | `paradedb_score_from_relation_wrapper` |
| `with_index` | `paradedb.searchqueryinput` | immutable | `with_index_wrapper` |
| `parse_with_field` | `paradedb.searchqueryinput` | immutable | `parse_with_field_bfn_wrapper` |

Provisioning refuses ambient PUBLIC function, sequence, table or schema grants
that exceed this profile, including grant options on otherwise permitted
privileges and SELECT on otherwise invisible zero-column relations. It does
**not** revoke grants for other applications. Administrator provisioning bounds
its entire operation to 2-second lock waits and 30-second statements, preserves
stricter caller limits, and restores caller settings on success or failure.
The synthetic fixture removes those grants only inside its newly created test
database. Existing shared-database ACL preparation needs separate operational
review before this reader can be provisioned there.

### Exact native generic-plan shape

The following difference is reproducible on pg_search 0.23.0 for **all three**
partitioned lexical branches, even though the role already has literal owner
RLS:

```sql
-- Explicit owner parameter plus a generic plan: unsupported query shape.
WHERE user_id = $1 AND search_id OPERATOR(pg_catalog.@@@) $2

-- Supported custom and generic plans: fixed owner constant plus bound query.
WHERE user_id = '<immutable credential owner>'
  AND search_id OPERATOR(pg_catalog.@@@) $1
```

The RLS-only query from the earlier partition test also works. The new fixed
builder retains an explicit owner condition, generating its constant with
`psycopg.sql.Literal` from the session's immutable credential. No method accepts
an owner, SQL fragment, field name or database role. Query tokens, candidate IDs,
limits and other request values remain bound parameters. Quoted/backslash owner
IDs are covered by a real fixture regression. Attachment/fact branches use
`id` and their fixed fields with the same owner-constant rule.

Tests exercise the real bounded `row_to_json` wrapper and named cursor with
forced generic planning, as well as prepared custom/generic direct statements.
Scores remain unchanged after foreign-corpus edits. An unsupported query shape
is a fixed error, never a successful empty lexical result.

### Query and lifecycle bounds

`SearchReader.session(owner_id, deadline, check_active)` owns one direct-login
read-only repeatable-read transaction. It acquires shared SQL/search admission
before connecting and releases only after cancellation and connection close
are acknowledged. Repeated cancellation, revocation, deadline expiry, and
cancellation of a running `pg_sleep` test query are covered. Queries on one
snapshot are sequential; concurrent method calls are rejected.

Default limits: 30 seconds total session time, 10 seconds per SQL statement,
4 MiB per result page, 128 MiB total internal transfer and 200,000 total rows.
These internal vector/data limits are distinct from guest JSON limits. Vector
pages have at most 128 rows and validate exact configured byte length before
encoding or transfer. Missing, wrong-model and invalid vectors carry explicit
status. No raw vector operation is offered to the guest.

Hydration clips message bodies and embedding snippets in SQL and returns original
byte lengths and completeness flags. Body hydration defaults to 200 characters;
a trusted caller may request up to 400,000, with the same explicit completeness
flag. Summary hydration chooses the newest owner/message summary with a
stable model tiebreak and includes model/timestamp metadata. Selection caps and
keyset pages report `complete`, `reason`, and a continuation cursor when usable.
These flags describe the selected operation, not exhaustive semantic recall.

Reproduce with the existing original virtualenv, using only the disposable
loopback-55440 DSN as `GMS_TEST_PG_DSN`:

```sh
PYTHONPATH=src /home/ssilver/development/gmail-search/.venv/bin/python -m pytest -q \
  tests/test_gateway_search_reader.py tests/test_gateway_search_queries.py
```

This qualifies the fixed database slice. It does not enable guest search routes,
provider query embeddings, ScaNN orchestration, reranking, or full search parity.

## Fixed-reader maintenance policy (2026-09-15)

**Qualification remains blocked.** Strengthening the fixture to commit REINDEX
of every owner BM25 leaf before subsequent INSERT/DELETE/VACUUM exposed a native
`XX000: assertion failed: item_pointer_is_valid(ctid)` in the actual numeric
fixed-reader message branch. Both the proposed custom/unprepared policy and an
in-memory replay of the previous reader policy fail this stronger case. The
failing statement selects/scorers `messages.search_id` with the immutable owner
literal; this is not evidence that every numeric attachment or fact key fails.
Relation-specific and repeated TEXT checks are in progress. No rollout or
post-maintenance safety claim follows from the earlier passing fixture.

The proposed `SearchReader` policy requires `plan_cache_mode=force_custom_plan`
in its exact role settings and effective initial session settings, sets it
locally for the snapshot, and connects with `prepare_threshold=None`. Missing or
drifted role settings are refused before credential rotation; an effective
database-role override is refused before local settings could hide it. Existing
roles require an explicitly reviewed administrator update; provisioning does not
repair them silently. Owner-literal query shapes and the numeric four/TEXT five
function grants are unchanged.

The standalone ordinary generic/parallel probe can fail with `XX000` after own
INSERT/DELETE/VACUUM, including fresh generic connections. The fixed reader
already disables parallel workers and uses named cursors, so that standalone
probe alone did not establish a fixed-reader failure. The stronger REINDEX
fixture now establishes a separate actual numeric-message reader failure under
both old and proposed policies. Historical custom/generic successes elsewhere
in this document remain limited observations.

`tests/test_gateway_search_reader_plan_policy.py` commits leaf REINDEX before
three rounds of each of three owners' INSERT/DELETE/VACUUM. Each profile/relation
case runs 27 post-maintenance snapshots, repeats its lexical query and, for
messages and attachments, its restricted-candidate query. Facts has no restricted
candidate API. Own scores may change after own edits; other owners' results and
scores must remain unchanged. Prepared statement inventories stay empty.
Separate tests check missing/drifted role settings and database-role overrides.

Earlier results below apply **only to the original fresh-leaf fixture without
explicit pre-cycle REINDEX**. Six focused tests passed in 7.25 seconds and the
broader reader/query/profile/native search/service/facts batch passed 126 tests
in 45.19 seconds without skips. The exact old-policy replay also passed both
original fixture profiles. One tiny timing sample gave numeric prior/new mean
snapshot 76/76 ms and TEXT prior/new 64/78 ms. These measurements do not qualify
the strengthened scenario or establish production latency or overhead.

Run the focused tests with the existing disposable loopback-55440 DSN supplied
as `GMS_TEST_PG_DSN`:

```sh
PYTHONPATH=src /home/ssilver/development/gmail-search/.venv/bin/python -m pytest -q -s \
  tests/test_gateway_search_reader_plan_policy.py
```

No legacy store caller, readiness gate, mixed-owner migration, provider or
production configuration changes are part of this planner-policy slice.
