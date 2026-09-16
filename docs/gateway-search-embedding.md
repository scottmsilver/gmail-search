# Internal Gemini query embeddings

## Approved design and implementation plan

Candidate-only fixed `gemini-embedding-2`, 3072 dimensions, with the query prefix
`task: search result | query: `. No production configuration or index relabeling.

1. Add failing synthetic tests for fixed requests, budget accounting, malformed
   responses, revoked bindings and shared capacity.
2. Implement the immutable profile, strict bounded JSON/vector handling and
   fixed HTTP transport; no SDK retries or ambient credentials/proxy settings.
3. Exercise cancellation during reservation, transport teardown and settlement;
   retain capacity until each owned operation drains. Quarantine settlement
   failures for explicit trusted reconciliation.
4. Run focused tests and independent review; record supported API references and
   unqualified compatibility/deployment boundaries below.

## API profile and source checks (2026-09-15)

The new adapter supports only `gemini-embedding-2` at 3,072 dimensions. Google
records its GA release on April 22, 2026. This is an explicit new adapter choice,
not a change to existing mailbox/index model configuration. The legacy preview
profile is rejected. [Google release notes](https://ai.google.dev/gemini-api/docs/changelog),
[model specification](https://ai.google.dev/gemini-api/docs/models/gemini-embedding-2).

Embedding 2 uses a query instruction, not the `RETRIEVAL_QUERY` task-type field.
The adapter prepends the fixed search prefix shown above. Document formatting is
separate corpus-provisioning work; the guest cannot choose either profile.
[Google embedding task guidance](https://ai.google.dev/gemini-api/docs/embeddings#task-types).

Requests use the fixed HTTPS `v1beta/models/gemini-embedding-2:embedContent`
endpoint, one `content.parts.text`, and `embedContentConfig` with
`outputDimensionality: 3072` and `autoTruncate: false`. The response is one
`embedding.values` vector. Optional `usageMetadata.promptTokenCount` describes
input tokens; accepted modality details must contain only a matching TEXT count.
The currently documented nested config replaces deprecated top-level config
fields. [Google embedding REST reference](https://ai.google.dev/api/embeddings).

The deprecation table lists the abbreviated `embedding-2-preview` with an
August 10, 2026 shutdown schedule. Google describes these dates as the earliest
possible shutdown dates. This is a documented past schedule, not an observed
endpoint test; no authenticated provider request was made.
[Google deprecation schedule](https://ai.google.dev/gemini-api/docs/deprecations).

The model pages, guide, release notes and targeted official-document search did
not establish vector-space equivalence between preview and GA Embedding 2.
Whether existing preview vectors can be reused remains unqualified; this work
neither requires nor rules out reembedding. Do not relabel an existing preview
index or mix its vectors with the new adapter. A future compatibility/migration
qualification must cover the actual model versions, dimensions, corpus format
and query prefix. The guide's explicit incompatibility warning concerns
Embedding 001 versus Embedding 2, which does not settle preview-versus-GA
compatibility.

## Trusted interface and limits

Construct `GeminiQueryEmbedder(registry, transport, profile=..., admission=...)`.
Its read-only `.model` and `.dimensions` allow RunSearchService to compare the
pinned index binding before calling
`await embed(verified_lease, text, deadline=monotonic_deadline, check_active=...)`.
The asynchronous callback must freshly authorize the retrieval capability and
return `True`; the adapter also rechecks active durable registry state and all
immutable run-lease fields. A renewed lease expiration is allowed. The adapter
checks the conversation fence and workspace version.

The complete prefixed text is limited to 4,096 UTF-8 bytes; blank, NUL-containing,
invalid Unicode and oversized queries are rejected before reservation. Expanded
queries beyond this limit fail explicitly. Encoded HTTP requests are capped at
32 KiB. Responses have a 256 KiB aggregate cap, 64 KiB chunk cap, JSON nesting
limit of eight and duplicate-key rejection. The sole vector must contain exactly
3,072 finite scalar numbers, be nonzero, and have element magnitudes at most
1,000,000. Unknown response structure is refused; unknown usage is conservatively
charged without discarding an otherwise valid vector.

`GeminiEmbeddingHTTPTransport` owns a fixed HTTP client with explicit API-key
headers, TLS verification, environment/proxy discovery disabled, redirects
disabled, no SDK retry behavior and at most four connections. Non-200 status,
non-JSON responses and compression are refused without exposing provider error
bodies. Credentials remain in trusted host memory. Tests inject an httpx mock
transport; the product provides no endpoint override. The controller must call
`aclose()` only after active embedding operations have drained.

## Budget, cancellation and capacity

The embedding adapter inherits its reviewed lifecycle from the private
`gateway/search_provider.py`, shared with the fixed thread reranker. The
extraction preserves all 42 embedding tests, including cancellation during
normal EOF close and the final authorization/deadline ordering regressions.

Before each billable attempt, the adapter reserves the full 8,192-token model
input ceiling multiplied by the profile's positive integer internal unit rate.
The rate is trusted accounting configuration, not a current price claim.
Each attempt receives a fresh internal reservation key. No cache, query-derived
idempotency, global cost log, provider retry or guest embedding route exists.

Only a completed valid response, successful transport teardown and fresh
authorization permit settlement at the accepted input usage. Missing/invalid
usage, incomplete transport, failure, cancellation or deadline expiry charge the
full reservation. Provider-reported usage cannot exceed the reservation. No
vector is published without another authorization check after settlement.
That final check completes the registry binding lookup first, then freshly
authorizes the capability and checks the deadline. All task drains precede it;
there is no further asynchronous work before returning the vector.

Composition must inject the same `DataAdmission` instance into all providers
that share the intended quota. This instance must allow no more than four
global operations and two per owner; there is no hidden adapter default. It is
separate from database capacity. Existing inference-route admission is not
rewired by this slice: sharing across inference, embeddings and reranking is an
explicit composition requirement. Limits are process-local, not cross-process.

The adapter watches authorization every 50 ms and applies the earlier of the
caller's absolute deadline and 30 seconds. Cancellation drains reservation
threads, transport cleanup and settlement despite repeated cancellation before
releasing capacity. A slow cleanup can exceed the operation deadline. Provider
transport implementations are trusted to acknowledge actual resource teardown.
Transport exit is separately owned and drained, including cancellation arriving
after normal response EOF while the transport is already closing.

A failed settlement retains its admission lease and durable budget reservation.
`await reconcile()` explicitly retries settlement only, then releases capacity;
it never repeats a provider call. Quarantine size is bounded by admission.
The controller must retain the adapter until reconciliation finishes. Process
restart recovery of outstanding durable reservations remains controller work;
process-local capacity and quarantine are not durable provider-job inventory.

## Synthetic verification

The focused suite passes 42 tests; together with the existing provider lifecycle
suite, 61 tests pass. Tests exercise fixed HTTP requests, refused redirect/error/
compression responses without retry, budget exhaustion, duplicate/deep JSON,
malformed vectors and usage, forged bindings, lease renewal, revocation and
deadlines, cancellation during reservation and slow teardown/settlement, shared
capacity hold, and explicit reconciliation after failed settlement.
The review regressions reproduce cancellation during a gated normal EOF close,
revocation or deadline expiry during the final threaded binding lookup, and
deadline expiry during final capability authorization.

No real provider calls, production credentials, production index loads, live
application routes, model renames or reembedding were performed. Actual model
availability, billable usage semantics, retrieval quality, preview-index
compatibility and public composition remain release qualification work.
