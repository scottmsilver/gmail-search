# Fixed internal thread reranker

## Approved design and execution

The candidate uses a new fixed `gemini-3.1-flash-lite` profile for internal thread
reranking. No existing production model configuration is changed. Implementation
steps were: preserve the reviewed embedding lifecycle in a private shared helper;
add failing tests for the reranker contract; implement strict ordinal ranking and
HTTP handling; verify shared cancellation, authorization and budget behavior;
request independent review.

Owned implementation files are `gateway/search_provider.py`,
`gateway/search_reranker.py`, the embedding adapter updated to inherit the shared
lifecycle, and focused embedding/reranker tests. There is no public reranking
route and no change to the original search engine's production defaults.

## Official API profile checked 2026-09-15

Google lists `gemini-3.1-flash-lite` as stable, with a 1,048,576-token input limit
and 65,536-token output limit. Its preview model page explicitly says that preview
shut down May 25, 2026. The new profile rejects the preview name. No authenticated
endpoint availability test was performed. [Stable model specification](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-lite),
[preview status](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-lite-preview).

The adapter sends one text-only `generateContent` request to the fixed URL:

```
https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent
```

It uses `generationConfig.maxOutputTokens: 512`, JSON output with a fixed
`responseJsonSchema`, and `thinkingConfig` with `thinkingLevel: MINIMAL` and
`includeThoughts: false`. There are no tools, cached content, media, history,
URL-context references or guest-selected generation settings. MINIMAL does not
guarantee zero thinking. [Google GenerateContent reference](https://ai.google.dev/api/generate-content),
[Gemini 3 thinking guidance](https://ai.google.dev/gemini-api/docs/generate-content/gemini-3).

The reference describes separate input, candidate and thought token usage, and a
total that includes all three. This adapter requires those three counts and an
exact total before lowering the charge from the reserved amount. Missing
`thoughtsTokenCount` is unknown usage, even when it might represent zero.
[Google usage metadata](https://ai.google.dev/api/generate-content#UsageMetadata).

## Interface and summaries

Construct `GeminiThreadReranker(registry, transport, profile=..., admission=...)`.
Call:

```python
order = await reranker.rerank(
    verified_lease, query, tuple_of_thread_results,
    deadline=absolute_monotonic_deadline,
    check_active=fresh_capability_authorization,
)
```

Input must contain 1–30 exact `ThreadResult` objects in a tuple with unique,
nonempty thread IDs. IDs are bounded to 512 UTF-8 bytes. The adapter snapshots the
IDs before its first await. Provider output can only select those snapshotted IDs;
subsequent mutation of the input objects does not change the returned mapping.

The provider receives ordinal positions and concise summaries. It receives no
thread IDs, message IDs, capability, owner/run ID, paths or database credentials.
The summary profile retains at most 512 subject characters, the display-name
portion of the first three participants at 256 characters each, message count,
and the first match's first 100 snippet characters. The first-match/participant
selection follows the original reranker's concise summary approach; explicit
subject and participant caps are additional limits. It does not send full bodies.

Source strings and collections are bounded before copying or encoding. Query
text has a 4,096-byte UTF-8 ceiling; blank, invalid Unicode, NUL-containing or
oversized input fails before reservation. Source subject/participant strings have
4,096-byte caps, source snippets 20,000 bytes, and participant/match lists at most
100 entries. The complete encoded request must fit 96 KiB. An expanded query
beyond the transport cap fails explicitly.

The system instruction treats summaries/query as data. This reduces accidental
instruction following but is not a security guarantee about model behavior.
The strict ordinal response contract limits what a model response can publish;
retrieval relevance and prompt-injection effects on ordering still need quality
qualification.

## Response contract

The entire response is bounded to 256 KiB in chunks no larger than 64 KiB, with
JSON nesting at most eight and duplicate-key rejection. Accept exactly one
candidate at index zero with `finishReason: STOP`, one model text part, and no
thought output or function calls. A bounded thought signature may be discarded.
Present model-version metadata must exactly match the configured stable model.
Unknown response fields, blocked prompt feedback, truncation and unsupported
parts fail closed pending qualification.

The inner text is at most 2,048 UTF-8 bytes and must be exactly a JSON object
containing `order`: an integer permutation of every supplied ordinal, once each.
Missing/duplicate/foreign indices, booleans, numeric strings, markdown fences and
extra output fields fail. The adapter never fills in missing indices or applies
an undocumented fallback. The return value is a tuple of original thread IDs,
without model commentary or provider diagnostics.

## Explicit conservative budget policy

`GeminiRerankerProfile` requires positive integer input/output unit rates and the
explicit policy string `full-model-ceilings-v1`. It does not infer a price or
choose a smaller reservation from request length. Each attempt reserves:

```
1,048,576 * input_units_per_token + 65,536 * output_units_per_token
```

This deliberately uses the complete model ceilings despite the small prompt and
requested 512-token response. The current qualification does not use MINIMAL or
an assumed relationship between visible-output and hidden-thinking limits to
reduce the reservation. Units are trusted host accounting configuration, not
current provider-price claims. A run must have enough available budget for this
reservation; failed/unknown calls can consume the full amount. Reducing it later
requires an explicit reviewed billing-bound qualification.

A valid complete response charges input tokens at the input rate and candidate
plus thought tokens at the output rate. Counts must fit the reserved ceilings,
visible candidate count must not exceed 512, and the total must match. Cache,
tool-use or non-text modality billing is unsupported. Missing, inconsistent or
unknown usage charges the full reservation, while a valid permutation can still
be returned. Failure, cancellation, incomplete response and deadline expiry also
charge the full reservation. No provider call is retried automatically.

## Shared lifecycle and deployment constraints

Embedding and reranking inherit the same private `_SearchProvider` lifecycle.
Composition must inject the same `DataAdmission` instance for their shared
provider quota, with at most four operations globally and two per owner. Tests
also exercise refusal across distinct owners. This capacity is process-local;
sharing with existing inference routes and cross-process coordination require
explicit composition and qualification.

The shared implementation rechecks immutable run binding, active owner and
conversation fence, while allowing lease-expiration renewal. It watches the
fresh capability callback, bounds work by the caller deadline and 30 seconds,
and retains budget/capacity while cancellation drains reservation, transport
exit and settlement. A separately owned transport exit survives cancellation
arriving during normal EOF close. Cleanup can outlast the operation deadline.

After all task drains and settlement, the final check performs the threaded
binding lookup, freshly authorizes the capability and checks the deadline; no
further asynchronous work follows before returning. Failed settlement retains
capacity in a bounded quarantine until explicit `reconcile()` settles it; that
method never repeats a provider call. Retain the adapter until reconciliation
finishes. Durable reservation recovery after process restart remains trusted
controller work.

HTTP credentials stay in the trusted transport. HTTPS verification is enabled;
environment/proxy discovery, redirects and retries are disabled. Non-200,
compressed and non-JSON responses are refused. The controller owns transport
lifetime and closes it after active calls drain.

## Verification

The focused reranker suite passes 45 tests. Together with all 42 embedding tests
and 19 existing provider tests, 106 tests pass. Coverage includes actual httpx
mock requests, ordinal-only prompts, full 30-item permutations, malformed and
unsupported responses, conservative usage accounting including thoughts,
input mutation, shared capacity, EOF-close cancellation and final-binding
revocation. All previously reviewed embedding cancellation/settlement/deadline
regressions remain unchanged and pass after the extraction.

No real provider calls, production keys, real mail, live routes, original-engine
model changes or provider-cost writes were used. Actual model availability,
response metadata shape, billing behavior, retrieval/reranking quality and full
public composition remain release qualification work.
