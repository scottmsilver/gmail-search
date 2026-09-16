# Trusted owner ScaNN index adapter

`gateway/search_index.py` provides the owner-index dependency for the run search
service. It reuses trusted indexes across runs; it creates no mailbox/index copy
per run and has no bootstrap-owner or global-path fallback. Guest input cannot
select a path, owner, model, generation, loader or matrix.

## Controller and service interface

A trusted controller supplies a frozen
`IndexBinding(owner_id, generation, model, dimensions, source_id)`. The source ID
is an immutable provisioning identifier, not evidence inferred from embedding
IDs. Existing index manifests do not establish owner/model provenance.

The synchronous `load_scann_index(binding, absolute_path, limits=...)` returns a
`LoadedIndex` after validating and loading a complete generation. Loading scans
bounded metadata and index data and can be expensive: run it outside the event
loop, under trusted provisioning lifecycle control, before publication. No index
loading happens during a run's `acquire` call.

The controller transfers ownership using `await registry.publish(loaded)`.
Validation and publication have no suspension point; rejected publication leaves
resource ownership with the caller. Accepted publication schedules retired
cleanup off the event loop. The registry rejects changed/reused generation or
source bindings, the same searcher object reused for another publication, and
the same physical generation directory assigned to another owner. Retained source
and generation bindings are bounded tombstones for this registry's lifetime.

The search service pins the binding before requesting its query embedding:

```python
async with registry.acquire(authenticated_owner_id) as index:
    # The trusted embedding adapter uses index.binding.model and dimensions.
    vector = await embed_query(query, binding=index.binding)
    ids, scores = await index.search(
        vector, top_k=2000, absolute_deadline=operation_deadline,
    )
    # Revalidate and hydrate every candidate through the immutable owner reader.
```

A missing registry entry raises sanitized `IndexUnavailable`. Only an explicit
trusted `mark_pending(owner_id, model, dimensions)` produces `PendingIndex`.
A genuinely built empty index returns empty candidates; a torn empty ID file
with leftover native assets is refused. Registry errors contain no path, query
text or native exception details.

## Lifetime, cancellation and limits

Generation rotation keeps the old resource alive until its leases and native
searches drain. A lease continues to select its original generation/model during
embedding and query work. Each generation executes at most one native search at
a time. The registry also bounds simultaneous searches, active leases, resident
including retired generations, and lifetime binding metadata. Defaults are two
searches, eight leases, sixteen resident generations and 4,096 bindings; these
are trusted host policy settings, not guest parameters.

Search and close run in worker threads. Cancellation and the absolute deadline
shield and drain native work before releasing its capacity. Repeated cancellation
cannot close an index beneath a running search. Exiting a lease cancels and drains
any background search started through that lease. Cleanup failure retains the
resident generation and capacity; `aclose` reports failure and can retry cleanup.
Registry shutdown refuses new work and waits for outstanding leases to finish.

**ScaNN does not provide a hard thread-kill API.** A deadline or cancellation can
therefore wait beyond the deadline for native work to finish. This adapter does
not acknowledge teardown early. Query admission and service cancellation must
remain held until lease cleanup returns. The per-search deadline is capped at
thirty seconds before this mandatory drain; it is not a hard native execution
limit. Loading and native parsing are trusted controller work, not sandboxed
processing of user-supplied index files.

Query vectors must be one-dimensional, finite, nonzero numeric float32-compatible
values with the bound dimensions, at most 8,192 dimensions and magnitude at most
1,000,000 per element. Nested/scalar-invalid values are rejected before NumPy
conversion. Candidate limits are 1–10,000; output must contain matching bounded
lists of unique positive signed-64-bit IDs and finite numeric scores. These
checks do not establish candidate ownership: the owner reader must revalidate
IDs before any guest-visible publication.

## Strict disk profile

The selected absolute generation must resolve to itself, belong to the current
trusted service UID, contain no symlinks or special files, and have no group/world
write permission. Deployment must arrange immutable generation lifetime; this
adapter does not freeze writable owner files or coordinate legacy builder GC.
It checks the file inventory before and after loading. Trusted same-owner sealed
shard hardlinks remain compatible; native objects are not shared between loaded
generations in this first implementation.

Default configurable load limits are 32 GiB total files, 4,096 files/directories,
2,000,000 IDs, 256 shards, 64 MiB ID/JSON payloads, 4 MiB native/manifest metadata
and 10,000 manual-rerank candidates. These bound accepted inputs, not all native
parser allocation overhead or peak process memory. The registry's generation
limit counts published resources; administrative loading occurs before its
publication-capacity check. Host memory sizing and administrative load admission
therefore require separate operational qualification, including concurrent loads
and resident generations.

The loader validates unique ID lists, exact shard counts and concatenated ID
coverage, dimensions, bounded native configuration, required native assets,
asset paths confined to the chosen generation, and finite NumPy/mmap vectors in
bounded blocks. Missing shards or rerank corpus are failures. It reuses the
existing positional/docid/truncated-query/manual-rerank search implementation,
while replacing its tolerant loading paths.

The native convenience loader is never called: it can unpickle arbitrary globals
and synthesize missing asset metadata. Docid pickle input instead accepts only
a bounded protocol-4/5 string-list encoding with a restricted unpickler, no global
or persistent references, and exact positive-ID membership. Native construction
then uses explicitly validated asset text and docids. Configuration cannot name
external files. The one accepted nonfinite configuration value is the installed
serializer's `FixedPoint(enabled=false).noise_shaping_threshold=NaN` sentinel;
query vectors, datasets, rerank corpus and results remain strictly finite.

This is not a sandbox for the native ScaNN parser. Trust in owner-specific index
production and immutable source provenance remains necessary; directory names,
ID lists and current-process reuse checks do not prove corpus ownership across
arbitrary copied or relabeled indexes.

## Verification and remaining integration

The focused suite passed **38 tests**, using fake lifecycle controls and small
real ScaNN serialized indexes. It covers legacy positional and v2 docid format,
changed mutable docid order, mmap manual reranking, an actual AH/tree/reorder
index, missing required native assets, malformed paths/pickle/data, wrong binding,
repeated cancellation, background-task drain, slow/failed cleanup, retained
capacity and genuine-empty versus torn-empty state. No production index was
loaded and no real provider was called.

Service-level ANN coverage, structured-filter exact-dot search, database candidate
revalidation, owner budget accounting, provider transport and full search parity
remain separate integration work. The inherited ScaNN/manual-rerank path normalizes
vectors and uses cosine-like scores; the legacy restricted-vector path uses raw
dot products and belongs to the separately streamed owner-reader implementation.
