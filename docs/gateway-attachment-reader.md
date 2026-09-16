# Owner attachment metadata and stored text

## Stage 1 scope

`gateway/attachment_reader.py` implements `OwnerAttachmentReader` over the
existing restricted `QueryGateway`. It performs only fixed SELECT operations.
There are no new database grants, raw-path reads, filesystem operations, parsers,
provider requests, extraction writes or public routes. Thread response wiring and
raw-file delivery are separate stages.

The tests first established the absent reader contract, then exercised its typed
results, owner isolation, paging and cancellation. The implementation retains the
existing analytical compiler, fixed-owner database login/RLS and shared database
admission as independent boundaries.

## Trusted API

Construct `OwnerAttachmentReader(query_gateway)` and call:

```python
metadata = await reader.describe(
    owner_id, attachment_id, deadline=deadline, check_active=check_active,
)
page = await reader.text_page(
    owner_id, attachment_id, offset=0, limit=20000,
    deadline=deadline, check_active=check_active,
)
listing = await reader.list_for_thread(
    owner_id, thread_id, after_attachment_id=0, limit=100,
    deadline=deadline, check_active=check_active,
)
```

This is an internal dependency, not an authorization endpoint. The trusted
service must derive `owner_id` from its verified run capability and provide an
asynchronous callback that freshly authorizes that same run and returns `True`.
The owner argument is never accepted directly from a guest. A future public
service must choose the operation/audience checks before invoking the reader.

The caller supplies a finite absolute monotonic deadline and must choose a
bounded operation duration. This reader does not impose an additional duration
cap. It enforces the supplied deadline around authorization/query work and checks
it after the final authorization await. QueryGateway separately enforces its
configured query deadline, at most 30 seconds.

Each operation executes one fixed query with the attachment/message join on both
`user_id` and message ID. The compiler emits parameterized SQL. Integer attachment
IDs/cursors and restricted thread-ID grammar are checked before query construction.
Every returned metadata row must match the trusted owner and requested binding.
Raw path columns are neither selected nor granted.

## Typed results and availability

All three result types are frozen dataclasses. `AttachmentMetadata` contains:

- Internal `owner_id` binding; attachment, message and thread IDs.
- Optional filename, MIME, size and fetch status, preserving database NULL values.
- `text_chars`, which is NULL when no text value is stored.
- `stored_text_state`: `missing`, `empty` or `present`.
- `extraction_complete`, always `None` because this schema has no authoritative
  extraction-completion field.

An empty stored string is distinct from missing text, but neither proves that
extraction completed. Present text can also be a partial extraction. Metadata
works for any stored MIME and zero-byte/unfetched records; it does not promise
that raw bytes exist or that a parser supports the format. Filename/MIME/status
are untrusted display data, never filesystem paths, HTTP routing or parser choices.
A metadata result does not contain inline-image recommendations or signed URLs.

`describe` requires exactly one complete matching row. Missing, ambiguous,
foreign, malformed and database-truncated results produce the same sanitized
`AttachmentReadUnavailable`. There is no fallback to a privileged reader.

`AttachmentTextPage` includes the metadata, stored text page (or `None`), offset,
limit, total characters, next offset, `page_complete`, `stored_text_complete`,
and `pagination_limited`. `page_complete` means the requested stored-text slice
is exact; `stored_text_complete` additionally means offset zero and the full
stored value was returned. Neither flag describes extraction completeness.
Missing text has both flags false and no cursor. An empty stored value at offset
zero has both true while `extraction_complete` remains unknown.

Text slicing uses PostgreSQL `substr` plus `length` in the same query, before
result transfer. Offset is 0–2,147,483,646, keeping the one-based substring start
within signed int4; limit is 1–100,000 characters. The default is 20,000.
Returned text length must equal the exact expected slice. Out-of-range offsets
return the database's empty slice without inventing a continuation.

`AttachmentMetadataPage` contains an ordered tuple of metadata, the input cursor
and limit, `source_complete`, `complete`, `next_attachment_id` and
`pagination_limited`. It fetches `limit+1` rows ordered by attachment ID, with a
1–100 item limit. `source_complete` means the queried tail completed; `complete`
also requires the initial zero cursor. Incomplete database results preserve a
usable last-ID cursor when possible. If no progress is possible, the result
explicitly marks pagination limited and provides no cursor. It never invents an
attachment count. An empty complete list does not distinguish a missing thread
from a thread with no attachments.

Cursor requests are independent database snapshots. Completion describes the
observed selection, not a durable mailbox snapshot across concurrent changes.

## Bounds and lifecycle

Attachment IDs are positive signed-int8 values; cursors may also be zero. Thread
and message IDs use the existing 1–256 ASCII identifier grammar. Owner IDs are
bounded to 2,048 UTF-8 bytes. Metadata strings are bounded to 1,024 bytes for
filename, 256 for MIME and 128 for fetch status. Control characters, invalid
Unicode and oversized metadata are refused.

The SELECT bounds metadata strings with a one-character overflow sentinel before
transfer. Validation rejects the sentinel/oversized value instead of silently
presenting a truncated identifier or filename. Sizes are optional nonnegative
signed-int8 values. Text length is an optional nonnegative signed-int4 value.
QueryGateway's row and aggregate byte limits remain effective; incomplete
single-item reads fail rather than misrepresenting a partial object.

The reader watches authorization every 50 ms. Cancellation, deadline expiry and
revocation cancel/drain the owned query and watcher before returning. Repeated
cancellation cannot bypass that drain. The concrete QueryGateway retains database
admission through its own connection cleanup. All decoding and task cleanup
precede the final fresh authorization/deadline check; no further await occurs
before publication. Error messages contain no query, row or database diagnostics.

## Proposed thread integration (not implemented here)

The trusted thread service can call `list_for_thread` with the same owner/run
binding and attach the first metadata page to its response. Include each item's
`message_id` so the client can associate it with a displayed message, and retain
all completion/cursor fields. A subsequent fixed metadata-list operation can
continue by `thread_id` and `after_attachment_id`. Do not label a truncated page
as the thread's complete attachment inventory or infer byte availability from
metadata alone. This approach requires no new analytical schema columns/grants.

## Later bounded stages (not implemented here)

1. Separate generic opaque raw-source eligibility (including zero-byte files)
   from the existing parser-only MIME/nonempty restrictions. Preserve owner-hash
   path derivation, descriptor-relative no-follow traversal, mutation checks and
   expected-size binding. Legacy shared paths require separate migration/mapping.
2. Add run-authorized fixed binary download routes and a guest downloader that
   verifies bounded manifests, size/hash/EOF and creates private generated local
   files. No signed URLs or automatic raw-byte upload into provider context.
3. Add a versioned fourth attachment capability to bootstrap/tool configuration,
   closed attachment tool schemas, shared batch/byte limits and newly pinned
   guest images. Preserve historical three-capability profiles and evidence.
4. Qualify separate parser protocol/image changes for explicit text/render modes,
   page counts/continuation and archive member provenance; then qualify additional
   formats. Preserve the current parser allowlist and image until replacement
   qualification passes. OCR, Office/text/calendar/HTML/HEIC parsing and complete
   archive semantics remain beyond the current isolated parser profile.

Stored text for those formats is readable here when already present; that is not
new extraction support or full attachment tool parity.

## Verification

The attachment reader suite passes 32 tests, including four real synthetic
PostgreSQL fixtures. Combined with the existing thread retrieval suite, 53 tests
pass. Real tests use the preserved private PostgreSQL 16.15 instance, create
random databases/reader roles and remove them afterward. They cover colliding
attachment/message/thread IDs for two owners, generic MIME/zero-byte metadata,
NULL versus empty text, a two-million-character offset under a 700-byte gateway
response budget, the maximum accepted PostgreSQL offset, denied foreign joins,
and lack of direct `raw_path` column privilege.

Synthetic controls also verify malformed bindings/pages, cursor and byte-limit
semantics, cancellation/revocation/deadline cleanup and expiry during final
authorization. No production database, real mail, file storage, parser or provider
was used.
