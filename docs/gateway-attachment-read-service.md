# Run attachment JSON read service

`RunAttachmentReadService` adds run-capability authorization and bounded public
JSON projection to the reviewed `OwnerAttachmentReader`. It implements metadata,
already-stored text pages, and thread attachment manifests. It does not access
raw files, invoke a parser, perform extraction, upload content, or call a provider.
Optional private HTTP wiring is implemented separately; guest mapping and
binary lifetime/capacity are subsequent stages.

## Trusted construction and methods

Construct `RunAttachmentReadService(capabilities, reader)` using an
`OwnerAttachmentReader` backed by the existing restricted `QueryGateway` and
shared `DataAdmission`. Every owner comes from the run capability. Neither owner,
filesystem path, SQL, MIME override, nor provider/model is a public selector.

```python
await service.describe(token, attachment_id=1)
await service.text_page(token, attachment_id=1, offset=0, limit=20000)
await service.list_for_thread(token, thread_id='thread', after_attachment_id=0, limit=100)
```

All use the `attachment` audience. Describe and thread listing require operation
`meta`; text pages require `text`. These privileges do not substitute for one
another. `authorize(token, operation)` accepts only these two fixed operations
and is available for trusted transport authentication before reading a request
body and immediately before response publication.

The service preserves reader bounds: positive signed-int8 attachment IDs;
nonnegative signed-int8 manifest cursors; 1–256-character ASCII thread IDs;
text offsets 0–2,147,483,646; text limits 1–100,000 characters; manifest limits
1–100 items. Unknown keyword selectors are rejected by the fixed signatures.

## Public JSON

Describe returns these explicit fields:

```json
{
  "attachment_id": 1,
  "message_id": "message",
  "thread_id": "thread",
  "filename": "file.txt",
  "mime_type": "text/plain",
  "size_bytes": 0,
  "fetch_status": "fetch_failed",
  "text_chars": null,
  "stored_text_state": "missing",
  "extraction_complete": null,
  "cite_ref": "thread"
}
```

Optional metadata remains null when unknown. The public projection omits
`owner_id` at every nesting. Metadata is display data, not proof that raw bytes
exist or that a parser can handle the MIME type. Citation references preserve
the bound thread identity.

Text pages add `extracted_text`, `offset`, `limit`, `total_chars`, `next_offset`,
`page_complete`, `stored_text_complete`, and `pagination_limited` to the same
flat metadata. The existing `extracted_text` field name is retained. Missing text
is null; empty stored text is an empty string. `page_complete` describes the
exact requested slice. `stored_text_complete` additionally requires offset zero
and the entire stored value. `extraction_complete` remains null in all cases:
this schema cannot establish whether extraction finished. No extraction or raw
fallback is attempted when stored text is missing or empty.

Thread manifests return `thread_id`, `cite_ref`, `attachments`,
`after_attachment_id`, `limit`, `source_complete`, `complete`,
`next_attachment_id`, and `pagination_limited`. Every attachment is the public
metadata above, with an additional `id` alias equal to `attachment_id` for
existing thread-manifest consumers. Items are strictly increasing by attachment
ID and all bind to the requested owner/thread. `source_complete` describes the
queried tail; `complete` also requires the initial zero cursor. An incomplete
empty page remains explicitly incomplete and pagination-limited. No inventory
count or thread existence is inferred from an empty response. Separate cursor
requests do not share a durable database snapshot.

## Validation and lifecycle

The service accepts only the exact reviewed result dataclasses. Before public
projection it rechecks owner and requested attachment/thread identities,
canonical metadata states, exact expected text slice lengths, cursor progress,
and consistency of completion flags. Metadata validation reuses the reader's
bounded pure decoder. It names every public field rather than recursively
serializing internal objects. Reader errors and malformed responses produce the
fixed `Attachment data is unavailable.` error without paths, SQL or diagnostics.

One absolute 30-second deadline starts before initial capability authorization.
The reader receives that deadline and a callback freshly authorizing the same
operation, owner and run. The reader already owns its authorization watcher and
query cleanup, so the service adds no redundant watcher. The service owns and
shield-drains the reader task; repeated cancellation cannot abandon reader/query
cleanup or an authorization thread. QueryGateway independently retains shared
admission until its connection closes and may enforce a stricter deadline.

After owned cleanup, the service validates and projects the result and verifies
strict UTF-8 JSON encoding within a total 4 MiB budget. It then performs a final
fresh authorization/deadline check and returns with no later await or output
processing. HTTP composition must also check authorization after its own
transport cleanup. These guarantees do not make later ASGI/socket sends atomic
with capability revocation or recall bytes already sent.

## Verification

The focused suite passes 32 tests, including two actual PostgreSQL owner cases.
Fixtures create their own random databases and reader roles. Owners have
colliding message, thread and numeric attachment IDs. All three methods return
only the bound owner's data; metadata works for generic MIME, zero-byte and
unfetched records; missing/empty/present stored text remains distinct; manifests
retain continuation/completion fields. The real tests also prove shared capacity
is enforced. No production mailbox, raw file source, parser or provider is used.

Synthetic tests cover operation/audience separation, revoked tokens, exact public
fields, malformed/tampered reader output, bounds, output-size refusal, partial
text and stalled manifests, repeated cancellation during reader cleanup and
initial authorization, revocation during reader cleanup, and expiration or
revocation during public projection before final authorization. Eight initial
missing-module tests were confirmed failing before implementation.

```bash
GMS_GATEWAY_TEST_DSN="$SYNTHETIC_GATEWAY_DSN" PYTHONPATH=src \
  /home/ssilver/development/gmail-search/.venv/bin/python -m pytest -q \
  tests/test_gateway_attachment_read_service.py
```

## Optional private HTTP integration

Explicit `attachment_reads=` injection installs POST `/v1/attachment/meta`,
`/v1/attachment/text`, and `/v1/attachment/list`. The transport authenticates the
fixed operation before bounded JSON reads, rejects extra selectors, drains owned
HTTP cleanup, then repeats authorization/deadline checks before constructing a
response. Ten HTTP tests passed. The combined service, HTTP, publication and
actual two-owner thread-manifest transport checks passed 53 tests under root
review. No raw download or parser fallback is installed by this composition.
