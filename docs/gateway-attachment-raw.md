# Authenticated opaque attachment downloads

## Scope and composition

`attachment_raw_service.py` and `attachment_raw_http.py` implement one fixed
internal raw download operation. They reuse the owner-bound opaque source;
they do not parse files, write host files, call providers, issue signed URLs,
change database grants, or expose stored filesystem paths. Generic MIME types
and zero-byte files are supported. This does not expand parser eligibility.

Trusted composition constructs:

```python
admission = DataAdmission(global_concurrency=2, owner_concurrency=1)
raw_service = RunRawAttachmentService(capabilities, raw_source, admission=admission)
add_raw_attachment_middleware(app, raw_service)
```

`raw_source.load_raw(owner_id, attachment_id)` must be asynchronous and return
exactly `RawAttachmentInput`. Configure `OwnerAttachmentSource` with the raw
locator (`QueryAttachmentLocator.locate_raw`), preserving owner-hash path
derivation, expected-size binding, descriptor-relative no-follow opens and
mutation checks. The source must close its descriptors before returning or
acknowledging cancellation. There is no legacy shared-directory fallback.

Call the middleware helper **last**, after registering buffering HTTP middleware
and before starting the application. It installs a pure-ASGI handler outside
FastAPI/Starlette `BaseHTTPMiddleware` for exactly `/v1/attachment/raw`. Other
paths delegate unchanged. Future outer middleware must preserve direct awaited
ASGI sends; wrapping this route in a detached or buffered body producer would
move the ownership boundary and requires a new qualification. Starlette's outer
server-error middleware does not buffer these body sends.

The raw route supplies its own fixed errors, `Cache-Control: private, no-store`
and `X-Content-Type-Options: nosniff`, since it deliberately bypasses the inner
response-buffering middleware. This is application wiring, not a new listener.
No public deployment is performed by these modules.

## Request and authorization

Only `POST /v1/attachment/raw` is accepted. The decoded path and original ASGI
`raw_path` must both match the literal fixed path. Percent-encoded aliases,
trailing slashes, mounted root prefixes, and query strings are refused.

The request requires one `Authorization: Bearer <64 lowercase hex characters>`,
one `Content-Type: application/json`, and one canonical decimal Content-Length
of at most 4,096 bytes. Duplicate headers, cookies, alternate user identity,
range headers, raw-path headers, content encoding and transfer encoding are
refused. HTTP body framing is fixed; chunked transfer encoding is not supported.
The application bounds header count to 64 and aggregate header bytes to 16 KiB;
the serving HTTP stack must also bound its own ingress allocations.

The JSON body has exactly one key:

```json
{"attachment_id": 123}
```

The ID is a positive signed-int8 integer, excluding booleans. Duplicate JSON keys,
additional fields, malformed JSON, oversize input and mismatched declared length
fail before loading. The body deadline is three seconds. Capability authorization
precedes reading the body; it requires audience `attachment`, operation `raw`.
A parser-only capability does not authorize raw downloads. Owner/run binding
comes exclusively from the capability, never from request fields or headers.

Repeated authorization must retain every immutable `RunLease` field. Renewal of
`lease_expires` is allowed. The service checks authorization and its monotonic
deadline before/after each authorization await. Its independent watcher runs at
most every 50 ms. The whole operation, including preflight and send backpressure,
has a trusted duration of at most 30 seconds; tests can configure a shorter one.

## Response framing

All authorization, source validation, source cleanup, hash computation and header
construction required for a successful response finish before HTTP 200. The
response is `application/octet-stream`, with exact Content-Length. Its body is:

1. Four bytes: unsigned big-endian JSON header length, 1–4,096 bytes.
2. That many ASCII/UTF-8 JSON bytes, with the exact fields below.
3. Exactly `size_bytes` opaque payload bytes, followed by HTTP body completion.

```json
{"version":1,"operation":"raw","attachment_id":123,"mime_type":"application/octet-stream","size_bytes":0,"sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"}
```

Payloads are bounded to 0–10 MiB. MIME uses the source's bounded ASCII media-type
grammar. The packet contains no filename, source path, owner, token, arbitrary
URL or parser options. Filename display metadata remains a separate read; the
guest will choose its own generated private download path. SHA-256 is computed
over the complete opaque bytes, including the conventional empty-file digest.

Each awaited ASGI send contains at most 64 KiB of body. A fresh authorization and
deadline check precedes HTTP headers, the framed header, each payload chunk and
the final body-completion send. Hashing yields between bounded chunks so the
independent cancellation/deadline watcher can run.

Before response headers, failures produce fixed JSON errors: 400 invalid request,
401 missing/malformed authentication, 403 denied capability, 404 unavailable or
invalid source, 405 method, 413 request limit, 429 payload capacity, 504 deadline,
or 503 other service failure. Missing, foreign, malformed and inaccessible source
records share the same unavailable response. Storage/network diagnostics are not
returned. Once a start send has been attempted, failures abort the stream without
retrying headers or appending a JSON error to the binary packet.

## Payload ownership and cancellation

Construction does not authenticate, load files or acquire capacity. The actual
ASGI call owns preflight, loading and every response send. A constructed handler
that is never called therefore owns no payload or reservation.

The required shared `DataAdmission` instance permits at most two raw transfers
globally and one per owner. It is separate from database, parser and provider
capacity, and must be shared by every raw service instance in the process. It
is acquired before loading any attachment bytes. Cross-process coordination is
not implemented: multi-process deployments require a separately qualified shared
payload budget.

`RunRawAttachmentService.deliver(token, attachment_id, deadline=..., publish=...)`
keeps capacity while its producer invokes the trusted asynchronous
`publish(source, check_active)` callback. That callback must await its actual
transport sends and must not retain source bytes independently. HTTP uses the
provided owned callback, not a lazy `StreamingResponse` generator. This handles
header-send failure and disconnects before the first body iteration.

Disconnect, expiry, revocation and cancellation cancel owned tasks once, then
shield/drain the loader, publisher, authorization threads and watchers. Repeated
caller cancellation cannot interrupt this cleanup. Capacity is released only
after cleanup completes; an indefinitely uncooperative dependency keeps the
operation and capacity pending. The deadline is not a hard kill for filesystem
syscalls, SQLite threads or a noncooperative ASGI server. Trusted dependencies
must honor their documented cleanup contracts.

The producer replaces failed loader, validator and publisher exception chains
with fresh fixed errors after dropping the original tracebacks. Those traceback
frames can otherwise retain raw byte buffers beyond capacity release. The HTTP
publisher also clears its source reference in `finally`. Tests retain exceptions
and inspect weak references to verify the payload is released on failure.

Admission bounds concurrent payload owners, not total process RSS. The current
source temporarily holds a bytearray and immutable bytes during conversion, so
budget roughly twice the 10 MiB source ceiling per simultaneous load, plus chunk
buffers, Python overhead and HTTP server buffers. Payload admission remains held
through application-boundary send completion; that acknowledgment is not a
remote TCP delivery acknowledgment. Already sent or buffered bytes cannot be
recalled after revocation.

## Qualification and remaining work

Synthetic tests cover empty/max-size framing, owner/run binding, parser-capability
refusal, request and source bounds, shared owner/global capacity, deadline and
revocation before and after headers, header/body send failures, and disconnected
clients. Gated loader/send teardown tests assert admission remains held through
repeated cancellation. A complete FastAPI application test places a deliberately
failing `BaseHTTPMiddleware` inside this handler and confirms the raw route owns
the real outer send. Exception-retention regressions cover loader, validation
and publisher failures using weak references. Existing opaque source/locator
tests are run alongside the raw suites.

The candidate relay and versioned metadata/text capability configuration have
separate source qualification below. Binary guest downloader review, raw tool
activation and newly pinned runtime images remain separate gates. A downloader
must enforce the bounded frame, exact size, SHA-256 and EOF; use generated `O_EXCL`/no-follow 0600 temporary files under a private 0700
directory; remove every partial on failure; and publish its local path only
after verification. No automatic provider upload is part of raw downloading.

No real mail, providers, production application, external listener or deployed
worker configuration was used. Full attachment tool parity is not yet qualified.

### Candidate relay qualification (2026-09-15)

The candidate synthetic `vsock_http_relay.py` now forwards the fixed metadata,
text, inventory and raw attachment routes. Before uploading a raw request it
refuses a declared body above 4,096 bytes. Before forwarding raw success it
requires status 200, exactly one `application/octet-stream` Content-Type, one
canonical decimal Content-Length with `4 < length <= 10489860`, and no transfer
or content encoding. It preserves the validated length for the guest's binary
frame decoder. Truncated bodies close the connection with the original declared
length; they never append a second HTTP response or a JSON error to binary data.

Independent review found no relay blocker. Its 53 synthetic tests passed in
3.15 seconds using the original virtual environment and candidate `PYTHONPATH=src`
(`pytest tests/test_worker_vsock_relay.py -q`); Ruff passed. Persisted regression
cases include signed/zero-padded/whitespace/combined/short lengths, duplicate MIME,
non-200 statuses, coding rejection, and empty/partial/non-UTF-8 truncated bodies.
These tests use disposable local UNIX sockets and HTTP upstreams. This evidence
qualifies candidate relay source behavior; it neither updates historical runtime
pins nor establishes a deployed raw-download workflow. Versioned guest metadata/
text capability configuration is documented separately in
`guest-mail-tool-profiles.md`; binary downloader integration remains a separate
review and runtime qualification gate.
