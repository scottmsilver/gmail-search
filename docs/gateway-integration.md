# Gateway integration checkpoint

The candidate remains isolated from live public/private services. These routes
are worker-facing, require run capabilities, and are mounted only by explicit
dependencies on `create_gateway_app`. They are not Internet browser endpoints.

## Completed connections

- `POST /v1/thread`: `retrieval` audience, `thread.get` operation, server-derived
  owner, fixed text projection through the existing compiler and immutable owner
  reader. Supports bounded message/body pagination with explicit incomplete and
  pagination-limit flags. No HTML parsing, raw paths or privileged fallback.
- HTTP disconnect cancels and awaits retrieval cleanup. The route was tested
  with an ASGI disconnect event, not just direct coroutine cancellation.
- The synthetic vsock relay forwards only the exact `/v1/thread` path to its
  fixed upstream. It does not accept caller-selected upstreams or owner headers.
- `QueryAttachmentLocator` selects bounded attachment metadata with the existing
  owner reader and derives `attachments/owners/sha256(owner)/message/filename`.
  No new database grant or `raw_path` selection is needed.
- `OwnerAttachmentSource` follows that binding with descriptor-relative,
  symlink-free traversal, regular-file/hardlink checks, bounded reads, mutation
  detection and cancellation cleanup. It does not decode attachment bytes.
- Optional `POST /v1/attachment/parse` composes capability authorization, the
  owner reader, bounded file loader and parser-job interface. The two-owner HTTP
  test uses a synthetic parser job. Separately, the actual cancellable VM backend
  passed launch/cancel/teardown qualification with the rebuilt image. Output is bounded text plus base64 page bytes, without host decoding.

## Parser lifecycle qualification

The real backend now returns a job handle before launch/I/O, interrupts sockets
on cancellation, stops and reaps the VM, joins the worker, and only then releases
its lock. Failed teardown keeps its job binding and admission for retry. An
independent review and real nested-VM cancellation/long-text/watchdog checks
passed. The controller service separately reserves per-owner/global admission
before reading files, shields cleanup against repeated cancellation and checks
authorization after teardown. These counters require one event loop/process.

The rebuilt parser image includes the PDF truncation correction; its SHA256 is
`89bf4da702506dadacc5cd08b9f4d737f27abb592eb35b7a768cc22d590f9068`.
The prior image remains preserved. See the SSH qualification record for the latest outer-VM lifecycle evidence.

## Remaining in this slice

- Qualify deployment-wide admission and the persistent production worker service.
  The composed synthetic SSH/parser path now passes actual nested-VM checks;
  production provisioning remains separate.
- Qualify public preview behavior and legacy storage layouts. New-layout lookup
  fails closed when a derived file is missing; it does not prove ingestion
  provenance and does not search the old shared attachment tree.
- Qualify remaining retrieval/search parity and public preview handling.
- Connect the remaining retrieval/search tools, runtime orchestration, browser
  sessions/conversations and production worker service; finish all release gates
  in the approved plan. This checkpoint does not enable invitees or full tools.

## Verification and review

Synthetic PostgreSQL tests use colliding thread/message/attachment IDs for two
owners. Loader tests cover foreign paths, symlinks, hard links, FIFO files,
oversized/growing files and cancellation. Independent architecture/spec and
code-quality reviews found and prompted fixes for unusable pagination cursors.
Current focused counts are recorded in the implementation-status document;
they do not substitute for full-system qualification or actual Google consent.

## Integration verification, September 15

Large messages now use a closed, parameterized DB-side `substr` projection,
fetching only the requested body page and full character length. Two-owner tests
cover bodies above 2 MB and continuation beyond the former offset ceiling.

`create_gateway_app` optionally mounts reviewed Anthropic/Gemini streaming
services supplied by trusted startup. The relay accepts the exact fixed Gemini
streaming URL and normalizes one of Bearer, `x-api-key`, or `x-goog-api-key` into
a single Bearer capability. Mixed or duplicate credentials fail closed. Provider
keys stay in the trusted gateway. Request-key/retry limitations are documented
in `gateway-inference-http.md`.

The host `SSHAttachmentBackend` durably binds jobs before upload and talks only
to a pinned-host-key, service-specific SSH forced command. The worker's private
Unix-socket manager validates bounded frames, binds OS peer identity and immutable
job context, and renews a short independent VM lease only on increasing poll
sequences. Cancellation needs a stop acknowledgment before capacity is released;
unresolved jobs are retained for reconciliation. No capability, mailbox path,
provider key, or host private SSH key enters the parser VM.

Independent adapter review passed, including stop-before-start tombstoning.
Actual SSH-to-nested-VM qualification returned synthetic PDF text and image bytes,
renewed across the initial lease, suppressed results after capability revocation,
and reaped the VM when the manager was frozen and polling ceased. Restricted SSH
checks denied shell execution, SFTP, PTY, and forwarding. This is synthetic
qualification, not a deployed production service.

Combined gateway, attachment, and relay verification with the private synthetic
PostgreSQL fixture: **548 passed, 12 skipped**. Focused inference/relay integration:
**35 passed**. These counts overlap. No production database or live provider was
used. Full application/runtime and two-account browser qualification remain open.

The relay now preserves the artifact upload endpoint's expected `201 Created`
response; a real-upstream fixture replaces the old synthetic sink bypass in the
regression. Other relay operations still require `200`, and upstream errors are
still sanitized to `400` rather than exposing provider diagnostics.

Optional `POST /v1/events` accepts a bounded event object using an `events` /
`append` capability. It authenticates before reading the body, derives the run
and owner from the capability, and rechecks the fence in the SQLite transaction.
Repeated request cancellation drains the write before the handler exits. The
relay admits only the exact POST path. Browser replay remains a separate trusted
adapter; guest usage events are display data and never determine billing.
Focused event/relay/artifact HTTP verification: **29 passed**.

The candidate frontend production build passed after applying passive Markdown
rendering to both answers and expanded reasoning. Remote image syntax and raw
HTML cannot cause automatic image loads through those Markdown components;
independent review and two rendering regressions passed. This build does not
establish that the new full-agent runtime is connected to the browser yet.

## Invited browser result adapters

`auth.result_routes.create_result_router` adds separate candidate paths for
`GET /api/agent-artifacts/{id}?conversation_id=...` and
`GET /api/agent-events/{run_id}?conversation_id=...&after=...&limit=...`.
It is not mounted in the existing public application. Authentication uses the
invited identity store's opaque browser session; worker Bearer tokens and
owner-selection headers do not authenticate these routes. Both storage calls
check the exact owner/conversation, and the session is checked again after the
read before returning bytes. Foreign/unknown objects share a sanitized response.
Artifacts retain forced attachment/octet-stream/sandbox headers. Event pages are
bounded to 100 entries and preserve an explicit replay cursor. Database/file
thread work is drained if the request is cancelled. Initial synthetic isolation,
mid-read revocation, replay and artifact checks: **13 passed** including existing
store/worker-route tests. Independent review found no ownership or privacy defects in these browser routes.

See `guest-mail-tools-integration.md` for the reviewed four-tool guest client and
one-shot invocation wrapper. The next actual-VM qualification uses real scoped
mail/artifact services and synthetic provider responses; it is not a billed
provider test or proof of complete typed-tool parity.

## Review completion after retry

The stdio MCP adapter now shares one core across concurrent calls, drains an
output frame before releasing its write lock on cancellation, and poisons a
stalled transport after a five-second output deadline. Repeated shutdown signals
drain outstanding calls. Oversized response errors retain the validated request
ID so clients can finish the correct request. All findings were reproduced and
fixed with regressions, then independently reviewed. Combined guest client, CLI,
MCP, bootstrap, browser-result and controller-cleanup checks: **69 passed**.

The actual Pi/Alice and Claude/Bob workflow qualification is documented in
`deploy/public/worker/GUEST_MAIL_TOOLS_QUALIFICATION.md`. It uses real owner-scoped
mail and artifact services with synthetic data and a mocked inference endpoint.
It exercises the Bash wrapper, not the newly added typed MCP adapter inside the
VM. The latter still needs actual runtime integration.

The relay now propagates guest disconnection while waiting for upstream response
headers or body bytes. A bounded per-request watcher shuts down the captured
upstream socket; watcher cleanup precedes releasing relay capacity. The fixed
guest client/bridge keep the connection open through the response, and no HTTP
pipelining or request-side half-close is supported by this profile. Synthetic
header/body cancellation tests and an independent SSE disconnect test passed.
Combined relay/retrieval/event/inference-lifecycle checks with the private
PostgreSQL fixture: **49 passed**. This changes the outer relay, not an image pin;
historical actual-VM records retain their original source hashes.
