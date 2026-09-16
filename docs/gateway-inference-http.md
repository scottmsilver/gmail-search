# Worker inference HTTP boundary

`gateway.inference_http.add_inference_routes` mounts a worker-only adapter for
already-qualified, run-bound services. The trusted process startup code supplies
the Anthropic and/or Gemini service instances and the relay's Bearer-token
extractor. `create_gateway_app(..., anthropic=..., gemini=...)` now composes
these optional services. It is not a public browser API. The worker relay accepts
the exact fixed Google URL and normalizes a single `x-goog-api-key` capability
header; duplicate or mixed credential headers are rejected.

## Fixed requests

- Anthropic: `POST /v1/messages` with no query parameters.
- Google: `POST /v1beta/models/gemini-3.8-flash:streamGenerateContent?alt=sse`.
  The one `alt=sse` pair is required; all other or repeated parameters fail.

The adapter authenticates the run capability before it reads a request byte. It
then takes a non-queuing local admission slot, reads at most 8 MiB of JSON in
three seconds, and refuses duplicate keys, non-object roots, non-finite JSON,
and the wrong media type. Profile, owner, model, upstream address, effort, and
client compatibility profile are not HTTP request parameters: the underlying
run service derives them from immutable server bindings.

Both routes return the service's already-validated SSE bytes incrementally.
One bounded first chunk is prefetched so pre-stream authorization, profile,
compiler, reservation, replay, and upstream failures can have a sanitized HTTP
status. A later provider error terminates the SSE response without a fabricated
completion event or provider diagnostics. Client disconnect and task
cancellation close the service iterator before its budget settlement is allowed
to finish.

## Request keys and retries

`X-Gateway-Request-Key` is optional and, when present, must be a single ASCII
key matching `[A-Za-z0-9][A-Za-z0-9._:-]{0,127}`. The adapter echoes the
accepted/generated key in its response. The current relay drops unknown client
headers, and the qualified Pi/native Claude captures send no custom request-key
header, so an omitted key receives a server-generated UUID key.

A supplied stable key can prevent duplicate dispatch: an already-reserved key
returns `409` and is never retried by this adapter. There is no durable result
spool, so it cannot replay a prior stream. Retrying a request without a stable
key is a new billable provider call. The adapter itself never retries upstream
requests.

## Local admission

Before reading a body, each authenticated lease takes one of four process-wide
slots and one of two slots for its derived owner. The single `_Admission`
instance is shared by both provider routes, rejects excess work with `429`, and
holds its slot through prefetch, SSE streaming, cancellation, iterator close,
and settlement cleanup. This is deliberately process-local and assumes one
ASGI event loop; multi-process deployment needs an upstream/controller limit as
well. It deliberately queues neither request bodies nor excess requests.

## Staged limitations

The boundary has synthetic HTTPX `MockTransport` coverage for the captured
native Claude and Pi/Gemini request shapes. It does not make a live provider
call, qualify arbitrary CLI versions, persist completed SSE streams, or offer a
token-count endpoint. Token totals are not inferred from request text: provider
usage is settled only by the qualified run services' trusted terminal metadata;
missing or interrupted usage conservatively consumes the reservation.
