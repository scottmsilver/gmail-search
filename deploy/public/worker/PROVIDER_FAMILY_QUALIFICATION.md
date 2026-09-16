# Provider families and staged qualification

Inventory from `web/lib/config.ts`, `web/lib/battleVariants.ts`,
`deploy/pi/models.json` and the three agent runtime modules, 2026-09-15.

| Advertised selection | Wire API required | Staged qualification |
| --- | --- | --- |
| Pi `google/gemini-3.8-flash` (default) | Google `models.streamGenerateContent` | Strict text/function normalizer, fixed HTTP/SSE adapter, mocked service integration |
| Pi `openrouter/meta/muse-spark-1.3` | OpenRouter OpenAI-compatible Chat Completions | Missing |
| Pi `anthropic/claude-opus-5` | Anthropic Messages | Family implemented; this specific model and adaptive thinking remain unqualified |
| Claude `sonnet`, `opus`, `haiku`, `opusplan` | Anthropic Messages | Pinned native CLI synthetic exchange qualified for fixed Sonnet 4.6/high only; all alias mappings and plan/execution transitions remain to qualify |

Battle selects three native aliases (excluding opusplan) and all three Pi models.
There is no direct OpenAI choice in the current picker. Historical OpenRouter
Gemini/Anthropic aliases normalize to the direct provider choices.
Runtime prefix-based context estimates are not an authorization allowlist.

## Google implementation

`gateway/gemini.py` has a separate immutable `GeminiProfile`, persisted in the
same per-run profile slot with an explicit `google-generate-content` family.
Cross-family binding and reads fail. Anthropic profiles carry an explicit family;
older profile JSON without that field retains its existing Anthropic meaning.
No old profile is interpreted as Google. Shared service hooks reuse authorization,
reservation, deadlines and cancellation only; they do not translate Google bodies
into Anthropic messages.

The server fixes the exact model `gemini-3.8-flash`, endpoint, thinking level,
token ceilings and integer accounting rates. Defaults use HIGH; the validated
client hint cannot change it. Guest bodies cannot set a model or upstream URL.
Local function declarations, calls/results, text, thought signatures and thought
text are supported. Unknown fields, remote files, hosted code/search, caches,
multiple candidates, images/audio/video and additional billing dimensions reject.
Text and function JSON can contain inert URLs. No schema reference is fetched.

The HTTP adapter sends the server credential in `x-goog-api-key` only. It bypasses
client default cookies/auth/query settings, disables environment proxies and
redirects, and its owned client has zero retries. HTTP/frame/body/service limits
and cancellation close the upstream response. No credential is placed in a URL
or runtime configuration.

Reserve full configured input context plus requested output before opening HTTP.
Only terminal usage followed by clean EOF settles actual usage. Output charge
includes `candidatesTokenCount + thoughtsTokenCount`; unknown, contradictory,
cached or tool-use usage conservatively charges the full reservation. Current
strict SSE contract requires one candidate, an exact modelVersion and terminal
usage on the finish frame. Live-provider deviations require explicit qualification.

## Evidence and release limitations

`tests/test_gateway_gemini.py` uses real HTTPX Request/Response and fragmented
MockTransport SSE through the capability/budget service. It checks fixed routing,
server thinking, header isolation, thinking-token accounting, immutable family
bindings, replay, revocation, timeout/cancellation, redirects, compression and size
limits. No provider call is billed. The public runtime package's Pi 0.84.4 Google
converter was inspected and uses functionDeclarations/parametersJsonSchema;
there has been no actual Pi Google mock conversation yet.

This candidate does **not** establish full model-menu parity. Keep full release
gated on real Pi Google exchange, inline images, complete model/effort profiles,
OpenRouter, native alias routing and full-session tests. Do not silently remove
choices or substitute models to pass the gate. Integer test rates are synthetic,
not published prices; live rates and complete billable ceilings require trusted
configuration before any live call.

Primary references checked 2026-09-15:

- https://ai.google.dev/api/generate-content
- https://ai.google.dev/gemini-api/docs/function-calling
- https://ai.google.dev/gemini-api/docs/thinking

The API reference currently advertises Gemini 3.8 Flash and documents the
generateContent streaming endpoint, thinking configuration and usage dimensions.
This verifies the protocol reference, not account-specific availability.
