# Explicit persistent MCP raw profile

## Profile boundary

The source profile `mail-raw-mcp-v3` adds opaque attachment downloads to the
persistent guest MCP core. It has an explicit configuration version of 3:

```json
{"version":3,"tool_profile":"mail-raw-mcp-v3","capabilities":{"sql":"<token>","retrieval":"<token>","artifact":"<token>","attachment":"<token>"}}
```

Tokens are still separately issued, opaque 64-character lowercase hexadecimal
capabilities. No model argument selects a profile, owner, credential or route.
The version/profile pair and exact capability audience set must match. Trusted
bootstrap additionally carries its existing fixed `runtime` selector and must
explicitly request `expected_profile='mail-raw-mcp-v3'`; a default legacy
entrypoint refuses v3. The bounded one-shot bootstrap framing is unchanged.

Historical bare three-capability v1 files retain `legacy-mail-v1` and its four
tools. Version 2 retains `mail-read-v2`, eight tools, and attachment meta/text
only. Their parser, runtime image and qualification pins are unchanged. This
source change does not build an image, wire a trusted capability issuer, activate
a production caller or qualify a native agent runtime.

The v3 constructor requires loopback port **18080** and a trusted timeout no
greater than 30 seconds. It cannot silently send JSON to 18081 while the binary
transport uses 18080. Existing v1/v2 port and timeout choices remain available.

## Tools and argument contract

All eight tool names remain exact:

1. `describe_schema`
2. `sql_query_batch`
3. `get_thread_batch`
4. `publish_artifact_batch`
5. `search_emails_batch`
6. `find_facts`
7. `query_emails_batch`
8. `get_attachment_batch`

Only v3 adds this closed item variant to the existing attachment batch schema:

```json
{"attachment_id":123,"mode":"raw"}
```

It accepts a positive signed-int8 ID and no offset, limit, filename, path, MIME,
URL, owner or parser options. Meta/text behavior and the default text mode remain
unchanged. Raw attachment IDs must be unique within one batch; duplicates reject
the batch before any I/O. An ID may still appear once in raw mode and separately
in metadata/text mode.

The tool retains the existing ordered envelope:

```json
{"results":[{"input":{"attachment_id":123,"mode":"raw"},"result":{"relative_path":"gms-downloads-<random>/attachment-<random>.bin","size_bytes":123,"sha256":"<digest>"}}]}
```

Raw results contain only the generated relative path, byte size and SHA-256, or
a fixed per-item error. No payload/base64, source filename/path, MIME or signed
URL enters model context. A later explicit guest file operation can inspect its
own bytes; automatic provider upload is not part of this tool.

MCP `tools/list` exposes the raw schema only for v3. The initial structural
argument check recognizes the closed union before configuration is loaded, but
the selected profile is checked again before dispatch. Guessing raw arguments
with v1/v2 still fails before attachment network access or download workspace creation.

## Persistent ownership and shared capacity

One `GuestMailMCP` instance retains one `GuestMailTools` core. The core constructs
one `asyncio.Semaphore(2)` shared by JSON and binary operations. It lazily opens
and holds its trusted workspace directory with no-follow/directory flags on the
first raw request. The downloader requires UID ownership and mode 0700 before
any connection, creates its private child directory and receives this same
semaphore. The core retains its original workspace descriptor until teardown;
the downloader owns its duplicated descriptors.

One mixed attachment batch groups all valid raw IDs into a single
`download_many` call, sharing its cumulative 20 MiB attempted-payload budget.
Metadata/text items use the existing JSON request path. The raw group does not
enter the JSON per-item semaphore wrapper, avoiding a double-acquisition deadlock.
Both groups use the same absolute deadline, share two active socket slots, and
reassemble results in input order. Binary payload bytes go to files, while the
small returned path metadata also counts against the existing JSON result budget.

The downloader and its 20-file/64 MiB counters persist across calls on that core.
These are **per-instance** limits, not durable per-run accounting. The exact file,
frame, descriptor, cleanup and same-UID limits are documented in
`gateway-guest-attachment-download.md`.

Every raw group and JSON child is cancelled/drained before a mixed batch returns.
A final local deadline/cancellation check follows that drain. If slow JSON cleanup
expires the deadline after a raw file was verified, the core returns a fixed
deadline error rather than publishing its path late. That fully verified file
may remain charged because the downloader already returned it to the core.
Likewise, an outer MCP response lost after a successful core return can leave
verified charged files. There is no atomic filesystem-plus-RPC handoff claim.

## Shutdown contract

`GuestMailTools.aclose()` stops new dispatches, cancels/drains tracked dispatches,
awaits the persistent downloader's cleanup, and only then closes its original
workspace descriptor. Its close operation runs in a separately owned task;
repeated caller cancellation cannot interrupt cleanup. A close failure leaves
the core closed and its unresolved resources retained. A later trusted close
can retry supported cleanup, without reopening tool admission.

An uncertain workspace `os.close` is never retried while that descriptor number
is still valid: it could already refer to a different open handle, including
another handle for the same directory. Capacity/resource retention and guest
teardown are preferable to closing an unrelated descriptor. Observed `EBADF`
can acknowledge that the number is closed.

`GuestMailMCP.close()` similarly owns its complete close task, rejects new calls,
cancels/drains active MCP calls and always calls `aclose()` on a constructed core.
This holds even when `close()` is called directly, without the stdio main loop's
outer drain. A missing core lifetime API or failed raw cleanup is an error,
not a reason to silently skip teardown. Test doubles explicitly implement the
lifetime contract. Native runtime/controller termination remains the final
recovery mechanism when cleanup cannot be acknowledged.

## One-shot CLI and trusted issuer requirements

The one-shot CLI wrapper rejects **all v3 invocations before core construction**.
It continues to support v1/v2 and now closes those cores after each invocation.
The private configuration reader itself accepts v3 so the persistent MCP entrypoint
can consume it; the CLI rejection occurs at invocation, not in the shared reader.

No durable cross-process raw quota is implemented. A new CLI/process could create
a new downloader and fresh counters, so CLI raw activation remains gated. This
restriction is not an isolation boundary against a malicious same-UID shell
inside the VM; the VM disk/tmpfs/resource ceiling bounds its independent writes
and process restarts.

Future trusted controller composition must deliberately issue the attachment
capability for `meta`, `text`, and `raw`, with a bounded TTL under the active run's
deadline and controller renewal/revocation. It must provide only that run owner's
capabilities. The profile does not request `parse`; renderer/parser integration
is separate. This document specifies the required issuer contract without adding
an unused production composition function.

## Qualification status

Synthetic tests cover exact version/profile selection, preserved v1/v2 behavior,
v3-only schema discovery, fixed port/deadline refusal, mixed-batch grouping/order,
shared raw/JSON socket admission, repeated-call quota persistence, invalid raw
options, private workspace checks and explicit CLI rejection. They exercise MCP
and direct core close during already-started socket teardown, repeated cancellation,
cleanup failure/retry, uncertain workspace-close descriptor reuse, and mixed-batch
deadline expiry after a completed raw download.

The existing standalone downloader and real synthetic encoder/relay contract
suites run alongside these tests. No real mail, provider call, new runtime image,
production application or deployment was used. A separately pinned image and
actual native agent/MCP workflow qualification remain required before activation.
Rendered pages, additional isolated parser formats and full attachment parity
remain separate work.
