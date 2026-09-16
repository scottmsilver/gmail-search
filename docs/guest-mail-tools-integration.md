# Guest mail tool integration

The candidate's `deploy/public/worker/guest_mail_tools.py` is a guest-only,
standard-library client. Trusted startup supplies separate run capabilities for
SQL, retrieval, artifact and (in v2) attachment operations. Model arguments cannot choose an owner,
server, port, session, or credential. The client connects only to the fixed
loopback bridge, which forwards to the closed worker relay.

| Tool | Operation | Capability audience / operation |
| --- | --- | --- |
| `describe_schema` | GET `/v1/schema` | `sql` / `schema` |
| `sql_query_batch` | POST `/v1/sql` | `sql` / `query` |
| `get_thread_batch` | POST `/v1/thread` | `retrieval` / `thread.get` |
| `search_emails_batch` | POST `/v1/search` | `retrieval` / `search` |
| `query_emails_batch` | POST `/v1/query-emails` | `retrieval` / `query.emails` |
| `find_facts` | POST `/v1/find-facts` | `retrieval` / `facts.find` |
| `publish_artifact_batch` | POST `/v1/artifacts?filename=…` | `artifact` / `artifact.commit` |
| `get_attachment_batch` | POST `/v1/attachment/meta` or `/v1/attachment/text` | `attachment` / `meta` or `text` |

The table describes the explicit `mail-read-v2` profile. Historical bare
three-token configuration selects `legacy-mail-v1`, which exposes only schema,
SQL, thread retrieval and artifact publication. See
[the closed profile contract](guest-mail-tool-profiles.md). Existing runtime
entrypoints reject v2 until a new image is qualified.

The separate `mail-raw-mcp-v3` profile retains these eight names and adds a
closed `mode: "raw"` item to `get_attachment_batch`. It runs only in persistent
MCP, shares the core's two request slots, and returns a verified private guest
file path, size and hash. The one-shot CLI rejects v3. See
[the v3 source contract](gateway-guest-raw-profile.md). It has not been built
into a qualified runtime image.

The facts tool takes one query and returns a single result object. Other tools
keep their existing single/batch contracts. Batches contain at most 20 items. One client instance permits two simultaneous
requests, includes admission in the absolute deadline, bounds response bytes,
and drains aborted sockets on cancellation. Interrupted operations can have
unknown outcomes; the client does not retry uploads or queries automatically.
Thread access preserves explicit message/body paging and completeness metadata.
It returns text only and rejects unsupported raw/Markdown/message-selection
options. Artifact files are opened inside the guest workspace with directory
file descriptors and symlink/FIFO/traversal checks, then uploaded as bytes.
Downloads remain attachments with `application/octet-stream`; model MIME
claims do not turn them into active browser content.

## One-shot invocation

Run `python -I /tmp/runtime/guest_mail_tool_cli.py` inside the guest, passing one
JSON object on stdin and closing stdin:

```json
{"name":"get_thread_batch","arguments":{"thread_ids":["example"],"body_limit":20000}}
```

Trusted guest startup installs `/tmp/gms-run` as mode 0700 owned by the runtime
UID, `/tmp/gms-run/capabilities.json` as a regular single-link mode 0600 file with
the closed profile envelope, and `/tmp/gms-run/work` as its workspace. Legacy
configuration contains exactly three audience tokens; v2 contains a version,
profile name and exactly four audience tokens.
The wrapper uses port 18080, accepts no configuration flags or environment-based
routing, and caps raw stdin before JSON parsing. Input and output each have a
five-second pipe deadline. SIGTERM/SIGINT cancel and drain the client. Output is
one bounded JSON object; malformed Unicode/nonfinite values become a fixed error
rather than an exception traceback. Tokens and private configuration are never
printed. A one-shot process has its own local concurrency limit; the relay and
gateway remain the authorities for aggregate admission.

## Verification and release boundary

The current facts HTTP, guest core, MCP, CLI and relay suite passed 101 tests.
The private HTTP search/facts composition also passed tests against both owners
in disposable partitioned PostgreSQL with native ScaNN and synthetic embeddings.
These results cover routing, capability binding, limits and result isolation;
they are not actual-provider or rebuilt-runtime qualification.

`guest_mail_mcp.py` exposes four legacy or eight v2 tools over bounded stdio JSON-RPC.
It shares one core across calls, limits active calls, drains partial response
writes on cancellation, stops stalled output transports, and preserves request
IDs in oversized-response errors. Client cancellation closes its gateway socket.
The gateway drains its owned service/transport work before the final permission
check and response construction. Later ASGI/network delivery cannot be atomic
with revocation and already delivered bytes cannot be recalled.

The historical native Claude and Pi VM proofs exposed the original four mail
tools plus Bash, using mocked inference. Their source/image pins remain unchanged.
See the worker's `GUEST_MAIL_MCP_QUALIFICATION.md` and
`GUEST_PI_MAIL_MCP_QUALIFICATION.md`. Search and facts require fresh builds and
actual workflow qualification before a release can claim those tools work in
an isolated runtime.

Raw/rendered attachment tool integration, richer mail formats,
workspace restore, runtime-event translation and browser artifact/conversation
wiring remain to be completed before full-tool release. The public owner-only
runtime has not been replaced by this adapter.

## Structured filtering addition

`query_emails_batch` accepts up to 20 closed filter objects and uses the same
retrieval token, request slots and aggregate response budget. The guest/MCP/CLI/
relay suite now passes 103 tests. The optional metadata HTTP route and transport
publication suite pass 18 tests. The underlying metadata service passed
independent review and actual database checks; this does not update the pinned
runtime images or qualify production tool parity.

## Thread attachment paging

The v2 guest forwards bounded `attachment_after_id` and `attachment_limit`
options for thread calls. The guest core/MCP/CLI/relay suite passes 110 tests.
The service accepts these options only when its optional attachment reader is
installed (nondefault paging otherwise fails). Messages expose attachments from
that page, with an explicit `attachments_complete` flag; the separate
`attachment_inventory` also includes rows for messages outside the body page.
Inventory cursors and completeness do not change the existing body-completeness
flags. Message bodies and inventory use separate owner-bound snapshots.

## Latest source checks

Independent profile/config/bootstrap/core/CLI/MCP and compatibility tests passed
168 tests. The relay's 32 tests also pass with exact metadata/text/list/raw
routes, a 4096-byte raw request limit, strict upstream binary framing and
preserved Content-Length. These are synthetic source checks. The subsequently
reviewed v3 source integrates the downloader into persistent MCP; v1/v2 do not
advertise raw mode. Its final independent config/bootstrap/core/MCP/CLI/v3/
downloader/encoder-relay run passed **207 tests** after directory-descriptor
cleanup fixes. This does not change historical runtime image qualification.
