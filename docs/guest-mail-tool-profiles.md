# Guest mail tool profiles (candidate source only)

The candidate now has an explicit profile boundary. This source change does not qualify a new guest image. Historical image pins, qualification reports, runtime profiles, and the four-tool Pi extension remain unchanged. Full raw/rendered attachment tools and new runtime qualification are separate work.

## Closed configuration

`deploy/public/worker/guest_tool_config.py` is the shared standard-library parser. It copies tokens into an immutable configuration with a secret-free representation. Tokens are the existing 64 lowercase hexadecimal capability format. No URL, port, owner, role, model, path, provider secret, or extra audience is accepted in configuration or tool arguments.

| Profile | Persisted capabilities.json | Advertised tools |
| --- | --- | --- |
| `legacy-mail-v1` | Historical bare object with exactly `sql`, `retrieval`, `artifact` | `describe_schema`, `sql_query_batch`, `get_thread_batch`, `publish_artifact_batch` |
| `mail-read-v2` | Exactly `version: 2`, `tool_profile: "mail-read-v2"`, and `capabilities` containing exactly `sql`, `retrieval`, `artifact`, `attachment` | The four legacy tools plus `search_emails_batch`, `find_facts`, `query_emails_batch`, `get_attachment_batch` |

A bare four-token object is invalid. A version-one envelope, mixed version/profile, missing audience, extra field, or noncanonical token is invalid. Existing unpinned search/facts/metadata fixtures now use explicit v2 configuration; historical three-token configurations do not advertise these capabilities.

The trusted startup bootstrap is distinct from the persisted file:

- V1 remains exactly `{version: 1, runtime: "pi" | "claude", capabilities: <three tokens>}`.
- V2 is exactly `{version: 2, runtime: "pi" | "claude", tool_profile: "mail-read-v2", capabilities: <four tokens>}`.

`guest_run_bootstrap.read_config`, `receive`, `validate_config`, and `persist_config` default to `expected_profile="legacy-mail-v1"`. All existing host/guest entrypoints retain this default and reject v2 before launching their workflows. Explicit `expected_profile="mail-read-v2"` is currently exercised only through source tests; there is no new v2 entrypoint or image profile. The bootstrap frame remains length-prefixed, at most 4096 bytes, closed-schema JSON, and requires EOF.

`persist_config` preserves the selected version through `write_capability_file`. The writer creates capabilities.json exclusively in an existing private 0700 run directory, uses directory-relative no-follow opens, verifies ownership and a single regular link, and creates mode 0600. It never overwrites an earlier run's file. Startup must fail if persistence fails. The CLI retains its independent private-directory/file checks and 4096-byte read cap at fixed `/tmp/gms-run/capabilities.json`; workspace remains `/tmp/gms-run/work`. Trusted future startup is responsible for ensuring mounted gateway operations match the profile and issuing run-bound audience tokens. The envelope is not a route-readiness test.

## Core, CLI and MCP

`GuestMailTools` accepts a parsed configuration or its exact persisted shape. Its read-only `tool_profile` and `tool_names` properties govern dispatch. Legacy thread requests also reject the v2-only `attachment_after_id` and `attachment_limit` fields.

The CLI parses the fixed private file once per invocation. MCP lazily creates one core and caches one profile on the first tools/list or valid tools/call. Subsequent listing and calls share that core, its two-socket admission limit, and the original profile. Unavailable configuration produces a fixed error without advertising tools; malformed calls are rejected before opening configuration. The MCP tool schemas are filtered by profile, including omission of attachment inventory paging from v1 thread arguments.

Existing fixed loopback routing, input/output framing, response budgets, deadline handling, repeated-cancellation drain, and no-retry behavior remain in force. There is no model-selected config path, endpoint, or port. Only the three existing runtime-builder copy loops gained `guest_tool_config.py`, to keep future v1 builds internally coherent; no images were built.

## Attachment metadata and stored text mapping

`get_attachment_batch` accepts exactly `{items: [...]}`, with 1–20 items. Each item accepts only:

| Field | Meaning |
| --- | --- |
| `attachment_id` | Required integer, 1 through 9223372036854775807 |
| `mode` | `text` by default, or `meta` |
| `offset` | Text character offset, default 0, range 0–2147483646 |
| `limit` | Text characters, default 20000, range 1–100000 |

Metadata mode rejects any supplied offset or limit, including default-valued ones. Text mode posts `{attachment_id, offset, limit}` to `/v1/attachment/text`; metadata mode posts `{attachment_id}` to `/v1/attachment/meta`. Both use only the attachment-audience token. Trusted v2 composition initially issues metadata/text operations only; raw/parse issuance and guest binary/file handling are not part of this slice.

Results preserve input order and the existing envelope: `{results: [{input: <item>, result: <gateway JSON or fixed error>}]}`. The core preserves absent (`null`) versus empty stored text and all gateway completeness/paging metadata. It neither starts extraction nor falls back to raw bytes. Unsupported raw/rendered modes or selector fields return per-item errors without a socket request. MCP additionally rejects invalid item schemas before dispatch, as it does for the other batch tools.

The existing per-instance two-request limit spans all tool types. A batch has one 30-second default deadline, at most 20 items, a 4 MiB per-response byte limit, and an 8 MiB aggregate response budget. Cancellation closes and drains owned local sockets/tasks before completing. Gateway errors are fixed and redacted; there are no automatic retries or claims that disconnected remote work did not run.

## Verification

Tests exercise legacy/v2 parser rejection and persistence, default legacy bootstrap refusal of v2, profile-specific listing/calls and cached-profile stability, official MCP SDK stdio interoperability using a synthetic private v2 file, attachment audience/routes/paging, null/empty preservation, invalid-mode refusal, redacted errors, socket cancellation, and existing framing/admission regressions. All are synthetic and make no provider calls. Run the focused source suite with the repository's original virtual environment and candidate `PYTHONPATH=src`.
