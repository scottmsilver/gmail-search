# Native Claude typed MCP: actual synthetic VM qualification

On 2026-09-15, native Claude discovered and called the guest's typed MCP tools
inside an actual jailed Firecracker VM. This extends the earlier Bash wrapper
proof without replacing its image or evidence. The MCP process used the reviewed
`guest_mail_mcp.py` adapter, the existing guest tool core, and real authenticated
host SQL, retrieval and artifact services. The provider was a deterministic
mock; inference gateway behavior was not under test.

## Observed sequence

The native Claude client advertised and invoked this exact sequence:

1. `mcp__mail__describe_schema`
2. `mcp__mail__sql_query_batch`
3. `mcp__mail__get_thread_batch`
4. `Bash`, solely to create a CSV from the verified preceding MCP results
5. `mcp__mail__publish_artifact_batch`

The mock provider validated each returned tool result before emitting the next
call. Schema, SQL, retrieval and publication were actual typed MCP calls, not
Bash invocations of the wrapper. The artifact ID returned by MCP matched the
committed record in the real `ArtifactStore`.

The disposable PostgreSQL database had Alice and Bob records with colliding
message and thread IDs. This fresh VM received only Alice's SQL, retrieval and
artifact capabilities. Its queries returned one Alice message and body, and it
published this exact 68-byte CSV (CRLF line endings):

```csv
id,subject,body
shared-message,ALICE_ONLY_SUBJECT,ALICE_ONLY_BODY
```

Bob's records were absent from results and provider request bodies. Artifact
read as Bob was denied. The run took 6.25 seconds and six mock provider requests.
The host revoked its three capabilities after completion. The outer controller
acknowledged teardown and the final backend inventory was empty.

`guest-mail-mcp-qualification.json` records the discovered tool names, exact
sequence, artifact bytes, enforcement, source hashes and cleanup evidence.

## Separate pinned image and fixed configuration

`SyntheticAgentMCPBackend` selects only the `agent_mcp` profile and fixed
read-only `agent-mcp.squashfs`, with SHA256:

```text
54303c8873abc96c27ea8cc99930e4713016a5d1b091376ed858a472f58ba7ba
```

The existing agent, attachment and Bash-wrapper tool images and pins were
preserved and reverified after the run. The new image contains only the retained
public runtime and explicitly listed guest Python sources. No capability or
private key is built into the image; no guest-modified filesystem is extracted
or mounted on the host.

`guest-mail-mcp-smoke.py` accepts the existing strict one-shot bootstrap only
with `runtime: "claude"`. The three audience-separated capabilities arrive over
the same fixed per-jail vsock8002 path and are installed as mode0600 under the
UID1000-owned mode0700 `/tmp/gms-run`. Configuration is bounded to 4096 bytes,
rejects duplicate/unknown fields and trailing input, and has a five-second
absolute read deadline. The bootstrap listener closes after one delivery.

Claude receives a fixed local stdio server named `mail`, launched with:

```text
/usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 /usr/bin/python3 -I /tmp/runtime/guest_mail_mcp.py
```

The MCP process inherits no CLI environment or platform credentials. It reads
only the fixed ownership-checked capability file. Claude's own provider key is
a fixed nonsecret synthetic value. Its built-in tool list contains Bash solely
for CSV creation; four MCP tools come from the adapter's discovery response.

## Boundary and cleanup

The actual VMM ran as UID65534 with seccomp mode2, a private PID namespace,
1 GiB memory, zero swap, 128 processes and one CPU. The fixed machine has no NIC;
guest CLI and MCP processes run as UID/GID1000. HTTP follows the fixed guest
loopback bridge, vsock8000, reviewed outer relay, and a temporary loopback-only
admin SSH reverse forward to the ephemeral host gateway.

The real gateway used separate PostgreSQL reader roles and the real
`RunQueryService`, `RunRetrievalService`, and `ArtifactStore`. No synthetic
artifact bypass was enabled. This proof installed the relay's guest-disconnect
watcher fix and records its exact hash. The outer controller includes the
reviewed cleanup fix that attempts VM stop even if relay shutdown fails.

The temporary gateway, reverse forward, disposable database and reader roles
were removed. VM inventory was empty, the outer VM powered off, and QEMU exited
with code0. No host live unit, sshd, production database or network configuration
changed. Private mail and real provider calls were not used.

## Limits of this proof

Only native Claude's four listed typed MCP tools were qualified. Pi's typed
extension, additional mail tools, and complete tool parity require separate
work. The bootstrap remains a synthetic qualification harness; production run
admission, recovery and deployment review remain outstanding. The existing
synthetic worker hostname/root guard and immutable image checks remain enabled.
