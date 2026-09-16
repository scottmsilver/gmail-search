# Pi typed MCP: actual synthetic VM qualification

On 2026-09-15, the public Pi 0.84.4 CLI used the pinned `pi-mcp-adapter`
2.32.1 extension and reviewed `guest_mail_mcp.py` server inside an actual jailed
Firecracker VM. The gateway ran real SQL, retrieval and artifact services against
a disposable synthetic database. Only the provider was mocked; inference gateway
behavior was not under test.

## Observed behavior

Every one of the six mock provider requests advertised exactly these five tools.
The provider requested them in this order and validated each returned result:

1. `mail_describe_schema`
2. `mail_sql_query_batch`
3. `mail_get_thread_batch`
4. `bash`, solely to create a CSV from the preceding MCP results
5. `mail_publish_artifact_batch`

No proxy or administration tool was advertised. The four mail operations used
typed MCP calls through one persistent stdio server and `GuestMailTools` instance.
The trusted synthetic guest bootstrap sampled process arguments every 20ms and
observed one MCP process with a peak of one. This is a fixture observation, not
proof against a malicious guest spoofing process information.

Alice and Bob had colliding message and thread IDs in the disposable database,
with separate PostgreSQL reader roles. This VM received only Alice's three
short-lived audience-specific capabilities. SQL and thread results contained
only Alice's message. The MCP receipt matched the committed real `ArtifactStore`
record containing this 68-byte CSV (CRLF endings):

```csv
id,subject,body
shared-message,ALICE_ONLY_SUBJECT,ALICE_ONLY_BODY
```

Bob's artifact read was denied. The final run took 11.344 seconds. Capabilities
were revoked, the run finished, and the controller acknowledged VM teardown.
The temporary gateway, SSH reverse forward, database and reader roles were
removed. Inventory was empty before powering down the outer VM; QEMU exited0.
Sanitized enforcement and provenance are in `guest-pi-mail-mcp-qualification.json`.

## Immutable public runtime inputs

`SyntheticAgentPiMCPBackend` selects only `agent_pi_mcp` and fixed read-only
`agent-pi-mcp.squashfs`, pinned to SHA256:

```text
478461755f2f344572bb0784685205fb173795bb5e3acae6fe523fb73dc62bbc
```

All prior agent, attachment, Bash-wrapper and native Claude MCP pins and images
were retained and reverified. The builder uses the retained public Node/Pi/Claude
runtime inputs plus freshly installed public dependencies from the exact
`deploy/pi/pi-pkgs/package-lock.json`. Installation used
`npm ci --ignore-scripts --no-audit --no-fund` with empty npm user/global config
and an isolated cache. No dependency lifecycle scripts ran. Lockfile, package
metadata and package licenses are retained in the image; evidence records the
adapter's registry URL, integrity, MIT license hash and lockfile hash. Production
container contents and guest-modified filesystems were not used.

## Fixed startup and adapter contract

The existing strict one-shot per-jail vsock8002 bootstrap accepts only
`runtime: "pi"` for this fixture. Its frame has a maximum 4096-byte strict JSON
body, rejects duplicate/unknown fields and trailing input, and enforces a fixed
read deadline. The listener closes after delivering one owner's SQL, retrieval
and artifact capabilities. The guest installs its fixed capability file mode0600
under UID1000's mode0700 run directory. No capabilities are built into the image.

Pi starts with an empty home and an explicit environment allowlist. Only its
fixed mock-provider configuration, public runtime paths and exact
`MCP_DIRECT_TOOLS` selectors are supplied. `--no-extensions` disables discovery,
while one explicit `--extension /tmp/runtime/guest-pi-mail-mcp.ts` loads the
reviewed configuration. Skills, workspace context, themes and prompt template
discovery are disabled. Pi's `--tools` list names exactly the five tools above;
this flag filters extension tools as well as built-ins.

The extension uses `createMcpAdapter({config: ...})` with inline configuration.
Host configuration discovery, imports, plugin paths, authentication/OAuth,
resources, sampling, elicitation, script mode and proxy tools are disabled.
Four exact tool names are selected. `lazy-keep-alive` with zero idle timeout
retains one MCP server after discovery; fixed `MCP_DIRECT_TOOLS` selectors make
session startup await discovery with a fresh metadata cache. The stdio command is:

```text
/usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 /usr/bin/python3 -I /tmp/runtime/guest_mail_mcp.py
```

The MCP process inherits no CLI provider environment. There is no additional
per-call wrapper or Node HTTP implementation. The existing core owns request
serialization, cancellation and gateway access.

## Boundary, verification and limits

The VMM ran as UID65534 with seccomp mode2, a private PID namespace, one CPU,
1 GiB memory, zero swap and 128 processes. The machine has no NIC. Guest CLI and
MCP processes run as UID/GID1000. Requests cross only the fixed guest bridge,
vsock8000, outer relay and temporary loopback-only SSH reverse forward.

The focused backend, cleanup, bootstrap, MCP, core and wrapper suite passed
86 tests. The initial VM attempt exposed the CLI's extension allowlist behavior
and failed before any tool call; its stop acknowledgment was retained privately.
A subsequent successful workflow exposed a duplicate host-harness run-finish
call. That cleanup was corrected and the final fresh run completed with exit0
and all cleanup verified. These changes required no adapter or core modification.

This qualifies Pi's four listed typed MCP operations with synthetic data and a
mock provider. It does not establish full tool parity, production package
installation, production bootstrap or production run admission/recovery. The
synthetic worker hostname/root guard and fixed immutable image checks remain
enabled. No real mail, provider credentials, production database, host live
units, sshd or production network configuration changed.

## Post-qualification builder amendment (2026-09-15)

Security review found that the original packing script compared package metadata
but did not verify the actual Pi CLI and dependency bytes. The builder now checks
`pi-mcp-runtime-inputs.json` with `verify_pi_mcp_runtime_inputs.py`. Its five
canonical SHA256 tree fingerprints cover all copied `bin`, `lib`, `node_modules`,
`package.json` and `package-lock.json` content, entry paths/types, executable
permissions and internal symlink targets. Extra, missing or modified files fail
verification. Absolute, dangling and escaping symlinks and special files fail
before packing.

The manifest was derived from a fresh extraction of the exact qualified squashfs
after verifying the image hash above, then independently matched against the
retained public image staging tree. It identifies those already qualified bytes;
it does not independently prove the earlier npm installation procedure. No
private run directory or production dependency installation supplied its inputs.

Future builds require root execution and root-owned input files and ancestors,
with no group/world write or setuid/setgid permissions. Root-owned sticky
ancestors such as `/tmp` are allowed. Stage the **public pinned image** in a private
root-owned directory, verify its SHA256, extract with `unsquashfs -no-xattrs`,
and remove group/world write permissions from extracted files/directories before
passing the extracted image directory and its `pi-pkgs` subdirectory to the
builder. Existing developer-owned retained assets remain unchanged. The builder
creates a private copy without inherited ACLs or xattrs, verifies that copy
again, and packs that copy. A mismatch removes the incomplete build directory.
The builder, verifier, manifest and current guest sources remain trusted,
reviewed operator inputs; they are not guest-supplied configuration.

Eighteen regression tests cover changed CLI/adapter bytes with unchanged package
metadata, tree membership, executable permissions, internal and escaping
symlinks, special files, owner/write-permission rules and snapshot changes. The
builder/backend/cleanup suite passed 41 tests, and the shell syntax check passed.
Fresh public-image extraction and snapshot fingerprints also matched. Ownership
checks were tested with metadata fixtures and ordinary-user rejection; no
root-owned host staging or actual image build was performed for this amendment.

This changes only future builder validation. The historical qualification JSON,
runtime/worker source hashes and image pin above are unchanged. No VM was rerun,
and the VM evidence does not claim to exercise this later builder amendment.
Amendment source hashes and verification scope are recorded separately in
`pi-mcp-runtime-builder-amendment.json`.
