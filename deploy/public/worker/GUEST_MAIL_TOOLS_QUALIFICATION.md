# Actual guest mail tools: synthetic Bash wrapper qualification

On 2026-09-15, actual Pi and native Claude ran the fixed guest mail-tool wrapper
inside separate jailed Firecracker VMs. Each VM received three short-lived
capabilities for exactly one synthetic owner. The host gateway used the real
`RunQueryService`, `RunRetrievalService`, separate owner PostgreSQL readers, and
`ArtifactStore`. Inference responses came from a deterministic mock provider;
this run did not test the inference gateway or contact a provider.

This qualifies four tools through the CLI's Bash tool: `describe_schema`,
`sql_query_batch`, text `get_thread_batch`, and `publish_artifact_batch`.
It does not establish complete tools or MCP parity, enable production bootstrap,
or mount the host application inside a guest.

## Observed result

The disposable database contained Alice and Bob records with identical message
and thread IDs. Each actual CLI invoked Bash to execute a fixed workflow, which
called `guest_mail_tool_cli.py` separately for schema, SQL, thread retrieval, and
artifact publication. The workflow generated its CSV inside the guest.

| Runtime | Run owner | SQL and thread result | Real published CSV | Duration |
| --- | --- | --- | --- | --- |
| Pi | Alice | `ALICE_ONLY_SUBJECT`, `ALICE_ONLY_BODY` | 68 bytes | 5.031 s |
| Native Claude | Bob | `BOB_ONLY_SUBJECT`, `BOB_ONLY_BODY` | 64 bytes | 5.796 s |

Both artifacts were committed by the real artifact store, and reads as the
other owner were denied. The mock provider observed the actual Bash tool result;
capabilities were absent from provider request bodies and controller output.
The three run capabilities were revoked after each run. Both VMs acknowledged
teardown. The final worker inventory was empty.

The real VMMs ran as UID65534 with seccomp mode 2, a private PID namespace,
1 GiB memory cgroup, zero swap, 128 processes, and one CPU. The fixed image
configuration has no NIC. CLI processes drop to UID/GID1000 inside the guest.

Sanitized measurements, image/source checksums, CSV bytes, and enforcement
records are in `guest-mail-tools-qualification.json`.

## Separate immutable profile

`SyntheticAgentToolsBackend` selects only the fixed `agent_tools` profile and
`agent-tools.squashfs`. Its SHA256 is:

```text
4b981cd6a2e1eb2f28acad2d02ac6c365b4ef48b03345eb098578b9a1dd71402
```

The existing agent image pin remains
`0546f3aad681b012a1b44b90ac6f18188a6e25419220806e6a7a6ed4245bb32a`;
the attachment image pin remains
`89bf4da702506dadacc5cd08b9f4d737f27abb592eb35b7a768cc22d590f9068`.
Both installed images were rechecked unchanged after qualification. The original
runtime and new image remain preserved privately. No guest-modified filesystem
was extracted or mounted on the host.

`prepare-agent-tools-runtime.sh` verifies the existing public Node and Claude
binary hashes, copies their retained public runtime libraries and the explicit
Python script list, and builds a new read-only SquashFS. A new build can produce
a new digest and requires explicit review before updating `AGENT_TOOLS_PIN`.
No capability, private key, owner database credential or user configuration is
included in the image.

## One-shot synthetic bootstrap contract

`agent-tools-backend-smoke.py` is a trusted root controller restricted to the
existing synthetic worker hostname. Its stdin carries exactly one frame:

- Four-byte unsigned big-endian JSON length, at most 4096 bytes.
- A duplicate-free JSON object containing exactly `version: 1`,
  `runtime: "pi" | "claude"`, and `capabilities`.
- `capabilities` has exactly `sql`, `retrieval`, and `artifact`, each a 64-character
  lowercase hexadecimal token. There are no owner, path, command, port, image,
  resource-limit or upstream selectors.
- EOF follows immediately; input has an absolute five-second read deadline.

The host creates a fresh Registry run and issues the three audience-separated
capabilities with a 150-second TTL, bounded by the run's 180-second lease and
hard deadline. Only that run's owner capabilities are sent over dedicated admin
SSH stdin to the outer controller. They remain transient memory there.

The controller starts the separately pinned inner VM and binds one private,
per-jail Unix endpoint for guest-initiated vsock port8002. The immutable guest
bootstrap connects once, reads the same bounded frame and EOF with a five-second
deadline, and installs `capabilities.json` as UID1000, mode0600, under the fixed
UID1000 mode0700 `/tmp/gms-run`. The workspace is `/tmp/gms-run/work`.
The controller closes the bootstrap listener after that connection. Failure
causes teardown; the VMM's independent hard watchdog also remains active.
The guest removes its capability file at workflow exit.

## Transport and cleanup

The guest's fixed local HTTP bridge uses vsock8000. The trusted outer HTTP relay
forwards only its existing allowlisted routes to loopback18081. An ephemeral,
loopback-only SSH reverse forward connected that port to the temporary host
FastAPI gateway. The relay was invoked without `--synthetic-artifacts`; artifact
bytes traversed the real authenticated artifact HTTP route and store.

Only the dedicated synthetic outer VM's existing administration key was used
for setup, inspection and the scoped reverse forward. No SSH private key entered
the outer or inner VM. The host mock provider used a fixed nonsecret synthetic
key, separate from the three real short-lived tool capabilities.

The host gateway, reverse forward, disposable PostgreSQL database and temporary
reader roles were removed. The nested inventory was verified empty, the outer
VM powered off, and the foreground QEMU process exited with code0. No host live
unit, sshd, production database or network configuration changed.

## Remaining work

The one-shot bootstrap is a qualification harness, not a deployed general run
admission service. Production run binding, worker admission/recovery, capability
rotation, and deployment review remain necessary. Tool coverage is limited to
the four operations above; raw/Markdown thread conversion and message selection
remain unavailable. A separate MCP adapter or other tool support needs its own
qualification. Private mail and production traffic remain disabled.

## Subsequent review correction

The actual-run checksums in the JSON record describe the tested snapshot. After
that run, independent review found that a relay teardown timeout could skip the
controller's explicit VM stop (the separate hard watchdog still applied). The
controller now uses a guaranteed `finally` for VM teardown and escalates a relay
wait timeout to kill/wait. Three synthetic failure regressions passed independently.
The JSON record separately records the corrected controller hash; the historical
run hashes and image pin are unchanged. This correction did not require another
VM boot.

A subsequent outer-relay correction propagates guest EOF to the gateway while
waiting for headers or streaming body data. It passed independent synthetic
checks; its new source hash is also recorded separately in the JSON amendment.
The original actual-run relay hash is retained, and this amendment does not
claim that the older VM run exercised the newer relay.
