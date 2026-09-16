# Full-agent Pi VM qualification — 2026-09-15

**Passed on synthetic data only.** A newly pinned, immutable runtime booted in an actual jailed inner Firecracker VM under the preserved bounded outer worker. Pi 0.84.4 received a fresh question through the versioned bootstrap and called all eight typed MCP mail tools through one persistent server. The native Bash tool checked a downloaded file and created a CSV. Real gateway services streamed 21 events and stored the 42-byte CSV artifact; ten local mock inference requests received ten transport close acknowledgements.

## Evidence and exact profile

- Image SHA-256: `2d0c6d9e861740dbed914d6a73cd9a787fac6bc434f103c2bc00b3b2c6477bde` (`agent_full`; historical pins preserved).
- Machine-readable sanitized evidence: [full-agent-runtime-qualification.json](full-agent-runtime-qualification.json), including the fixed prompt, advertised names, executed sequence and executed source hashes. The executed sources are retained privately under `qualified-source`; the outer smoke helper subsequently had only an unused import removed.
- Tool sequence: schema, SQL batch, thread, semantic search, facts, metadata query, attachment meta/text/raw batch, native Bash, artifact publication, final citation.
- The exact 10,485,760-byte raw attachment passed SHA-256 verification and remained outside model/event content. Only its relative path, size and hash were returned. The new fixed bridge limits each direction to 12 MiB.
- Real disposable PostgreSQL owner readers, strict owner-only native ScaNN generation, QueryGateway, retrieval/search/facts/attachment services, Events and ArtifactStore were used. Embedding and inference were local synthetic implementations; no provider credential or real mailbox was used.
- Alice/Bob had colliding message, thread, attachment and proposition IDs. Alice SQL returned only Alice's row; foreign markers were absent from model requests/events, and Bob could not download Alice's artifact.
- The runner completed and the controller obtained worker stop ACK. Final worker inventory was empty. Capabilities were revoked, the Registry run cancelled, temporary database/roles removed, and loopback gateway/reverse tunnel stopped.
- Outer shutdown completed with QEMU exit 0; its local SSH listener was absent afterward. The successful guest workflow took 10.827 seconds, inside a 180-second inner watchdog and a separate 900-second outer timeout.

The observed VMM ran as UID 65534, seccomp mode 2 and PID 1 in its PID namespace. Cgroup limits were memory 1 GiB, swap zero, 128 PIDs and one CPU. These observations are qualification evidence for the controlled fixture; the existing synthetic-host backend guard remains mandatory.

## Build provenance and retained assets

`prepare-full-agent-runtime.sh` verifies the reviewed public runtime input manifest, snapshots only its fixed public trees, checks the retained package JSON/lock and adds the fixed guest sources. It builds inside root-owned private staging in the outer worker. The prior immutable Pi runtime input SHA-256 was `478461755f2f344572bb0784685205fb173795bb5e3acae6fe523fb73dc62bbc`; neither guest-modified storage nor production containers were extracted.

The outer's build-only Debian packages came from the official [squashfs-tools directory](https://deb.debian.org/debian/pool/main/s/squashfs-tools/) and [lzo2 directory](https://deb.debian.org/debian/pool/main/l/lzo2/):

- `squashfs-tools_4.5.1-1_amd64.deb`: `3f96b16c9b985ea03cf70f90a05f4dc5c15039e845326f8f4293a3280fd0b2f4`
- `liblzo2-2_2.10-2_amd64.deb`: `4f08e092c76e425295a498cd547dc9b8f6a595473f3020ab8c96309b29872636`

Private assets are retained under `worktrees/full-agent-assets-20260915/full-agent-runtime-qualification`: current and prior immutable images, package archives, build logs, `qualify_full.py`, per-attempt logs and Registry files. Capabilities in those Registry files are revoked; they are not published with this report. The qualification script is a disposable synthetic fixture, not a deployment controller. Its composition can be reused for a later browser-to-VM integration proof.

Two earlier bounded guests stopped cleanly after inference rejected incompatible advertised schema hints. Their images/logs are preserved. The final v3 schemas retain strict runtime format/ID validation while using the existing inference schema subset; signed numeric IDs are limited to JavaScript's exact integer range. The inference gateway implementation was not relaxed.

## Limits and remaining app integration

This establishes eight-tool execution in a real guest, not a deployed browser app, live provider/model availability, cancellation during this particular workflow, quality of real-model tool choices, rendering support or production database readiness. The mock provider prescribed the useful sequence and checked each preceding real tool result. Seven native filesystem tools were advertised; only Bash was executed in this proof.

The 65,536-byte RPC line/StreamReader, normalization and event limits still reject large textual tool responses or repeated final transcripts. Raw binary transfer bypasses that model-context path. A later bounded record/display policy is needed before claiming general large-thread support. Raw file quotas remain per persistent MCP instance, with VM disk limits still an operational requirement.

The smoke controller passes run capabilities through a per-jail one-shot listener, but its VM lease identifier is independent of the gateway Registry fixture run. Production browser composition still needs its actual WorkerController driver, matching lease ownership, cancellation/drain behavior and answer persistence/download route proof. No production backend guard, live service, network configuration or mailbox was changed.

Focused tests: **152 passed in 2.04s** for full-runtime, v3, MCP, CLI, runner and unchanged inference validation. Actual VM proof passed separately with final cleanup recorded above.
