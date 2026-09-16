# Real CLIs inside the nested Firecracker guest — synthetic qualification

On 2026-09-15, Pi 0.84.4 and native Claude 2.1.272 each completed a real
Bash → Python tool call **inside the inner Firecracker VM**, wrote a file
containing `42`, returned the tool result to the synthetic provider, and printed
the expected completion marker. Both then uploaded the file's two bytes through
the fixed artifact relay and received HTTP 201. The trusted outer verifier read
the received objects and checked their content and SHA256. No guest filesystem
was mounted or read by the outer verifier.

Machine-readable evidence is in `inner-vm-compatibility.json`. The SHA256 for
both received `42` objects is:

```
73475cb40a568e8da8a045ced110137e159f890ac4da883b6b17dc651b3a8049
```

## Boundary exercised

- 1 vCPU; host memory cgroup 1 GiB; swap disabled; 128 host VMM tasks;
  `cpu.max=100000 100000`; jailer UID/GID 65534; default seccomp mode 2;
  separate PID and network namespaces; no Firecracker NIC.
- Original pinned vendor SquashFS root remained read-only. A second read-only
  drive contained only public Node/Pi/Claude runtime files and two minimal
  synthetic guest scripts. Its complete SHA256 is pinned by `RUNTIME_PIN` in
  `firecracker_backend.py`:
  `0546f3aad681b012a1b44b90ac6f18188a6e25419220806e6a7a6ed4245bb32a`.
- Writable files, including each runtime's fresh HOME and working directory,
  lived on guest `/tmp` tmpfs capped at 256 MiB with `nodev,nosuid`. Runtime
  processes ran as guest UID/GID 1000 with empty supplementary groups,
  bounded files/descriptors/processes, and a scrubbed environment.
- Guest-local HTTP listeners forwarded only to host CID 2, ports 8000/8001.
  The outer Unix-socket relay allowed fixed inference/artifact routes and
  forwarded inference to the already profile-bound synthetic gateways on
  loopback ports 18081/18082. This follows Firecracker's documented
  [guest-initiated vsock mapping](https://github.com/firecracker-microvm/firecracker/blob/main/docs/vsock.md#guest-initiated-connections).
- The gateway used real request normalization, capability checks, budget
  reservation and provider transport with HTTPX MockTransport. No actual model
  provider request, provider secret, mailbox content, host home or application
  checkout entered the guest. The visible API-key strings were synthetic.
- The smoke controller stopped the VMM and relays on exit; no inner VM remains.

## Reproduce with reviewed public inputs

`prepare-agent-runtime.sh PUBLIC_RUNTIME_INPUTS` builds the immutable runtime
image. It validates the existing Node and Claude binary pins and copies only the
public `bin/` and `lib/` inputs plus `guest-vsock-bridge.py` and
`guest-agent-smoke.py`. Build outputs are private temporary directories. A new
build has a new image digest and requires explicit review/update of the fixed
backend pin before boot.

Install the standalone backend/relay inside the clean synthetic outer worker,
place the verified runtime image at
`/var/lib/gmail-worker/images/runtime.squashfs`, start the two standalone
synthetic gateway modes described in `CLI_COMPATIBILITY.md`, then execute
`agent-backend-smoke.py` there as root. Its hostname guard refuses the development
host. It creates no production service or deployment.

## Limits of this proof

This is runtime and transport compatibility, not a completed production rollout.
The synthetic profile endpoints are fixed test bindings; production still needs
per-run relay/capability binding, production image qualification and hardening,
managed controller/sweeper services, persistent versioned workspaces, browser
artifact delivery, and adversarial multi-tenant qualification. The guest bridge
is untrusted convenience code; authorization belongs in the outer gateway.
The outer worker's offline network and inner no-NIC configuration contain
outbound attempts, but this smoke is not a comprehensive network penetration
test. The mocked provider, rather than a live model, requested the tool calls.
