# Dedicated Linux execution worker spike

These scripts prepare a **synthetic infrastructure probe**, not the public agent
runtime. They never copy application data, provider keys, host credentials,
source trees or home directories into the worker. Existing public execution
must remain retrieval-only until the full design passes its acceptance tests.

## Footprint and boundary

- Debian 12 genericcloud amd64 image from Debian's HTTPS distribution endpoint;
  verify SHA512 against its vendor manifest and retain the resolved digest.
  The manifest uses the same HTTPS trust boundary as the image; this is not
  detached-signature verification. `latest` is resolved once per preparation.
- One 12 GiB sparse disk overlay, 4 vCPUs and 4 GiB RAM. Actual disk consumption
  begins near the downloaded image size and grows up to the overlay limit.
- Private `/tmp/gmail-worker-spike.*` directory and newly generated temporary
  SSH key. Never use a host identity, agent forwarding or a shared filesystem.
- QEMU KVM acceleration with host CPU exposes nested virtualization; confirm
  `KVM_CREATE_VM` inside the worker, not merely a visible CPU flag.
- Restricted QEMU user networking, only host loopback port 22092 forwarded to
  guest SSH. No TAP devices, bridges, host firewall changes or system services.
  Guest general outbound networking is disabled. This bootstrap network is not
  the final guest vsock capability channel.
- Foreground process terminates after 15 minutes, with a 10-second kill grace.
  There is no persistent VM or automatic restart.

## Run

Prerequisites: Linux, current-user `/dev/kvm` access, QEMU, `qemu-img`, `xorriso`,
Python 3.11+, curl, OpenSSH and GNU timeout. Download preparation requires
network access; boot requires host KVM access outside a restricted sandbox.

```bash
bash deploy/public/worker/prepare.sh
# Use the exact private directory printed by prepare.sh:
bash deploy/public/worker/boot.sh /tmp/gmail-worker-spike.EXAMPLE
```

In a second terminal, after cloud-init finishes (usually within a few minutes):

```bash
worker_dir=/tmp/gmail-worker-spike.EXAMPLE
ssh -F /dev/null -o BatchMode=yes -o ConnectTimeout=5 \
  -o IdentitiesOnly=yes -o ForwardAgent=no -o StrictHostKeyChecking=accept-new \
  -o UserKnownHostsFile="$worker_dir/known_hosts" \
  -i "$worker_dir/id_ed25519" -p 22092 worker@127.0.0.1 \
  'bash -s' < deploy/public/worker/probe.sh
```

Run the probe only against this clean worker. The first SSH connection records
the host key in its private known-hosts file. QEMU fails closed if port 22092 is
already occupied. Stop with Ctrl-C or wait for the hard timeout. After QEMU has
exited, remove only the exact dedicated temporary directory if no longer needed.

## Next stage

Nested KVM success establishes hardware feasibility only. Install pinned,
verified Firecracker/jailer binaries and a versioned synthetic kernel/rootfs
inside the worker using reviewed build inputs. Configure the production jailer,
seccomp, cgroups and vsock-only relay there before any untrusted agent execution.
No agent microVM may boot with mail, provider credentials or public traffic yet.

Required further evidence: actual Firecracker boot, clean Pi/native Claude
gateway compatibility, shell/Python chart/XLSX output, streaming/cancellation,
negative network tests, worker crash cleanup and the proposal's tenant tests.
Do not reinterpret successful nested KVM creation as those checks passing.

## Recorded synthetic check (2026-09-15)

On sukkot, this worker booted Debian kernel `6.1.0-53-cloud-amd64` with
cloud-init detecting `DataSourceNoCloud [seed=/dev/vdb]`. The probe returned
`nested parameter: Y` and `PASS: nested KVM API 12 and KVM_CREATE_VM`.
The initial probe found no Firecracker/jailer inside the worker; the subsequent
microVM smoke below installed the pinned pair there.

The verified 323 MiB base image SHA512 was:

```text
08fea112563461f251f3c95a5c5cf8cb25eb60f74cec03e85a97ff91d3efef3059d35837598bbb476008f20db6d3bdc7143c5f2f2a9a6da394a0acc601fd5986
```

The seed must use virtio block transport. The initial CD-ROM attachment failed
to initialize cloud-init/SSH with this cloud kernel; the virtio attachment
booted and completed cloud-init within five seconds.

## Jailed nested microVM smoke

`prepare-microvm.sh` stages checksum-pinned public artifacts in a prepared
worker directory. It copies only the known Firecracker/jailer 1.14.0 binaries
from `/usr/local/bin` and downloads the official Firecracker v1.14 CI kernel
`vmlinux-6.1.155` and Ubuntu 24.04 squashfs root. All four SHA256 values are
pinned in the script and rechecked inside the worker. The upstream S3 multipart
ETags were also verified when obtaining the initial kernel/rootfs pins. These
are HTTPS provenance and integrity checks, not a claim of signed release
attestation. Different local binaries fail verification and require review.

```bash
bash deploy/public/worker/prepare-microvm.sh "$worker_dir"
# Boot the outer VM in another terminal using boot.sh, then:
ssh_options=(-F /dev/null -o BatchMode=yes -o ConnectTimeout=5
  -o IdentitiesOnly=yes -o ForwardAgent=no -o StrictHostKeyChecking=yes
  -o "UserKnownHostsFile=$worker_dir/known_hosts" -i "$worker_dir/id_ed25519")
scp "${ssh_options[@]}" -P 22092 -r "$worker_dir/microvm" worker@127.0.0.1:/home/worker/
ssh "${ssh_options[@]}" -p 22092 worker@127.0.0.1 'sudo bash -s' \
  < deploy/public/worker/microvm-smoke.sh
ssh "${ssh_options[@]}" -p 22092 worker@127.0.0.1 'sudo poweroff'
```

The inner guest has 1 vCPU, 256 MiB RAM, a read-only vendor squashfs root,
vsock CID 3, and **no network device**. No guest filesystem is mounted in the
outer host kernel. The jailer changes UID/GID to 65534, creates its chroot and
PID namespace, and preserves Firecracker's default seccomp filtering. The
smoke supervisor adopts the jailer's forked child and reaps the actual VMM;
timeout kills that child rather than merely the exited jailer parent.

Actual run on 2026-09-15 passed: root mounted read-only, `/bin/sh` executed the
standalone serial marker `GMAIL_SYNTHETIC_MICROVM_HELLO`, and guest reboot
caused `Firecracker exiting successfully. exit_code=0` in about four seconds.
Guest `poweroff` halted this configuration without exiting the VMM; the
qualified smoke uses the guest reboot exit path. Timeout cleanup of the halted
attempt was also exercised. A final process check found no remaining
Firecracker processes before the outer worker was powered off.

This is **not a production controller**. The production worker still needs
cgroup CPU/memory/process limits, a separate network namespace, online leases,
actual capability relay, persistent orphan cleanup and production host
hardening. A configured vsock device alone proves no gateway functionality.
The scripts exercise only synthetic trusted code. Both agent runtimes,
inference, artifact uploads and adversarial isolation remain unqualified.

## Synthetic lifecycle backend (2026-09-15)

`firecracker_backend.py` is a standalone standard-library implementation of the
trusted `WorkerBackend` interface. It refuses execution unless UID 0 and the
hostname is `synthetic-execution-worker`. It is deliberately unavailable as a
production host launcher. Copy only this file into the clean outer worker and
install it as `/opt/gmail-worker/firecracker_backend.py`, owned by root, mode
0500; the only other inputs are the already pinned public microVM fixtures.
`backend-smoke.py` runs there via `sudo python3 -`; it contains no application
source, mailbox data, provider keys or host credentials.

Each generated 32-character handle gets a jailed Firecracker process in its own
PID and network namespaces. The VMM has no NIC and only a vsock device (no relay
is installed). A root-owned metadata directory records the lease and process
state. The detached supervisor adopts/reaps the actual VMM, waits for seccomp
activation before readiness, bounds serial output, and enforces lease expiry
and an immutable monotonic hard deadline independently of its launching
controller. A restarted backend can inventory and stop orphan cgroups, including
a VMM whose supervisor was killed. Cancellation retains a failed teardown for
retry instead of claiming completion while its supervisor still runs.

The smoke uses a 256 MiB host memory cgroup with swap disabled, a 192 MiB guest,
one CPU (`cpu.max=100000 100000`), 64 host tasks (`pids.max`), and 1 MiB serial
output. `pids.max` bounds VMM host threads; it does **not** count processes inside
the guest. Guest process resource use is contained by guest memory and the VMM
CPU/memory limits. The root drive is read-only; no writable block volume or host
filesystem share exists. Disk limits currently gate the fixed images; a writable
workspace volume and its quota are not implemented.

Actual tests exercised the guest shell marker, UID 65534, active seccomp mode 2,
cgroup CPU/memory/pids settings, no guest NIC, lease-triggered teardown after the
launching controller exited, explicit cancellation after renewal, and cleanup
after supervisor loss. These are synthetic checks, not adversarial production
qualification. A production backend still needs pinned runtime images, a
capability relay, per-run identity/channel binding, persistent workspace disks,
a managed external sweeper/watchdog service, output transfer, and dedicated-host
hardening. This backend has not been connected to public routes or real agents.

## Actual inner-VM agent compatibility

`INNER_VM_COMPATIBILITY.md` and `inner-vm-compatibility.json` record the next
synthetic step: actual Pi and native Claude ran Bash/Python **inside** the jailed
inner VM and uploaded artifact bytes through guest-initiated vsock. The guest
used a pinned second read-only runtime drive and a bounded tmpfs workspace.
`agent-backend-smoke.py` runs this proof against the two fixed synthetic gateways;
`guest-vsock-bridge.py` provides the guest-local HTTP transport convenience.
This replaces the earlier runtime-compatibility uncertainty for these pinned
versions, while the production prerequisites listed in that report remain.
