# Dedicated production worker installation candidate

This is a separate enrollment and installation profile for a **dedicated Linux
VM**, not a release qualification or permission to run on the mail host. The
existing synthetic entrypoint continues to require `synthetic-execution-worker`.
The production entrypoint requires root, hostname `gmail-execution-worker`, a
root-owned enrollment bound to that VM's machine ID, cgroup v2, KVM, and fixed
SHA-256 pins. No browser/RPC field or environment variable can select an image,
path, hostname, executable, or execution profile.

The production profile uses the existing `agent_full` guest and jailer lifecycle:
read-only images, no guest NIC, isolated PID/network namespaces, default seccomp,
UID/GID 65534, resource cgroups, independent online/hard lease enforcement, and
per-run vsock relay. Production discards guest serial bytes while retaining the
serial byte quota, so guest console output cannot persist mail in host log files.
The production supervisor accepts only `agent_full`. It
verifies preinstalled assets and never stages files from `/home/worker`. Manager
startup and service stop reap old physical inventory. Production still needs the
release checklist's actual adversarial and end-to-end acceptance evidence.

## Host prerequisites and observed inventory

Read-only inspection on 2026-09-15 found one disposable synthetic directory,
`/tmp/gmail-worker-spike.9E9kUZQs`, no running QEMU/Firecracker process, and no
configured production worker SSH alias. This is not a comprehensive infrastructure
inventory. Its `microvm` staging directory has the base Firecracker, jailer,
kernel and rootfs assets; it does **not** contain `agent-full.squashfs`. The
qualified image (Pi and Claude Code runners, dispatched by `guest_agent.py`) is SHA-256
`2b16ee2db771f49b635d3a836cb8090f321beff2483454556f184c0bb891e062`.
A separately provisioned persistent VM and a verified export of that existing
runtime image remain deployment prerequisites. Do not relabel the disposable VM
or copy its disk, home, temporary identity keys, or qualification services.

Provision a clean amd64 Linux VM with Python 3.11+, OpenSSH, iproute2, writable
cgroup v2 CPU/memory/PID controllers, working nested `/dev/kvm`, no swap, adequate
storage for the pinned read-only images, and at least 2 GiB RAM. Validate
`KVM_CREATE_VM` there. Set and independently record its hostname and machine ID.
Use a managed lifetime and clock synchronization; the browser/worker deadline
margin permits only five seconds of drift. Configure host network policy to
admit only the trusted controller's SSH connection; the worker needs no general
outbound access. Do not attach mail stores, provider credentials, other host
homes, or shared filesystems.

## Install files, with services stopped

1. Provision separate unprivileged accounts `gmail-full-agent-rpc` and
   `gmail-gateway-tunnel`, with distinct groups, no sudo, and no writable trusted
   configuration. The RPC account needs a valid shell for sshd's forced command.
   The tunnel account opens no session channels (`MaxSessions 0`). Lock password
   authentication through sshd policy; ensure public-key authentication remains
   usable under the host's account-lock policy.
2. Transfer only the reviewed install source files and five pinned assets into
   the clean VM. The installer consumes a repository-shaped source tree but only
   installs its explicit module allowlist and `full_agent_rpc.py`; it does not
   install the application or its secrets. Assets are `firecracker`, `jailer`,
   `vmlinux`, `rootfs.squashfs`, and `agent-full.squashfs`.
3. Fill `production.json.example` with the independently recorded VM machine ID.
   As root **inside that VM**, run:

   ```sh
   python3 deploy/public/worker/production/install.py REVIEWED_SOURCE PINNED_ASSETS ENROLLMENT_JSON
   ```

   Installation performs no downloads, account creation, service start or SSH
   reload. It installs a pending SSH policy at
   `/etc/gmail-worker/sshd-worker.conf.pending`. The service remains stopped.
4. Put separately generated public keys in root-owned
   `/etc/gmail-worker/rpc-authorized-keys` and
   `/etc/gmail-worker/tunnel-authorized-keys` (mode 0644). Pin the worker SSH host
   key on the controller through an independently verified channel. Never use
   the synthetic VM's key pair or accept-new for production.
5. Review/install the pending sshd include and validate with `sshd -t` and
   `sshd -T -C user=ACCOUNT,host=HOST,addr=CONTROLLER_IP` for **both** accounts.
   Existing global sshd configuration can affect the result: verify forced RPC
   command/no forwarding; tunnel remote-only TCP forwarding with only
   `127.0.0.1:18081`, no local forwarding, no session/PTY/SFTP/agent forwarding.
   Then reload SSH through the reviewed deployment procedure.

## Fixed gateway tunnel and RPC

The trusted API listens on **127.0.0.1:8092**. The separate controller user unit
`gmail-worker-gateway-tunnel.service` creates only the reverse listener
**worker 127.0.0.1:18081 → controller 127.0.0.1:8092**. Replace `worker-host`
(and port if needed) with the enrolled VM's actual address. Install the distinct
private key as `~/.config/gmail-search/worker-tunnel-key` mode 0600, and the
verified host-key file as `worker-known-hosts`. Do not use the RPC key here.
All inherited SSH config, proxies and agents are disabled. SSH keepalives and
systemd restart own tunnel lifetime. Missing tunnel causes gateway requests to
fail; it does not authorize alternate destinations.

The RPC transport uses its own dedicated key and the same pinned host identity,
connecting as `gmail-full-agent-rpc`; that account retains `DisableForwarding yes`. No public worker listener is added. The root manager listens only on its
private Unix socket and accepts that account's peer UID.

## Qualification before admission

After review, enable/start the manager and the tunnel with **synthetic data only**.
Repeat real SSH framing, forwarding denial, cancellation, manager crash/orphan
cleanup, lease expiry, startup reconciliation, negative guest network probes,
owner isolation, and full browser→gateway→VM acceptance against this exact
installation. Test graceful stop and forced manager death; verify no orphan
cgroups, namespaces, VMMs or relays remain. Confirm no private mail or provider
keys appear on disk or in service logs. Unit tests and valid systemd syntax do
not replace these host-level checks. Public admission must stay closed until
the root deployment's complete acceptance checks pass.
