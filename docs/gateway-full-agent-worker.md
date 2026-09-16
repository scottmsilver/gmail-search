# Fixed full-agent worker manager

This synthetic-only adapter connects the existing `WorkerController` to the separately pinned full-agent image. It changes no guest image, attachment parser RPC, production guard or public listener. The manager's physical backend namespace must be dedicated to this service: startup stops all old VMs in that namespace before admitting work.

## Host composition

`gateway.full_agent_remote.SSHFullAgentBackend(registry, transport=SSHTransport(...), envelope_for=...)` implements `launch`, `renew`, `stop`, `inventory` and `close`. Pass its `prepare_input(lease,prompt)` to `compose_browser_runs`; the required trusted envelope factory issues the six same-run capabilities and returns `guest_agent_bootstrap.encode_config(...)` bytes. It must pin the inference profile and return only this run's authority. The adapter verifies the closed envelope and exact prompt before retaining it. Each admission/renewal rechecks the actual Registry run, fence and immutable lease fields.

Use `FULL_LIMITS`: one CPU, 1 GiB RAM, 128 PIDs, 1 GiB disk accounting, 8 MiB output accounting and 180-second wall limit. The existing synthetic backend remains responsible for actual kernel enforcement; these configuration fields alone are not proof of disk/output quota enforcement. The worker lease is at most 20 seconds and can extend only on an explicit authorized heartbeat. A duplicate launch does not renew it. The fixed browser heartbeat interval is ten seconds.

Prepared envelopes are transient, bounded to eight instances, and removed after launch ACK or stop. Invalid/expired pending inputs are pruned on later preparation, and `close` clears all pending inputs. An abandoned prelaunch input may remain in memory until that pruning/close; its run/capability revocation remains authoritative. No bootstrap or capability is stored in either metadata database. The host persists handle/run/context/digest/sequence/state so a restarted controller can stop or reconcile an existing worker; it cannot replay lost bootstrap bytes.

## Fixed private transport

New `full_agent_rpc.py` is stdlib-only and copied unchanged to the worker. One big-endian uint32 header length (maximum 4,096 bytes), strict duplicate-free JSON header, declared bootstrap payload (maximum 32,772 bytes) and EOF form one request. Response is at most 4,096 bytes, echoes request ID/operation/handle/context digest and has no payload. Operations are `launch`, `renew`, `stop` and `inventory`. Paths, commands, images, upstreams, models and resource ceilings are not protocol inputs.

`full_agent_rpc_frontend.py` is an unprivileged forced SSH command connected only to `/run/gmail-full-agent-rpc/manager.sock`. It ignores `SSH_ORIGINAL_COMMAND` and never executes it. The separate fixed `gmail-full-agent-rpc` account must have no general shell commands, forwarding, PTY, user RC, environment injection or SFTP access. The host pins its dedicated private key and known-host file and disables inherited SSH configuration/proxies/agents. Four bounded Unix connections use peer credentials and five-second framing deadlines. SSH requests have a 30-second bound.

The fixed controller account is trusted across its whole worker namespace. Normal operations bind owner/run/conversation/fence/workspace version/deadline and bootstrap SHA-256. Administrative orphan `stop` permits an all-zero context digest, authorizing by fixed peer identity plus handle; it is not exposed as a browser capability. Unknown stops first persist a tombstone and then obtain physical teardown ACK. A delayed launch cannot bypass the tombstone.

## Owned lifecycle and crash behavior

`full_agent_manager.Manager` reserves the single physical slot and persists an immutable binding before its lifecycle thread starts. Exact launch retry returns the existing state; changed input/context, stopped handles and replayed renewal sequences are rejected. Inventory includes launching/stopping bindings until cleanup acknowledges both relay and VM removal. Backend launch failure, bootstrap failure, lease expiry and explicit stop all take the same cleanup path; failed cleanup retains the slot and durable stopping state for retry.

The fixed `RuntimeSession` owns one per-jail vsock-8002 listener, delivers the envelope once to the jailed VMM UID, closes it, and owns the existing HTTP relay on vsock-8000 to trusted outer loopback port 18081. Its relay runs in-process with tracked sockets and non-daemon request threads: manager process death closes those descriptors rather than leaving an orphan relay subprocess. Teardown closes guest sockets, drains relay threads and obtains backend stop ACK. Independent VMM supervision retains the hard/online watchdog when the manager disappears.

The private state directory and files are checked for owner/mode/type; SQLite uses DELETE journaling and FULL synchronization, and initial directory entries are fsynced. An exclusive manager flock prevents competing owners. Restart reaps physical inventory and marks prior jobs stopped before opening the RPC server. Database metadata is bounded to 10,000 records; tombstones are never silently recycled.

## Qualification boundary

Focused tests cover lost launch ACK and compensation, stop-before-launch, cancelled bootstrap, pending launch cancellation, restart orphan reaping, renewal replay, changed bindings, rejected peers/envelopes, exact framing, stop failures retaining capacity and real temporary Unix frontend/server composition. The eight-tool image itself was previously qualified separately in [the guest report](../deploy/public/worker/GUEST_FULL_AGENT_QUALIFICATION.md).

Actual manager/SSH and browser HTTP→VM→PostgreSQL composition passed; see [the qualification report](../deploy/public/worker/FULL_AGENT_MANAGER_QUALIFICATION.md), including in-flight cancellation and SSH access denials. The subsequently pinned larger-result image also passed the [80 KiB thread qualification](../deploy/public/worker/FULL_AGENT_LARGE_RESULT_QUALIFICATION.md); display events remain bounded previews and the configured maximum sizes are not all resource-qualified. No production deployment or live provider use is asserted.
