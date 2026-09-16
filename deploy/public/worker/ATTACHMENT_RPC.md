# Staged attachment worker RPC

This seam connects the trusted mailhost controller to the existing isolated
attachment parser through SSH. It does not enable production, add a public
listener, change the parser image, or remove the synthetic-worker host guard.
The worker must be dedicated to this manager: startup reconciliation owns the
worker backend's entire inventory.

## Trust and installed layout

The root-owned files under `/opt/gmail-worker` are:

- `attachment_manager.py`, `attachment_backend.py`, and `firecracker_backend.py`;
- `attachment_rpc_frontend.py`;
- an exact copy of `src/gmail_search/gateway/attachment_rpc.py` named
  `attachment_rpc.py` beside those scripts.

The root manager uses `/var/lib/gmail-attachment-rpc` (0700), a 0600 SQLite
metadata database and an exclusive process lock. The manager creates one Unix
socket at `/run/gmail-attachment-rpc/manager.sock`. Its parent is root-owned
0750, with the dedicated frontend account's group; the socket is root-owned
0660 with that group. Socket connections must have the exact configured
`gmail-attachment-rpc` UID according to Linux `SO_PEERCRED`.

The dedicated SSH account is unprivileged and cannot modify installed code,
manager state, VM images or backend state. Its key must be restricted to the
fixed command below, with forwarding, PTY, agent forwarding, X11 and user rc
execution disabled. The account must have no other authorized keys or login
route. This is a configuration contract, not an installed service or key:

```text
restrict,command="/usr/bin/env -i PATH=/usr/bin:/bin /usr/bin/python3 -I /opt/gmail-worker/attachment_rpc_frontend.py" ssh-ed25519 <trusted-controller-public-key>
```

The frontend ignores `SSH_ORIGINAL_COMMAND`, accepts no arguments, and runs
neither a shell nor sudo. It sends one validated request to the fixed socket,
validates the response and echoed binding, then exits. The root manager's fixed
entrypoint is `/usr/bin/python3 -I /opt/gmail-worker/attachment_manager.py`.
Neither entrypoint takes request-selected destinations or configuration paths.

## Wire contract, version 1

Each SSH process and each Unix connection carries exactly one request frame,
then request EOF; the response is exactly one frame followed by EOF. A frame is
an unsigned 32-bit big-endian header length, UTF-8 JSON header, and exactly the
number of raw payload bytes in `payload_size`. Headers are at most 4096 bytes.
Requests allow at most 10 MiB input; responses, including framing and headers,
allow at most 8 MiB. JSON objects reject duplicate keys, unknown fields and
non-finite values. MIME types and parser options use fixed allowlists.

Every request has these fields:

| Field | Value |
| --- | --- |
| `version` | Integer `1` |
| `op` | `start`, `poll`, or `stop` |
| `request_id`, `job_id` | Canonical UUID strings |
| `context_sha256` | Lowercase SHA-256 of canonical context JSON |
| `payload_size` | Exact integer byte count |

`start` additionally includes `context`, `mime_type`, `options`, and
`input_sha256`. Its context has exactly `owner_id`, `run_id`, `conversation_id`,
`fence`, `attachment_id`, and Unix-seconds `deadline`. The options have exactly
`dpi` (72–200) and `pages` (up to eight distinct positive page numbers).
Canonical context JSON uses sorted keys, compact separators, ASCII escaping and
no NaN/Infinity; use the shared `context_digest()` helper. The payload is the
opaque attachment bytes. Context describes a binding; it does not authenticate
the sender. Only the peer UID provides controller identity.

`poll` adds a positive `renew_seq` greater than every previously accepted poll
sequence for that job. `stop` has no extra fields. Both have empty payloads.
There are no path, image, command, port, upstream, or resource-limit fields.

Responses echo `version`, `op`, `request_id`, `job_id`, `context_sha256` and add
`status`, `code`, `payload_size`, `payload_sha256`. Status is `running`, `done`,
`stopped`, or `error`. Only `done` can contain a payload: opaque parser result
JSON still requiring the trusted host's result validation. An error or broken
transport is never a stop acknowledgement.

## Lifecycle and failure behavior

The manager persists a job binding before calling the backend. A duplicate
`start` with the same job, controller, context, input digest, MIME and options
returns the existing status without launching again. A changed binding is
rejected. A `stop` for an unknown job creates a durable tombstone, so a delayed
`start` cannot launch after the stop acknowledgement. Tombstones survive
manager restart and are never automatically evicted.

One parser job owns worker capacity until backend teardown succeeds. Returning
result bytes does not release capacity. A failed stop returns `stop_pending`,
retains its durable binding and capacity, and may be retried with the same job
ID. On startup the manager acquires an exclusive lock, stops/reaps backend
inventory, verifies reconciliation, and invalidates all old jobs and results
before accepting requests. Failed reconciliation prevents startup.

Only explicit polls can renew the three-second worker lease. The trusted host
must freshly authorize the original run capability before each poll. Replayed
or decreasing sequences never renew. A poll before VM launch readiness consumes
its sequence but cannot extend the initial lease. Neither repeated `start` nor
transport activity renews a lease. Manager expiry initiates teardown; the
independent VMM supervisor also enforces the short lease and an immutable hard
deadline of at most 45 seconds, including if the manager dies.

Private bytes are transient bounded memory, never manager metadata. The manager
has at most four connection handlers and one parser. Every connection's input
has an absolute five-second deadline; slow or trailing input is rejected. Each
handler admits at most 10 MiB, and the frontend also bounds its input and output.
The frontend allows 30 seconds for manager transport/teardown and five seconds
for output, using nonblocking writes so a blocked SSH client cannot defeat its
output deadline. The SSH daemon/account configuration must separately bound
session/process admission; the manager's handler cap does not bound sshd's
frontend process count.

The metadata quota is 10,000 job IDs. Exhaustion refuses new jobs and tombstones
without deleting old bindings. It requires deliberate administrative recovery;
existing stop retries still work. There is no automatic garbage collection
because deleting a tombstone would allow delayed requests to resurrect a job.

## Local verification

The synthetic protocol and manager tests use a fake parser and real temporary
Unix sockets, with no provider calls, private mail or VM launch:

```sh
PYTHONPATH=src /home/ssilver/development/gmail-search/.venv/bin/python -m pytest tests/test_attachment_rpc.py tests/test_attachment_worker_manager.py -q
```

Coverage includes strict framing, content binding, UID checks, stop-before-start,
lost-response retries, renewal replay, pending launch lease expiry, restart
orphan reconciliation, stop failure capacity retention, metadata permissions and
quota, trailing input, slow clients, and blocked frontend output. Full SSH to
root manager to nested VM qualification is a separate acceptance check; these
unit tests do not claim it has run.


## Actual SSH and nested VM qualification (2026-09-15)

`attachment-rpc-qualification.json` records an actual run through the host
`SSHTransport`, `SSHAttachmentBackend`, and `RunAttachmentService`, a new
service-specific SSH key, the unprivileged forced frontend, the private root
manager, and the nested Firecracker parser. The existing pinned parser image
`89bf4da702506dadacc5cd08b9f4d737f27abb592eb35b7a768cc22d590f9068`
was used unchanged. The service key's private half stayed on the mailhost.

Observed checks:

- The synthetic PDF returned `SYNTHETIC ATTACHMENT 42` and one 417×417 page;
  explicit authorized polls renewed the lease through guest boot, and successful
  service completion included stop acknowledgement and empty VM inventory.
- Revoking the host capability while an actual VMM was running suppressed output
  and completed teardown before returning denial.
- With polling abandoned and the manager frozen using SIGSTOP, the independent
  supervisor reaped the nested VM before the 4.23-second observation.
- Killing and restarting the manager with a live nested VM reconciled that VM;
  the old start failed closed and its stop remained acknowledged.
- Same-job start retry with a fresh request ID kept exactly one VM. Replayed
  renewal sequence, foreign context binding and delayed start after stop failed.
- General shell command, sudo and SFTP requests did not execute. An actual PTY
  request and direct TCP forwarding were denied by sshd. A valid stop RPC paired
  with an arbitrary `touch` command still returned its normal acknowledgement;
  the command marker was absent. Effective settings disabled user rc, user
  environment, password/keyboard-interactive auth, PTY and all forwarding.
- Manager code hashes, private-state modes and effective account restrictions
  were captured. The manager exited, removed its socket and left empty inventory;
  the outer VM was powered off and its foreground QEMU process exited normally.

The bootstrap administration identity was used only to configure and inspect the
synthetic outer VM. No host sshd, network or live service configuration changed.
The supported profile remains the dedicated synthetic outer worker. Production
admission, operational recovery, SSH process limits and deployment review remain
outstanding; public routes and private mail were not enabled by this proof.
