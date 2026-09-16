# Standalone guest attachment downloader

## Supported scope

`deploy/public/worker/guest_attachment_transport.py` and
`guest_attachment_download.py` implement a standard-library downloader for a
**persistent guest tool process**. They are not enabled in the tool core, CLI,
bootstrap configuration, MCP schema or runtime images by this change. No host
file storage, parser or provider is invoked by these modules. Tests use synthetic
loopback responses and temporary workspaces.

The host protocol is documented in `gateway-attachment-raw.md`. The raw relay
must preserve its strict Content-Length framing. A legacy close-delimited JSON
relay is insufficient; the binary client does not relax its protocol to accept
missing length. Relay changes and full guest/runtime integration have their own
qualification and review.

## Trusted API

```python
downloader = GuestAttachmentDownloader(
    workspace_fd, attachment_capability, slots=shared_core_socket_semaphore,
)
results = await downloader.download_many((123, 456), deadline=absolute_deadline)
await downloader.aclose()
```

`workspace_fd` is a held descriptor supplied by trusted guest startup. It must
refer to a directory owned by the process UID with mode 0700. The downloader
duplicates it, creates a fresh random 0700 subdirectory relative to it, and holds
that subdirectory descriptor. No model argument selects a directory or path.
The caller retains ownership of its original descriptor.

The opaque capability is exactly 64 lowercase hexadecimal characters. Trusted
bootstrap must supply only this run's attachment capability with operation
`raw`; the host derives the actual owner from it. The downloader cannot select
an owner, URL, port, endpoint, filename, MIME override, parser mode or provider.

The persistent core must supply its **same shared two-socket semaphore** used
by JSON/mail/artifact operations. There is no default private socket pool. The
constructor checks the semaphore type and rejects a current count above two,
but a semaphore's current count does not prove its original capacity. Correct
shared construction and balanced use remain trusted composition requirements.
Operations on one downloader instance use one event loop.

`download_many` accepts a tuple of 1–20 unique positive signed-int8 attachment
IDs. It preserves input order and returns a tuple of either:

```json
{"relative_path":"gms-downloads-<random>/attachment-<random>.bin","size_bytes":123,"sha256":"<64 hex characters>"}
```

or a fixed `{"error":"..."}`. It never returns the raw packet, file bytes,
source filename, source path, MIME or credentials into model context. Cancellation
and deadline expiry propagate after cleanup. Other item failures do not cause
automatic retries. The absolute monotonic deadline includes waiting for shared
socket capacity and is capped at 30 seconds.

## Fixed HTTP and packet transport

The transport opens only `127.0.0.1:18080` and posts
`/v1/attachment/raw`. It sends fixed JSON `{"attachment_id":...}`, the attachment
Bearer capability, `Content-Length`, `Accept-Encoding: identity` and
`Connection: close`. It uses no proxy/environment routing or redirects.

Response HTTP headers are bounded to 16 KiB and 64 fields. Only HTTP 200 with
exact `application/octet-stream` and a canonical positive Content-Length is
accepted. Duplicate/malformed headers, transfer encoding, content encoding,
missing length, non-200 statuses and oversized responses fail closed.

The packet begins with a four-byte big-endian header length, 1–4,096. Its UTF-8
JSON header rejects duplicate keys and non-finite constants and must contain
exactly `version`, `operation`, `attachment_id`, `mime_type`, `size_bytes`, and
`sha256`. Version/operation must be `1`/`raw`; the attachment ID must equal the
request; MIME and digest use their bounded ASCII grammars; size is 0–10 MiB.
HTTP Content-Length must equal `4 + header_length + size_bytes` exactly.

The client reads at most 64 KiB of payload at a time into the file sink while
computing SHA-256. The complete digest and declared size must match, and an
additional read must observe EOF, not a trailing byte. A server that sends the
complete payload but never closes still times out. The receive buffer is not an
aggregate 10 MiB packet; HTTP/asyncio/socket buffers have their own bounded-stack
allocation overhead beyond the application's current chunk.

The socket is closed, aborted and its local close acknowledgment shield-drained
on every exit. Separate close-task ownership covers cancellation that arrives
after EOF while normal `wait_closed()` is already running. A cancelled connection
attempt is also drained; if it returns a socket during cancellation, that socket
is retrieved and closed. Local close acknowledgment is not an upstream gateway
or remote TCP acknowledgment; the relay/host retains its own cleanup ownership.

## Files, quota and publication

Each item reserves a file slot before its connection. After validated headers it
reserves declared disk bytes, then creates a generated `.partial` file using
descriptor-relative `O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC` and mode 0600. File checks
require a regular UID-owned inode with the expected size, mode and link count.
Writes handle partial `os.write` results in bounded chunks and yield between
writes. There is no parser, extension inference, filesystem extraction or
automatic model/provider upload.

The fixed counters bound each **persistent downloader instance** to:

| Bound | Limit |
| --- | ---: |
| Committed plus reserved files | 20 |
| Committed plus reserved payload bytes | 64 MiB |
| Items in one batch | 20 |
| Cumulative attempted declared payload in one batch | 20 MiB |
| One payload | 10 MiB |

Batch byte charges are not refunded after hash/IO failures, so repeated failed
items cannot cause unbounded payload reads within the batch. Exceeding a quota
refuses the item before writing its payload. Bytes already sent by a peer or
buffered by the network stack cannot be recalled by this application bound.
Successful files remain charged even if the shell deletes them. There is no
arbitrary-path delete/refund API.

These are **not durable per-run quotas**. Restarting a CLI/process or constructing
another downloader does not reuse these counters. Each instance creates a fresh
directory and cannot treat existing files as free recovered quota. Full CLI and
cross-process/run accounting remain unqualified. The VM disk/tmpfs/resource
ceiling is the outer bound on all guest writes, including hostile restarts and
shell-created files.

After exact size/hash/EOF and socket-close acknowledgment, publication runs while
the original file descriptor and shared socket slot are still held. It checks
the temporary name against that descriptor, creates a random final name with
`os.link` (which fails instead of overwriting), verifies the final inode against
`fstat` of the original descriptor, and removes the temporary link. It rechecks
the final file, private-directory binding, deadline and cancellation, then closes
the original descriptor before releasing the socket slot.

Per-file final names remain owned by the batch until `download_many` returns.
Completed items can release their sockets while later batch items proceed,
avoiding a deadlock when a batch exceeds two items. Cancellation/expiry before
batch return removes both partial files and all unreturned final files. The
batch drains all child/socket cleanup before its final local deadline and
cancellation checks. No further await occurs before marking its successes
committed and returning the small result tuple.

There is no atomic filesystem-plus-RPC publication primitive. If an outer MCP
response is lost after `download_many` returns, fully verified files can remain
in the guest and remain charged. Integration must preserve this explicit outcome,
not claim those files were necessarily delivered to the model or automatically
deleted. The files are ephemeral guest data, not crash-durable artifacts; no
`fsync` durability guarantee is made.

## Cleanup and failure recovery

Failures remove only tracked names that still identify the expected regular
inode. When still owned, the original descriptor remains open during rollback
name checks and deletion, and is then closed. Cleanup never overwrites or
deletes a pre-existing final-name collision. Unexpected inode substitution is
refused; it does not authorize deleting the replacement file.

Socket/file cleanup failure retains the corresponding quota and any still-held
socket slot, poisons the downloader against new operations and reports a fixed
error. `aclose()` cancels/drains active batches and retries supported cleanup,
such as a previously failing unlink or local socket close. It leaves committed
files in the workspace and closes its directory descriptors when cleanup permits.
Repeated caller cancellation cannot bypass the owned close/drain task.

Concurrent and repeated `aclose()` calls share one cached cleanup task. A later
explicit retry is allowed only after that task completed with failure. Both the
private-directory handle and duplicated workspace-directory handle use the same
uncertain-close rule as file handles: neither is retried while its descriptor
number remains valid, even when a replacement points to the same directory.

A failed `os.close` has an especially conservative rule: the descriptor number
may already have been reused, even for a newly opened handle to the **same inode**.
The downloader never retries closing an uncertain still-valid descriptor number.
It retains quota/capacity and requires teardown; observing `EBADF` can acknowledge
that the number is closed, but matching inode/device is insufficient proof of
handle ownership. Synthetic regressions cover reuse for both the same inode and
an unrelated file without closing the replacement.

File operations use bounded local guest syscalls. An absolute deadline is not a
hard kill for a stuck kernel operation or uncooperative socket cleanup. Trusted
guest termination/VM teardown remains the final recovery boundary.

## Same-UID and revocation limits

No-follow opens, private modes, generated names and inode checks prevent ordinary
traversal and accidental overwrites. They do not create an isolation boundary
against a malicious shell running with the same UID inside the guest. That shell
can read or mutate file contents, move names, change permissions or create its
own files. SHA-256 proves the bytes received from the authenticated stream; it
does not prove lasting file-content/path immutability against that shell after
an observed snapshot. Completed tentative-file rollback has saved identities,
not continuously held descriptors, after its successful file close.

The jailed VM, owner-specific capability and VM resource ceiling are the security
boundaries. Host revocation stops further authorized transfer, but cannot recall
bytes already in the guest. After EOF the downloader has no independent fresh
host-authority endpoint; it enforces its local deadline and the verified stream,
without claiming a new capability authorization at local filesystem publication.

## Verification

The standalone suite covers zero/max files, fixed HTTP/request framing, malformed
headers/manifests, wrong IDs and digests, truncation, exact EOF, shared capacity,
cumulative batch accounting, instance file/byte quotas, partial writes and ENOSPC.
It exercises repeated cancellation during receive and normal EOF close, late
socket returns, unknown close acknowledgments, private-directory substitution,
final inode substitution, no-overwrite collisions, held-descriptor publication,
batch cancellation, and post-link deadline/cancellation rollback.

Only controlled synthetic HTTP servers and temporary directories were used.
Actual gateway-encoder/relay contract tests are a separate root-owned test slice.
No CLI raw tool, runtime image or production deployment is qualified here.
