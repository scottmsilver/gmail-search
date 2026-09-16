# Attachment parser isolation: synthetic qualification

Status: standalone parser-only inner Firecracker job qualified with synthetic
PDF/image/ZIP files. Not wired into public routes or mailbox ingestion. The
public owner-only guard must remain in place.

## Interface and boundary

`gateway/attachment_sandbox.py` retains the synchronous
`AttachmentSandbox.parse(owner_id, attachment_id, dpi=100, pages=())` adapter
for trusted internal callers. New run-bound callers use
`gateway/attachment_service.py:RunAttachmentService.parse(token, attachment_id,
...)`: it derives the owner solely from the `attachment`/`parse` capability and
has no caller-supplied owner parameter. Its trusted async loader must return the
exact owner/attachment binding and bounded opaque bytes. `OwnerAttachmentSource`
can compose that loader with the restricted owner query locator; it opens only
the derived owner-hash storage path using held no-follow descriptors.

The service reauthorizes while loading and while a parser job runs. On token/run
revocation or request cancellation it cancels the loader, invokes the job's
blocking `stop()` acknowledgement, and drains cleanup before it returns any
error. A backend's `start()` contract must return a job handle without blocking
on VM launch; `job.stop()` must close its transport, stop and reap the full VM,
and join its worker. The service reauthorizes again after successful teardown,
immediately before publishing a parsed result. The host never imports a
PDF/image/archive decoder. It validates a bounded JSON manifest and the fixed
PNG envelope without decompressing image bytes.

`attachment_backend.py` runs only as root inside the clean outer worker named
`synthetic-execution-worker`. Its process lock admits one parser job per outer
worker. Each job starts a fresh fixed `attachment` Firecracker profile:

- One vCPU; 768 MiB VMM cgroup; swap disabled; 128 host threads/processes.
- Existing pinned read-only vendor root and a separate pinned read-only parser
  SquashFS. No agent runtime, mailbox mount, workspace restore, API key or NIC.
- Guest Python drops to UID/GID1000 before importing native parsers. Temporary
  storage is a 128 MiB tmpfs. Guest file/process/descriptor limits are also set.
- One fixed guest-initiated vsock connection to port 8002. The host sends a
  length-prefixed <=1 KiB control record and <=10 MiB of one attachment's bytes.
  No filesystem path is accepted or transmitted. The guest returns <=8 MiB JSON.
- Backend always stops the VM after success, bad output, disconnect or timeout.
  The independent supervisor imposes a 45-second hard deadline, kills the whole
  VMM cgroup and reaps its process even if the controller disappears.

The asynchronous handle interface separates transport completion (`wait`) from
acknowledged teardown (`stop`). The service and synchronous `parse` wrapper always
call `stop`. It interrupts transport sockets, stops/reaps the VM, and joins its
controller thread before releasing the process lock. Repeated successful stops
are idempotent; a failed stop retains both capacity and the job in
`pending_jobs()` for a trusted retry. New admission also refuses pre-existing
worker inventory after a controller restart until reconciliation completes.

The guest accepts PDF, PNG/JPEG/GIF/WebP/TIFF and flat ZIP archives containing
those formats. It checks PDF page dimensions before text extraction/rasterizing:
maximum 4000 pixels per side, 4 million pixels per page, 8 requested pages, 72–200 DPI.
Image dimensions are checked before full Pillow decode. Returned UTF-8 text is
capped at 200,000 bytes. ZIP traversal names never become filesystem paths; at
most 20 entries are inspected, nested archives are skipped, and decompressed
members/aggregate bytes are bounded. Oversized results fail closed.

## Reproduce

1. Fetch only the two exact public wheels in `attachment-runtime-inputs.json`.
   Their hashes were verified against exact-version
   [PyMuPDF metadata](https://pypi.org/pypi/PyMuPDF/1.28.2/json) and
   [Pillow metadata](https://pypi.org/pypi/Pillow/12.3.0/json).
2. Run `prepare-attachment-runtime.sh WHEEL_DIRECTORY`. It checks wheel hashes
   before unpacking trusted public inputs and builds a read-only SquashFS with
   the fixed guest entrypoint. Never mount a guest-modified filesystem on host.
   SquashFS build timestamps can change its digest; any rebuild requires explicit
   trusted review and updating `ATTACHMENT_PIN`, never accepting a caller pin.
3. `prepare-attachment-fixtures.py` generates known synthetic fixtures with local
   PyMuPDF/Pillow; it never parses existing documents. Copy those generated files
   as `/home/worker/attachment-fixtures` in the clean outer worker.
4. Install only `firecracker_backend.py`, `attachment_backend.py`, and the minimal
   `attachment_sandbox.py` validator under `/opt/gmail-worker`; install the pinned
   image as `/var/lib/gmail-worker/images/attachment.squashfs`. The smoke script
   runs inside the outer VM via `sudo python3 attachment-backend-smoke.py`.
   Run `sudo python3 attachment-cancellation-smoke.py` there for the actual
   cancellation-during-launch proof; it pauses only the trusted controller after
   the real VMM starts, leaving that guest waiting for parser input.

The retained proof image SHA256 is
`89bf4da702506dadacc5cd08b9f4d737f27abb592eb35b7a768cc22d590f9068`.
It includes the PDF text truncation fix. The previous image
`4a445c5bee81a6ab660f83a9c3ef0a193e6c1973df2758361f4e45086797a376`
remains retained as a rollback artifact in the synthetic assets directory.
The public runtime uses PyMuPDF 1.28.2 and Pillow 12.3.0, compatible with the pinned
vendor guest's Python 3.12. No host site-packages or user configuration are copied.

## Actual observed results

`attachment-qualification.json` records the successful run:

- Synthetic PDF: text `SYNTHETIC ATTACHMENT 42`, one 417×417 PNG.
- Same PDF inside a ZIP member named `../../hello.pdf`: same text and PNG, with
  no archive member ever written to a filesystem.
- Synthetic 8×8 PNG: normalized successfully.
- A synthetic single-page PDF containing over 200,000 text bytes: exactly
  200,000 output text bytes and `truncated=true` in the rebuilt guest image.
- 100,000-point-square PDF, 100,000×100,000 PNG header and malformed PDF: rejected.
- 25 unsupported ZIP entries followed by a PDF: truncated, no files/pages/text.
- Separate parser controller exited immediately after launch; the independent
  8-second test lease killed and reaped its orphan VMM. Observed UID 65534,
 seccomp mode 2, memory.max 805306368, swap 0, pids 128, CPU 100000/100000.
- Actual parser VMM cancellation while the trusted launch handoff was paused:
  `start` returned in approximately 0.0008 seconds; `stop` completed in 0.058
  seconds, reaped the VMM and joined the controller thread. The second stop was
  idempotent and the worker inventory was empty. These are synthetic observations,
  not production latency guarantees.

Local tests cover owner binding, malformed inputs before launch, revocation and
caller cancellation during load/parse/teardown, output suppression after late
revocation, capacity retention through teardown, input/result bounds, geometry,
malformed protocol teardown and existing worker lifecycle. Synthetic checks are
not a complete adversarial qualification of MuPDF/Pillow/KVM.

## Remaining before multiuser public use

The host APIs and ingestion still invoke legacy parsers. Callers must be wired
to the qualified service with owner-scoped DB loading and trusted persistence;
there is deliberately no automatic host fallback. Office/HTML/calendar/text,
HEIF, OCR, animated multi-frame rendering and network fetching are not implemented
by this first parser job. Existing attachment features remain intact behind the
owner-only gate until equivalent guest functionality is ready.

The standalone backend still needs production service registration,
deployment-wide admission controls, and durable cleanup/retention of completed
or crash-interrupted metadata/serial logs. This staged service is not mounted in
public routes and legacy previews retain their separate limits; it does not claim
feature parity. The fixed 45s watchdog remains active. Guest output is untrusted
and must use safe download/preview handling; never feed returned PNGs into
another native host decoder.
