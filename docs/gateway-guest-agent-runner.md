# Fixed Pi full-agent runner

This integration accepts an actual user question and streams Pi events into the existing run event store. A separate synthetic image has passed an actual jailed Pi run with all eight typed mail tools, native Bash, live events, a 10 MiB raw download and artifact publication. Historical image pins, backend guards and tool-only bootstrap defaults are unchanged. See [qualification evidence](../deploy/public/worker/GUEST_FULL_AGENT_QUALIFICATION.md); no deployment, real provider call or production data access is claimed.

## Trusted input contract

`deploy/public/worker/guest_agent_bootstrap.py` exposes `validate_config`, `encode_config`, `read_config` and `receive`. The fixed entrypoint is `guest_agent_pi.py`; its sole permitted extension is `guest-agent-mail-mcp.ts`.

The new envelope is:

```json
{
  "version": 1,
  "profile": "mail-agent-pi-v1",
  "prompt": "The authenticated user's question",
  "tool_config": {
    "version": 3,
    "tool_profile": "mail-raw-mcp-v3",
    "capabilities": {
      "sql": "64 lowercase hexadecimal characters",
      "retrieval": "64 lowercase hexadecimal characters",
      "artifact": "64 lowercase hexadecimal characters",
      "attachment": "64 lowercase hexadecimal characters"
    }
  },
  "inference_capability": "64 lowercase hexadecimal characters",
  "events_capability": "64 lowercase hexadecimal characters"
}
```

The prompt is nonblank UTF-8, excludes NUL and is at most 16,384 **bytes**. The envelope is one big-endian uint32 byte length followed by at most 32,768 UTF-8 JSON bytes and exact EOF. Duplicate keys, unknown fields, mismatched profiles and malformed capabilities fail startup. `encode_config` snapshots validated primitives. Guest receive uses the existing one-shot host CID 2, vsock port 8002 seam with a five-second aggregate deadline. The controller must bind that listener to the selected jail, send once, close it and stop/reap the VM on failure. Old 4,096-byte tool-only bootstrap readers deliberately reject this envelope.

No owner, run ID, path, URL, executable, model or resource limit is selected through the envelope. The issuer must bind all six capabilities to the same active run. Required operations are SQL `schema/query`; retrieval `thread.get/search/facts.find/query.emails`; artifact `artifact.commit`; attachment `meta/text/raw`; inference `generate`; events `append`. The gateway remains the authority for these bindings and fresh authorization.

## Fixed runtime

The guest creates `/tmp/gms-run`, private home/work directories and a mode-0600 v3 capability file. Native Pi runs as UID/GID 1000 with a fresh session, a minimal environment and fixed process/file limits. The MCP server receives an empty environment plus fixed PATH/LANG, and loads only the four tool capability tokens from the private file. One persistent MCP core retains its shared two-socket admission and raw downloader lifetime.

The allowed tools are the reviewed eight mail tools plus Pi's `read`, `bash`, `edit`, `write`, `grep`, `find`, and `ls`. Extension/config discovery, skills, context files, themes and prompt templates are disabled except for the one fixed MCP extension. Fixed mail guidance explains tool use, untrusted retrieved content and `[art:OBJECT_ID]` citations from successful artifact receipts. The user's prompt goes through the Pi RPC stdin JSON `prompt` command; it is never shell or command-line interpolation.

The provider is the fixed local `gateway` profile, Anthropic messages at `127.0.0.1:18080`, model `claude-sonnet-4-6`. The host must compose the reviewed `pi-0.84.4` compatibility profile for that model. The inference capability supplies `ANTHROPIC_API_KEY`; `models.json` contains only the explicit `${ANTHROPIC_API_KEY}` environment template. This template syntax was inspected in the preserved Pi bundle's `parseConfigValueTemplate`. No mock key or actual provider credential is embedded in this entrypoint. Model availability and live provider use are not asserted.

## Events and terminal contract

The runner posts sequentially to fixed `/v1/events`, using only the events capability. It reuses the reviewed `GuestMailTools._request` JSON framing and owned socket close code through a small internal adapter; each acknowledgement must be exactly `{"seq": positive_integer}`. There are no retries. Each event is at most 65,536 bytes; local totals are capped at 10,000 events and 8 MiB. An event request has a five-second deadline. Known capability strings are replaced before publishing CLI-derived records. This is not a security boundary against a malicious same-UID guest encoding its own capabilities.

| Pi record | Gateway event |
| --- | --- |
| startup | `{"type":"status","state":"running"}` |
| `tool_execution_start` | `{"type":"tool_start","name":...,"args":...}` |
| `tool_execution_end` | `{"type":"tool_result","name":...,"result":...,"is_error":boolean}` |
| assistant `message_end` | `{"type":"text","text":...}` |
| `agent_end`, after CLI process cleanup | `{"type":"status","state":"runner_completed"}` |

Tool events stream as they arrive; assistant prose currently streams once per completed assistant message, not token by token. Hidden reasoning, raw stderr and whole CLI records are not published. Answerless `agent_end`, provider/extension errors, malformed/oversize JSON records and missing EOF fail the run. The RPC input is bounded at 65,536 bytes per line and 8 MiB aggregate; the turn has a 120-second deadline. Provider usage from the guest never changes the authoritative gateway ledger.

`runner_completed` is **untrusted display output**, not worker stop acknowledgement. The browser controller must obtain backend stop ACK, then persist the answer and mark its own final status. The serial marker `GMS_AGENT_RUNNER_COMPLETE` appears only after the owned bridge cleanup and capability-file removal; any top-level failure prints only `GMS_AGENT_RUNNER_FAILED` and exits 1. A revoked events capability may prevent any final guest event. Controllers therefore also monitor worker failure, cancellation and deadlines.

Process creation is owned across cancellation, including a process acquired after cancellation arrives. Cleanup sends group SIGTERM, drains stdout, waits up to five seconds, then sends group SIGKILL and waits for the leader/pipe cleanup. Repeated cancellation cannot abandon that cleanup. Same-UID descendants can create other sessions; the outer VM lease/watchdog and stop ACK remain the isolation and final reclamation boundary. Local cleanup is not proof that a gateway/provider operation drained.

## Qualified image and remaining composition

- Root browser controller supplies this envelope through the per-jail bootstrap and owns worker heartbeat, stop, event replay, answer persistence and owner-checked artifact downloads.
- The separate `agent_full` image/backend profile includes this entrypoint, v3 core, raw transport/downloader and verified public Pi dependencies. The builder checks the retained public-input fingerprint manifest and package lock; no historical image pin was changed.
- The separate fixed `guest-agent-vsock-bridge.py` bounds each direction at 12 MiB, uses at most 64 KiB reads and supports the qualified 10 MiB raw packet. The historical 8 MiB bridge is unchanged.
- The current 120-second guest turn limit must fit controller leases, capability TTL/renewal and the guest VM watchdog. Failure to fit is an explicit failed run, not permission to extend authority.
- Raw quotas remain per persistent MCP downloader, with a separate VM disk ceiling required; the one-shot CLI still rejects v3 raw activation. Attachment rendering remains outside this profile.
- Production retained-heap BM25 readiness remains blocked separately. This runner can be composed against fresh synthetic owner-isolated data without changing the production migration gate.

Unit tests use synthetic subprocesses and fake HTTP streams: arbitrary prompt delivery, live event ordering, closed schemas, byte framing, error/answerless output, fixed configuration, capability separation/redaction, event ACK parsing, process-spawn cancellation and repeated-cancel cleanup. The separate actual VM proof is recorded in the linked qualification report.

Validation on 2026-09-15: **97 passed in 1.88s** across the new runner's 30 cases plus historical bootstrap, v3 raw profile, MCP and CLI tests. This required scoped temporary loopback-server permission; the first sandbox run's 21 setup failures were denied local binds, not behavioral assertions. Scoped Ruff F checks and `git diff --check` passed. No image rebuild or provider call occurred.

## v3 advertised schema and output limits

The actual Pi JavaScript transport rounds signed-64-bit JSON schema maxima. The v3 profile therefore accepts numeric attachment IDs and attachment cursors only through `9007199254740991`, the largest exactly represented JavaScript integer. Its advertised schema expresses that limit and fixed string formats in descriptions where the existing inference schema subset does not support those numeric bounds or `pattern`; MCP and core runtime checks still enforce the exact bounds and formats. v1/v2 definitions and the inference validator are unchanged. All current v3 definitions pass the actual inference compiler after JavaScript numeric roundtrip.

The current 65,536-byte Pi RPC line/StreamReader bound, normalization bound and display-event bound can reject otherwise valid large thread/text tool results. A final `agent_end` can also repeat a large transcript. This proof uses short textual outputs; only raw attachment payload bytes travel outside Pi/model context. Supporting large tool responses requires a separately bounded RPC-record policy plus explicitly truncated display events; increasing only the event limit would not fix it. The current runner fails the run on these overflows.

Latest focused validation: **152 passed in 2.04s** across full-runtime, v3, MCP, CLI, runner and unchanged inference suites. The earlier 97-test result above records the initial source-only handoff.
