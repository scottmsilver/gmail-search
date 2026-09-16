# Large tool-result runtime qualification

**Passed with the actual invited HTTP app, fixed SSH manager and jailed Pi VM on synthetic data only.** The current candidate image is:

`968d41e5875b18044282901c6848f1a257833f2f875ec4ac2cddf83bf29da703`

The candidate pin changed only after successful end-to-end verification. The previous qualified `2d0c6d9e…` image and its historical evidence remain preserved. [Machine-readable evidence](full-agent-large-runtime-qualification.json) records the exact hashes and results.

## What this closes

The previous runner capped each local Pi RPC record at 65,536 bytes, and the pinned MCP adapter's default output guard independently truncated model-facing text at 50 KiB. The new image keeps display events bounded while allowing larger local records:

- Pi RPC: 32 MiB per record, 128 MiB aggregate per run.
- Display events: unchanged 65,536 bytes per event and 8 MiB aggregate. Oversized tool arguments/results become redacted previews with `display_truncated: true`; the original Pi result is retained.
- Fixed MCP output guard: enabled with an 8 MiB text ceiling, 100,000-line ceiling and 16 KiB details ceiling. This remains fixed extension configuration, not a guest-selected limit.

The actual fixture returned an **80,028-byte thread body**, including a distinctive final marker, using explicit `body_limit: 100000`. The mock provider verified the complete body exactly. The corresponding browser tool event carried `display_truncated: true`. The largest serialized display event was 21,109 bytes. Pi completed subsequent tool calls and its final transcript without the old RPC overflow.

## Complete workflow

The same HTTP request exercised all eight typed mail tools and native Bash, verified a separate 10 MiB raw attachment, published a CSV, streamed 21 events, obtained worker stop ACK, committed one PostgreSQL answer receipt, and downloaded the artifact over the authenticated browser endpoint. Owner conversation retrieval and event replay passed; foreign-owner artifact/conversation access was denied. Ten mock provider requests received ten close acknowledgements.

The final run took 11.753 seconds. It used the new conservative 175-second browser run deadline with the unchanged 180-second worker wall guard, while the disposable outer clock was deliberately **1.077 seconds behind** the host. Earlier attempts correctly refused an exact 180-second host deadline when the worker clock was behind; no worker guard was relaxed. Clocks must still be managed operationally: the five-second margin does not support arbitrary drift.

The manager exited cleanly, inventory was empty, all fixture databases/roles and listeners were cleaned, and the outer VM exited with its SSH listener absent. The outer retained its independent 900-second boot timeout. No host live unit, production mailbox, production runtime or provider credential changed.

## Reproduction and limits

Private assets live under `worktrees/full-agent-assets-20260915/full-agent-large-runtime-qualification`: current and intermediate immutable images, verified public-input build logs, exact exercised runner/extension sources, `qualify_browser.py`, per-attempt logs and clock-margin evidence. The intermediate runner-only image `2079b20c…` is preserved as **unqualified** because it retained the adapter's 50 KiB default.

The input manifest, package lock, public dependency/license trees and root-owned build procedure are unchanged from the prior qualified image. Only the fixed runner and extension settings changed. The retained public inputs were verified again before each build.

This proves the specific 80 KiB result through the actual Pi/MCP/app path. It does not establish that every response at the configured 8/32/128 MiB ceilings fits inference budgets or VM memory, and it is not a real-model quality/provider availability test. Inference and query embedding were local synthetic implementations; real owner-bound readers, native ScaNN, gateway services, ArtifactStore, browser routes, manager, VM and PostgreSQL persistence were exercised. The app was served on temporary loopback listeners, not publicly deployed.

The earlier actual in-flight cancellation and SSH-denial results remain documented in [the manager qualification](FULL_AGENT_MANAGER_QUALIFICATION.md). This larger-image pass did not repeat those unchanged paths.

Final source verification: **99 passed in 7.11 seconds** across the new manager/adapter/Unix transport, full-runtime, runner, WorkerController and BrowserRuns suites. The broader runner/v3/MCP/CLI/inference set passed **155 tests in 2.18 seconds** before the final pin promotion. Scoped Ruff F checks and `git diff --check` passed.
