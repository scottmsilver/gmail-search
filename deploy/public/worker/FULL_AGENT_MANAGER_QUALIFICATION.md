# Browser → remote worker → PostgreSQL qualification

On 2026-09-15, the composed invited browser API successfully ran the actual separately pinned Pi VM through the new fixed SSH manager. **Synthetic data and mock inference/embedding only; no deployment or real provider call.**

[Sanitized machine-readable evidence](full-agent-manager-qualification.json) records the unchanged image `2d0c6d9e861740dbed914d6a73cd9a787fac6bc434f103c2bc00b3b2c6477bde`, source hashes, cleanup and separate cancellation/access-control results.

## Actual successful path

Two distinct temporary loopback listeners served the bearer-authenticated tool gateway and `create_invited_app`. Real `IdentityStore` sessions represented two imported synthetic owners. A browser HTTP request passed through `create_invited_app`, `compose_browser_runs`, the actual `WorkerController` and `SSHFullAgentBackend`, a dedicated forced-command SSH frontend and root Unix manager, then a jailed Firecracker guest running Pi.

The guest called all eight typed MCP mail tools plus native Bash, downloaded a 10 MiB raw file with matching SHA-256, and published a CSV through the real ArtifactStore. Gateway events arrived between mock inference calls. After worker stop ACK, the existing PostgreSQL conversation tables received one rich answer and one idempotency receipt. Browser HTTP downloaded the owner's artifact, read the saved conversation and replayed the run; the other owner's artifact/conversation requests were denied. Ten mock provider calls received ten close acknowledgements; 21 gateway events were recorded. This workflow took 24.877 seconds.

A second actual HTTP run was cancelled while its mock provider stream was open. Cross-owner cancellation was denied. Owner cancellation drained the relay/provider, obtained VM stop ACK, set the browser state to cancelled and persisted no answer. The budget held zero reserved units afterward and charged the conservative unknown-usage reservation of 204,096 units. These are synthetic accounting units, not a price claim. That check took 15.635 seconds. The gateway logged its expected generic stream-failed exception when cancellation interrupted the stream; no bootstrap or provider credentials were logged.

The dedicated `gmail-full-agent-rpc` account denied general commands, PTY, SFTP and reverse forwarding. It has only the fixed frontend command and private Unix socket access; the private service key remained on the trusted host. No public worker network listener was added. The outer-only OpenSSH configuration was validated before reload; an initial invalid Match-block placement was corrected before installation completed.

## Cleanup and boundaries

Both disposable database fixtures and extra reader roles were removed; capability/run authority was revoked and temporary browser/gateway listeners and reverse forwards stopped. The manager exited on SIGTERM, physical inventory was empty, and the outer VM exited with its SSH listener absent. Each outer boot retained the existing independent 900-second watchdog; each VM retained its fixed lease/deadline enforcement. No host live unit, host SSH configuration or production data changed.

The private assets directory `worktrees/full-agent-assets-20260915/full-agent-manager-qualification` retains the exact dedicated keys, fixed install script, reusable proof scripts, per-attempt logs, executed source snapshots and sanitized evidence. `qualify_browser.py` composes the real services; `qualify_cancel.py` adds the held-provider cancellation case. First fixture attempts stopped before VM launch after discovering existing conversation tables and the broker's explicit `trust_env=False` requirement; neither required a product relaxation.

This is an HTTP integration proof, not a browser UI automation proof or real-model quality test. Mock inference prescribed the tool sequence and checked each preceding real result. Source changes increasing Pi RPC limits were intentionally **not** included in this image; large textual tool-result support requires the subsequent separately pinned image proof. The current deployed/public application is not changed by these experiments.

Focused manager/adapter/browser/worker tests: **57 passed in 11.82 seconds**. The manager contains durable metadata only, and its synthetic backend guard and historical image profiles remain unchanged.
