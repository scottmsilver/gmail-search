# Invited browser app qualification — 2026-09-15

The production Next.js build was exercised with headless Chromium through a
loopback TLS proxy. The browser resolved `gms.browser.test` only to that proxy.
The dedicated invited app used real IdentityStore/Registry/Events, disposable
PostgreSQL conversation tables, and a fake worker. No real mail, Google login,
production service or live provider was accessed.

Observed assertions:

- An authenticated synthetic user submitted a question and saw its answer.
- Reload fetched the saved rich transcript and displayed the answer.
- A follow-up answer included the prior saved answer as context.
- A second synthetic user could not see the first user's conversation.
- Stop called the dedicated cancellation proxy, returned HTTP 200 with state
  `cancelled`, and completed worker cleanup.
- No browser `pageerror` occurred.

The first test attempted `networkidle` navigation and timed out. The successful
run waited for DOM readiness and specific visible answers instead. Navigation
and Stop caused expected request-abort notifications; those were not treated as
JavaScript errors. Browser cookies were injected from private synthetic session
state: this does not qualify Google's interactive login or consent flow.

`browser-proof.cjs` preserves the exact assertions. It expects the private
synthetic harness at `/tmp/gms-browser-app-proof-v2`; that temporary directory
contains only test sessions and test TLS keys and is not a deployment artifact.
The harness listeners are shut down after qualification. The independently
qualified real VM path is recorded in
[the manager report](../../deploy/public/worker/FULL_AGENT_MANAGER_QUALIFICATION.md).

Still required before public cutover: browser search/attachment routes, automatic replay/reconnect,
real invitation/Gmail registration and consent checks, production data/index
qualification, and deployment/rollback assembly.
