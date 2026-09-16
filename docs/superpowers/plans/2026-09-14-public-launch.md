# Public Gmail Search Launch Implementation Plan

**Goal:** Deliver owner-only public Gmail Search using the existing identity broker without exposing shared agent workspaces or privileged tools.

**Architecture:** Separate web/backend processes on loopback3001/8091; explicit route policy, request/concurrency limits and existing strict public auth. Public chat uses a bounded server-side Gemini function loop with fixed authenticated retrieval functions only. Private Pi/Claude app remains on3000/8090. Separate gmail-search broker registration, secret and exact callback. Dedicated outbound tunnel with default404.

## Work and verification

- [x] Broker: isolated source copy, separate app callback/secret/audience, preserve terminal flow; adversarial identity tests and build. Deploy only after reviewing exact diff.
- [x] Runtime: runtime_public.py explicit tools/arguments and injected owner; no shell/files/arbitrary URL tools; bounded rounds, output and timeout; mocked adversarial tests. Wire public mode before workspace/credential paths.
- [x] Web: public route/method/auth boundary, headers/body cap, owner-only truthful model presentation; route tests, TypeScript and isolated build.
- [x] Backend: public route allowlist and authentication before handlers, body/rate/concurrency limits; synthetic middleware tests; skip shared runtime startup probes/credential sync in public mode.
- [x] Candidate: separate restricted service configs and secrets; loopback candidate with test identity handoff, cross-user/wrong-signature/replay/logout probes, streamed synthetic prompt and private-service health.
- [x] Broker deploy: independent Gmail secret and exact return URL; preserve terminal registration and test start flows.
- [x] Ingress: dedicated exact-host tunnel only after candidate checks, then anonymous external probes and owner browser sign-in. Keep previous services running and rollback by disabling new tunnel.

No commits or pushes. Preserve all existing dirty work. Owner browser Google login cannot be manufactured; request only that final validation when the working URL is ready. Never represent automated synthetic login as real Google authentication.

Manual real Google/off-LAN browser check remains pending; user has the live URL. Automated HTTPS synthetic-handoff and real-model checks passed.
