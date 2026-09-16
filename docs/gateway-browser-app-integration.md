# Browser integration for the full worker app

Status: candidate implementation, **not deployed**. The public site still uses
its existing release. These tests do not establish an actual Pi VM/provider run.

## Implemented path

The existing chat page and `/api/chat` proxy can consume the full worker's
session and events through `auth/run_routes.py`. An invited session starts a
`BrowserRuns` run, and its prompt reaches trusted worker input preparation.
The browser receives tool activity, then an answer only after worker stop ACK
and the trusted answer persistence callback succeed. A `persist_ok` event stops
the existing proxy from overwriting the saved rich answer with a text-only copy.

Replay uses the same run ID and conversation, not a new worker. The Python SSE
route checks the invited cookie and account generation before every private
event. Foreign owners cannot read, cancel, or claim an existing conversation.
POST admission and Stop require the configured exact HTTPS Origin. An initial
response that cannot send its first session frame cleans up its admitted worker,
including cancellation while the send is blocked. The response header declares
server transcript ownership before any body frames: losing `persist_ok` cannot
trigger a destructive fallback conversation PUT. Neither
worker capabilities nor caller-supplied owner headers authenticate these routes.

Stop uses a separate authenticated request and waits for the controller's
termination acknowledgement. Browser state remains retryable when cleanup is
pending. Stream disconnection alone leaves an already admitted run available
for replay, bounded by the worker lease. A request cancelled during admission is
owned until its newly admitted worker is cleaned up. Startup/shutdown must own
`BrowserRuns.close()` and worker reconciliation.

The invited auth router now returns `multi_tenant: true` on `/api/auth/me` and
Gmail status, as required by the existing auth gate and Settings card. Gmail
status selects the separate POST-based consent flow; Settings uses a top-level
form, and new proxies preserve its state cookie and callback handoff. This is
synthetic contract verification, not actual Google consent or a deployed broker.

## Dedicated app composition and remaining deployment work

`gmail_search.invited_app.create_invited_app` now assembles invited auth,
conversation CRUD, run and result routes with a private browser request boundary.
It disables documentation endpoints, rejects caller identity headers, checks
mutation Origins and permits only the configured public host or loopback Host.
It does not import the legacy server. One API process owns each Registry.
Startup drains abandoned browser runs, reconciles worker inventory, and drains
durable Gmail credential cleanup on a threadpool before serving. Failed cleanup
prevents startup and closes recovered runs. Shutdown waits for owned run cleanup.

A qualified deployment still must:

1. Supply the trusted dependencies to this factory and bind its listener to
   loopback behind Next.js. Do not mount invited identity onto legacy mail APIs.
2. Use `compose_browser_runs` from `gateway/browser_conversations.py` with an
   fixed-owner writer registry: `connect=None, transaction_factory=writers.connection`.
   The registry selects an immutable PostgreSQL login for each owner. Its claim and rich
   answer persistence use the existing shared conversation tables plus a small
   run receipt table installed by an explicit migration. Persistence runs outside
   Registry SQLite, checks controller authorization before PG commit and is
   idempotent by run ID. Cross-store revocation/commit is not atomic. The invited
   browser conversation list/read/save/delete routes now use this store. Browser
   submissions contain only user text: an unchanged prefix plus at most one new
   turn. Existing assistant/tool history remains server-owned. The sidebar's
   response envelope is preserved. Follow-up worker prompts include recent saved
   text within the existing 16 KiB bound, excluding rich tool payloads.
3. Supply fixed owner budgets and actual per-jail input preparation for the new
   guest runner. The full VM image and bridge were qualified with all eight mail tools,
   native Bash, a 10 MiB attachment and a CSV artifact using a mock provider.
   The separate manager/SSH-to-browser composition is undergoing qualification.
4. Set `GMS_FULL_WORKER_ROUTES=1` only for that composed release. It opens the
   frontend's authenticated GET replay and DELETE Stop proxy routes. Existing
   deployments continue denying these paths by default.
5. Connect automatic browser reconnect to the replay adapter. The HTTP adapter
   exists; automatic transport resumption is not implemented yet.
6. The factory mounts the reviewed result router: opaque artifact citations and frontend
   conversation-scoped download proxy are now implemented, behind the same
   rollout flag. Validate download behavior in an actual browser.
7. Exercise real browser login, actual worker/tool/provider execution, reload,
   Stop, artifact access and a second invited account before public cutover.

The UI knows the run after its session frame arrives. Stop before that frame and
navigation between active conversations still need browser-level qualification
(the UI now retains active run handles per conversation);
unit coverage is not evidence those timing windows are resolved.

## Evidence (2026-09-15)

- Root: invited run/result routes, controller lifecycle and underlying worker/
  event tests: **57 passed in 3.48s**. Actual temporary identity, run, capability and event stores; fake
  worker backend and in-memory conversation persistence; no real mail/provider.
- Frontend proxy/control/boundary/artifact/Markdown checks: **33 passed**,
  including explicit rollout-gate authentication and terminal notifications.
- Candidate `npm run build` passed (Next.js 15.5.15); it reported the existing
  multiple-lockfile tracing-root warning. Do not package an inferred home-wide
  dependency trace for deployment; use a clean release dependency installation.
- Shared PostgreSQL claim/idempotent transcript/revocation/controller composition:
  **4 passed in 0.66s**, using newly created disposable databases, no mail tables.
- Frontend TypeScript, scoped Ruff F and diff whitespace checks passed.
- Independent browser review: both reported findings fixed and rechecked.
  Additional failed-first-send and blocked-send cancellation cases pass.
- Invited authentication/Gmail status contract and lifecycle: **25 passed**;
  frontend consent/boundary checks: **7 passed** (overlap with earlier suites).
- Guest runner handoff: **97 passed** (separate report in
  `gateway-guest-agent-runner.md`); the pinned full VM image was subsequently qualified (see
  `../deploy/public/worker/GUEST_FULL_AGENT_QUALIFICATION.md`).

No production mailbox migration, public restart or live provider call occurred.
The retained-heap search-engine blocker remains separate and unresolved.

### Additional app assembly evidence

- Dedicated invited app test passed: identity, chat, replay, cross-user denial,
  unavailable legacy/admin APIs, host/header rejection and shutdown cleanup.
- Owner-scoped PostgreSQL CRUD, persisted follow-up prompt and prompt bounds:
  **12 passed in 4.23s** in disposable databases.
- Frontend chat regression tests **22 passed**, compact user-history test passed,
  and TypeScript passed after the transport change.
- New source-only large-result fix separates bounded local Pi RPC output from
  the unchanged 64 KiB event / 8 MiB event-stream bounds. Oversized tool-display
  copies get an explicit `display_truncated` preview after redaction; Pi's
  original tool result is untouched. The local RPC ceiling is 32 MiB per record,
  128 MiB per run to account for repeated final transcript records. The replacement image was subsequently qualified through the actual app,
  including the adapter's fixed 8 MiB model-facing output guard and an 80,028-byte
  thread body preserved exactly through to mock inference.

### Browser citation reads

The optional `mail=BrowserMail(query_gateway, attachment_reader=...)` dependency
mounts `/api/thread/{id}` and `/api/thread_lookup` in the dedicated factory. These
fixed queries run under the existing owner-specific reader credentials; a
browser cannot choose an owner, SQL text, role or DSN. The HTTP adapter rechecks
the exact session generation after the read. Prefix ambiguity is explicit.
Thread bodies use bounded pages, and the UI labels partial bodies/threads.
Missing sender headers are normalized for the existing drawer. Attachment
inventory may use the reviewed metadata reader; browser raw attachment download
and the full browser search route remain separate integration work.

Actual owner-reader PostgreSQL tests use identical message/thread IDs for two
owners and verify separate results. Invited citation-route tests cover injected
owner query rejection and session revocation during a read. Both pass.

The integrated browser/auth/persistence/runner suite passed **112 tests in
6.05 seconds** before the subsequent null-header regression (which separately
passes). A fresh production web build passed. Headless browser UI verification
passed through loopback TLS; see
[the browser report](../qualification/browser-app/README.md). Its temporary
listeners and matching disposable database were removed after verification.

### Clock margin found during actual image qualification

A disposable worker clock approximately 0.726 seconds behind the host caused
an exact 180-second host deadline to fail the manager's strict 180-second
admission guard. Browser admission now subtracts five seconds from its existing
ceiling (175 seconds for the full-agent profile). The remote hard limit is
unchanged. Clocks still require operational synchronization; this margin covers
small skew and is not permission to extend a run. A fixed-clock regression
proves the host deadline is inside the worker ceiling.


The replacement full-agent image is now pinned to
`968d41e5875b18044282901c6848f1a257833f2f875ec4ac2cddf83bf29da703` after
successful actual app-to-VM qualification, including a deliberately 1.077-second
worker clock lag. The strict remote 180-second wall limit remained unchanged;
the 175-second host deadline succeeded. The complete 80,028-byte result reached
mock inference, the event preview was explicitly shortened, and all eight mail
tools, Bash, 10 MiB raw download, artifact ownership, PostgreSQL persistence and
replay passed. The old qualified image remains available as evidence. This does
not qualify real Google login, real-model answer quality or public deployment.


### Production writer preparation

Run `BrowserConversations.install(admin_connection)` as an explicit offline
migration before provisioning application writers. It transactionally creates
or validates the four-column receipt table, primary key, NOT NULL fields and
conversation foreign key with cascading deletion; enables RLS; and revokes
PUBLIC access. It can upgrade the earlier receipt table and can be repeated.
It rejects incompatible constraints instead of repairing them implicitly.
Then reprovision every configured application writer using
`provision_application_writer` so each receives owner-and-conversation-bound
receipt SELECT/INSERT privileges and policies. Direct receipt UPDATE/DELETE
remains unavailable. No public startup or request runs this migration.

Fresh disposable PostgreSQL verification on 2026-09-15: **24 tests passed** for
writer privileges, owner isolation, transcript persistence, idempotency,
revocation, receipt cascade and repeatable schema upgrade. This is candidate
verification; no production schema or credentials were changed.
