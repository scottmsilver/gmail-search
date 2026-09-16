# Production readiness, 2026-09-15

**Not deployed. The invited full-worker application is not ready for cutover.**
The current public release has not been restarted or migrated by this work.

## Fixes completed in the candidate

- Browser transcripts can use `WriterRegistry.connection(owner)` so every
  operation runs with the fixed user's PostgreSQL identity. Receipt policies
  require both the owner and a conversation owned by that owner. Grants permit
  SELECT/INSERT; conversation deletion cascades receipts.
- The offline receipt migration is repeatable, validates column and constraint
  compatibility, and enables RLS for earlier compatible receipt tables. Existing
  writer roles must be reprovisioned before enabling the new runtime.
- Invited startup recovers abandoned runs, drains durable Gmail disconnect tasks
  off the event loop before admission, and closes recovered runs if draining fails.
- A dedicated production worker enrollment profile, installer, manager service,
  restricted SSH configuration, and private gateway tunnel template exist.
  The synthetic worker guard remains intact. Production guest serial output is
  discarded while its byte quota remains enforced.

## Unfinished application work

`create_invited_app` remains a dependency-injected factory. A production launcher
must instantiate and own its real fixed-user readers/writers, search indexes,
provider transports, budget bindings, worker transport and gateway listener.
It must enforce the existing release gate and close resources on failure.
The live service still starts the old `gmail_search.public_server`.
The new factory and worker installer do not make that old entrypoint multi-user.

**Superseded 2026-09-15 (later the same day):** the pg_search assertion was
diagnosed to root cause and fixed locally — see
[root cause](qualification/retained-reader-root-cause.md) and
[the migration plan](superpowers/plans/2026-09-15-migration-after-ctid-fix.md).
Phase two is now implemented and passes synthetically on the patched engine
(`paradedb-patched:0.23.0-ctidfix`). The paragraph below records the state
before that fix.

The retained-table migration stops at INDEX_PENDING. The existing pg_search
retained-table maintenance assertion remains unresolved; no production migration
or READY publication is qualified. See
[the evidence](qualification/retained-reader-maintenance-blocker.md).
Existing preview vectors cannot be relabeled as stable model vectors, and the
owner-specific raw attachment layout still needs production migration evidence.

## Infrastructure and configuration prerequisites

### Worker

No persistent enrolled production VM was found in the inspected local setup.
A dedicated VM and a verified export of the pinned full-agent image are required.
See [worker installation](../deploy/public/worker/production/README.md).
No worker was installed, enrolled or restarted by this change.

### Gmail broker

Read-only cloud inventory found active `broker` and `identityBroker` functions,
but no `gmailBrokerV1`. None of the six dedicated `GMAIL_V1_*` secrets exists.
The candidate Hosting rewrite and metadata TTL policy are not deployed.

Create a dedicated OAuth Web client in project `silver-oauth-broker` with redirect:

```
https://auth.oursilverfamily.com/v1/gmail/callback
```

Store credentials privately; do not paste secrets into chat or commit them.
The dedicated broker needs these Secret Manager keys:

- `GMAIL_V1_GOOGLE_CLIENT_ID`
- `GMAIL_V1_GOOGLE_CLIENT_SECRET`
- `GMAIL_V1_APP_BEARER`
- `GMAIL_V1_HANDOFF_SECRET`
- `GMAIL_V1_STATE_SECRET`
- `GMAIL_V1_TOKEN_ENC_KEY`

Non-secret broker parameters are `GMAIL_V1_PUBLIC_ORIGIN=https://auth.oursilverfamily.com`
and `GMAIL_V1_APP_CALLBACK=https://gms.oursilverfamily.com/api/auth/gmail-callback`.
Configure Firestore TTL for `gmail_v1_metadata.expires_at`. App configuration
must use the same dedicated app bearer and handoff secret; it must not reuse
identity-login or session secrets. The app-side startup mapping remains unfinished.
Enable the new frontend routes only with the qualified invited service.

## Remaining cutover evidence

After implementing and configuring the above, qualify the exact persistent
installation with synthetic data, then verify real Google login/consent and
owner isolation with two invited accounts. Verify actual provider execution,
restart cleanup, cancellation, private artifacts and rollback. Do not route
invitees into the legacy owner-only service as a rollback.
