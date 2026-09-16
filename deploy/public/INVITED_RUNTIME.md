# Invited service launcher

Candidate source, not a deployed release. Start with:

```sh
GMS_INVITED_CONFIG=/absolute/private/invited-runtime.json python -m gmail_search.invited_server
```

`--check-config` validates the private JSON and fixed-user credential names only;
it makes no readiness claim and contacts no database, provider or worker.
The real entrypoint requires a ready, pinned TEXT-partition release before
initializing any runtime. It never migrates PostgreSQL, creates budgets, imports
users, regenerates indexes or supplies privileged fallback credentials.

## State and configuration

Use `invited-runtime.json.example` for the exact closed JSON contract. Copy it to
a private file owned by the service account, mode 0600. Unknown/duplicate fields,
wrong-owner DSNs and invalid paths fail. Provider cost fields are conservative
internal budget units, not price claims. All configured budgets must already
belong to the specified owner; exhausted budgets remain exhausted after restart.

The existing mode-0700 state directory contains:

- `registry.sqlite`: administrator-prepared runtime registry and release gate.
- `identities.sqlite`: imported and provisioned invited identities, each matching
  the configured canonical owner ID, verified email and Google subject.
- `artifacts/`: service-owned mode-0700 artifact storage.
- `runtime.lock`: acquired by the launcher to reject concurrent API processes.

Prepare the actual fixed reader/search/writer roles through the existing reviewed
provisioning path. Apply the explicit receipt migration and reprovision writer
roles before startup. Existing sealed owner indexes must have provenance for
`gemini-embedding-2`, 3072 dimensions; do not relabel preview vectors. Attachment
files must use the qualified owner-specific layout. The startup database checks
open only fixed-user logins and do not read message contents.

Only preconfigured and preprovisioned users can complete admission. To add an
invitee, provision their fixed credentials, budget, identity and search index
outside this process, update the private configuration, and restart the service.
This launcher does not yet automate the administrator onboarding workflow or
mail ingestion.

## Process and network lifetime

One process owns browser API 127.0.0.1:8091 and bearer gateway 127.0.0.1:8092.
The existing Next.js proxy forwards browser traffic only to 8091. The dedicated
worker tunnel reaches only 8092; its credentials are separate from the RPC key.
Use the enrolled production worker and host-key pinning documented in
[worker installation](worker/production/README.md).

The API owns the three provider transports, native indexes, worker adapter and
run cleanup. Failure to assemble or start either listener closes acquired
resources. Closing the release gate also denies subsequent reader/writer access.
The launcher reconciles abandoned workers and drains durable Gmail disconnect
tasks before either listener opens. Gateway readiness then precedes browser
admission. SIGTERM/SIGINT shuts down both listeners.

The service template uses a separately prepared `invited-current` release with
its own venv. Do not build a release from the worktree's symlinked venv or copy
home-wide Next.js tracing output. Install only a reviewed source/dependency
snapshot. Its auth env has no DB administrator or provider credentials; those
belong only in the fixed private runtime config as appropriate.

Do not enable the service or frontend `GMS_FULL_WORKER_ROUTES=1` until the
[remaining deployment gates](../../docs/production-readiness-20260915.md) pass.
The unit conflicts with the old public API because both use 8091. Rollback must
first disable invited ingress; the legacy owner-only app is not an invited-user
fallback.
