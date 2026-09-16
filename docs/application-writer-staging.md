# Staged application writer and admission provisioning

These modules are unmounted candidates. They do not authorize a release or modify production roles. Public startup retains its owner-only guard.

## Minimum writer profile

The direct immutable-owner login uses fixed restrictive RLS alongside its explicit permissive policy. There is no runtime SET ROLE, administrator login, bootstrap owner fallback, or worker schema initialization.

| Table | Allowed operations |
| --- | --- |
| conversations | Selected column reads/inserts; update title/updated_at; delete |
| conversation_messages | Selected column reads/inserts; delete; owner checked through conversation |
| agent_sessions | Selected column reads/inserts; update plan/status/final_answer/finished_at; conversation must share owner |
| agent_events | Selected column reads/inserts; owner checked through session |
| costs | Selected column reads/inserts only |

Only the three required serial sequences grant USAGE. Provisioning refuses missing RLS, unexpected table types/columns, custom triggers/rules, inherited role membership, privileged roles, and effective privileges beyond the allowlist, including PUBLIC privileges. Existing installations need an independently reviewed administrator migration for missing RLS; the provisioner does not repair or alter legacy roles implicitly. Every direct transaction checks the immutable role binding and invitation activity, and checks activity again before commit. Trusted callers must not commit inside the registry context.

Excluded: mailbox writes, users, sync_state, model_battles, legacy conversation_claude_session and agent_artifacts. The profile is sufficient only for the specified conversation/session/event/cost paths. Mail reads use the separate reader credential.

## Admission integration

`AdmissionProvisioner` runs only in a separate trusted administrator service. The invited router's `provision_account` callback should call that service through an authenticated controller channel; public controllers and workers must never receive its administrator DSN. It accepts server-derived `Account` plus independently verified Google identity, validates exact canonical users ID/email/subject, and provisions reader and writer in one database transaction. A session advisory lock serializes credential rotation and atomic vault installation. Admission reports success only after both credentials are installed. Sink failure denies admission; retry rotates both logins.

The trusted runtime endpoint must explicitly name the database and contain no login, service or options overrides. The installer must durably and atomically store both credential objects outside SQLite, and refresh the immutable runtime registries. It must not log passwords. The service caller must authenticate and validate the invitation before calling this module; it is not a guest-callable arbitrary SQL API.

Existing users with NULL google_sub require an explicit audited `bind_existing_subject(connection, known_old_owner_id, verified_identity)` import. Email claims alone never reassign the legacy owner ID. New canonical identities use the server-created invitation owner UUID; conflicting email/subject uniqueness fails closed. Canonical users remain administrator-only and need their own reviewed RLS migration before wider runtime grants.

## Separate ingestion profile still required

Ingestion writes messages, attachments, embeddings, topics, summaries/failures, propositions/processed markers; it also updates specific user status columns and per-owner namespaced sync_state keys. It needs a distinct fixed-owner login and exact privileges for those operations, including parent-scoped children and immutable owner keys. This conversation writer does not qualify ingestion. No ingestion worker may run schema DDL or fall back to the bootstrap owner. Inventory and privilege qualification for that larger profile remain release prerequisites.

## Verification boundary

Tests use a disposable synthetic PostgreSQL database with two owners and permissive legacy policies to prove the restrictive writer policies still isolate writes and reads. They exercise canonical identity rejection, explicit legacy binding, and vault failure/retry. They do not qualify production migrations, live credentials, broker deployment, or the full public application.
