# Shared database and isolated full agents implementation plan

> **For agentic workers:** Use superpowers:executing-plans or superpowers:subagent-driven-development task by task. Preserve the existing uncommitted work; do not commit or deploy an unqualified candidate.

**Goal:** Invited users can use the full Gmail agent tools through the public site, accessing only their own mailbox, without copying the mailbox database.

**Architecture:** Keep shared PostgreSQL storage, with LIST(user_id) partitions for messages, attachments and propositions so native BM25 statistics stay within each mailbox. Trusted gateways select immutable owner-specific database identities and mediate all mail, inference and artifact access. Arbitrary agent code runs on an isolated Linux execution worker with no platform credentials or unrestricted network.

**Tech stack:** PostgreSQL/psycopg, Python/FastAPI, Next.js, existing Pi and Claude runtimes; Firecracker or qualified gVisor worker.

**Approved specification:** `docs/superpowers/specs/2026-09-15-full-agent-isolation.md` and user's approval on 2026-09-15. The existing public release remains owner-only until complete qualification. There is no fallback to privileged SQL or the shared private agent container.

**Progress:** See `docs/shared-database-implementation-status.md` for verified
components and remaining launch requirements. Login is fixed/deployed; Tasks
2–4 have staged tested libraries and a scoped SQL HTTP adapter. Task 5 now has actual Pi and native Claude shell/Python, streamed inference
and artifact upload inside a synthetic jailed microVM with a mocked upstream.
Production service integration and advertised provider qualification remain.
Importer key collisions have staged schema/caller/BM25 fixes recorded in
`deploy/public/OWNER_KEY_MIGRATION.md`. This overall plan is not complete.

The approved user-partition implementation and qualification are tracked in
[the partition subplan](2026-09-15-owner-partitioned-search.md). Its candidate
migration/provisioning/admission and native ranking checks passed independent
review. Production capacity/recovery and the full public release remain gated.

## 1. Establish prerequisites and diagnose login

- [ ] Inspect actual runtime, database schema, deployment state and worker availability without exposing credentials or mail.
- [ ] Reproduce the browser-login boundaries and inspect sanitized failures. Fix a demonstrated cause with a regression test; distinguish this from completing actual Google sign-in.
- [ ] Verify a dedicated Linux worker can provide the proposed isolation. If one is absent, prepare the software and identify the exact provisioning requirement; never substitute the mail host's ordinary Docker runtime.

## 2. Immutable database readers

First define `gateway/schema.py` as the shared table/column/built-in-type allowlist. Ownership is the stable server user ID; gateway APIs receive it only from verified authentication, never a submitted query. A trusted active-owner callback will be replaced by invitation/lease validation at integration. Test PUBLIC policy composition, effective inherited grants including TEMP, security-definer functions and mismatched owner bindings on role reuse.

Files: new `src/gmail_search/gateway/database.py`, `deploy/public/provision_reader.py`, `tests/test_gateway_database.py`, `tests/test_gateway_database_integration.py`.

- [ ] Write failing tests for deterministic owner-specific roles, fixed restrictive policies, minimal grants, rejected privileged role reuse, revoked readers, unknown owners, DSN identity mismatch and password-safe errors.
- [ ] Provision SELECT-only roles transactionally, from trusted owner identity. Refuse missing RLS, unexpected object ownership/memberships and unsafe inherited permissions rather than modifying unrelated privileges. Scope policy changes to each new role and approved mail tables.
- [ ] Connect directly as the selected role; never connect as an administrator then SET ROLE. Keep credentials outside runtime inputs and guest environments. Bound concurrent connections and query duration.
- [ ] Test on an ephemeral PostgreSQL fixture with two synthetic owners: unfiltered SELECT, foreign owner predicates, mutable app.user_id, SET ROLE, writes, catalog/functions access and role reuse. Do not copy production mail.

## 3. Analytical SQL compiler and gateway

Files: new `src/gmail_search/gateway/analytics.py`, `src/gmail_search/gateway/schema.py`, `tests/test_gateway_analytics.py`; dependencies in `pyproject.toml`/`uv.lock` if needed.

- [ ] Write adversarial compiler tests before implementation: nested writes, system relations, functions, casts/operators, comments/multiple statements, ambiguous/unknown columns, invalid aliases and oversized inputs.
- [ ] Parse to a syntax tree and emit fresh parameterized SQL from an explicit supported node set. Resolve every table/column against a closed mailbox schema. Support joins, grouping, selected aggregates/windows, nonrecursive CTEs and subqueries; reject every unsupported feature.
- [ ] Execute through owner reader with read-only transactions, deadlines, result/byte caps and cancellation. Schema tool returns the explicit schema without host paths, embeddings, credentials or operational tables.
- [ ] Enforce `work_mem`, administrator-set `temp_file_limit`, per-owner/global concurrency and lock timeout. Discard cancelled connections. Document exact allowed functions/operators/types and CTE/window scope rules with rejection tests.
- [ ] Verify exact whole-mailbox counts and joins against two-owner synthetic data, plus timeout/cancellation and incomplete-extract reporting. Keep existing raw SQL disabled until this path is qualified.

## 4. Run capabilities and durable lifecycle

Files: new `src/gmail_search/gateway/capabilities.py`, `src/gmail_search/gateway/registry.py`, gateway route module and tests; additive registry migration under `deploy/public/`.

- [ ] Test forged/expired/revoked credentials, audience mismatch, changed owner/conversation/run, concurrent budget reservation and stale workspace fencing.
- [ ] Store run leases, deadlines, idempotency keys, event cursors and budget reservations durably. Bind tokens to exact operations and revalidate active invitation and lease on each operation and commit.
- [ ] Implement cancellation propagation and orphan sweeper. Verify process restart, worker loss, stream expiry and simultaneous battle branches.

## 5. Isolated worker and provider gateway

Files: new worker implementation and deployment directory; provider gateway module; adapters in existing runtime files only after a successful synthetic spike.

- [ ] Boot a clean fixed image on dedicated Linux worker; verify external network, LAN, DNS, metadata and other guests are unreachable from hostile guest code.
- [ ] Run Pi and native Claude through a gateway holding provider credentials. Strict schemas exclude remote fetches, provider-side tools and arbitrary model/upstream selection. Verify no provider keys in guest.
- [ ] Demonstrate streaming, cancellation, shell/Python, chart/XLSX output, model choices and workflow subprocesses with synthetic inputs. Require functional evidence before enabling browser selection.

## 6. Workspaces, artifacts and full runtime integration

- [ ] Implement owner/conversation/run namespaces, fenced workspace versions and separate battle outputs. Restore data and the selected transcript only into a clean runtime home.
- [ ] Publish bounded bytes with server-generated storage keys, not host-side resolution of guest paths. Test traversal, symlinks, stale commits, foreign IDs, content types and cross-session downloads.
- [ ] Wire retrieval, SQL, attachments, events, artifacts and existing UI capabilities to the scoped worker path. Test multi-turn continuity, both battle branches and reconnect replay.

## 7. Invitation and account-bound Gmail onboarding

Files: separate identity-store and invitation modules; integration in `auth/routes.py`, `auth/public.py`, public startup and relevant web routes/components.

- [ ] Test active/revoked invitations, stable provider subject, exact verified email matching, empty new accounts, foreign object IDs and administrator non-bypass.
- [ ] Separate sign-in from Gmail consent. One-use signed consent state binds expected user/account; reject wrong-account callback before credential storage or ingestion.
- [ ] Provision user access before admission; never widen the original owner's fixed DB policy. Revoke sessions, leases and credential use promptly on invitation removal/disconnect.

## 8. Qualification and rollout

- [ ] Run focused unit tests as each component changes, then the combined auth/boundary/runtime/database suite. Build frontend in a candidate release directory.
- [ ] Independently review security boundaries and exercise all capabilities with two synthetic owners, malicious mail, quotas, cancellation, restart and fault injection.
- [ ] Verify actual broker/Google login and separate Gmail consent using two invited accounts; user interaction may be required to complete Google authentication.
- [ ] Deploy only the qualified complete candidate behind the existing hostname; retain rollback that disables invitee ingress instead of routing invitees into the owner-only app.
- [ ] Record commands, results, residual limitations and exact release. Do not claim completion while any capability or isolation acceptance criterion remains unverified.

## Verification commands

Use `UV_CACHE_DIR=/tmp/gmail-search-uv-cache uv run --no-sync pytest tests/test_gateway_*.py -q` for the new components, and the existing public auth, boundary, containment and runtime tests for regressions. Live DB fixtures must use generated synthetic data and dedicated temporary resources. No production mailbox copies or real provider calls are required for unit tests.
