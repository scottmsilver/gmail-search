# Shared-database full-agent implementation status

Updated 2026-09-15. **The full public rollout is not complete.** The live site
remains owner-only and retrieval-only. Implementation is isolated in
`/home/ssilver/development/gmail-search/worktrees/full-agents-20260915`
(the former `/tmp/gms-full-agents-worktree` remains an alias) because private services import the original
checkout directly. No production database migration or broker change has been
applied. No per-user mailbox copies are introduced.

## Approved owner partitions

The broad candidate regression run passed **639 tests in 138.20 seconds**: gateway,
invitation/consent, result access, guest mail tools/MCP, relay, application writer,
partition provisioning/ranking/lifecycle/admission, crawler collision and runtime
builder checks. Both synthetic PostgreSQL fixtures were explicitly configured;
there were no skips. Independent final migration and complete-schema lifecycle review passed
**36 tests**, with no remaining blocking findings. Review fixes cover stored
database dependencies, allocator names/comments and the actual network address
used by the synthetic-only apply guard. Empty-owner cleanup also passed five
independent checks. The existing owner-key, derived-data and
ingestion regression run also passed **64 tests** on the synthetic ParadeDB.

The user selected physical LIST(user_id) partitions in the shared database.
The [implementation plan](superpowers/plans/2026-09-15-owner-partitioned-search.md)
passed independent review. It covers messages, attachments, and propositions,
including native BM25 indexes for each owner. The migration contract and synthetic conversion rehearsal are implemented
and independently reviewed; no production DDL has run.
Administrator partition provisioning and mandatory invitation-admission integration
are implemented in the candidate. All three native lexical branches passed
foreign-corpus insert/update/delete invariance checks, including custom/generic
query plans. The combined real partition-admission, application-writer and
ranking run passed **17 tests**. Five crawler collision regressions and seven
partition-reader checks also passed independent verification. Small-scale setup
costs are recorded in [the overhead report](owner-partition-scale-qualification.md).

Seven new PostgreSQL reader checks passed against partitioned parents: fixed
owner RLS, hidden raw columns, denied direct child reads, and rejection of
reader/PUBLIC child SELECT, UPDATE, or DELETE privileges. These tests qualify
existing reader behavior, not the migration or full ranked-search gateway.

## Live browser-login fix

Release `login-fix-20260915` fixes OAuth's final cross-site document navigation.
Public HTTPS checks previously returned 200 for return navigation, 401 for
unauthenticated mail requests and 403 for cross-site API requests. Actual Google
sign-in still needs a browser retry; do not equate HTTP checks with completing
Google authentication. Previous release: `security-20260914-final`.

## Shared mailbox ownership

The demonstrated importer collisions are fixed in the candidate, with positive
regressions replacing the two expected failures. Messages use `(user_id,id)`;
attachment conflict targets and dependent references include owner. Owner
propagation covers ingestion, summary/failure storage, embeddings, crawl caches,
attachment extraction, inbox/search joins and clustering. New attachment bytes
use hashed owner directories; existing file paths remain readable without copies.
Sync checkpoints are per owner, initialized from that owner's history only.

`deploy/public/migrate_owner_keys.py` explicitly rebuilds the messages BM25 index
around a unique internal `search_id`, preserving its other configuration. Fresh
schema and search callers agree on that key. Startup reapplication does not
silently migrate legacy production keys. A separate derived-data migration adds
owner/message references for propositions and processed markers; see
`deploy/public/OWNER_KEY_MIGRATION.md` for ordering and deletion behavior.

Qualification includes actual colliding-ID search and index-rebuild rollback on
the installed image: PostgreSQL 16.13 / pg_search 0.23.0. The initial combined
ownership, schema, search and summary run passed **111 tests**. The main migration
suite passed **43 tests** on that image; the combined main and derived migration
suite subsequently passed **52 tests**. Owner-qualified SQL endpoint checks
passed **17 tests** after correcting invalid synthetic fixtures.

Read-only production preflight passed the existing six-table catalog/data checks
and reported `requires_bm25_rebuild=true`. Catalog-only size checks found messages
17 GB plus 925 MB indexes; embeddings 20 GB plus 396 MB indexes. The shared data
volume had approximately 114 GB free. These are observations, not a measured
maintenance-window or peak-WAL estimate. Backup recovery, production-scale rewrite
and coordinated caller rollout remain prerequisites. Migration CLIs still refuse
production application.

## Trusted gateway and data transfer

- Immutable per-owner SQL readers, restrictive RLS, direct restricted logins,
  privilege audit, closed analytical SQL compiler, cancellation and query limits.
- Durable hashed capabilities, leases, fencing and shared spend reservations.
- Bounded artifact byte upload with server-generated object keys, per-owner
  reservations, revocation checks, private downloads and interrupted-upload cleanup.
  No guest path is opened on the host. Optional worker HTTP routes are staged.
- Opaque workspace snapshots with atomic version promotion and explicit battle
  branch selection. No guest filesystem is mounted by this storage code.
- Bounded durable event replay with owner/conversation checks and retention hooks.

The initial combined gateway/ownership/invitation run passed **345 tests** against
synthetic PostgreSQL, before subsequent provider/route/derived additions. Further
focused tests qualify those additions. The latest combined gateway, invitation,
consent, application writer and relay run passed **465 tests**, with one legacy
database-dependent transaction test skipped because its separate fixture was not
enabled. These component tests are not complete release qualification.

Production reader provisioning currently fails closed on inherited `PUBLIC`
privileges: the installed database exposes sequence privileges on
`paradedb._typmod_cache_id_seq` and `topology.topology_id_seq`. The stock image also
contains additional extension objects. Review effective grants and preserve
necessary existing service access before provisioning; do not weaken the reader
checks or silently revoke unrelated grants. Tests use clean disposable databases.

## Provider and full CLI work

The staged Anthropic service binds immutable server model/client profiles,
reserves pessimistic spend before dispatch, prevents duplicate request dispatch,
and validates authorization during streaming. The HTTP adapter fixes the
upstream URL/headers, disables redirects/environment proxies/retries, bounds and
validates SSE, and settles only trusted terminal usage. Interrupted/unknown usage
keeps the conservative charge. A separate Gemini text/function normalizer and
HTTP/SSE transport are staged; the combined provider suite passed **201 tests**.
An actual Pi Google request was subsequently captured against a loopback mocked
upstream and passed the normalizer regression. Inline images, OpenRouter and
additional advertised model profiles still require qualification.

Real Pi 0.84.4 and native Claude 2.1.272 executed shell/Python and produced `42`
through the normalizer, run service and HTTP/SSE adapter with a mocked upstream.
Both CLIs subsequently ran inside the inner jailed Firecracker VM with no NIC,
fresh homes and an immutable public runtime image. Requests and streamed replies
traversed vsock and the trusted gateway. Each uploaded the actual two-byte `42`
artifact, whose digest was verified outside the guest. No host filesystem mount,
mailbox data, platform credentials or billed model calls were used. Versioned compatibility profiles remove known cache,
metadata and schema-identifier hints; model and effort remain server-controlled.
No live model quality or complete workflow compatibility is claimed.

## Worker isolation

Actual nested Firecracker tests passed shell execution, readonly root, no NIC,
jailed UID 65534, seccomp, PID/network namespaces, cgroup memory/swap/CPU/process
limits, independent lease expiry after controller death, explicit cancellation
and supervisor-crash cleanup. `gateway/worker.py` adds durable lifecycle bindings,
quotas and reconciliation. The backend remains explicitly restricted to the clean
synthetic worker. A bounded fixed-loopback vsock relay has ten socket regressions.

This is not a qualified production worker service. Runtime image/relay integration,
host hardening, credential handoff, writable workspace restoration, resource abuse,
orphan reconciliation and load/restart checks still need complete qualification.
The host process limit limits VMM threads; guest process exhaustion is contained by
the VM's memory/CPU bounds and needs its own runtime limits.

## Invitations and Gmail consent

New unmounted auth routes integrate signed broker identity, invited-user admission,
trusted provisioning, opaque sessions, separately bound Gmail consent, status,
disconnect and logout. Separate durable invitation and credential generations
prevent revocation/reconnect from reviving old sessions or credentials. Existing
owner import requires an explicit old database ID and verified Google subject;
there is no email-based ownership reassignment.

A separate broker candidate at
`/home/ssilver/development/gmail-search/worktrees/broker-consent-20260915`
(`/tmp/gms-broker-consent` remains an alias) provides account-bound
Gmail consent and token access. It checks Google issuer/audience/nonce/subject/email
before encrypted token storage, uses one-use PKCE state, and fences refresh and
revocation races. Its initial **26 tests** passed and TypeScript built; the patch
applied cleanly in dry-run against the original dirty broker checkout.

External prerequisites: a separate Google Web OAuth client allowing
`https://auth.oursilverfamily.com/v1/gmail/callback`, six scoped broker secrets,
reviewed new-function/hosting deployment, and app callback
`https://gms.oursilverfamily.com/api/auth/gmail-callback`. The existing broker and
WezTerm flows remain live unchanged. Two actual accounts must complete browser
sign-in/consent before multi-user release qualification.

New auth routes authenticate before bounded body reads and limit consent attempts
independently of pending-state deletion. Broker requests use fixed origins and
explicit headers with bounded responses. The focused auth suite passed **118 tests**.

A separate owner-bound application writer and administrator admission provisioner
are staged (**12 synthetic PostgreSQL tests**). Their grants cover five conversation,
event and cost tables only. They require reviewed RLS prerequisites and an atomic
credential-vault installation; ingestion credentials remain separate work. See
`docs/application-writer-staging.md`.

## Attachment parser boundary

Existing attachment extraction, PDF previews, Office conversion, image loading and
some URL content parsing still execute on the mail host. Thread timeouts do not
terminate native parsers. Parser-only microVM jobs passed seven synthetic PDF/image/ZIP cases and an
independent orphan-watchdog check, with **37 focused tests**. A new owner-scoped
database-to-file loader and optional capability-authenticated attachment route
are connected with a synthetic parser interface. Cancellation-aware real backend
handles passed independent review and actual nested-VM qualification; see `docs/gateway-integration.md`. Office, HTML/browser parsing and every remaining host
parser caller must be covered before multi-user release.

## Integration resumed

Typed text-thread retrieval now connects run capabilities to the existing
restricted database reader and optional worker HTTP route. Two-owner collision,
pagination, authorization and HTTP-disconnect checks passed with existing SQL
HTTP regressions (**18 tests**). The exact relay route was separately verified.
Attachment locator/loader checks passed **14 tests**; the composed attachment
HTTP path passed **2 tests** using a synthetic parser job. The new loader needs
no `raw_path` grant and never falls back to shared legacy storage.

The real parser backend passed cancellation during launch/input handoff, teardown
retry/idempotence and orphan-watchdog qualification. Its rebuilt image includes
the PDF truncation fix. Controller admission and repeated-cancellation cleanup
are tested; the authenticated SSH controller/manager path now passes synthetic real-VM
qualification, including revocation and manager-loss cleanup. Anthropic/Gemini
HTTP streaming is mounted optionally in the worker gateway, and DB-side bounded
large-message paging is implemented and reviewed. Combined gateway, attachment
and relay tests: **548 passed, 12 skipped** with the private synthetic PostgreSQL
fixture. Production worker/application integration remains open.

See `docs/gateway-integration.md` for exact scope and remaining integration gates.

## Remaining release work

1. Qualify and deploy compatible ownership migrations/callers with backup recovery
   and measured capacity; finish effective privilege remediation and separate
   application/ingestion writer credentials.
2. Complete inner-guest runtimes, production controller/relay, advertised provider
   transports, tools/retrieval/attachments, workspace/transcript/artifact handling,
   event replay, battle integration and all cancellation/restart paths.
3. Wire invited auth/provisioning and the new broker into the candidate public app
   and UI; complete disconnect, deletion and retention policy implementation.
4. Independently review and adversarially exercise the complete two-owner system,
   including malicious mail, quotas, worker loss and actual browser flows.
5. Deploy only the qualified candidate. Rollback must disable invitee ingress; an
   owner-only release must never become a fallback that exposes the owner's mail.

## Safe tests

Use the isolated candidate's source with the original Python interpreter directly;
never resync its symlinked virtualenv. `GMS_GATEWAY_TEST_DSN` is for disposable
synthetic databases; `GMS_TEST_PG_DSN` explicitly enables older schema-isolated DB
fixtures on a disposable instance. Tests no longer implicitly connect to the
ordinary development database. See each fixture's requirements. Do not place
production credentials or mailbox contents in test output or guest environments.

## Latest integration checks

- Actual SSH parser-manager qualification now includes capability revocation,
  manager freeze, manager restart, durable tombstones, replay rejection and
  restricted SSH negative checks. Evidence is recorded in
  `deploy/public/worker/attachment-rpc-qualification.json`; that synthetic outer
  was shut down cleanly after the parser qualification.
- The fixed worker relay supports Google capability headers and the exact Gemini
  streaming route, artifact `201 Created`, and event append. Reviewed event,
  relay and artifact HTTP checks passed **29 tests**.
- Guest schema/SQL/thread/artifact client plus fixed-path stdin wrapper passed
  **42 tests**, independently rerun. This is a four-tool transport core; typed
  MCP and actual CLI workflow integration are still being qualified.
- Candidate browser result routes enforce invited sessions and owner/conversation
  checks, including revocation after storage reads. Combined invited auth,
  consent and result tests passed **40 tests**. These routes are not yet mounted
  in the public application and have passed independent ownership/privacy review.
- The candidate frontend production build passed after passive Markdown rendering
  was shared by answers and reasoning. No build output has been deployed.

All counts are focused and may overlap earlier results. Shared database
migration/recovery, remaining ranked search/provider/full-tool parity, production
worker/application integration, and actual two-account Google flows still block
multi-user release.

After resuming interrupted reviews, the MCP concurrency/framing/correlation
findings and the qualification controller's teardown finding were fixed and
independently verified. The latest focused integration suite passed **69 tests**;
this overlaps earlier counts. Actual isolated Pi/Claude synthetic mailbox-to-CSV
workflows passed with separate owners and real artifact storage. Production and
typed-MCP-in-VM qualification remain open.

The resumed combined gateway, guest protocol, invited auth/results/consent and
relay suite passed **551 tests** with the private synthetic PostgreSQL fixture.
Native Claude also completed the real typed MCP workflow in a separate pinned
VM: schema, SQL, thread, Bash CSV creation, then MCP artifact publication.
Independent review found no remaining issues in that profile/controller proof;
see `deploy/public/worker/GUEST_MAIL_MCP_QUALIFICATION.md`.

A separate ranked-search probe confirms RLS row isolation and owner-filtered
TopK recall on the pinned ParadeDB version, but global BM25 scores/rankings
change with another owner's corpus. Tenant-local scoring is therefore an open
architecture/quality gate; passing RLS is not treated as proof of ranking
independence. The user has been asked about the migration-versus-ranking tradeoff.

## Actual Pi typed MCP qualification

The Pi runtime now has an independently reviewed synthetic VM proof for four
typed mail tools plus Bash used to create a CSV from retrieved data. Every one
of six mocked-provider requests exposed only the fixed five-tool set. The
owner-scoped gateway produced Alice's 68-byte CSV and denied Bob access. VM,
capability and temporary service cleanup completed. Independent review reran
86 tests. Evidence: `deploy/public/worker/GUEST_PI_MAIL_MCP_QUALIFICATION.md`.

Review found that the future builder trusted dependency metadata without
checking all executable input bytes. The builder now verifies a committed tree
manifest derived from the already-pinned image, requires trusted root-owned
staging, rejects unsafe symlinks, and rechecks its private packing snapshot.
All 18 new checks passed independent root verification. The original image pin
and historical run evidence remain unchanged; no replacement image was built.
See `pi-mcp-runtime-builder-amendment.json` alongside the qualification report.
This remains synthetic runtime qualification, not a production rollout.

## Partition rollout boundary

The current numeric-key migration profile starts after the existing owner-key
prerequisite; that prerequisite still has production-size rewrite/index costs.
The attach step preserves existing mail heaps, TOAST storage and BM25 files in
synthetic qualification. A separately qualified text-key mechanism may remove
the numeric-key prerequisite, but its complete source migration and caller
profile are not implemented. See `docs/gateway-text-key-partition-probe.md`.

Before a live cutover, qualify production peak space/WAL, backup restoration,
and the compatible caller release. Complete the parent plan's gateway search,
worker/controller, broker/consent and two-account browser checks. The current
public release remains owner-only/retrieval-only, and no live services changed.

## Search and facts gateway implementation

The candidate now has an independently reviewed fixed owner reader (32 tests),
strict native owner-index registry (38 tests), bounded vector scorer, pure hybrid
ranking and run search service. Forty-one legacy search/filter/parser checks
passed against the synthetic database. Three actual partitioned PostgreSQL plus
native ScaNN tests cover both owners, colliding mail IDs, exclusion of foreign
vector candidates before scoring, and shared SQL/search admission.

The facts service passed independent review and 35 tests, including both actual
owner readers. It streams bounded vector pages and reports incomplete coverage;
its 128 MiB transfer cap can truncate a production-sized 3072-dimensional corpus.
The low-dimensional 8,001-fact test does not qualify that production capacity.
See [facts service](gateway-facts-service.md).

Service review reproduced revocation during final watcher cleanup. Facts and
search now drain all owned work before their final authorization check. The
original search reproduction now raises AccessDenied; the focused search and
HTTP suite passed 39 tests. HTTP response cleanup revealed an analogous gap;
four new facts/search revocation/deadline regressions failed before the transport
fix and pass afterward. Independent transport review passed 48 tests with no skips. SQL and thread
retrieval now use the same cleanup ordering; their lifecycle and HTTP checks
passed an independent 46-test run, including real PostgreSQL cancellation.

Optional private routes `/v1/search` and `/v1/find-facts` are available only when
explicitly injected. Guest MCP/CLI now register `search_emails_batch` and
`find_facts` alongside the prior four tools, using the retrieval capability and
fixed relay paths. The combined facts HTTP, guest core, MCP, CLI and relay suite
passed 101 tests. These additions are not in the historical pinned VM images;
new image builds and actual runtime qualification remain required.

The stable embedding adapter passed independent lifecycle review: 61 combined
embedding/provider tests, including transport-close cancellation and final
revocation races. Existing preview-model indexes are not relabeled or assumed
compatible. The stable-model reranker and shared provider lifecycle extraction passed
independent review with 106 combined tests; actual provider/model compatibility and cost qualification
remain outstanding. No live provider calls were made for these tests.

Full tool parity, runtime/controller composition, production migration and
recovery, broker/consent deployment and two-account browser checks remain open.
The live site is unchanged; these results do not qualify a public rollout.


## Combined search and attachment follow-up

After the shared provider/lifecycle and guest additions, the combined search,
facts, provider, native index, fixed reader, legacy ranking and guest suite passed
**431 tests in 67.92 seconds**, with no skips and both synthetic database fixtures
configured. This count overlaps the focused results above; it is not additive.

Structured metadata filtering now uses one fixed, owner-qualified analytical
query without an embedding or index. Its service, HTTP route, guest/MCP/CLI and
relay path passed independent review with **151 tests**, including actual
owner-collision and matching-message ordering cases. Seven mail tools are now
registered in guest source; historical VM pins still contain four.

The new OwnerAttachmentReader passed independent review and **53 combined
reader/retrieval tests**. It supplies bounded metadata, stored-text pages and
thread inventories; NULL/empty text does not imply extraction completeness.
The opaque source now offers distinct raw reads for generic/empty files while
retaining parser MIME/nonempty restrictions. Real owner locator and raw-source
checks pass, including declared-size binding, path/link rejection and cleanup.
Neither source component installs a public download or full attachment tool.
Run-bound JSON reads, binary guest downloads and parser paging remain staged.

A concise list of the remaining gates is maintained in
[the full-agent release checklist](full-agent-release-checklist.md).
