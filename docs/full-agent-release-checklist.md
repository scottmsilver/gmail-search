# Full public Gmail Search release gates

Updated 2026-09-15. This is a candidate checklist, not a deployment record.
Live remains owner-only/retrieval-only. No production DDL or mailbox copies were
performed for the candidate work below. Detailed evidence is in
[implementation status](shared-database-implementation-status.md).

## Data isolation and migration

- Reviewed: shared PostgreSQL with LIST(user_id) partitions for messages,
  attachments and propositions, owner-specific native BM25 statistics, composite
  identifiers, fixed reader roles/RLS and admission provisioning.
- Reviewed: bounded analytical SQL compiler/executor and separate fixed search
  reader; shared per-process SQL/search admission and cancellation cleanup.
- Qualified synthetically: owner-colliding identifiers, foreign search candidates,
  foreign-corpus ranking invariance, rollback and dependency rejection.
- Pending production gate: choose and qualify the compatible migration/caller
  profile, measure peak space/WAL, restore a backup, and rehearse rollback.
  The existing numeric-key prerequisite still has rewrite/index costs. The
  cheaper TEXT profile now has reviewed partition provisioning and fixed-reader
  qualification (162 compatibility tests; 48 independent profile/service tests,
  overlapping). The single-owner direct legacy migration has been implemented
  and independently reviewed in disposable databases; it is ineligible for the
  live mixed-owner heaps. See [TEXT profile plan](superpowers/plans/2026-09-15-text-key-owner-partitions.md)
  and [matching caller inventory](text-partition-caller-inventory.md).
- Live preflight found three owners in the existing heaps. A single-owner attach
  is ineligible. The [mixed-owner plan](superpowers/plans/2026-09-15-mixed-owner-partitions.md)
  preserves dominant heaps, relocates minority rows and requires retained BM25
  rebuilds: deletion/vacuum left foreign ranking influence in the synthetic
  history probe. [Live metadata and counts](live-partition-preflight-2026-09-15.md).
- Synthetic phase-one mixed-owner redistribution is implemented and independently
  reviewed: 56 tests passed, including a real process exit after PostgreSQL commit
  and restart without copying rows again. Phase two now exists as a separate
  `publish()` call — VACUUM the retained leaves, rebuild their BM25 indexes,
  qualify every owner — and passes synthetically on the patched engine; release
  readiness still requires the live rehearsal and cutover gates in
  [the migration plan](superpowers/plans/2026-09-15-migration-after-ctid-fix.md).
  (Before 2026-09-15 the controller stopped at INDEX_PENDING.)
  All writes are restricted
  to disposable databases. Production migration remains blocked.
  DELETE/ATTACH/REINDEX in one transaction
  failed score isolation; a separately committed rebuild restored clean scores
  in the targeted probe. Stronger actual-reader testing then reproduced a native
  pg_search assertion after retained-table REINDEX and subsequent writes/VACUUM.
  Custom/unprepared planning did not resolve every case. Actual retained TEXT
  heaps also failed, including after replacing BM25 indexes with new OIDs.
  See the [retained-reader blocker](qualification/retained-reader-maintenance-blocker.md).
- Reviewed controller [maintenance gate](gateway-maintenance.md): durable closure,
  epoch-bound authority, old-token revocation, bounded publication locking and
  crash/retry checks. Root independently ran 80 gate/capability/worker tests.
  The first synthetic PostgreSQL phase witness is now tested. Phase-two
  qualification, production orchestration, credential publication wiring and
  actual process/session fencing remain required.
- Capacity limit: measured empty partitions/indexes add about 8.4 MiB per owner
  in the small test. This is neither a mailbox clone nor a large-scale estimate.

## Model-facing tool coverage

| Capability | Candidate implementation | Runtime release gate |
| --- | --- | --- |
| SQL and schema | Reviewed service, HTTP, guest MCP/CLI | Existing synthetic VM proof; requalify new full image |
| Thread text | Reviewed bounded body/message paging | Richer formats, message selection and runtime manifest qualification |
| Hybrid search | Reviewed native index, fixed reader, service, HTTP, guest | Actual model/index compatibility and rebuilt runtime |
| Facts | Reviewed bounded service, HTTP, guest | Production-dimension corpus capacity and rebuilt runtime |
| Structured mail filters | Reviewed fixed SQL, service, HTTP, guest | Rebuilt runtime |
| Attachment metadata/stored text | Reviewed owner reader, run service, HTTP and v2 guest mapping | Runtime qualification |
| Raw attachments | Reviewed opaque source, binary service, relay, private downloader and persistent v3 MCP integration | New image, controller and workspace lifecycle qualification |
| Rendered attachments | Qualified bounded parser VM for listed formats | Binary page transfer, paging/provenance and tool integration |
| Shell/files/artifacts | Synthetic Pi/Claude VM proofs | Full-tool image qualified with mock provider; workspace restore/export and complete browser wiring remain |

The guest source explicitly selects four legacy tools, eight v2 tools, or eight
v3 MCP tools with raw attachment mode. V3 is unavailable in the one-shot CLI.
Historical VM images expose four
mail tools plus Bash; their original pins are unchanged. A tool in source is not
a qualified installed runtime feature. Attachment parser support does not yet
cover all legacy formats or OCR.

## Provider, worker and application composition

- Reviewed candidate embedding/reranking adapters reserve budget before calls,
  share provider admission, retain capacity through teardown, and reject stale
  authorization before publication. Stable model profiles are explicit.
- Pending: actual provider request/response, cost and quality qualification.
  Existing preview vectors are not relabeled or assumed compatible with stable
  embeddings. Reranking's conservative reservation policy needs budget-fit review.
- Full-tool Pi image and actual HTTP/SSH-manager/VM/PG run plus in-flight Stop qualified with mock providers.
  Large-result Pi replacement image and conservative clock margin also qualified.
  Pending: full-tool Claude image, all selected model transports,
  dedicated worker deployment, capacity under concurrent users and restart tests.
- Implemented candidate browser/controller integration: invited start/replay/Stop,
  owner-scoped opaque artifact downloads, and rich shared-PostgreSQL transcript
  persistence. Browser auth/Gmail status response contracts, consent POST and its
  distinct callback proxy now match the invited backend. Independent review found
  and corrected lost-ack transcript overwrites and unpublished-worker cleanup.
  See [browser app integration](gateway-browser-app-integration.md).
- Implemented: dedicated invited API factory and owner-scoped conversation CRUD.
  Implemented: optional fixed owner-reader citation routes and partial-view notices.
  Pending: remaining browser search/attachment routes,
  persistent worker deployment, workspace restoration/export, battle branches,
  automatic browser reconnect and full running-app acceptance.

## Sign-in and rollout

- Reviewed candidate invitation identity, separate Gmail consent and owner-bound
  sessions/credentials. The current login navigation fix has separate live HTTP
  evidence; actual Google browser sign-in still needs verification.
- Pending: dedicated Google client/broker secrets and broker/app composition,
  followed by two invited-account browser sign-in and Gmail-consent checks.
- Pending: final independent review, full acceptance suite and cutover rehearsal.
  Rollback must disable invitee ingress rather than send invitees to the
  owner-only application. Deploy only after the complete profile is qualified.
