# TEXT-key Owner Partitions Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement the approved first slice task by task. Parent coordinates independent review. No commits or production operations are part of this task.

**Goal:** Qualify explicitly selected TEXT-key owner partitions and fixed internal search readers without adding a numeric message key.

**Architecture:** A closed immutable schema profile supplies canonical message keys/index fields. Existing numeric APIs remain the default. Administrator and runtime catalog audits verify the selected profile; credentials and role identities bind it explicitly. No model input, capability claim, schema autodetection, or SQL-error fallback selects a profile.

**Tech Stack:** Python, psycopg 3, PostgreSQL 16.13, pg_search 0.23.0, pytest; disposable synthetic ParadeDB on 127.0.0.1:55440 only.

## Approved first slice

Create `src/gmail_search/gateway/partition_profiles.py`; modify only `partitions.py`, `search_reader.py`, `provision_search_reader.py`, `search_queries.py`, and the approved narrow qualification ordering in `search_service.py` in that directory. Add `tests/test_gateway_partition_profiles.py` with a complete canonical synthetic TEXT fixture and focused profile tests. Existing numeric qualification records and schema installers remain unchanged.

The numeric profile retains messages PK `(user_id,id)`, UNIQUE `(user_id,search_id)`, and BM25 `(search_id,id,subject,body_text,from_addr,to_addr)` keyed by `search_id`. TEXT requires no `search_id` column, PK `(user_id,id)`, and BM25 `(id,subject,body_text,from_addr,to_addr)` keyed by `id`. Attachments and propositions retain their reviewed owner-qualified numeric keys and canonical indexes in both profiles. Existing private partition names/comments continue to bind owner/table; actual keys, column types and index definitions prove the profile.

### 1. Closed profiles and partition administration

- [x] Write failing tests for immutable closed choices, invalid profile rejection, TEXT three-parent provisioning/idempotence/read-only inspection/empty cleanup, and wrong-profile refusal.
- [x] Implement fixed profile definitions and thread an additive keyword `profile` through partition verification, provisioning, inspection and empty cleanup. Preserve numeric defaults and internal numeric compatibility constants used by the existing migration.
- [x] Require message key types and absence of `search_id` under TEXT before partition mutation; retain all existing ACL, bound, collation, lock and ancestry checks.

### 2. Bound reader credentials and typed lexical queries

- [x] Write failing tests for profile-bound credentials and runtime mismatch before admission/connect, exact column grants, and TEXT message lexical IDs.
- [x] Add immutable `schema_profile` fields to `SearchCredential` and `SearchProfile`, default numeric. Keep existing numeric role names/comments unchanged; use a distinct TEXT role name and binding. Select fixed message columns/index audit from the profile. TEXT candidate restrictions on the indexed key also require exactly `paradedb.terms_with_operator(paradedb.fieldname,text,anyelement,boolean)`, an invoker C query-construction helper. Audit its extension membership/library/symbol/signature/security attributes and grant it only to TEXT readers; numeric retains four functions.
- [x] Select only the reviewed message BM25 key. Keep immutable SQL owner literals and bound search inputs. Return TEXT-specific message lexical rows while numeric message/attachment/fact rows retain strict numeric IDs.

### 3. Actual synthetic qualification

- [x] Build full canonical TEXT messages/attachments/propositions parents plus fixed reader relations, without invoking or modifying the numeric migration or global schema installer.
- [x] Verify colliding owners, all three lexical branches, direct-child denial, immutable RLS despite owner GUC changes, wrong profile refusal, and custom/generic prepared plans using the actual fixed owner-literal query shape.
- [x] Insert/update/delete only synthetic foreign-owner rows and prove unchanged owner scores; verify owner cleanup/reprovision and rollback/drift refusal.
- [x] Run new tests RED before implementation, then GREEN. Run existing numeric partition, reader/query, migration and search integration tests to preserve compatibility. Use `GMS_TEST_PG_DSN='postgresql://postgres:synthetic-owner-test@127.0.0.1:55440/postgres' PYTHONPATH=src /home/ssilver/development/gmail-search/.venv/bin/python -m pytest -q ...` with explicit local-fixture escalation.
- [x] Obtain independent review and record measured results here. Do not claim production release qualification.

## Later gates: explicitly outside this slice

1. **Direct legacy migration.** Separately implement a strict legacy TEXT-source to owner-partitioned TEXT-target migration. Fold owner-qualified keys/known incoming FKs and derived-table transitions into one transaction; preserve existing heap, TOAST and BM25 relation identities; refuse unknown dependencies/defaults/policies/sequences. Reuse catalog/lock/rollback mechanisms but do not run `migrate_owner_keys.py` (its identity addition and BM25 rebuild defeat this cost objective). Qualify full canonical schema, external dependencies, serial advancement, attach failure rollback and preserved files on synthetic data before any proposed production DDL.
2. **Compatible caller/schema release.** Thread trusted profile selection through admission and installed schema configuration, `store/queries.py`, `propositions.py`, search engine callers and model-facing SQL examples in `server.py`, `agents/analyst.py`, `agents/mcp_tools_server.py`, and `store/db.py`. Replace column-existence/error fallback with fail-closed profile verification. Do not silently change the global `pg_schema.sql` profile or automatically migrate at application startup. Audit every actual composed provider/index path to ensure wrong-profile denial precedes expensive work.
3. **Cost and cutover qualification.** Measure new owner-qualified B-tree/FK validation costs, lock duration, WAL and rollback/restore requirements on representative synthetic scale. Drain writers/readers and approve a concrete restore/cutover plan before production work. Historical observed messages size was 17 GB plus 925 MB total message indexes; neither is a measured rewrite/WAL estimate. The tiny TEXT probe preserved heap/TOAST/BM25 identities in 0.0183 seconds with an 8192-byte heap and 3,022,848-byte BM25 main fork. It used only three message index fields and no full product migration. This first slice supplies full canonical reader/provisioning evidence; it does not establish production cost or zero internal writes.

## Verification results

Implementation completed in the candidate worktree. Parent independently reviewed and approved this slice; independent profile-denial regressions plus profile/service suites passed 48 tests in 11.34 seconds.

- Initial RED: missing `partition_profiles` module. Actual TEXT fixture then exposed `terms_with_operator` permission denial for `id=ANY(...)`; the approved TEXT-only exact helper grant resolved it. A separate RED test proved null-strictness drift was accepted, then the explicit catalog audit fixed it.
- Composed profile-ordering regression was RED because index acquisition happened before credential mismatch; moving the already-existing preliminary context snapshot before acquisition fixed it. Model/dimension checks remain before embedding and no reader snapshot remains open during provider calls. Pending-index tests now expect this preliminary qualification.
- Focused TEXT/profile plus service suite: **43 passed in 9.40 seconds** before adding the final two restricted churn variants.
- Final combined command (fixture environment as above): `python -m pytest -q tests/test_gateway_partition_profiles.py tests/test_gateway_partitions.py tests/test_gateway_partition_provision.py tests/test_gateway_search_reader.py tests/test_gateway_search_queries.py tests/test_gateway_search_service.py tests/test_gateway_search_integration.py tests/test_owner_partitions.py tests/test_owner_partition_lifecycle.py tests/test_owner_partition_ranking.py`: **162 passed in 55.57 seconds**, no skips. This includes 17 new profile tests, all four custom/generic × unrestricted/restricted foreign-churn combinations, exact helper grants, and real catalog drift before index/provider work.
- Ruff passed on the six changed gateway modules plus new profile module and the changed/new tests.

These are synthetic compatibility/security results, not direct legacy migration or production-cost measurements. No provider, production query, global schema switch or historical numeric qualification record was changed.
