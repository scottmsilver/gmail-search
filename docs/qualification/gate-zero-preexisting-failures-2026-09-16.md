# Gate 0: the failures were the test cluster, not the code

2026-09-16. **This document previously claimed 25 pre-existing gateway failures
and a hanging test. That was wrong.** The whole suite passes. The failures were
caused by the disposable PostgreSQL container I built using password
authentication; the fixtures require `trust`.

## The corrected result

Per-file sweep, patched engine, `trust` auth, port 55440:

**1189 passed, 0 failed, 0 errors.** No hangs.

| Cluster | Result |
| --- | --- |
| Password auth (what I measured first) | 1156 passed, 25 failed, 0 errors, one hang |
| **Trust auth (what the fixtures require)** | **1189 passed, 0 failed, 0 errors** |

## Second correction, later on 2026-09-16: that figure does not reproduce

The line above is left standing because it is what was recorded, but it is not
the state of the suite. A bounded per-file sweep on the same patched cluster,
same trust auth, same port, over all 218 test files:

**2912 passed, 24 failed, 234 skipped.**

The pass count alone shows the earlier run covered far less than the whole
suite — 1189 against 2912 is not a difference two fixes could account for. I did
not record what that sweep actually ran, so I cannot say what it missed; the
number should be treated as unexplained rather than as a baseline.

Of the 24 failures, **two are real defects in committed code**, the rest were
environment or a test pinning behaviour the patch has since removed.

### Two real defects, both from `0a7e3c9`

Both are tests committed for code that does not exist. Both files are unmodified
in the worktree, so they fail from HEAD.

| | |
| --- | --- |
| `tests/test_browser_mail.py` | Passes `attachment_source=` to `BrowserMail.__init__`, which accepts only `gateway` and `attachment_reader`. 1 test. |
| `tests/test_invited_mail_routes.py` | Exercises `/api/attachment/{id}`, `/api/attachment/{id}/meta` and their rejection cases. `create_mail_router` defines only `/api/thread_lookup` and `/api/thread/{thread_id}`; `attachment_metadata` and `attachment_download` appear nowhere in `src/gmail_search/auth/`. 9 tests. |

The invited attachment read path is therefore **specified and tested but not
implemented**. That is a Track C gap, not a migration blocker, but it is a real
one and it was not visible in the earlier figure.

### The rest

| Failure | Cause | Resolution |
| --- | --- | --- |
| `test_attachment_worker_manager` (1) | `mkdir(mode=0o755)` is masked by the ambient umask. Under `umask 0077` the "unsafe" directory came out `0700` — safe — so the expected refusal never fired. | Test fixed to `chmod` explicitly; passes under umask 022, 077 and 002. |
| `test_full_agent_runtime` (2) | `socket.AF_VSOCK` is absent from uv's portable CPython 3.12.9 and present in the system Python 3.12.3. | Open. The test cannot pass in the interpreter the project mandates. |
| `test_bm25_deleted_statistics` (6) | **The test pinned the pg_search bug as the expected result** — it asserted `{'sqlstate':'XX000','error':'assertion failed: item_pointer_is_valid(ctid)'}` for the generic plan. On the patched engine the generic plan returns rows, so the assertion no longer holds. | Repinned to assert rows and to name the ctid patch if the error ever returns. |
| `test_bm25_deleted_statistics` (3 more) | Only under the sweep: 9 failed concurrently, 6 in isolation. The sweep drops every `gms_%` database between files, including ones this probe had just created. | Not a defect. Never run the sweep alongside anything on the same cluster. |
| `test_partitioned_admission` (2) | Mine — a call site missed when `AdmissionProvisioner` gained a required `partition_profile`. | Fixed. |

That `test_bm25_deleted_statistics` row is worth dwelling on. A test written to
reproduce the upstream defect now fails *because the defect is gone* — an
independent confirmation of the patch from a test that predates it. Repinning it
to assert the absence turns it into the regression guard the patched image needs:
if a rebuild ever pulls the floating `paradedb/paradedb:latest-pg16` tag, or the
engine is rolled back to stock 0.23.0, that test fails and says why.

### What the earlier sweep also missed

Five files skip entirely without `GMS_GATEWAY_TEST_DSN`, which the sweep does not
set. Run with it pointed at the same cluster they are green
(`test_application_writer` 22, `test_browser_conversation_persistence` 11,
`test_gateway_database_integration` 5, `test_owner_derived_data` 5,
`test_owner_key_schema_migration` 43) — but only after fixing three monkeypatched
`provision_owner_partitions` stubs whose signatures had drifted from the real one.
Set both DSN variables when you want the real baseline.

## Root cause

`tests/test_gateway_database_integration.py`:

```python
def reader_dsn(dsn, owner):
    cfg = conninfo_to_dict(dsn)
    cfg['user'] = reader_role(owner)
    cfg.pop('password', None)      # ← deliberate
    return make_conninfo(**cfg)
```

The fixture strips the password on purpose: the gateway's per-owner reader roles
are provisioned without one, so the disposable cluster must accept them without
a password. On a password-auth cluster every reader connection fails with

```
OperationalError: connection failed: fe_sendauth: no password supplied
```

which `gateway/database.py` deliberately masks as
`RuntimeError('Analytical query could not be completed')` so PostgreSQL detail
cannot leak. That mask is correct for production and is exactly what made this
hard to see: every symptom looked like a gateway defect.

The "hang" has the same cause.
`test_inflight_owner_revocation_closes_connection` takes an `ACCESS EXCLUSIVE`
lock and then waits for the gateway query to appear as lock-blocked:

```python
while not api.active_queries:
    await asyncio.sleep(.01)
```

The connection died on authentication, so the query never registered, and that
wait has no bound. Not a deadlock — a failed precondition plus an unbounded
wait. Worth bounding regardless, since it turns any early failure into a hang.

## Run the cluster this way

```bash
docker run -d --name gms-parade-patched \
  -e POSTGRES_PASSWORD=<pw> -e POSTGRES_HOST_AUTH_METHOD=trust \
  -e RUST_BACKTRACE=full --shm-size=1g -p 127.0.0.1:55440:5432 \
  paradedb-patched:0.23.0-ctidfix \
  -c shared_buffers=1GB -c effective_cache_size=4GB -c work_mem=32MB \
  -c maintenance_work_mem=512MB -c max_parallel_workers=8 \
  -c max_parallel_workers_per_gather=2 -c max_parallel_maintenance_workers=2 \
  -c max_worker_processes=8 -c max_connections=50 -c max_wal_size=2GB
```

Three properties the fixtures require, each of which cost time to discover:

- **`POSTGRES_HOST_AUTH_METHOD=trust`** — or every gateway reader fails to
  connect and the failures look like code defects.
- **Port 55440** — several fixtures assert the exact host, port, dbname and user
  and error out otherwise.
- **`--shm-size=1g`** — Docker's 64 MB default surfaces as a misleading
  `DiskFull` under parallel work, with plenty of real disk free.

## How the wrong conclusion survived a control

I A/B-tested the failures against the original unpatched image and got
byte-identical results, and treated that as proof the code was at fault. **Both
containers were mine, and both had the same wrong auth.** The comparison could
only ever have exonerated the patch — it could not distinguish "codex's bug"
from "my cluster", because the variable I should have been testing was held
fixed in both arms.

The general form: *a control that varies the thing you suspect, while holding
fixed the thing you never thought to suspect, proves much less than it appears
to.* When a whole subsystem fails at once, suspect the environment before the
code.

## Also verified on this cluster

The ctid reproducer, run alone: **13 passed**. An earlier run of it reported
3 failed / 3 errors — that measurement was contaminated by the per-file sweep
running concurrently, which drops every `gms_%` database between files,
including the ones the reproducer had just created. Never run the sweep
alongside anything that uses the same cluster.
