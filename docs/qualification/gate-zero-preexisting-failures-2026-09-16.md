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
