# Owner partition setup overhead

Measured 2026-09-15 on the dedicated synthetic PostgreSQL 16.13 / pg_search
0.23.0 container (2 CPUs, 2 GiB). No production data or services were used.
The generated database was removed after measurement.

The fixture in `tests/test_gateway_partition_provision.py` created the three
canonical partitioned parents and BM25 indexes. For owners `scale_1` through
`scale_25`, the probe inserted a synthetic users row, timed
`provision_owner_partitions`, and sampled catalog size and a parent BM25
`EXPLAIN (ANALYZE,FORMAT JSON)` query with an explicit owner predicate.
All partitions were empty. Timings are single observations, not percentiles.

| Owners | Physical mail partitions | New owner provisioning | Query planning | Database bytes | FK constraints |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3 | 0.0682 s | 1.284 ms | 17,341,463 | 10 |
| 5 | 15 | 0.0626 s | 0.670 ms | 52,517,911 | 34 |
| 10 | 30 | 0.0962 s | 0.822 ms | 96,427,031 | 64 |
| 25 | 75 | 0.1736 s | 0.754 ms | 228,236,311 | 154 |

The additional 24 owners used 210,894,848 bytes, about **8.4 MiB per empty owner**
in this fixture. These bytes include empty native indexes and catalog/storage
allocation. There are no mailbox copies. Data, embedding/index growth, backups,
WAL and migration headroom are additional costs.

This qualifies only small-scale setup overhead on the tested environment. It is
not a claim about thousands of users, concurrent ingestion/query latency,
production maintenance windows, or physical-storage billing. The current
provisioner serializes catalog changes and scans the existing partition catalog;
large-account-count deployment needs separate qualification.

Correctness under real direct owner logins, including colliding numeric IDs,
foreign corpus insert/update/delete, custom/generic prepared plans, counts,
LIMIT, index pruning and denied direct child access, is tested separately in
`tests/test_owner_partition_ranking.py` for messages, attachments and propositions.
