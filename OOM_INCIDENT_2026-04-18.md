# gmail-search memory leak — OOM incident 2026-04-18

## TL;DR

At **05:02:40 PDT on 2026-04-18**, a running `gmail-search` process consumed **~35 GiB of anonymous RSS** and triggered a global kernel OOM on a 62 GiB / 0-swap machine. A second `gmail-search` process was also resident at ~4 GiB. The OOM killed the main offender, systemd tore down the whole `vncserver.service` cgroup as a side effect, and the entire desktop session (i3 + wezterm + tmux server + all child shells) died.

**You are being asked to find the leak in this codebase.** The host-level contributing factors (no swap, `OOMPolicy=stop`) have already been mitigated separately; that's not what needs investigating here.

## Kernel OOM evidence (journalctl)

```
Apr 18 05:02:40 sukkot kernel: avahi-daemon invoked oom-killer: gfp_mask=0x140cca(...), order=0
Apr 18 05:02:40 sukkot kernel: Free swap  = 0kB
Apr 18 05:02:40 sukkot kernel: Total swap = 0kB

Apr 18 05:02:40 sukkot kernel: [    2806]  1000  2806     4863    2326 ... vncserver
Apr 18 05:02:40 sukkot kernel: [  276005]  1000 276005  9616916 8781872 ... gmail-search
Apr 18 05:02:40 sukkot kernel: [  360923]  1000 360923  1651680 1023391 ... gmail-search

Apr 18 05:02:40 sukkot kernel: oom-kill:constraint=CONSTRAINT_NONE,...,global_oom,
  task_memcg=/user.slice/user-1000.slice/user@1000.service/app.slice/vncserver.service,
  task=gmail-search,pid=276005,uid=1000

Apr 18 05:02:40 sukkot kernel: Out of memory: Killed process 276005 (gmail-search)
  total-vm:38467664kB, anon-rss:35126368kB, file-rss:1120kB, shmem-rss:0kB, UID:1000
  pgtables:69800kB oom_score_adj:200

Apr 18 05:02:43 sukkot systemd[2491]: vncserver.service: Consumed 2d 17h 10min 34s CPU time,
  53.4G memory peak, 0B memory swap peak.
```

Decoded numbers for the killed process:
- `total-vm` ≈ **38.5 GB** virtual
- `anon-rss` ≈ **35.1 GB** resident anonymous memory (almost all of it)
- `file-rss` ≈ 1 MB (negligible)
- `pgtables` ≈ 70 MB (consistent with ~35 GB mapped)
- `oom_score_adj=200` — inherited from the `vncserver.service` parent unit, not set by the app itself

The second process (`pid 360923`) was at ~4 GB RSS. The service cgroup's `memory peak` was **53.4 GB** over 2d 17h of CPU time, which is consistent with sustained growth rather than a single spike.

## How it was launched

**Not as the systemd user unit.** That unit (`~/.config/systemd/user/gmail-search.service`) had run cleanly 20:53:37 → 20:55:55 on 2026-04-17 and exited at 113.9 MB peak:

```
Apr 17 20:55:55 sukkot systemd[2491]: gmail-search.service: Consumed 2.378s CPU time,
  113.9M memory peak, 0B memory swap peak.
```

The two processes that blew up at 05:02:40 were running inside `vncserver.service`'s cgroup — i.e. launched by hand from a wezterm shell under the VNC :1 session, almost certainly with `gmail-search watch --interval 300` (that is the unit's `ExecStart` line). The 2d 17h cumulative CPU time on `vncserver.service` and the long-lived process suggest they had been running for days.

Invocation path found on disk: `/home/ssilver/anaconda3/bin/gmail-search`.

## What to investigate

Hypothesis framework — the leak is almost certainly in the long-running `watch` code path, not the one-shot CLI, because:

1. The systemd unit runs **the same command** (`watch --interval 300`) and exits at 113 MB after one cycle — so the first sync itself doesn't leak catastrophically.
2. The failure mode requires many cycles over many hours to reach 35 GB.
3. The 2 resident processes at OOM time (parent + one child?) are consistent with a fork or subprocess-leak pattern.

Likely suspects, ordered by how often they cause this shape of leak:

- **Accumulating state in the watch loop.** Lists/dicts/caches that grow every interval and never get evicted. Per-message objects held indirectly by a logger, metric, or diff structure. Check every container declared outside the per-iteration scope.
- **Gmail API pagination buffers.** If `watch` pulls history or re-syncs on each tick and stores all messages in memory for a diff, a mailbox of N messages × T ticks grows without bound.
- **Embedding / vector index held in RAM.** The project name implies search; indexing code that rebuilds or appends an in-process index on each watch cycle will balloon. Check for FAISS, numpy arrays, or any `dict[str, np.ndarray]` that grows.
- **Subprocess / file-descriptor churn producing zombie memory.** If each cycle forks helpers and the parent keeps references to their output buffers, `anon-rss` climbs.
- **`googleapiclient.discovery` clients re-created each cycle** without closing the underlying `httplib2`/`requests` session, holding onto response bodies.
- **Logger retaining records.** A `MemoryHandler` or a handler with unbounded buffering.

Starting points in the repo:
- `src/` — main package; find the function invoked by `gmail-search watch`.
- `data/watch.log` — the app-level log file the process was writing; look for clues about what it was doing at the time.
- `config.yaml` — current watch config (interval, filters, anything that scales with mailbox size).
- `pyproject.toml` — dep list tells you whether FAISS/numpy/transformers are in play.

## Useful follow-up commands

Reproduce under memory profiling before touching code:

```bash
# Run one cycle with tracemalloc enabled and diff snapshots
python -X tracemalloc=25 -m <module> watch --interval 5 --max-cycles 20

# Or attach memray to a live watch process
memray attach <pid>                # record allocations
memray flamegraph memray-<pid>.bin # analyze

# Watch RSS over time externally
while :; do ps -o pid,rss,cmd -p $(pgrep -f 'gmail-search watch'); sleep 30; done
```

A 30-minute run with `tracemalloc` diffing snapshot N vs N+5 should surface the offending allocation site even if the leak is slow.

## Out of scope for you

These have already been done at the OS layer and do **not** need fixing in this repo:
- `OOMPolicy=continue` drop-in added to `vncserver.service` so a single-process OOM no longer tears down the desktop.
- 16 GiB zstd zram swap device activated, so transient memory pressure no longer goes straight to a global OOM.

The leak itself is still a real bug and is what you're here to fix.
