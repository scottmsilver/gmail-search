"""Phase 0c — per-user ScaNN load latency benchmark.

The PER_USER_LOGIN_2026-04-27.md plan assumes per-user ScaNN indexes can
be eager-loaded at boot and that the kernel page cache evicts cold users
gracefully. If that assumption breaks (loads too slow, RAM doesn't release,
first-query latency too high after eviction), the entire per-user-index
strategy needs a redesign before any auth work happens.

Today's V3 index already has 13 self-contained shards (~43k vectors each
at 768 dims). Each shard IS a "synthetic user" — same shape and rough
size as a real per-user index would be. So we don't need to splice
anything; we just load each shard as if it were a separate user.

What we measure, per shard:
  * load_cold_ms      — load_searcher with files NOT in page cache
                         (we evict via posix_fadvise(POSIX_FADV_DONTNEED))
  * load_warm_ms      — load_searcher with files in page cache
  * rss_delta_mb      — process RSS growth from this load
  * first_query_ms    — first .search() call after a cold load
  * warm_query_ms     — second .search() (best-case steady state)

What we measure across the run:
  * sum(rss_delta)    — total RAM held by all loaded indexes
  * sum(load_cold_ms) — total boot time if we eager-load all users cold

Run:
    python scripts/bench_per_user_scann_load.py [--index-dir DIR] [--max-shards N]
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import scann


def _evict_dir_from_page_cache(dir_path: Path) -> None:
    """POSIX_FADV_DONTNEED tells the kernel to drop cached pages for
    these files. No sudo, no global drop_caches side effects, only
    affects the files we actually point at.

    Linux honors this for clean pages immediately (the .npy/.pb files
    here are read-only post-build, so they're always clean). Files
    that are still mmap'd by another process won't drop until that
    mapping is released — which is fine here because each .load_searcher
    we're benchmarking against is its own short-lived mapping.

    Assumes a flat shard layout (all files at the top of `dir_path`,
    no nested subdirs). True today for ScaNN; if that ever changes,
    extend to recurse — otherwise nested files won't get evicted and
    cold-vs-warm timings become indistinguishable."""
    for child in dir_path.iterdir():
        if not child.is_file():
            continue
        try:
            fd = os.open(child, os.O_RDONLY)
        except OSError:
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def _rss_mb() -> float:
    """CURRENT resident set size in MiB, read from /proc/self/status.
    Not ru_maxrss — that's a high-water mark and never decreases, so it
    can't show per-shard deltas after we drop a previous shard's refs."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                # Format: "VmRSS:\t  123456 kB"
                return int(line.split()[1]) / 1024.0
    return 0.0


def _read_index_dim(index_dir: Path) -> int:
    """Manifest knows the AH-stage dim; queries must match."""
    manifest = json.loads((index_dir / "manifest.json").read_text())
    return int(manifest.get("index_dim") or manifest["dimensions"])


def _time_load(shard_dir: Path) -> tuple[object, float]:
    """Return (loaded searcher, wall-clock ms). gc.collect first so
    timing isn't polluted by collecting the previous shard's tensors."""
    gc.collect()
    t0 = time.perf_counter()
    searcher = scann.scann_ops_pybind.load_searcher(str(shard_dir))
    return searcher, (time.perf_counter() - t0) * 1000.0


def _time_query(searcher: object, query: np.ndarray, k: int = 20) -> float:
    t0 = time.perf_counter()
    searcher.search(query, final_num_neighbors=k)
    return (time.perf_counter() - t0) * 1000.0


def _build_query_vector(dim: int) -> np.ndarray:
    """Random unit-norm vector at the index's AH dim. We don't care
    about result quality — just that the search executes a real distance
    calculation against the AH codebook + reorder dataset."""
    rng = np.random.default_rng(seed=42)
    v = rng.standard_normal(dim).astype(np.float32)
    n = float(np.linalg.norm(v))
    if n > 0:
        v /= n
    return v


def _bench_one_shard(shard_dir: Path, query: np.ndarray) -> dict[str, float]:
    """Per-shard measurement. Order matters:
       1. Evict from page cache.
       2. Load (cold).
       3. First query (cold-ish — AH codebook + dataset rows touched
          for the first time post-eviction).
       4. Warm query.
       5. Throw away the searcher.
       6. Re-evict (so the next "warm load" timing isn't measuring a
          load that's still warm from this iteration's queries).
       7. Wait... actually for warm load we WANT the cache hot, so
          we re-load WITHOUT evicting first (cache is warm from step
          3+4 having read everything).

    Reordering: do warm load FIRST after step 4 (cache is hot), THEN
    re-evict for the next shard."""
    rss_before = _rss_mb()
    _evict_dir_from_page_cache(shard_dir)

    searcher_cold, load_cold_ms = _time_load(shard_dir)
    rss_after_cold = _rss_mb()
    first_query_ms = _time_query(searcher_cold, query)
    warm_query_ms = _time_query(searcher_cold, query)
    del searcher_cold
    gc.collect()

    # Page cache is now hot from the load + 2 queries. Loading again
    # measures the all-warm case — what re-loads-during-process-restart
    # would look like if the kernel hasn't evicted us.
    searcher_warm, load_warm_ms = _time_load(shard_dir)
    del searcher_warm
    gc.collect()

    return {
        "load_cold_ms": load_cold_ms,
        "load_warm_ms": load_warm_ms,
        "rss_delta_mb": rss_after_cold - rss_before,
        "first_query_ms": first_query_ms,
        "warm_query_ms": warm_query_ms,
    }


def _bench_eager_load_all(shards: list[Path]) -> tuple[float, float, list[float]]:
    """Boot scenario: load ALL users' indexes back-to-back, keep them
    resident, measure aggregate wall-clock + final RSS. This is what
    the plan's "eager load all users at boot" actually costs."""
    for s in shards:
        _evict_dir_from_page_cache(s)
    gc.collect()
    rss_start = _rss_mb()
    t0 = time.perf_counter()
    held: list[object] = []  # keep refs so they don't get GC'd mid-loop
    per_load_ms: list[float] = []
    for s in shards:
        ti = time.perf_counter()
        held.append(scann.scann_ops_pybind.load_searcher(str(s)))
        per_load_ms.append((time.perf_counter() - ti) * 1000.0)
    total_ms = (time.perf_counter() - t0) * 1000.0
    rss_after = _rss_mb()
    del held
    gc.collect()
    return total_ms, rss_after - rss_start, per_load_ms


def _print_table(rows: list[dict[str, float]], shard_names: list[str]) -> None:
    header = (
        f"{'shard':<10} {'cold_load_ms':>14} {'warm_load_ms':>14} {'rss_mb':>10} {'1st_q_ms':>10} {'warm_q_ms':>10}"
    )
    print(header)
    print("-" * len(header))
    for name, r in zip(shard_names, rows):
        print(
            f"{name:<10} "
            f"{r['load_cold_ms']:>14.1f} "
            f"{r['load_warm_ms']:>14.1f} "
            f"{r['rss_delta_mb']:>10.1f} "
            f"{r['first_query_ms']:>10.2f} "
            f"{r['warm_query_ms']:>10.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("data") / "scann_index__20260428T042747Z_cb24ee",
        help="Sharded ScaNN index dir (default: today's V3 768d build).",
    )
    parser.add_argument("--max-shards", type=int, default=0, help="Limit (0 = all).")
    args = parser.parse_args()

    if not args.index_dir.exists():
        raise SystemExit(f"index dir not found: {args.index_dir}")

    shards = sorted(
        (p for p in args.index_dir.iterdir() if p.is_dir() and p.name.startswith("shard_")),
        key=lambda p: int(p.name.split("_", 1)[1]),
    )
    if args.max_shards:
        shards = shards[: args.max_shards]
    if not shards:
        raise SystemExit(f"no shard_* dirs under {args.index_dir}")

    index_dim = _read_index_dim(args.index_dir)
    query = _build_query_vector(index_dim)

    print(f"Index: {args.index_dir}")
    print(f"Synthetic users (= shards): {len(shards)}  index_dim={index_dim}")
    print()

    print("=== Per-shard cold/warm measurements ===")
    rows = [_bench_one_shard(s, query) for s in shards]
    _print_table(rows, [s.name for s in shards])

    avg_cold = sum(r["load_cold_ms"] for r in rows) / len(rows)
    avg_warm = sum(r["load_warm_ms"] for r in rows) / len(rows)
    avg_rss = sum(r["rss_delta_mb"] for r in rows) / len(rows)
    avg_1st = sum(r["first_query_ms"] for r in rows) / len(rows)
    avg_q = sum(r["warm_query_ms"] for r in rows) / len(rows)
    print()
    print(
        f"avg per-user: load_cold={avg_cold:.0f}ms  load_warm={avg_warm:.0f}ms  "
        f"rss={avg_rss:.0f}MiB  1st_q={avg_1st:.1f}ms  warm_q={avg_q:.1f}ms"
    )

    print()
    print("=== Eager-load-all-at-boot scenario ===")
    total_ms, total_rss_mb, per = _bench_eager_load_all(shards)
    print(
        f"boot: loaded {len(shards)} indexes in {total_ms / 1000:.2f}s, " f"resident RSS growth={total_rss_mb:.0f}MiB"
    )
    print(f"per-load (cold-but-back-to-back): min={min(per):.0f}ms max={max(per):.0f}ms")

    print()
    print("=== Verdict thresholds (from plan) ===")
    if avg_cold < 2000:
        print(f"  ✓ avg cold load {avg_cold:.0f}ms < 2000ms target")
    else:
        print(f"  ✗ avg cold load {avg_cold:.0f}ms ≥ 2000ms target — design risk")
    if avg_1st < 5000:
        print(f"  ✓ avg first-query {avg_1st:.1f}ms < 5000ms target")
    else:
        print(f"  ✗ avg first-query {avg_1st:.1f}ms ≥ 5000ms target — design risk")
    print(f"  → 8-user ceiling estimate: {8 * avg_rss:.0f}MiB resident, " f"{8 * avg_cold / 1000:.1f}s boot")


if __name__ == "__main__":
    main()
