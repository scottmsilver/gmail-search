"""Benchmark search latency across a variety of query types."""

import os
import statistics
import time

import requests

BASE = os.environ.get("BENCHMARK_URL", "http://127.0.0.1:8080")

QUERIES = [
    # Exact name lookups
    "Frank Rimerman tax",
    "Landmarks West billing",
    "Scott Silver",
    # Semantic / conceptual
    "construction draw request",
    "crypto appraisal",
    "home renovation budget",
    # Specific topics
    "IRS form 8283",
    "electrical bid analysis",
    "insurance claim",
    # Vague / broad
    "meeting tomorrow",
    "follow up",
    "thank you",
]


def bench_query(query: str, k: int = 20, runs: int = 3) -> dict:
    times = []
    result_count = 0
    for _ in range(runs):
        t0 = time.perf_counter()
        try:
            resp = requests.get(f"{BASE}/api/search", params={"q": query, "k": k})
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            if resp.status_code == 200:
                data = resp.json()
                result_count = len(data)
            else:
                times[-1] = elapsed  # keep the time but mark error
                result_count = -1
        except Exception:
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            result_count = -1
    return {
        "query": query,
        "results": result_count,
        "min_ms": round(min(times) * 1000),
        "median_ms": round(statistics.median(times) * 1000),
        "max_ms": round(max(times) * 1000),
        "p95_ms": round(sorted(times)[int(len(times) * 0.95)] * 1000),
    }


def main():
    # Warmup — first query loads the ScaNN index + initializes Gemini client
    print("Warming up...")
    requests.get(f"{BASE}/api/search", params={"q": "warmup", "k": 1})

    print(f"\nBenchmarking {len(QUERIES)} queries (3 runs each, k=20)")
    print(f"{'Query':<35} {'Results':>7} {'Min':>6} {'Median':>7} {'Max':>6}")
    print("-" * 70)

    all_medians = []
    for q in QUERIES:
        r = bench_query(q)
        all_medians.append(r["median_ms"])
        print(f"{r['query']:<35} {r['results']:>7} {r['min_ms']:>5}ms {r['median_ms']:>6}ms {r['max_ms']:>5}ms")

    print("-" * 70)
    print(f"{'Overall median':<35} {'':>7} {'':>6} {round(statistics.median(all_medians)):>6}ms")
    print(f"{'Overall p95':<35} {'':>7} {'':>6} {round(sorted(all_medians)[int(len(all_medians) * 0.95)]):>6}ms")
    print(f"{'Overall max':<35} {'':>7} {'':>6} {max(all_medians):>6}ms")


if __name__ == "__main__":
    main()
