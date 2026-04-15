"""Benchmark search latency across a variety of query types."""

import os
import sqlite3
import statistics
import time

import requests

BASE = os.environ.get("BENCHMARK_URL", "http://127.0.0.1:8080")
DB_PATH = os.environ.get("BENCHMARK_DB", "data/gmail_search.db")

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


def clear_query_cache():
    """Clear the embedding cache so we measure real cold latency."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM query_cache")
        conn.commit()
        conn.close()
    except Exception:
        pass


def bench_query_cold(query: str, k: int = 20) -> dict:
    """Benchmark a single query with no cache (cold start)."""
    clear_query_cache()
    t0 = time.perf_counter()
    try:
        resp = requests.get(f"{BASE}/api/search", params={"q": query, "k": k})
        cold_ms = (time.perf_counter() - t0) * 1000
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", data) if isinstance(data, dict) else data
            result_count = len(results)
        else:
            result_count = -1
    except Exception:
        cold_ms = (time.perf_counter() - t0) * 1000
        result_count = -1

    # Second call hits the embedding cache
    t1 = time.perf_counter()
    try:
        resp = requests.get(f"{BASE}/api/search", params={"q": query, "k": k})
        cached_ms = (time.perf_counter() - t1) * 1000
    except Exception:
        cached_ms = (time.perf_counter() - t1) * 1000

    return {
        "query": query,
        "results": result_count,
        "cold_ms": round(cold_ms),
        "cached_ms": round(cached_ms),
    }


def main():
    # Warmup — loads ScaNN index + initializes Gemini client
    print("Warming up...")
    clear_query_cache()
    requests.get(f"{BASE}/api/search", params={"q": "warmup", "k": 1})

    print(f"\nBenchmarking {len(QUERIES)} queries (cold + cached)")
    print(f"{'Query':<35} {'Results':>7} {'Cold':>7} {'Cached':>7}")
    print("-" * 62)

    cold_times = []
    cached_times = []
    for q in QUERIES:
        r = bench_query_cold(q)
        cold_times.append(r["cold_ms"])
        cached_times.append(r["cached_ms"])
        print(f"{r['query']:<35} {r['results']:>7} {r['cold_ms']:>6}ms {r['cached_ms']:>6}ms")

    print("-" * 62)
    print(
        f"{'Median':<35} {'':>7} {round(statistics.median(cold_times)):>6}ms {round(statistics.median(cached_times)):>6}ms"
    )
    print(
        f"{'P95':<35} {'':>7} {round(sorted(cold_times)[int(len(cold_times) * 0.95)]):>6}ms {round(sorted(cached_times)[int(len(cached_times) * 0.95)]):>6}ms"
    )


if __name__ == "__main__":
    main()
