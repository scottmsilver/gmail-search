#!/usr/bin/env python3
"""A/B API diff harness — offline comparison of two FastAPI instances.

WHAT THIS MEASURES
==================

Fires a fixed set of read-only API requests against two running
`gmail-search serve` instances (one backed by SQLite, one by Postgres, in
the intended migration setup) and deep-diffs the JSON responses. The
point is not to prove byte-identical behavior — ranking scores and
timestamps differ by backend — but to confirm **semantic equivalence**:
same message counts, same top results (modulo order), same thread bodies.

Endpoints covered (all read-only):
  /api/status, /api/search?q=..., /api/thread/{id}, /api/topics,
  /api/jobs/running, /api/sql_schema, /api/query?q=...

Per-request PASS criteria:
  status           exact match on messages + embeddings counts
  search           top-10 ID overlap ≥ 0.8, top-20 ≥ 0.7
  thread           byte-identical minus ignored fields
  jobs/running     same set of job_ids with same status
  topics           byte-identical
  sql_schema       byte-identical
  query            thread_id set overlap ≥ 0.8 @ top-10

HOW TO READ THE OUTPUT
======================

A table prints one row per request:
  request                                  status   notes
  GET /api/status                          PASS     msgs=12345 embs=12000
  GET /api/search?q=receipt                FAIL     top10 overlap=0.6

Failures dump full details (both bodies + computed diff) to
  /tmp/ab_api_diff_<run_id>/<slug>.json

Final summary:
  PASSED 18/20, FAILED 2, TOTAL 20
Exit code is 0 if all passed, 1 otherwise.

USAGE
=====

  python scripts/ab_api_diff.py --a http://localhost:8090 --b http://localhost:8091
  python scripts/ab_api_diff.py --a ... --b ... --queries scripts/ab_queries.json
  python scripts/ab_api_diff.py --a ... --b ... --sample-thread-ids 10 --verbose
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

# --- Config: fields to ignore when diffing ----------------------------------

# Fields that are inherently non-deterministic or differ by backend, regardless
# of where they appear in the response tree.
ALWAYS_IGNORE_FIELDS = {
    "updated_at",
    "created_at",
    "started_at",
    "timestamp",
    "latency_ms",
    "elapsed_ms",
    "query_embeds",
    "query_embed_cost_usd",
    "total_cost_usd",
    "budget_remaining_usd",
    "running_job",
    "last_polled_at",
}

# `date` is ambiguous: message.date is deterministic (we compare it), but
# running_jobs[].date isn't. Strip it only under a running_jobs parent.
DATE_ONLY_UNDER_RUNNING_JOBS = "date"

# Search results: score varies by ranker; strip it before structural diffs.
SEARCH_IGNORE_FIELDS = {"score"}

DEFAULT_QUERIES = [
    "receipt",
    "invoice",
    "capital one",
    "chase statement",
    "amazon order",
    "stratechery",
    "alessandra melloni",
    "cyber sécurité",
    "Informationssicherheit",
    "会議の議事録",
    "flight confirmation",
    "aws bill",
    "github security alert",
    "doctor appointment",
    "what ISP did we use during the pandemic",
    "kids school permission slip fall 2023",
    "refund policy",
    "password reset",
    "DocuSign completed contract",
    "property tax assessment",
    "verification code",
]


# --- Result dataclass -------------------------------------------------------


@dataclass
class CheckResult:
    request: str
    status: str  # "PASS" | "FAIL" | "ERROR" | "SKIP"
    notes: str = ""
    detail: dict[str, Any] = field(default_factory=dict)


# --- HTTP helpers -----------------------------------------------------------


def _slugify(s: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_")
    return slug[:80] or "unnamed"


def _fetch(
    client: httpx.Client,
    base: str,
    path: str,
    params: dict[str, Any] | None = None,
    verbose: bool = False,
) -> tuple[int, Any, float, int, dict[str, str]]:
    """GET base+path; return (status, json_or_text, elapsed_ms, size, headers)."""
    url = base.rstrip("/") + path
    t0 = time.monotonic()
    resp = client.get(url, params=params, timeout=60.0)
    elapsed_ms = (time.monotonic() - t0) * 1000.0
    try:
        body: Any = resp.json()
    except Exception:
        body = resp.text
    size = len(resp.content)
    if verbose:
        print(
            f"  [{base}] {resp.status_code} {path} ({elapsed_ms:.0f}ms, {size}B)",
            file=sys.stderr,
        )
    return resp.status_code, body, elapsed_ms, size, dict(resp.headers)


# --- Strip / normalize helpers ----------------------------------------------


def _strip_ignored(obj: Any, under_running_jobs: bool = False) -> Any:
    """Recursively remove always-ignored keys. Also removes `date` when the
    parent key path indicates we're inside running_jobs[].
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in ALWAYS_IGNORE_FIELDS:
                continue
            if under_running_jobs and k == DATE_ONLY_UNDER_RUNNING_JOBS:
                continue
            child_rj = under_running_jobs or k in {"running_jobs", "jobs"}
            out[k] = _strip_ignored(v, under_running_jobs=child_rj)
        return out
    if isinstance(obj, list):
        return [_strip_ignored(x, under_running_jobs=under_running_jobs) for x in obj]
    return obj


def _strip_search_scores(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in results:
        rc = {k: v for k, v in r.items() if k not in SEARCH_IGNORE_FIELDS}
        if "matches" in rc and isinstance(rc["matches"], list):
            rc["matches"] = [{k: v for k, v in m.items() if k not in SEARCH_IGNORE_FIELDS} for m in rc["matches"]]
        out.append(rc)
    return out


# --- Deep-diff --------------------------------------------------------------


def _deep_diff(a: Any, b: Any, path: str = "") -> list[dict[str, Any]]:
    """Try deepdiff; fall back to a local recursive diff. Always returns a
    list of {path, a, b, kind} dicts (possibly empty).
    """
    try:
        from deepdiff import DeepDiff  # type: ignore

        diff = DeepDiff(a, b, ignore_order=False, report_repetition=True)
        if not diff:
            return []
        # Serialize DeepDiff into our compact list-of-dicts shape.
        return [{"deepdiff": json.loads(diff.to_json())}]
    except ImportError:
        return _local_diff(a, b, path)


def _local_diff(a: Any, b: Any, path: str) -> list[dict[str, Any]]:
    diffs: list[dict[str, Any]] = []
    if type(a) is not type(b):
        if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
            diffs.append({"path": path or "$", "kind": "type", "a": a, "b": b})
            return diffs
    if isinstance(a, dict):
        keys = set(a) | set(b)
        for k in sorted(keys):
            if k not in a:
                diffs.append({"path": f"{path}.{k}", "kind": "added_in_b", "b": b[k]})
            elif k not in b:
                diffs.append({"path": f"{path}.{k}", "kind": "missing_in_b", "a": a[k]})
            else:
                diffs.extend(_local_diff(a[k], b[k], f"{path}.{k}"))
    elif isinstance(a, list):
        if len(a) != len(b):
            diffs.append({"path": path or "$", "kind": "length", "a_len": len(a), "b_len": len(b)})
        for i in range(min(len(a), len(b))):
            diffs.extend(_local_diff(a[i], b[i], f"{path}[{i}]"))
    else:
        if a != b:
            diffs.append({"path": path or "$", "kind": "value", "a": a, "b": b})
    return diffs


# --- Overlap metrics --------------------------------------------------------


def _jaccard(xs: list[str], ys: list[str]) -> float:
    sx, sy = set(xs), set(ys)
    if not sx and not sy:
        return 1.0
    if not sx or not sy:
        return 0.0
    return len(sx & sy) / len(sx | sy)


def _overlap_at_k(xs: list[str], ys: list[str], k: int) -> float:
    if k <= 0:
        return 1.0
    top_x = xs[:k]
    top_y = set(ys[:k])
    if not top_x:
        return 1.0 if not top_y else 0.0
    hits = sum(1 for x in top_x if x in top_y)
    return hits / min(k, len(top_x))


def _spearman_on_intersection(xs: list[str], ys: list[str]) -> float | None:
    """Spearman rank correlation on the intersection of the two lists.
    Returns None if intersection has < 2 elements.
    """
    common = [x for x in xs if x in set(ys)]
    if len(common) < 2:
        return None
    ry = {y: i for i, y in enumerate(ys)}
    rx = {x: i for i, x in enumerate(xs)}
    n = len(common)
    d2 = sum((rx[m] - ry[m]) ** 2 for m in common)
    return 1.0 - (6.0 * d2) / (n * (n * n - 1))


# --- Message-id extraction --------------------------------------------------


def _search_message_ids(search_body: dict[str, Any]) -> list[str]:
    """Flatten the search results into an ordered list of message_ids
    (first match from each thread, in result order). This is the unit we
    compare between backends.
    """
    out: list[str] = []
    for r in search_body.get("results", []) or []:
        for m in r.get("matches") or []:
            mid = m.get("message_id") or m.get("id")
            if mid:
                out.append(mid)
                break
    return out


def _search_thread_ids(search_body: dict[str, Any]) -> list[str]:
    return [r.get("thread_id") or r.get("id") or "" for r in search_body.get("results", []) or []]


def _query_thread_ids(query_body: dict[str, Any]) -> list[str]:
    return [r.get("thread_id") or r.get("id") or "" for r in query_body.get("results", []) or []]


# --- Individual checks ------------------------------------------------------


def check_status(a_body: Any, b_body: Any) -> CheckResult:
    req = "GET /api/status"
    if not (isinstance(a_body, dict) and isinstance(b_body, dict)):
        return CheckResult(req, "ERROR", "non-dict body", {"a": a_body, "b": b_body})
    am, bm = a_body.get("messages"), b_body.get("messages")
    ae, be = a_body.get("embeddings"), b_body.get("embeddings")
    ok = am == bm and ae == be
    note = f"msgs a={am} b={bm}, embs a={ae} b={be}"
    return CheckResult(
        req,
        "PASS" if ok else "FAIL",
        note,
        {"a": a_body, "b": b_body},
    )


def check_search(q: str, a_body: Any, b_body: Any) -> CheckResult:
    req = f"GET /api/search?q={q}"
    if not (isinstance(a_body, dict) and isinstance(b_body, dict)):
        return CheckResult(req, "ERROR", "non-dict body", {"a": a_body, "b": b_body})
    a_ids = _search_message_ids(a_body)
    b_ids = _search_message_ids(b_body)
    o10 = _overlap_at_k(a_ids, b_ids, 10)
    j20 = _jaccard(a_ids[:20], b_ids[:20])
    sp = _spearman_on_intersection(a_ids[:20], b_ids[:20])
    ok = o10 >= 0.8 and j20 >= 0.7
    note = f"top10_overlap={o10:.2f} jaccard20={j20:.2f}"
    if sp is not None:
        note += f" spearman={sp:.2f}"
    return CheckResult(
        req,
        "PASS" if ok else "FAIL",
        note,
        {
            "a_ids": a_ids,
            "b_ids": b_ids,
            "overlap_at_10": o10,
            "jaccard_at_20": j20,
            "spearman": sp,
            "a_facets": a_body.get("facets"),
            "b_facets": b_body.get("facets"),
        },
    )


def check_thread(thread_id: str, a_body: Any, b_body: Any) -> CheckResult:
    req = f"GET /api/thread/{thread_id}"
    a_s = _strip_ignored(a_body)
    b_s = _strip_ignored(b_body)
    diffs = _deep_diff(a_s, b_s)
    ok = not diffs
    return CheckResult(
        req,
        "PASS" if ok else "FAIL",
        "identical" if ok else f"{len(diffs)} diff(s)",
        {"a": a_body, "b": b_body, "diffs": diffs},
    )


def check_topics(a_body: Any, b_body: Any) -> CheckResult:
    req = "GET /api/topics"
    diffs = _deep_diff(_strip_ignored(a_body), _strip_ignored(b_body))
    return CheckResult(
        req,
        "PASS" if not diffs else "FAIL",
        "identical" if not diffs else f"{len(diffs)} diff(s)",
        {"a": a_body, "b": b_body, "diffs": diffs},
    )


def check_jobs_running(a_body: Any, b_body: Any) -> CheckResult:
    req = "GET /api/jobs/running"
    if not (isinstance(a_body, dict) and isinstance(b_body, dict)):
        return CheckResult(req, "ERROR", "non-dict body", {"a": a_body, "b": b_body})

    def _index(body: dict[str, Any]) -> dict[str, str]:
        jobs = body.get("running_jobs") or body.get("jobs") or []
        if not isinstance(jobs, list):
            return {}
        idx = {}
        for j in jobs:
            if isinstance(j, dict):
                jid = j.get("job_id") or j.get("id")
                if jid:
                    idx[str(jid)] = str(j.get("status", ""))
        return idx

    a_idx = _index(a_body)
    b_idx = _index(b_body)
    ok = a_idx == b_idx
    note = f"a={len(a_idx)} jobs, b={len(b_idx)} jobs"
    return CheckResult(
        req,
        "PASS" if ok else "FAIL",
        note,
        {"a": a_body, "b": b_body, "a_jobs": a_idx, "b_jobs": b_idx},
    )


def check_sql_schema(a_body: Any, b_body: Any) -> CheckResult:
    req = "GET /api/sql_schema"
    diffs = _deep_diff(_strip_ignored(a_body), _strip_ignored(b_body))
    return CheckResult(
        req,
        "PASS" if not diffs else "FAIL",
        "identical" if not diffs else f"{len(diffs)} diff(s)",
        {"a": a_body, "b": b_body, "diffs": diffs},
    )


def check_query(label: str, a_body: Any, b_body: Any) -> CheckResult:
    req = f"GET /api/query {label}"
    if not (isinstance(a_body, dict) and isinstance(b_body, dict)):
        return CheckResult(req, "ERROR", "non-dict body", {"a": a_body, "b": b_body})
    a_ids = _query_thread_ids(a_body)
    b_ids = _query_thread_ids(b_body)
    o10 = _overlap_at_k(a_ids, b_ids, 10)
    ok = o10 >= 0.8
    return CheckResult(
        req,
        "PASS" if ok else "FAIL",
        f"top10_overlap={o10:.2f}",
        {"a_ids": a_ids, "b_ids": b_ids, "overlap_at_10": o10},
    )


# --- Orchestration ----------------------------------------------------------


def _load_queries(path: Path | None) -> list[str]:
    if path is None:
        return DEFAULT_QUERIES
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return [str(q) for q in data]
    if isinstance(data, dict) and "queries" in data:
        return [str(q) for q in data["queries"]]
    raise ValueError(f"unrecognized queries file shape: {path}")


def _sample_thread_ids(client: httpx.Client, base: str, queries: list[str], n: int, verbose: bool) -> list[str]:
    """Pull thread_ids by running a few searches on A. We use a handful
    of queries so the sample isn't dominated by one result page.
    """
    seen: list[str] = []
    seen_set: set[str] = set()
    seed_queries = queries[:5] if len(queries) >= 5 else queries
    for q in seed_queries:
        if len(seen) >= n:
            break
        try:
            _, body, _, _, _ = _fetch(client, base, "/api/search", params={"q": q, "k": 10}, verbose=verbose)
        except httpx.HTTPError as e:
            if verbose:
                print(f"  search failed for seed query {q!r}: {e}", file=sys.stderr)
            continue
        if not isinstance(body, dict):
            continue
        for r in body.get("results", []) or []:
            tid = r.get("thread_id") or r.get("id")
            if tid and tid not in seen_set:
                seen.append(tid)
                seen_set.add(tid)
                if len(seen) >= n:
                    break
    return seen[:n]


def _compare_request(
    client: httpx.Client,
    a: str,
    b: str,
    path: str,
    params: dict[str, Any] | None,
    verbose: bool,
) -> tuple[Any, Any, dict[str, Any]]:
    """Fetch the same request from A and B. Returns (a_body, b_body, meta)."""
    a_code, a_body, a_ms, a_size, a_hdrs = _fetch(client, a, path, params, verbose)
    b_code, b_body, b_ms, b_size, b_hdrs = _fetch(client, b, path, params, verbose)
    meta = {
        "a_status_code": a_code,
        "b_status_code": b_code,
        "a_elapsed_ms": round(a_ms, 1),
        "b_elapsed_ms": round(b_ms, 1),
        "a_size": a_size,
        "b_size": b_size,
        "size_delta": b_size - a_size,
        "a_headers": a_hdrs if verbose else None,
        "b_headers": b_hdrs if verbose else None,
    }
    return a_body, b_body, meta


def _wrap_http_errors(fn):
    def inner(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except httpx.HTTPError as e:
            return CheckResult(
                request=kwargs.get("_req", "unknown"),
                status="ERROR",
                notes=f"http error: {e}",
            )

    return inner


def _attach_meta(result: CheckResult, meta: dict[str, Any]) -> CheckResult:
    # Fail if HTTP status codes differed — important signal.
    if meta["a_status_code"] != meta["b_status_code"]:
        result.status = "FAIL"
        result.notes = f"http status mismatch a={meta['a_status_code']} b={meta['b_status_code']}; " + result.notes
    result.detail["_meta"] = meta
    return result


def run(args: argparse.Namespace) -> int:
    queries = _load_queries(Path(args.queries) if args.queries else None)
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    out_dir = Path(f"/tmp/ab_api_diff_{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[CheckResult] = []
    with httpx.Client() as client:
        # 1. /api/status
        a_body, b_body, meta = _compare_request(client, args.a, args.b, "/api/status", None, args.verbose)
        results.append(_attach_meta(check_status(a_body, b_body), meta))

        # 2. /api/topics
        a_body, b_body, meta = _compare_request(client, args.a, args.b, "/api/topics", None, args.verbose)
        results.append(_attach_meta(check_topics(a_body, b_body), meta))

        # 3. /api/sql_schema
        a_body, b_body, meta = _compare_request(client, args.a, args.b, "/api/sql_schema", None, args.verbose)
        results.append(_attach_meta(check_sql_schema(a_body, b_body), meta))

        # 4. /api/jobs/running
        a_body, b_body, meta = _compare_request(client, args.a, args.b, "/api/jobs/running", None, args.verbose)
        results.append(_attach_meta(check_jobs_running(a_body, b_body), meta))

        # 5. /api/search for each query
        per_query_search_results: list[tuple[str, Any, Any]] = []
        for q in queries:
            a_body, b_body, meta = _compare_request(
                client,
                args.a,
                args.b,
                "/api/search",
                {"q": q, "k": 20},
                args.verbose,
            )
            results.append(_attach_meta(check_search(q, a_body, b_body), meta))
            per_query_search_results.append((q, a_body, b_body))

        # 6. /api/thread/{id} — sample thread ids from A's search results.
        thread_ids = _sample_thread_ids(client, args.a, queries, args.sample_thread_ids, args.verbose)
        if not thread_ids and args.verbose:
            print("  no thread ids sampled; skipping thread-fetch diffs", file=sys.stderr)
        for tid in thread_ids:
            a_body, b_body, meta = _compare_request(client, args.a, args.b, f"/api/thread/{tid}", None, args.verbose)
            results.append(_attach_meta(check_thread(tid, a_body, b_body), meta))

        # 7. /api/query — exercise a few structured queries. These
        #    endpoints don't take `q`; we vary realistic filters instead.
        query_cases: list[tuple[str, dict[str, Any]]] = [
            ("label=INBOX&limit=20", {"label": "INBOX", "limit": 20}),
            ("order=date_desc&limit=10", {"order_by": "date_desc", "limit": 10}),
            ("has_attachment=true&limit=20", {"has_attachment": "true", "limit": 20}),
        ]
        for label, params in query_cases:
            a_body, b_body, meta = _compare_request(client, args.a, args.b, "/api/query", params, args.verbose)
            results.append(_attach_meta(check_query(label, a_body, b_body), meta))

    # Dump failures + report.
    failed = [r for r in results if r.status in {"FAIL", "ERROR"}]
    for r in failed:
        slug = _slugify(r.request)
        (out_dir / f"{slug}.json").write_text(
            json.dumps(
                {"request": r.request, "status": r.status, "notes": r.notes, "detail": r.detail}, default=str, indent=2
            )
        )

    _print_report(results, out_dir, args.verbose)
    total = len(results)
    passed = sum(1 for r in results if r.status == "PASS")
    fails = sum(1 for r in results if r.status in {"FAIL", "ERROR"})
    print(f"\nPASSED {passed}/{total}, FAILED {fails}, TOTAL {total}")
    if failed:
        print(f"Per-failure details dumped to: {out_dir}")
    return 0 if fails == 0 else 1


def _print_report(results: list[CheckResult], out_dir: Path, verbose: bool) -> None:
    # Column widths: clip request to keep the table readable.
    req_w = min(60, max(20, max((len(r.request) for r in results), default=20)))
    note_w = 80
    hdr = f"{'request'.ljust(req_w)}  {'status'.ljust(6)}  {'notes'}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        req = r.request if len(r.request) <= req_w else r.request[: req_w - 1] + "…"
        note = r.notes if len(r.notes) <= note_w else r.notes[: note_w - 1] + "…"
        print(f"{req.ljust(req_w)}  {r.status.ljust(6)}  {note}")
        if verbose and r.detail.get("_meta"):
            m = r.detail["_meta"]
            print(
                f"    timing a={m['a_elapsed_ms']}ms b={m['b_elapsed_ms']}ms  "
                f"size a={m['a_size']}B b={m['b_size']}B Δ={m['size_delta']}B"
            )


def main() -> int:
    ap = argparse.ArgumentParser(description="A/B diff two gmail-search API instances.")
    ap.add_argument("--a", required=True, help="base URL of instance A (e.g. http://localhost:8090)")
    ap.add_argument("--b", required=True, help="base URL of instance B (e.g. http://localhost:8091)")
    ap.add_argument(
        "--queries",
        help="path to queries JSON (list of strings or {queries:[...]})",
    )
    ap.add_argument(
        "--sample-thread-ids",
        type=int,
        default=5,
        help="how many thread ids to pull from A for /api/thread/* diffs (default 5)",
    )
    ap.add_argument("--verbose", action="store_true", help="log headers, timing, size deltas")
    args = ap.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
