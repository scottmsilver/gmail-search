"""Round-3 judge orchestrator with subprocess-per-variant.

Why this rewrite: scann_compaction_judge2.py loaded indexes in one
long-running Python process. Even one-at-a-time with del+gc.collect()
didn't reliably release ScaNN's C++ allocations to the OS — so the
heap kept growing, swap saturated, the kernel killed the process.

Solution: each variant's searches run in a FRESH subprocess
(scann_search_one.py). When the subprocess exits, the OS guarantees
all memory is freed. The orchestrator just collects per-variant
search-result JSONs and runs the LLM judge in parallel at the end.

Outputs:
  /tmp/scann_compaction_eval/judge3_progress.log
  /tmp/scann_compaction_eval/judge3_results.json
  /tmp/scann_compaction_eval/searches_<variant>.json (one per variant)
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import numpy as np
import psycopg
from psycopg.rows import dict_row

DSN = "postgresql://gmail_search:gmail_search@127.0.0.1:5544/gmail_search"
EMBED_MODEL = "gemini-embedding-2-preview"
DIMENSIONS = 3072
WORK_DIR = Path("/tmp/scann_compaction_eval")
N_CACHED = 30
N_GENERATED = 20
TOP_K = 10
SNIPPET_CHARS = 300
JUDGE_MODEL = "gemini-3.1-pro-preview"
JUDGE_MAX_TOKENS = 8000
PARALLEL_WORKERS = 5
PROGRESS_LOG = WORK_DIR / "judge3_progress.log"
RESULTS_FILE = WORK_DIR / "judge3_results.json"
QUERY_SET_TEXTS = WORK_DIR / "judge2_queries.json"
QUERY_SET_EMBS = WORK_DIR / "judge2_queries.npy"

VARIANT_NAMES = ["V0", "V1", "V2", "V3", "V4", "V5", "A", "B", "C"]


_progress_lock = Lock()


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}\n"
    with _progress_lock:
        with PROGRESS_LOG.open("a") as f:
            f.write(line)
            f.flush()
        print(line, end="", flush=True)


def prepare_query_set() -> tuple[list[str], np.ndarray]:
    """Build the canonical 30-cached + 20-synthetic query set ONCE,
    then reuse across all variants. Persisted at judge2_queries.{json,npy}."""
    if QUERY_SET_TEXTS.exists() and QUERY_SET_EMBS.exists():
        texts = json.loads(QUERY_SET_TEXTS.read_text())["texts"]
        embs = np.load(QUERY_SET_EMBS)
        log(f"reused cached query set: {len(texts)} queries")
        return texts, embs

    log("=== building query set ===")
    # 30 random cached queries.
    conn = psycopg.connect(DSN, row_factory=dict_row)
    cached_rows = conn.execute(
        """SELECT query_text, embedding FROM query_cache
           WHERE model = %s AND OCTET_LENGTH(embedding) = %s
           ORDER BY random() LIMIT %s""",
        (EMBED_MODEL, DIMENSIONS * 4, N_CACHED),
    ).fetchall()
    conn.close()
    cached_texts = [f"[cached] {r['query_text']}" for r in cached_rows]
    cached_embs = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in cached_rows])

    # 20 synthetic queries via Gemini.
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    log(f"asking {JUDGE_MODEL} to generate {N_GENERATED} diverse queries ...")
    gen_prompt = (
        f"Generate exactly {N_GENERATED} diverse search queries that a user might "
        "type into a personal Gmail search system. Mix categories: financial, "
        "travel, subscriptions, family/personal, work/professional, health, real "
        "estate, school, legal, shopping. Each query should be 2-8 words, "
        "lowercase. ONE query per line, no numbering, no commentary."
    )
    r = client.models.generate_content(
        model=JUDGE_MODEL,
        contents=gen_prompt,
        config=types.GenerateContentConfig(temperature=0.7, max_output_tokens=4000),
    )
    raw = (r.text or "").strip()
    synth_texts = [line.strip().lower() for line in raw.splitlines() if line.strip()]
    synth_texts = [t for t in synth_texts if 2 <= len(t.split()) <= 12][:N_GENERATED]
    log(f"got {len(synth_texts)} synthetic queries")

    # Embed each via the same Gemini embedding model.
    log("embedding synthetic queries ...")
    synth_embs = []
    for t in synth_texts:
        er = client.models.embed_content(model=EMBED_MODEL, contents=t)
        synth_embs.append(np.array(er.embeddings[0].values, dtype=np.float32))
    synth_embs_arr = np.stack(synth_embs)

    all_texts = cached_texts + [f"[synth] {t}" for t in synth_texts]
    all_embs = np.concatenate([cached_embs, synth_embs_arr], axis=0)
    QUERY_SET_TEXTS.write_text(json.dumps({"texts": all_texts}))
    np.save(QUERY_SET_EMBS, all_embs)
    log(f"persisted {len(all_texts)} queries to {QUERY_SET_TEXTS}")
    return all_texts, all_embs


def run_one_variant(variant: str) -> bool:
    """Spawn scann_search_one.py as a subprocess. Returns True on success."""
    out_path = WORK_DIR / f"searches_{variant}.json"
    if out_path.exists():
        log(f"  {variant}: cached at {out_path.name}, skipping subprocess")
        return True
    log(f"  {variant}: spawning subprocess ...")
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "-u", "scripts/scann_search_one.py", variant],
        cwd="/home/ssilver/development/gmail-search",
        capture_output=True,
        text=True,
        timeout=600,  # 10 min per variant max
    )
    elapsed = time.time() - t0
    if proc.returncode != 0:
        log(f"  {variant}: FAILED in {elapsed:.1f}s — exit {proc.returncode}")
        log(f"    stderr tail: {proc.stderr[-500:]}")
        return False
    if not out_path.exists():
        log(f"  {variant}: subprocess succeeded but no output file — stdout: {proc.stdout[-200:]}")
        return False
    log(f"  {variant}: done in {elapsed:.1f}s ({out_path.stat().st_size // 1024} KB)")
    return True


def fetch_message_snippets(eids: set[int]) -> dict[int, dict]:
    if not eids:
        return {}
    conn = psycopg.connect(DSN, row_factory=dict_row)
    rows = conn.execute(
        f"""SELECT e.id AS embedding_id,
                  m.id AS message_id,
                  COALESCE(NULLIF(m.subject, ''), '(no subject)') AS subject,
                  COALESCE(NULLIF(LEFT(m.body_text, {SNIPPET_CHARS}), ''), '(empty body)') AS snippet,
                  m.from_addr,
                  to_char(m.date::timestamptz, 'YYYY-MM-DD') AS date
           FROM embeddings e
           JOIN messages m ON m.id = e.message_id
           WHERE e.id = ANY(%s)""",
        (list(eids),),
    ).fetchall()
    conn.close()
    out: dict[int, dict] = {}
    for r in rows:
        out[r["embedding_id"]] = {
            "id": r["embedding_id"],
            "message_id": r["message_id"],
            "subject": r["subject"],
            "snippet": r["snippet"],
            "from_addr": r["from_addr"],
            "date": r["date"],
        }
    return out


def build_judge_prompt(query: str, candidates: list[dict]) -> str:
    parts = [
        "You are an expert relevance judge. The user issued the search query below "
        "against their personal Gmail archive. Below are candidate emails returned "
        "by the search system. Grade each candidate's relevance to the query on a "
        "0-3 scale:",
        "",
        "  3 = Highly Relevant — directly answers or addresses the query intent",
        "  2 = Relevant — about the topic, useful context, but not the primary answer",
        "  1 = Tangential — mentions related terms but isn't really about this",
        "  0 = Irrelevant — wrong topic, false match",
        "",
        f"QUERY: {query}",
        "",
        f"CANDIDATES ({len(candidates)} emails):",
        "",
    ]
    for i, c in enumerate(candidates, 1):
        parts.append(f"--- candidate {i} (id={c['id']}) ---")
        parts.append(f"From: {c['from_addr']}")
        parts.append(f"Date: {c['date']}")
        parts.append(f"Subject: {c['subject']}")
        parts.append(f"Snippet: {c['snippet']}")
        parts.append("")
    parts.append(
        "Output ONLY one line per candidate, in the form `<id> <grade>`. " "No commentary, no preamble, no markdown."
    )
    return "\n".join(parts)


def call_judge(prompt: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(
        model=JUDGE_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=JUDGE_MAX_TOKENS),
    )
    return response.text or ""


def parse_grades(text: str, expected: set[int]) -> dict[int, int]:
    out: dict[int, int] = {}
    for line in text.strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        try:
            mid = int(parts[0])
            grade = int(parts[1])
        except ValueError:
            continue
        if mid in expected and 0 <= grade <= 3:
            out[mid] = grade
    return out


def ndcg_at_k(ranked: list[int], grades: dict[int, int], k: int = TOP_K) -> float:
    dcg = 0.0
    for rank, mid in enumerate(ranked[:k], start=1):
        g = grades.get(mid, 0)
        dcg += (2**g - 1) / math.log2(rank + 1)
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum((2**g - 1) / math.log2(r + 2) for r, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(ranked: list[int], grades: dict[int, int], k: int = TOP_K) -> float:
    if not ranked:
        return 0.0
    return sum(1 for mid in ranked[:k] if grades.get(mid, 0) >= 2) / k


def judge_one_query(qi: int, qtext: str, search_results: dict[str, list[int]]):
    union = set()
    for ranked in search_results.values():
        for mid in ranked:
            union.add(mid)
    snippets = fetch_message_snippets(union)
    candidates = [snippets[m] for m in sorted(union) if m in snippets]
    if not candidates:
        log(f"q{qi:02d} skip (no candidates)")
        return qi, qtext, {}, list(union)
    prompt = build_judge_prompt(qtext, candidates)
    t0 = time.time()
    try:
        raw = call_judge(prompt)
    except Exception as exc:
        log(f"q{qi:02d} JUDGE FAILED: {exc}")
        return qi, qtext, {}, list(union)
    elapsed = time.time() - t0
    grades = parse_grades(raw, union)
    for mid in union:
        grades.setdefault(mid, 0)
    graded = sum(1 for g in grades.values() if g > 0)
    log(f"q{qi:02d} ({elapsed:.1f}s, {len(union)} cands, {graded} graded>0): {qtext[:60]}")
    return qi, qtext, grades, list(union)


def main() -> None:
    PROGRESS_LOG.write_text("")
    log("=== judge3 orchestrator start ===")
    queries_text, queries_emb = prepare_query_set()
    log(f"query set ready: {len(queries_text)} queries")

    log(f"=== running {len(VARIANT_NAMES)} variants as subprocesses ===")
    successful: list[str] = []
    for v in VARIANT_NAMES:
        if run_one_variant(v):
            successful.append(v)
    log(f"variant search phase done: {len(successful)}/{len(VARIANT_NAMES)} succeeded")

    # Load all per-variant searches.
    all_results: list[dict[str, list[int]]] = [{} for _ in queries_text]
    for v in successful:
        data = json.loads((WORK_DIR / f"searches_{v}.json").read_text())
        for qi, ids_list in enumerate(data["results"]):
            all_results[qi][v] = ids_list

    # Parallel judging.
    log(f"=== judging {len(queries_text)} queries in parallel (workers={PARALLEL_WORKERS}) ===")
    judged: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as pool:
        futures = []
        for qi, qtext in enumerate(queries_text):
            futures.append(pool.submit(judge_one_query, qi, qtext, all_results[qi]))
        for fut in as_completed(futures):
            qi, qtext, grades, union_list = fut.result()
            judged[qi] = {"query": qtext, "grades": grades, "candidates": union_list}

    # Metrics.
    log("=== computing metrics ===")
    print()
    print("=" * 90)
    print(f"{'variant':<6s} {'NDCG@10':>10s} {'Prec@10':>10s} {'#H.Rel found / total':>25s}")
    print("-" * 90)
    summary = {}
    for v in successful:
        ndcgs, precs = [], []
        h_found = h_total = 0
        for qi in sorted(judged):
            grades = judged[qi]["grades"]
            ranked = all_results[qi].get(v)
            if ranked is None:
                continue
            ndcgs.append(ndcg_at_k(ranked, grades))
            precs.append(precision_at_k(ranked, grades))
            h_total += sum(1 for g in grades.values() if g == 3)
            h_found += sum(1 for mid in ranked if grades.get(mid, 0) == 3)
        avg_ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0
        avg_prec = sum(precs) / len(precs) if precs else 0.0
        rec = h_found / h_total if h_total else 0.0
        summary[v] = {
            "ndcg_at_10": avg_ndcg,
            "precision_at_10": avg_prec,
            "highly_relevant_found": h_found,
            "highly_relevant_total": h_total,
            "highly_relevant_recovery": rec,
        }
        print(f"{v:<6s} {avg_ndcg:>9.3f} {avg_prec:>9.1%}" f"   {h_found:>4d} / {h_total:<6d} ({rec:.1%})")

    out = {"n_queries": len(judged), "variants": summary}
    RESULTS_FILE.write_text(json.dumps(out, indent=2, default=str))
    log(f"DONE — results at {RESULTS_FILE}")


if __name__ == "__main__":
    main()
