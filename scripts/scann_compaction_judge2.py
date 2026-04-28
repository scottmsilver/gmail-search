"""Round 3: judge ALL 9 variants over 50 queries (30 cached + 20 LLM-generated).

Improvements over round 2:
- More variants (9 vs 4): adds round-1 V1/V2/V3/V4/V5 to the mix.
- More queries (50 vs 30): tighter confidence bands.
- Parallel LLM judging: 5 workers, 50 queries → ~2 min instead of ~10.
- Incremental progress log so a watcher can print live status.

Outputs:
  /tmp/scann_compaction_eval/judge2_progress.log  — line per query as it lands
  /tmp/scann_compaction_eval/judge2_results.json   — final aggregate
"""

from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import numpy as np
import psycopg
import scann
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
PROGRESS_LOG = WORK_DIR / "judge2_progress.log"
RESULTS_FILE = WORK_DIR / "judge2_results.json"


# ── Variant definitions ──────────────────────────────────────────────


def _load(path_part: str):
    return scann.scann_ops_pybind.load_searcher(str(WORK_DIR / path_part))


# Each variant is a callable: (ids, vectors, query_emb_3072d) -> list[int] of top-K embedding ids
def make_simple(searcher, truncate_dim: int | None):
    """Standard search: optionally truncate the query to match the
    index's dim, normalize, search."""

    def search(ids, vectors, emb):
        if truncate_dim:
            eq = emb[:truncate_dim].astype(np.float32, copy=True)
        else:
            eq = emb.astype(np.float32, copy=True)
        n = np.linalg.norm(eq)
        if n > 0:
            eq = eq / n
        local_idx, _ = searcher.search(eq, final_num_neighbors=TOP_K)
        return [ids[i] for i in local_idx]

    return search


def make_manual_rerank(searcher, ah_dim: int):
    """Variant C-style: AH search at lower dim, manual rerank vs full
    3072d. We close over `searcher` and `ah_dim`."""

    def search(ids, vectors, emb):
        eq_t = emb[:ah_dim].astype(np.float32, copy=True)
        n = np.linalg.norm(eq_t)
        if n > 0:
            eq_t = eq_t / n
        # Pull top-500 candidates from the AH index
        local_idx, _ = searcher.search(eq_t, final_num_neighbors=500)
        cand_ids = [ids[i] for i in local_idx]
        eq_full = emb.astype(np.float32, copy=True)
        nf = np.linalg.norm(eq_full)
        if nf > 0:
            eq_full = eq_full / nf
        cand_pos = [id_to_pos[c] for c in cand_ids]
        cand_v = vectors[cand_pos]
        cn = np.linalg.norm(cand_v, axis=1, keepdims=True)
        cn[cn == 0] = 1.0
        cand_v = cand_v / cn
        scores = cand_v @ eq_full
        top_pos = np.argpartition(scores, -TOP_K)[-TOP_K:]
        top_pos_sorted = top_pos[np.argsort(scores[top_pos])[::-1]]
        return [cand_ids[p] for p in top_pos_sorted]

    return search


# Set in main() after corpus loads.
id_to_pos: dict[int, int] = {}


# Variant DEFINITIONS (NOT loaded objects). Each entry is
# (path_part, factory). The factory takes the loaded searcher and
# returns the search-fn closure. We load each searcher one at a time
# in main() to avoid the 39 GB OOM that hit the v1 of this script
# (free RAM was 19 GB, swap saturated, kernel killed the python).
VARIANT_DEFS: list[tuple[str, str, callable]] = [
    ("V0", "index_v0_ah2_3072d", lambda s: make_simple(s, None)),
    ("V1", "index_v1_ah4_3072d", lambda s: make_simple(s, None)),
    ("V2", "index_v2_ah2_1536d", lambda s: make_simple(s, 1536)),
    ("V3", "index_v3_ah2_768d", lambda s: make_simple(s, 768)),
    ("V4", "index_v4_ah4_1536d", lambda s: make_simple(s, 1536)),
    ("V5", "index_v5_ah4_768d", lambda s: make_simple(s, 768)),
    ("A", "index_round2_A", lambda s: make_simple(s, None)),
    ("B", "index_round2_B", lambda s: make_simple(s, 1536)),
    ("C", "index_round2_C", lambda s: make_manual_rerank(s, 1536)),
]


# ── Corpus + queries ─────────────────────────────────────────────────


def load_corpus_memmap() -> tuple[list[int], np.memmap]:
    memmap_path = WORK_DIR / "corpus.memmap"
    ids = json.loads((WORK_DIR / "corpus_ids.json").read_text())
    n = len(ids)
    vectors = np.memmap(memmap_path, dtype=np.float32, mode="r", shape=(n, DIMENSIONS))
    return ids, vectors


def sample_cached_queries(n: int) -> list[tuple[str, np.ndarray]]:
    """Random sample from query_cache."""
    conn = psycopg.connect(DSN, row_factory=dict_row)
    rows = conn.execute(
        """SELECT query_text, embedding FROM query_cache
           WHERE model = %s AND OCTET_LENGTH(embedding) = %s
           ORDER BY random() LIMIT %s""",
        (EMBED_MODEL, DIMENSIONS * 4, n),
    ).fetchall()
    conn.close()
    out = []
    for r in rows:
        emb = np.frombuffer(r["embedding"], dtype=np.float32).copy()
        out.append((f"[cached] {r['query_text']}", emb))
    return out


def generate_synthetic_queries(n: int) -> list[tuple[str, np.ndarray]]:
    """Generate `n` realistic search queries via Gemini, then embed
    each one with the same model the corpus uses so we can search
    apples-to-apples. Saves to disk so re-runs reuse them."""
    cache_path = WORK_DIR / f"synthetic_queries_n{n}.json"
    cache_emb_path = WORK_DIR / f"synthetic_queries_n{n}.npy"
    if cache_path.exists() and cache_emb_path.exists():
        texts = json.loads(cache_path.read_text())
        embs = np.load(cache_emb_path)
        print(f"reused {n} cached synthetic queries")
        return [(f"[synth] {t}", e) for t, e in zip(texts, embs)]

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    print(f"asking gemini-3.1-pro-preview to generate {n} diverse search queries...")
    gen_prompt = (
        f"Generate exactly {n} diverse search queries that a user might type into a "
        "personal Gmail search system. Mix categories: financial, travel, subscriptions, "
        "family/personal, work/professional, health, real estate / home, school, legal, "
        "shopping/orders. Each query should be 2-8 words, lowercase, no quotes. Output "
        "ONE query per line, no numbering, no commentary."
    )
    r = client.models.generate_content(
        model=JUDGE_MODEL,
        contents=gen_prompt,
        config=types.GenerateContentConfig(temperature=0.7, max_output_tokens=4000),
    )
    raw = (r.text or "").strip()
    texts = [line.strip().lower() for line in raw.splitlines() if line.strip()]
    texts = [t for t in texts if 2 <= len(t.split()) <= 12][:n]
    if len(texts) < n:
        print(f"WARNING: only got {len(texts)}/{n} usable queries from generator")
    print(f"generated queries: {texts[:5]} ...")

    # Embed each via the same model the corpus uses so similarity is comparable.
    embs = []
    for t in texts:
        er = client.models.embed_content(
            model=EMBED_MODEL,
            contents=t,
        )
        # genai embed_content returns .embeddings[0].values (list of floats)
        vec = np.array(er.embeddings[0].values, dtype=np.float32)
        embs.append(vec)
    embs_arr = np.stack(embs)
    cache_path.write_text(json.dumps(texts))
    np.save(cache_emb_path, embs_arr)
    return [(f"[synth] {t}", v) for t, v in zip(texts, embs_arr)]


# ── Snippet fetch + judge ────────────────────────────────────────────


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
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=JUDGE_MAX_TOKENS,
        ),
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


# ── Metrics ──────────────────────────────────────────────────────────


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


# ── Progress logging ────────────────────────────────────────────────


_progress_lock = Lock()


def log_progress(msg: str) -> None:
    """Append a timestamped line to the progress log. Safe across threads."""
    line = f"[{time.strftime('%H:%M:%S')}] {msg}\n"
    with _progress_lock:
        with PROGRESS_LOG.open("a") as f:
            f.write(line)
            f.flush()
        print(line, end="", flush=True)


# ── Per-query worker ────────────────────────────────────────────────


def judge_one_query(qi: int, qtext: str, search_results: dict[str, list[int]]):
    """Build the union, fetch snippets, call the judge, parse grades.
    Returns (qi, qtext, grades_dict). Designed to run in a thread pool."""
    union = set()
    for v_name, ranked in search_results.items():
        for mid in ranked:
            union.add(mid)
    snippets = fetch_message_snippets(union)
    candidates = [snippets[m] for m in sorted(union) if m in snippets]
    if not candidates:
        log_progress(f"q{qi:02d} skip (no candidates)")
        return qi, qtext, {}, list(union)
    prompt = build_judge_prompt(qtext, candidates)
    t0 = time.time()
    try:
        raw = call_judge(prompt)
    except Exception as exc:
        log_progress(f"q{qi:02d} JUDGE FAILED: {exc}")
        return qi, qtext, {}, list(union)
    elapsed = time.time() - t0
    grades = parse_grades(raw, union)
    for mid in union:
        grades.setdefault(mid, 0)
    graded = sum(1 for g in grades.values() if g > 0)
    log_progress(f"q{qi:02d} ({elapsed:.1f}s, {len(union)} cands, {graded} graded>0): {qtext[:60]}")
    return qi, qtext, grades, list(union)


def main() -> None:
    PROGRESS_LOG.write_text("")  # truncate
    log_progress("=== loading data ===")
    global id_to_pos
    ids, vectors = load_corpus_memmap()
    id_to_pos = {pid: i for i, pid in enumerate(ids)}
    log_progress(f"corpus: {len(ids)} vectors x {DIMENSIONS} dims")

    cached = sample_cached_queries(N_CACHED)
    synthetic = generate_synthetic_queries(N_GENERATED)
    queries = cached + synthetic
    log_progress(f"queries: {len(cached)} cached + {len(synthetic)} synthetic = {len(queries)}")

    # Sequential variant load + search → free → next. Avoids the
    # 39 GB OOM that v1 hit (loading all 9 indexes at once).
    log_progress(f"=== searching {len(queries)} queries x {len(VARIANT_DEFS)} variants ===")
    t0 = time.time()
    all_results: list[dict[str, list[int]]] = [{} for _ in queries]
    import gc as _gc

    for v_name, path_part, factory in VARIANT_DEFS:
        v_t0 = time.time()
        log_progress(f"  loading {v_name} ({path_part}) ...")
        try:
            searcher = _load(path_part)
        except Exception as exc:
            log_progress(f"  {v_name}: load FAILED ({exc}); skipping")
            continue
        search_fn = factory(searcher)
        for qi, (qtext, emb) in enumerate(queries):
            all_results[qi][v_name] = search_fn(ids, vectors, emb)
        del searcher
        del search_fn
        _gc.collect()
        log_progress(f"  {v_name}: searches done in {time.time() - v_t0:.1f}s, freed memory")
    log_progress(f"all variant searches done in {time.time() - t0:.1f}s")

    # Parallel judging — N workers in a ThreadPool.
    log_progress(f"=== judging in parallel (workers={PARALLEL_WORKERS}) ===")
    judged: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as pool:
        futures = []
        for qi, (qtext, _) in enumerate(queries):
            futures.append(pool.submit(judge_one_query, qi, qtext, all_results[qi]))
        for fut in as_completed(futures):
            qi, qtext, grades, union_list = fut.result()
            judged[qi] = {"query": qtext, "grades": grades, "candidates": union_list}

    # Compute metrics per variant.
    log_progress("=== computing metrics ===")
    print()
    print("=" * 90)
    print(f"{'variant':<6s} {'NDCG@10':>10s} {'Prec@10':>10s} {'#H.Rel found / total':>25s}")
    print("-" * 90)
    summary = {}
    for v_name in ["V0", "V1", "V2", "V3", "V4", "V5", "A", "B", "C"]:
        ndcgs = []
        precs = []
        h_found = 0
        h_total = 0
        for qi in sorted(judged):
            grades = judged[qi]["grades"]
            ranked = all_results[qi].get(v_name)
            if ranked is None:
                # Variant load failed earlier — skip.
                continue
            ndcgs.append(ndcg_at_k(ranked, grades))
            precs.append(precision_at_k(ranked, grades))
            h_total_q = sum(1 for g in grades.values() if g == 3)
            h_total += h_total_q
            h_found += sum(1 for mid in ranked if grades.get(mid, 0) == 3)
        avg_ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0
        avg_prec = sum(precs) / len(precs) if precs else 0.0
        rec = h_found / h_total if h_total else 0.0
        summary[v_name] = {
            "ndcg_at_10": avg_ndcg,
            "precision_at_10": avg_prec,
            "highly_relevant_found": h_found,
            "highly_relevant_total": h_total,
            "highly_relevant_recovery": rec,
        }
        print(f"{v_name:<6s} {avg_ndcg:>9.3f} {avg_prec:>9.1%}" f"   {h_found:>4d} / {h_total:<6d} ({rec:.1%})")

    out = {"n_queries": len(judged), "variants": summary}
    RESULTS_FILE.write_text(json.dumps(out, indent=2, default=str))
    log_progress(f"DONE — results at {RESULTS_FILE}")


if __name__ == "__main__":
    main()
