"""LLM-judged relevance evaluation of ScaNN compaction variants.

Recall@K against the baseline only tells us "do we return the same
items as today" — NOT "do we return relevant items." A variant
returning DIFFERENT but EQUALLY RELEVANT results would score low
recall but be just as good.

This script asks a strong LLM to grade each candidate result's
relevance to the query (0-3 scale), then computes:
  - NDCG@10 per variant (industry-standard graded relevance metric)
  - Precision@10 (% of returned results graded ≥ 2)
  - % of "highly relevant" (grade=3) items each variant captures

Variants compared:
  V0 baseline (3072d, ah=2, reorder=100) — current production
  A   PCA(1536), ah=2
  B   MRL 1536d, ah=2, reorder=500
  C   MRL 1536d AH + manual rerank vs full 3072d in Python

Eval set: 30 queries randomly sampled from the round-2 cache.

Cost: ~$0.50 with Gemini 2.5 Pro.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import psycopg
import scann
from psycopg.rows import dict_row

DSN = "postgresql://gmail_search:gmail_search@127.0.0.1:5544/gmail_search"
MODEL = "gemini-embedding-2-preview"
DIMENSIONS = 3072
WORK_DIR = Path("/tmp/scann_compaction_eval")
JUDGE_N_QUERIES = 30
TOP_K = 10  # judge top-10 only — that's what users see
SNIPPET_CHARS = 300


def load_corpus_memmap() -> tuple[list[int], np.memmap]:
    memmap_path = WORK_DIR / "corpus.memmap"
    ids = json.loads((WORK_DIR / "corpus_ids.json").read_text())
    n = len(ids)
    vectors = np.memmap(memmap_path, dtype=np.float32, mode="r", shape=(n, DIMENSIONS))
    return ids, vectors


def load_eval_queries() -> list[tuple[str, np.ndarray]]:
    texts = json.loads((WORK_DIR / "eval_queries_n100.json").read_text())
    embs = np.load(WORK_DIR / "eval_queries_n100.npy")
    return list(zip(texts, embs))


def search_baseline(ids, queries):
    """Re-run V0 (production config) to get per-query top-K. We have
    its results already cached for recall computation but only as IDs;
    re-running is cheap and gives us a clean baseline result list."""
    print("loading baseline searcher...")
    searcher = scann.scann_ops_pybind.load_searcher(str(WORK_DIR / "index_v0_ah2_3072d"))
    out = []
    for qtext, emb in queries:
        n = np.linalg.norm(emb)
        eq = emb if n == 0 else emb / n
        local_idx, _ = searcher.search(eq.astype(np.float32), final_num_neighbors=TOP_K)
        out.append((qtext, [ids[i] for i in local_idx]))
    return out


def search_variant_A(ids, queries):
    print("loading variant A searcher...")
    searcher = scann.scann_ops_pybind.load_searcher(str(WORK_DIR / "index_round2_A"))
    out = []
    for qtext, emb in queries:
        n = np.linalg.norm(emb)
        eq = emb if n == 0 else emb / n
        local_idx, _ = searcher.search(eq.astype(np.float32), final_num_neighbors=TOP_K)
        out.append((qtext, [ids[i] for i in local_idx]))
    return out


def search_variant_B(ids, queries):
    print("loading variant B searcher...")
    searcher = scann.scann_ops_pybind.load_searcher(str(WORK_DIR / "index_round2_B"))
    out = []
    for qtext, emb in queries:
        eq = emb[:1536].astype(np.float32, copy=True)
        n = np.linalg.norm(eq)
        if n > 0:
            eq = eq / n
        local_idx, _ = searcher.search(eq, final_num_neighbors=TOP_K)
        out.append((qtext, [ids[i] for i in local_idx]))
    return out


def search_variant_C(ids, vectors, queries):
    """Manual-rerank variant: AH search top-500 in 1536d, then rerank
    against full 3072d in numpy."""
    print("loading variant C searcher...")
    searcher = scann.scann_ops_pybind.load_searcher(str(WORK_DIR / "index_round2_C"))
    id_to_pos = {pid: i for i, pid in enumerate(ids)}
    out = []
    for qtext, emb in queries:
        eq_t = emb[:1536].astype(np.float32, copy=True)
        n = np.linalg.norm(eq_t)
        if n > 0:
            eq_t = eq_t / n
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
        out.append((qtext, [cand_ids[p] for p in top_pos_sorted]))
    return out


def fetch_message_snippets(embedding_ids: set[int]) -> dict[int, dict]:
    """For a set of embedding IDs (BIGSERIAL ints), join through
    `embeddings.message_id` to fetch the parent message's content.
    Returns {embedding_id: {id (=embedding_id), message_id, subject,
    snippet, from_addr, date}}.

    The judge grades by EMBEDDING id (since that's what each ScaNN
    variant returned), but the relevance question is about the
    underlying message (chunks of the same message share the same
    snippet). Duplicate-message embeddings will get the same grade
    from the LLM — fine, NDCG handles it correctly."""
    if not embedding_ids:
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
        (list(embedding_ids),),
    ).fetchall()
    conn.close()
    out: dict[int, dict] = {}
    for r in rows:
        out[r["embedding_id"]] = {
            "id": r["embedding_id"],  # the judge keys grades on this
            "message_id": r["message_id"],
            "subject": r["subject"],
            "snippet": r["snippet"],
            "from_addr": r["from_addr"],
            "date": r["date"],
        }
    return out


def build_judge_prompt(query: str, candidates: list[dict]) -> str:
    """Build a single prompt asking the LLM to grade each candidate's
    relevance to the query. Output format: one line per candidate id
    with grade 0-3."""
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
        "Output ONLY one line per candidate, in the form `<id> <grade>`. "
        "No commentary, no preamble, no markdown. Example:\n"
        "abc123 3\ndef456 1\nghi789 0"
    )
    return "\n".join(parts)


def call_judge(prompt: str) -> str:
    """One-shot Gemini 3.1 Pro Preview call (Scott's preferred fallback;
    no Anthropic API key available locally to use Claude programmatically).
    The 3.x reasoning models burn output-token budget on internal thinking
    before any visible text — a cap of 2000 returned an empty string. 8000
    leaves room for ~3000 thinking tokens + ~5000 of grade lines."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(
        model="gemini-3.1-pro-preview",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=8000,
        ),
    )
    return response.text or ""


def parse_grades(text: str, expected_ids: set[int]) -> dict[int, int]:
    """Parse `<embedding_id> <grade>` lines. Embedding IDs are PG
    BIGSERIAL ints; LLM emits them as numeric strings, we coerce.
    Missing ids default to 0 (irrelevant) so a refusal doesn't
    inflate scores."""
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
        if mid in expected_ids and 0 <= grade <= 3:
            out[mid] = grade
    return out


def ndcg_at_k(ranked_ids: list[str], grades: dict[str, int], k: int = TOP_K) -> float:
    """Standard NDCG@k. Gain = 2^grade - 1; discount = log2(rank+1)."""
    dcg = 0.0
    for rank, mid in enumerate(ranked_ids[:k], start=1):
        g = grades.get(mid, 0)
        dcg += (2**g - 1) / math.log2(rank + 1)
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum((2**g - 1) / math.log2(r + 2) for r, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(ranked_ids: list[str], grades: dict[str, int], k: int = TOP_K) -> float:
    """Fraction of top-k items with grade >= 2."""
    if not ranked_ids:
        return 0.0
    return sum(1 for mid in ranked_ids[:k] if grades.get(mid, 0) >= 2) / k


def main() -> None:
    print("=== loading data ===")
    ids, vectors = load_corpus_memmap()
    all_queries = load_eval_queries()
    rng = random.Random(42)
    sample = rng.sample(all_queries, JUDGE_N_QUERIES)
    print(f"sampled {JUDGE_N_QUERIES}/{len(all_queries)} queries")

    print("\n=== running searches across variants ===")
    results = {
        "V0": search_baseline(ids, sample),
        "A": search_variant_A(ids, sample),
        "B": search_variant_B(ids, sample),
        "C": search_variant_C(ids, vectors, sample),
    }
    for name, lst in results.items():
        print(f"  {name}: {len(lst)} queries x top-{len(lst[0][1])}")

    # For each query, union the top-K across variants, fetch content,
    # ask the judge, parse grades.
    print("\n=== judging (Gemini 2.5 Pro) ===")
    per_query: list[dict] = []
    for qi, (qtext, _) in enumerate(sample):
        union: set[str] = set()
        for v_name in results:
            for mid in results[v_name][qi][1]:
                union.add(mid)
        snippets = fetch_message_snippets(union)
        candidates = [snippets[m] for m in sorted(union) if m in snippets]
        if not candidates:
            print(f"  q{qi:02d}: no candidates fetched, skipping")
            continue
        prompt = build_judge_prompt(qtext, candidates)
        t0 = time.time()
        try:
            raw = call_judge(prompt)
        except Exception as exc:
            print(f"  q{qi:02d}: judge call failed: {exc}; skipping")
            continue
        grades = parse_grades(raw, union)
        elapsed = time.time() - t0
        # default missing IDs to 0
        for mid in union:
            grades.setdefault(mid, 0)
        per_query.append(
            {
                "query": qtext,
                "candidates": list(union),
                "grades": grades,
                "judge_time_s": elapsed,
            }
        )
        graded = sum(1 for g in grades.values() if g > 0)
        print(f"  q{qi:02d} ({elapsed:.1f}s, {len(union)} candidates, {graded} graded>0): {qtext[:50]}")

    # Compute metrics per variant
    print("\n=== METRICS ===")
    print(f"{'variant':<8s} {'NDCG@10':>10s} {'Prec@10':>10s} {'#H.Rel found / total':>22s}")
    print("-" * 60)
    h_rel_found_baseline_count = 0
    summary = {}
    for v_name in ["V0", "A", "B", "C"]:
        ndcgs = []
        precs = []
        h_rel_found = 0
        h_rel_total_in_union = 0
        for qi, q in enumerate(per_query):
            ranked = results[v_name][qi][1]
            ndcgs.append(ndcg_at_k(ranked, q["grades"]))
            precs.append(precision_at_k(ranked, q["grades"]))
            qhr = sum(1 for g in q["grades"].values() if g == 3)
            h_rel_total_in_union += qhr
            h_rel_found += sum(1 for mid in ranked if q["grades"].get(mid, 0) == 3)
        avg_ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0
        avg_prec = sum(precs) / len(precs) if precs else 0.0
        h_rel_recovery = h_rel_found / h_rel_total_in_union if h_rel_total_in_union else 0.0
        summary[v_name] = {
            "ndcg_at_10": avg_ndcg,
            "precision_at_10": avg_prec,
            "highly_relevant_recovery": h_rel_recovery,
            "highly_relevant_found": h_rel_found,
            "highly_relevant_total_union": h_rel_total_in_union,
        }
        print(
            f"{v_name:<8s} {avg_ndcg:>9.3f} {avg_prec:>9.1%}"
            f"  {h_rel_found:>4d} / {h_rel_total_in_union:<6d} ({h_rel_recovery:.1%})"
        )

    # Persist for follow-up.
    out = {
        "n_queries": len(per_query),
        "variants": summary,
        "per_query": per_query,
    }
    out_path = WORK_DIR / "judge_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nfull results: {out_path}")


if __name__ == "__main__":
    main()
