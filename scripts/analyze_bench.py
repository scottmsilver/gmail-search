"""Summarize differences across the 6 bench runs so we don't have to
eyeball 40 six-way tables.

Metrics we care about:
- **Truncation signals**: summaries ending mid-sentence (no period / ends
  on a space or conjunction) — the 160-token budget can cut mid-sentence,
  a real quality regression.
- **"Thinking leak"**: summaries starting with "thought" or containing
  "Thinking Process" — Gemma reasoning-mode leaks into the output when
  the model doesn't know what to summarize.
- **Length stats**: median/p95 of summary length per config.
- **Per-message divergence**: messages where A-vs-C summaries differ
  a lot (character-level ratio < 0.5). Indicates the extra context
  materially changed the answer.
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path
from statistics import median

OUT_DIR = Path("scripts/bench_out")
LABELS = [
    "A_ctx2048_tok160",
    "A_ctx2048_tok300",
    "B_ctx4096_tok160",
    "B_ctx4096_tok300",
    "C_ctx8192_tok160",
    "C_ctx8192_tok300",
]

THINK_MARKERS = ("thought", "Thinking Process", "**Analyze the")


def _looks_truncated(s: str) -> bool:
    s = (s or "").rstrip()
    if not s:
        return False
    # Ends in a sentence terminator or closing quote? Probably complete.
    return s[-1] not in '.!?")'


def _thinking_leak(s: str) -> bool:
    return any(m in (s or "") for m in THINK_MARKERS)


def main() -> None:
    runs = {lbl: json.loads((OUT_DIR / f"{lbl}.json").read_text()) for lbl in LABELS}

    print("=" * 80)
    print("QUALITY METRICS PER CONFIG")
    print("=" * 80)
    print(f"{'label':<22} {'ok':>4} {'fail':>5} {'trunc':>6} {'think':>6} {'med_len':>8} {'p95_len':>8} {'med_ms':>7}")
    for lbl in LABELS:
        rs = runs[lbl]["results"]
        summaries = [r["summary"] for r in rs if r["summary"]]
        lens = [len(s) for s in summaries]
        times = [r["elapsed_ms"] for r in rs if r["summary"]]
        p95 = sorted(lens)[int(len(lens) * 0.95)] if lens else 0
        trunc = sum(1 for s in summaries if _looks_truncated(s))
        think = sum(1 for s in summaries if _thinking_leak(s))
        print(
            f"{lbl:<22} {runs[lbl]['totals']['ok']:>4} {runs[lbl]['totals']['failed']:>5} "
            f"{trunc:>6} {think:>6} "
            f"{int(median(lens)) if lens else 0:>8} {p95:>8} "
            f"{int(median(times)) if times else 0:>7}"
        )

    print("\n" + "=" * 80)
    print("MESSAGES WHERE A vs C DIVERGED MOST (char-ratio < 0.5)")
    print("=" * 80)
    a_by_id = {r["id"]: r for r in runs["A_ctx2048_tok160"]["results"]}
    c_by_id = {r["id"]: r for r in runs["C_ctx8192_tok160"]["results"]}
    divs = []
    for mid, a in a_by_id.items():
        c = c_by_id.get(mid)
        if not c or not a["summary"] or not c["summary"]:
            continue
        ratio = difflib.SequenceMatcher(None, a["summary"], c["summary"]).ratio()
        divs.append((ratio, a["bucket"], a["body_len"], a["subject"], a["summary"], c["summary"]))
    divs.sort()
    for ratio, bucket, body_len, subj, a_s, c_s in divs[:10]:
        print(f"\n[{bucket} body={body_len}] ratio={ratio:.2f}  subject={subj[:60]!r}")
        print(f"  A: {a_s[:200]}")
        print(f"  C: {c_s[:200]}")

    print("\n" + "=" * 80)
    print("FAILURES (summary empty or HTTP error)")
    print("=" * 80)
    for lbl in LABELS:
        for r in runs[lbl]["results"]:
            if r["error"] or not r["summary"]:
                print(f"  {lbl} | {r['bucket']} | {r['subject'][:50]!r} | err={r['error']}")


if __name__ == "__main__":
    main()
