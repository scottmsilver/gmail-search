"""Wave 4 analysis — rank all pipelines/configs by a combined
quality score, then dump a ranked table + 5 hand-picked eyeball
panels to COMPARISON_W4.md.

Scoring:
- specificity (numbers, dates, proper-noun pairs) — same metric as
  waves 2/3
- coverage (unique 3-gram overlap between body and summary) —
  proxy for "how much of the email is retrievable via this summary"
- failure rate (penalty)
- avg_ms (throughput view)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import median

OUT_DIR = Path("scripts/bench_out")

# All wave-4 runs + useful predecessors as reference points.
LABELS = [
    ("W4_baseline_v5", "prod baseline (ctx=8192 seqs=8)"),
    ("W4_cot_v6", "CoT: list facts then synthesise"),
    ("W4_structured_v7", "JSON extract → render"),
    ("W4_headline_v8", "headline + detail two-line"),
    ("W4_critique", "two-pass critique-and-revise"),
    ("W4_hierarchical", "thread splitter (reply markers)"),
    ("W4_few_shot", "few-shot from corpus"),
    ("W4_temp0.3", "temperature 0.3"),
    ("W4_temp0.5", "temperature 0.5"),
    ("W4_temp0.7", "temperature 0.7"),
    ("W4_think16k", "ctx=16384 + enable_thinking=1"),
    ("W4_ctx16k", "ctx=16384 seqs=1 + larger body"),
    ("W4_ctx32k", "ctx=32768 seqs=1 + full body"),
]

_NUM_RE = re.compile(r"\b\d[\d,.$]*")
_DATE_RE = re.compile(
    r"\b(?:\d{1,2}/\d{1,2}(?:/\d{2,4})?|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2}|\d{4}-\d{2}-\d{2})",
    re.IGNORECASE,
)
_PROPER_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z']+")


def specificity(s: str) -> int:
    if not s:
        return 0
    return len(_NUM_RE.findall(s)) + 2 * len(_DATE_RE.findall(s)) + len(_PROPER_RE.findall(s))


def _trigrams(text: str) -> set[tuple[str, str, str]]:
    toks = [t.lower() for t in _WORD_RE.findall(text)]
    return {(toks[i], toks[i + 1], toks[i + 2]) for i in range(len(toks) - 2)}


def coverage(body: str, summary: str) -> float:
    """Unique 3-grams from body that appear in summary, divided by
    3-grams in summary. Higher = more of the summary's wording
    genuinely comes from the email (not hallucinated or generic).
    """
    if not body or not summary:
        return 0.0
    body_tri = _trigrams(body)
    sum_tri = _trigrams(summary)
    if not sum_tri:
        return 0.0
    return len(body_tri & sum_tri) / len(sum_tri)


def main() -> None:
    runs = {}
    for label, _ in LABELS:
        path = OUT_DIR / f"{label}.json"
        if path.exists():
            runs[label] = json.loads(path.read_text())
        else:
            print(f"[skip] {label} not found")

    # Pull bodies for coverage scoring.
    import sqlite3

    conn = sqlite3.connect("data/gmail_search.db")
    conn.row_factory = sqlite3.Row
    bodies: dict[str, str] = {}
    for _, run in runs.items():
        for r in run["results"]:
            if r["id"] not in bodies:
                row = conn.execute("SELECT body_text FROM messages WHERE id = ?", (r["id"],)).fetchone()
                bodies[r["id"]] = (row["body_text"] if row else "") or ""
    conn.close()

    # Compute per-run aggregates.
    rows: list[dict] = []
    for label, desc in LABELS:
        if label not in runs:
            continue
        res = runs[label]["results"]
        oks = [r for r in res if r["summary"] and not r["error"]]
        lens = [len(r["summary"]) for r in oks]
        specs = [specificity(r["summary"]) for r in oks]
        covs = [coverage(bodies.get(r["id"], ""), r["summary"]) for r in oks]
        times = [r["elapsed_ms"] for r in oks]
        rows.append(
            {
                "label": label,
                "desc": desc,
                "ok": runs[label]["totals"]["ok"],
                "failed": runs[label]["totals"]["failed"],
                "med_len": int(median(lens)) if lens else 0,
                "spec_total": sum(specs),
                "spec_avg": round(sum(specs) / max(len(specs), 1), 1),
                "cov_avg": round(sum(covs) / max(len(covs), 1), 3),
                "med_ms": int(median(times)) if times else 0,
                "total_s": runs[label]["totals"]["total_seconds"],
            }
        )

    # Composite: weight spec_total heavily, cov_avg medium, penalise failures.
    for r in rows:
        r["score"] = r["spec_total"] + 500 * r["cov_avg"] - 20 * r["failed"]
    rows.sort(key=lambda r: -r["score"])

    # Render markdown.
    md: list[str] = []
    md.append("# Wave 4 creative bench — ranked results\n")
    md.append("Sample: 40 messages (scripts/bench_sample.json).\n")
    md.append(
        "Score = total specificity (numbers + dates×2 + proper-noun pairs) + 500×avg coverage (body→summary 3-gram overlap) − 20×failures.\n"
    )
    md.append("## Ranked\n")
    md.append("| # | label | ok/40 | med_len | spec_total | cov_avg | med_ms | total_s | score | desc |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(rows, 1):
        md.append(
            f"| {i} | `{r['label']}` | {r['ok']}/40 | {r['med_len']} | {r['spec_total']} | {r['cov_avg']:.3f} | {r['med_ms']} | {r['total_s']} | {round(r['score'], 1)} | {r['desc']} |"
        )

    # Hand-picked eyeball panels — pick 5 ids that stress different
    # dimensions (long thread, dense receipt, short ask, multi-party, numbers-heavy).
    probes = [
        ("Re: 2024 Subaru Outback", "long negotiation thread"),
        ("How do AI software", "dense technical article"),
        ("Vision Pro at Work", "newsletter with sports-streaming JV details"),
        ("CONGRATS", "promo + amounts"),
        ("Re: Promoted.ai March 2023", "multi-party business thread"),
    ]
    md.append("\n## Eyeball panels — 5 hard cases across pipelines\n")
    for needle, why in probes:
        md.append(f"### {needle!r} — {why}\n")
        for label, _ in LABELS:
            if label not in runs:
                continue
            for r in runs[label]["results"]:
                if needle in (r["subject"] or ""):
                    s = r["summary"] or "(empty)"
                    s = s.replace("\n", " ").replace("|", "\\|")
                    md.append(f"- **{label}**: {s[:400]}")
                    break
        md.append("")

    path = OUT_DIR / "COMPARISON_W4.md"
    path.write_text("\n".join(md))
    print(f"wrote {path}")
    # Also print the top-5 to stdout.
    print("\nTOP 5:")
    for r in rows[:5]:
        print(
            f"  {r['label']:<22}  score={r['score']:<7.1f}  spec={r['spec_total']:<4}  cov={r['cov_avg']:.3f}  ok={r['ok']}/40  med={r['med_ms']}ms"
        )


if __name__ == "__main__":
    main()
