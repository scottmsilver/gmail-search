"""Second-wave bench analysis: compare prompt variants + thinking modes
to find the best-quality configuration.

Metrics:
- success rate (non-empty summary)
- length distribution (we want richer content without drift into prose)
- a "specificity score": count of numbers, dollar amounts, dates,
  and capitalized-word runs. More of those → more concrete facts.
- per-bucket median length so we can tell whether extra prompt
  budget actually produced richer output on long emails.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import median

OUT_DIR = Path("scripts/bench_out")

# The configs we care about in wave 2. C2 is our re-baseline.
LABELS = [
    "C2_v1_tok300",  # current prod prompt
    "D_v2_tok400",  # richer (3 sentences, 400 chars)
    "E_v3_tok500",  # longest (4 sentences, 600 chars)
    "F_v4_tok400",  # structured triple
    "G2_v1_think_on",  # enable_thinking=1
    "H_v1_think_off",  # explicit thinking=0
]

_NUM_RE = re.compile(r"\b\d[\d,.$]*")
_DATE_RE = re.compile(
    r"\b(?:\d{1,2}/\d{1,2}(?:/\d{2,4})?|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2}|"
    r"\d{4}-\d{2}-\d{2})",
    re.IGNORECASE,
)
_PROPER_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+")


def specificity(s: str) -> int:
    if not s:
        return 0
    return len(_NUM_RE.findall(s)) + 2 * len(_DATE_RE.findall(s)) + len(_PROPER_RE.findall(s))


def main() -> None:
    runs = {lbl: json.loads((OUT_DIR / f"{lbl}.json").read_text()) for lbl in LABELS}

    buckets = ["short", "medium", "long", "very_long"]
    print("=" * 100)
    print(f"{'label':<22} {'ok':>3} {'fail':>4} {'med_len':>7} {'p95_len':>7} {'spec':>6} {'med_ms':>7}")
    print("-" * 100)
    for lbl in LABELS:
        rs = runs[lbl]["results"]
        oks = [r for r in rs if r["summary"]]
        lens = [len(r["summary"]) for r in oks]
        times = [r["elapsed_ms"] for r in oks]
        spec = sum(specificity(r["summary"]) for r in oks)
        p95 = sorted(lens)[int(len(lens) * 0.95)] if lens else 0
        print(
            f"{lbl:<22} {len(oks):>3} {len(rs) - len(oks):>4} "
            f"{int(median(lens)) if lens else 0:>7} {p95:>7} "
            f"{spec:>6} {int(median(times)) if times else 0:>7}"
        )

    # Median length per bucket — does extra budget translate to richer
    # content on long emails?
    print("\nMEDIAN SUMMARY LENGTH BY BUCKET")
    print(f"{'label':<22}" + "".join(f"{b:>11}" for b in buckets))
    for lbl in LABELS:
        rs = runs[lbl]["results"]
        cells = []
        for b in buckets:
            ls = [len(r["summary"]) for r in rs if r["bucket"] == b and r["summary"]]
            cells.append(f"{int(median(ls)) if ls else 0:>11}")
        print(f"{lbl:<22}" + "".join(cells))

    # Show sample outputs on a representative "very_long" message
    print("\nSAMPLE — 'Vision Pro at Work' (very_long, 31K)")
    sub_target = "Vision Pro at Work"
    for lbl in LABELS:
        for r in runs[lbl]["results"]:
            if sub_target in r["subject"]:
                print(f"\n[{lbl}]")
                print(f"  {r['summary'] or '(empty)'}")

    print("\nSAMPLE — 'How do AI software engineering agents work?' (very_long, 20K)")
    sub_target = "How do AI software"
    for lbl in LABELS:
        for r in runs[lbl]["results"]:
            if sub_target in r["subject"]:
                print(f"\n[{lbl}]")
                print(f"  {r['summary'] or '(empty)'}")

    print("\nSAMPLE — 'Re: Promoted.ai' (long, 8K)")
    sub_target = "Promoted.ai"
    for lbl in LABELS:
        for r in runs[lbl]["results"]:
            if sub_target in r["subject"]:
                print(f"\n[{lbl}]")
                print(f"  {r['summary'] or '(empty)'}")


if __name__ == "__main__":
    main()
