"""Wave 3: compare the top prompt candidates across different context
sizes so we can tell whether the wins are coming from the prompt or
from the extra body budget.

Compares:
  E_v3_tok500     — v3 @ ctx=8192 seqs=2 body=20000  (wave 2 best)
  I_v5_tok500     — v5 @ ctx=8192 seqs=2 body=20000
  J_v3_ctx4096    — v3 @ ctx=4096 seqs=4 body=8000   (throughput path)
  K_v3_ctx16384   — v3 @ ctx=16384 seqs=1 body=50000 (more-context path)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import median

OUT_DIR = Path("scripts/bench_out")
LABELS = ["E_v3_tok500", "I_v5_tok500", "J_v3_ctx4096", "K_v3_ctx16384"]

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

    print(f"{'label':<20} {'ok':>3} {'fail':>4} {'med_len':>7} {'p95_len':>7} {'spec':>6} {'med_ms':>7} {'total_s':>7}")
    for lbl in LABELS:
        rs = runs[lbl]["results"]
        oks = [r for r in rs if r["summary"]]
        lens = [len(r["summary"]) for r in oks]
        times = [r["elapsed_ms"] for r in oks]
        spec = sum(specificity(r["summary"]) for r in oks)
        p95 = sorted(lens)[int(len(lens) * 0.95)] if lens else 0
        print(
            f"{lbl:<20} {len(oks):>3} {len(rs) - len(oks):>4} "
            f"{int(median(lens)) if lens else 0:>7} {p95:>7} "
            f"{spec:>6} {int(median(times)) if times else 0:>7} "
            f"{runs[lbl]['totals']['total_seconds']:>7}"
        )

    # spec per bucket — tells us where each config is winning
    print("\nSPECIFICITY BY BUCKET")
    buckets = ["short", "medium", "long", "very_long"]
    print(f"{'label':<20}" + "".join(f"{b:>11}" for b in buckets))
    for lbl in LABELS:
        rs = runs[lbl]["results"]
        cells = []
        for b in buckets:
            s = sum(specificity(r["summary"]) for r in rs if r["bucket"] == b and r["summary"])
            cells.append(f"{s:>11}")
        print(f"{lbl:<20}" + "".join(cells))

    # The interesting samples — pick long ones to see where more
    # context/prompt actually matters.
    for probe in [
        ("Vision Pro at Work", "very_long 31K"),
        ("How do AI software", "very_long 20K"),
        ("Re: Promoted.ai", "long 8K"),
        ("EXCLUSIVE: Female Athletes", "very_long 12K"),
        ("Re: 2024 Subaru Outback", "very_long 17K"),
    ]:
        needle, lbl = probe
        print(f"\n── {lbl}: {needle!r} ──")
        for run_lbl in LABELS:
            for r in runs[run_lbl]["results"]:
                if needle in r["subject"]:
                    print(f"\n[{run_lbl}]")
                    print(f"  {r['summary'] or '(empty)'}")


if __name__ == "__main__":
    main()
