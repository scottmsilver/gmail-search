"""Merge the 6 bench_out/*.json runs into one markdown comparison
(scripts/bench_out/COMPARISON.md) for side-by-side visual review.

Layout: for every message, show subject+from+body_len, then 6 rows
(one per config) with the produced summary and elapsed_ms. This
makes quality drift easy to spot — same input, 6 outputs stacked.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT_DIR = Path("scripts/bench_out")
LABELS = [
    "A_ctx2048_tok160",
    "A_ctx2048_tok300",
    "B_ctx4096_tok160",
    "B_ctx4096_tok300",
    "C_ctx8192_tok160",
    "C_ctx8192_tok300",
]


def main() -> None:
    runs = {lbl: json.loads((OUT_DIR / f"{lbl}.json").read_text()) for lbl in LABELS}

    # Index each run by message id for easy cross-lookup.
    by_id: dict[str, dict[str, dict]] = {}
    for lbl, run in runs.items():
        for r in run["results"]:
            by_id.setdefault(r["id"], {})[lbl] = r

    lines: list[str] = []
    lines.append("# Summary quality bench — 6-way comparison\n")
    lines.append("Sample: 40 messages, stratified across body-length buckets.\n")
    lines.append("## Totals\n")
    lines.append("| Label | ok | failed | total_s | avg ms/msg |")
    lines.append("|---|---|---|---|---|")
    for lbl in LABELS:
        t = runs[lbl]["totals"]
        avg_ms = int(1000 * t["total_seconds"] / max(t["ok"] + t["failed"], 1))
        lines.append(f"| {lbl} | {t['ok']} | {t['failed']} | {t['total_seconds']} | {avg_ms} |")
    lines.append("")

    # Order messages by bucket then body length to make skimming natural.
    any_run = runs[LABELS[0]]["results"]
    bucket_order = {"short": 0, "medium": 1, "long": 2, "very_long": 3}
    ordered = sorted(any_run, key=lambda r: (bucket_order.get(r["bucket"], 99), r["body_len"]))

    lines.append("## Per-message summaries\n")
    for ref in ordered:
        mid = ref["id"]
        per = by_id[mid]
        subj = (ref["subject"] or "(no subject)").replace("|", "\\|")
        frm = (ref["from"] or "").replace("|", "\\|")
        lines.append(f"### [{ref['bucket']}] {subj}  —  `{mid}`")
        lines.append(f"- From: `{frm}`  ·  body_len={ref['body_len']}")
        lines.append("")
        lines.append("| Config | Summary | ms |")
        lines.append("|---|---|---|")
        for lbl in LABELS:
            r = per.get(lbl, {})
            s = r.get("summary") or (r.get("error") or "")
            s = s.replace("\n", " ").replace("|", "\\|")
            lines.append(f"| {lbl} | {s} | {r.get('elapsed_ms', '')} |")
        lines.append("")

    md_path = OUT_DIR / "COMPARISON.md"
    md_path.write_text("\n".join(lines))
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
