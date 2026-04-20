"""Pick 40 diverse messages for the summary-quality bench.

Selection strategy — we want the bench to exercise the failure modes
that matter most:

- 10 short bodies (< 500 chars) — personal notes, replies, calendar
  pings. Quality risk: over-summarizing trivial content.
- 10 medium bodies (1K-4K chars) — typical business mail. The sweet
  spot for the model.
- 10 long bodies (4K-12K chars) — threads, newsletters with real
  content. Where truncation starts mattering for 2048-context.
- 10 very long bodies (> 12K chars) — where only the 8192 config can
  see the whole thing. Quality risk: missing tail-of-document info.

We exclude auto-classified mail (promotions/social/forums labels)
since those bypass the LLM entirely — including them would just show
6 identical columns.
"""

from __future__ import annotations

import json
import random
import sqlite3
from pathlib import Path

DB = Path("data/gmail_search.db")
OUT = Path("scripts/bench_sample.json")

# Deterministic sample across runs.
random.seed(42)

BUCKETS = [
    ("short", 20, 500, 10),
    ("medium", 500, 4000, 10),
    ("long", 4000, 12000, 10),
    ("very_long", 12000, 200000, 10),
]

AUTO_LABELS = ("CATEGORY_PROMOTIONS", "CATEGORY_SOCIAL", "CATEGORY_FORUMS")


def _excluded_by_labels(labels_json: str) -> bool:
    try:
        labels = set(json.loads(labels_json or "[]"))
    except (TypeError, ValueError):
        return False
    return any(lbl in labels for lbl in AUTO_LABELS)


def main() -> None:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row

    chosen: list[dict] = []
    for bucket_name, lo, hi, want in BUCKETS:
        rows = conn.execute(
            """
            SELECT id, from_addr, subject, body_text, labels, date
            FROM messages
            WHERE LENGTH(body_text) >= ? AND LENGTH(body_text) < ?
            ORDER BY RANDOM()
            LIMIT ?
            """,
            (lo, hi, want * 4),  # oversample, filter autos
        ).fetchall()

        filtered = [r for r in rows if not _excluded_by_labels(r["labels"])][:want]
        for r in filtered:
            chosen.append(
                {
                    "bucket": bucket_name,
                    "id": r["id"],
                    "from_addr": r["from_addr"] or "",
                    "subject": r["subject"] or "",
                    "body_text": r["body_text"] or "",
                    "labels_json": r["labels"] or "[]",
                    "date": r["date"],
                    "body_len": len(r["body_text"] or ""),
                }
            )
        print(f"{bucket_name}: {len(filtered)}/{want} after auto-label filter")

    OUT.write_text(json.dumps(chosen, indent=2))
    print(f"wrote {len(chosen)} messages to {OUT}")


if __name__ == "__main__":
    main()
