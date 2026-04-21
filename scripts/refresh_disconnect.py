#!/usr/bin/env python
"""Refresh src/gmail_search/gmail/disconnect_email_hosts.txt from the
upstream Disconnect tracking-protection list.

Run occasionally (say, quarterly). No runtime dependency on network —
the URL crawler's denylist reads the bundled file, so this just
updates the bundle.

Usage:
    python scripts/refresh_disconnect.py
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

UPSTREAM = "https://raw.githubusercontent.com/disconnectme/disconnect-tracking-protection/master/services.json"
TARGET = Path(__file__).resolve().parent.parent / "src" / "gmail_search" / "gmail" / "disconnect_email_hosts.txt"


def main() -> int:
    print(f"fetching {UPSTREAM}")
    with urllib.request.urlopen(UPSTREAM, timeout=30) as resp:
        data = json.load(resp)

    hosts: set[str] = set()
    # We use ONLY the "Email" category — EmailAggressive flags broad
    # company domains (adobe.com, etc.) and false-positives real
    # content. 213 hosts in Email as of 2026-04-20.
    for entry in data.get("categories", {}).get("Email", []) or []:
        for _vendor, vdata in entry.items():
            for _url, value in vdata.items():
                if isinstance(value, list):
                    hosts.update(h.lower() for h in value)
    hosts.discard("dnt")  # Meta-field, not a host.

    if len(hosts) < 100:
        print(f"refusing to write: only {len(hosts)} hosts — upstream format may have changed", file=sys.stderr)
        return 1

    lines = [
        "# Email-tracker hosts sourced from disconnectme/disconnect-tracking-protection.",
        "# Regenerate with scripts/refresh_disconnect.py.",
        "# One host per line; blank lines and `#` comments are ignored.",
    ]
    lines.extend(sorted(hosts))
    TARGET.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(hosts)} hosts → {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
