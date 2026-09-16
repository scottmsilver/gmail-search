"""Resolve final-answer email citations without a model call.

Resolution proves that a source exists in the session owner's mailbox; it
cannot prove that the source supports the claim next to the citation.
"""

from __future__ import annotations

import re

from gmail_search.agents.session import session_owner

_REF = re.compile(r"\[ref:([^\]\n]+)\]")
_CODE_REF = re.compile(r"(`+)(\[ref:[^\]\n]+\])\1")


def normalize_citations(conn, session_id: str, text: str) -> str:
    """Map message IDs to owned thread IDs and mark unresolved sources.

    Exact thread IDs take precedence over message IDs. Missing owners never
    trigger a mailbox query. Resolve in batches to bound SQL argument sizes.
    """
    refs = sorted({m.group(1).strip() for m in _REF.finditer(text)})
    if not refs:
        return text
    owner = session_owner(conn, session_id)
    messages: dict[str, str] = {}
    threads: set[str] = set()
    if owner:
        for start in range(0, len(refs), 500):
            batch = refs[start:start + 500]
            rows = conn.execute(
                "SELECT id, thread_id FROM messages WHERE user_id = %s "
                "AND (id = ANY(%s) OR thread_id = ANY(%s))",
                (owner, batch, batch),
            ).fetchall()
            for row in rows:
                if row["thread_id"]:
                    threads.add(row["thread_id"])
                    messages[row["id"]] = row["thread_id"]

    def replace(match: re.Match) -> str:
        ref = match.group(1).strip()
        target = ref if ref in threads else messages.get(ref)
        return f"[ref:{target}]" if target else "[source unavailable]"

    # Models sometimes put a citation alone in inline code, preventing the UI
    # Markdown renderer from turning it into a clickable source chip.
    text = _CODE_REF.sub(lambda m: m.group(2), text)
    return _REF.sub(replace, text)
