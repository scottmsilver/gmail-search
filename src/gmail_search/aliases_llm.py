"""LLM-based term-alias discovery — the replacement for the co-occurrence
counting pipeline (`rebuild_term_aliases` in store/db.py).

Motivation (2026-07-07 eval, see docs/notes/ALIASES_LLM_2026-07-07.md):
the cooc pipeline kept 9.7 GB of pair-count run files (71% singletons that
can never clear its own min_occurrences=3 bar) to derive ~1.4k aliases that
blind judges scored as mostly noise. Asking an LLM directly — one call per
caps-token with real email snippets as context — produces a better table
for ~$0.11 per full rebuild and zero persistent state.

The eval's decisive finding is encoded here as the CAPS-DOMINANCE GATE:
a correct expansion for an ambiguous token is WORSE than no expansion
("net" -> Microsoft .NET redirected net-metering queries; "star" ->
Star Alliance hijacked stargazing). Query-time expansion cannot tell
"NET the acronym" from "net" the word, so only tokens that appear
predominantly in ALL-CAPS form in the corpus are allowed to expand.
End-to-end blind judging: ungated LLM table LOST 12-25 vs the cooc
baseline; caps-gated it WON 25-11.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from gmail_search.store.db import get_connection

logger = logging.getLogger(__name__)

# Vocabulary gates. df floor drops one-off tokens; the caps ratio is the
# ambiguity gate described in the module docstring; length matches the
# cooc pipeline's abbreviation regex.
MIN_DF = 5
MIN_CAPS_RATIO = 0.7
MAX_VOCAB = 4000
MAX_CONTEXTS = 3
CONTEXT_CHARS = 220
MIN_CONFIDENCE = 0.5
LLM_CONCURRENCY = 16

_WORD = re.compile(r"[A-Za-z]{2,5}\b")

_SYSTEM = """You analyze one user's personal email corpus. Given a short ALL-CAPS token and
snippets of real emails where it appears, decide whether the token is used as an
abbreviation / acronym / shorthand IN THIS MAILBOX, and if so, what it stands for.

Rules:
- Expansions must be supported by the snippets or be unambiguous common knowledge
  (e.g. HOA = homeowners association). Do not guess.
- A common English word merely written in caps (marketing shouting, subject-line caps)
  is NOT an abbreviation: is_abbreviation=false.
- A brand name that doesn't stand for anything useful for search (IKEA, LEGO) -> false.
- Expansions are lowercase phrases a search engine could add to a query (max 3, best first).
Reply with STRICT JSON only: {"is_abbreviation": bool, "expansions": ["..."], "confidence": 0.0-1.0}"""


def _scan_vocabulary(conn, uid: str) -> dict[str, dict]:
    """One streaming pass over messages: per lowercase token, ALL-CAPS doc
    frequency, other-case doc frequency, and up to MAX_CONTEXTS snippets
    from messages where it appeared in caps."""
    vocab: dict[str, dict] = {}
    last_id = ""
    while True:
        rows = conn.execute(
            "SELECT id, subject, body_text FROM messages WHERE user_id = %s AND id > %s ORDER BY id LIMIT 5000",
            (uid, last_id),
        ).fetchall()
        if not rows:
            return vocab
        last_id = rows[-1]["id"]
        for r in rows:
            text = f"{r['subject'] or ''}\n{r['body_text'] or ''}"
            caps_here: dict[str, int] = {}
            other_here: set[str] = set()
            for m in _WORD.finditer(text):
                w = m.group(0)
                k = w.lower()
                if w.isupper():
                    caps_here.setdefault(k, m.start())
                else:
                    other_here.add(k)
            for k, pos in caps_here.items():
                ent = vocab.setdefault(k, {"caps_df": 0, "other_df": 0, "contexts": []})
                ent["caps_df"] += 1
                if len(ent["contexts"]) < MAX_CONTEXTS:
                    lo = max(0, pos - CONTEXT_CHARS // 2)
                    snippet = text[lo : lo + CONTEXT_CHARS].replace("\n", " ").strip()
                    ent["contexts"].append(f"[subject: {(r['subject'] or '')[:90]}] …{snippet}…")
            # A message counts toward other_df even when it ALSO used the
            # caps form: "NET alert" over a body full of "net metering" is
            # evidence of ambiguity, which is exactly what the gate measures.
            for k in other_here:
                ent = vocab.setdefault(k, {"caps_df": 0, "other_df": 0, "contexts": []})
                ent["other_df"] += 1


def _ask_llm(backend, client, token: str, meta: dict) -> Optional[dict]:
    ctx = "\n".join(f"- {c}" for c in meta["contexts"])
    user = f'Token: "{token.upper()}" (appears in {meta["caps_df"]} emails)\n\nEmail snippets:\n{ctx}'
    try:
        content = backend.chat(
            client,
            [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
            max_tokens=200,
            json_format=True,
        )
        content = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        out = json.loads(content)
    except Exception as e:  # noqa: BLE001 — one token failing must not sink the rebuild
        logger.warning("alias llm call failed for %r: %s", token, e)
        return None
    if not out.get("is_abbreviation"):
        return None
    expansions = [e for e in (_sanitize_expansion(x) for x in out.get("expansions") or []) if e][:3]
    if not expansions or float(out.get("confidence", 0)) < MIN_CONFIDENCE:
        return None
    return {"expansions": expansions, "confidence": float(out.get("confidence", 0))}


# Expansions are appended verbatim to user queries BEFORE parse_query(), and
# the LLM read attacker-controllable email content — so a poisoned expansion
# could smuggle in query operators ("from:evil@x.com", "after:2020") that
# hijack any search containing the alias token. Deterministic gate, not a
# model judgment: plain lowercase word-phrases only.
_SAFE_EXPANSION = re.compile(r"^[a-z0-9][a-z0-9 &'\-\.]{0,39}$")


def _sanitize_expansion(raw: object) -> Optional[str]:
    s = str(raw).lower().strip()
    if not _SAFE_EXPANSION.fullmatch(s):
        return None
    words = s.split()
    if not 1 <= len(words) <= 4:
        return None
    return s


def _phrase_in_corpus(conn, phrase: str, uid: str) -> bool:
    """Does the phrase literally occur in the user's mail? Probes the
    ParadeDB BM25 index with a Tantivy quoted-phrase query (~75 ms), so the
    ~7k probes of a full rebuild stay in the minutes. Fail-open: if the BM25
    index is unavailable (fresh install, tests), grounding is skipped rather
    than silently dropping every alias."""
    # Tantivy needs field-qualified queries on this index (bare terms match
    # nothing); phrase-quote against both text fields.
    quoted = '"' + phrase.replace('"', " ") + '"'
    tantivy_query = f"subject:{quoted} OR body_text:{quoted}"
    try:
        row = conn.execute(
            "SELECT 1 FROM messages WHERE messages @@@ %s AND user_id = %s LIMIT 1",
            (tantivy_query, uid),
        ).fetchone()
    except Exception as e:  # noqa: BLE001 — no index -> no grounding, not no aliases
        # Roll back: a failed probe leaves the connection in an aborted
        # transaction, which would poison every later statement on it.
        try:
            conn.rollback()
        except Exception:  # noqa: BLE001
            pass
        logger.warning("alias grounding probe unavailable (%s); keeping expansion unverified", e)
        return True
    return row is not None


def rebuild_term_aliases_llm(
    db_path: Path,
    data_dir: Optional[Path] = None,
    *,
    user_id: Optional[str] = None,
) -> int:
    """Regenerate the user's term_aliases table via the LLM pipeline.

    Full regeneration every call — at ~$0.11 and a few minutes per run there
    is nothing worth caching between runs, so there is no watermark and no
    on-disk state. On success, the old cooc pipeline's run files (which can
    reach ~10 GB) are deleted. Requires OPENROUTER_KEY; without it the
    function logs and leaves the existing table untouched (returns -1) so a
    reindex never destroys a working alias table over a missing credential.
    """
    import httpx
    from gmail_search.auth.write_user import resolve_write_user_id
    from gmail_search.llm.openrouter import OpenRouterBackend

    try:
        backend = OpenRouterBackend()
    except RuntimeError as e:
        logger.warning("alias rebuild skipped: %s (existing aliases left in place)", e)
        return -1

    conn = get_connection(db_path)
    uid = resolve_write_user_id(conn, user_id=user_id)

    vocab = _scan_vocabulary(conn, uid)
    candidates = sorted(
        (
            (t, m)
            for t, m in vocab.items()
            if m["caps_df"] >= MIN_DF and m["caps_df"] / max(m["caps_df"] + m["other_df"], 1) >= MIN_CAPS_RATIO
        ),
        key=lambda kv: -kv[1]["caps_df"],
    )[:MAX_VOCAB]
    logger.info(
        "alias rebuild: %d caps tokens pass df>=%d + caps-ratio>=%.1f gates", len(candidates), MIN_DF, MIN_CAPS_RATIO
    )

    aliases: dict[str, dict] = {}
    with httpx.Client() as client:
        with ThreadPoolExecutor(LLM_CONCURRENCY) as ex:
            futs = {ex.submit(_ask_llm, backend, client, t, m): t for t, m in candidates}
            for f in as_completed(futs):
                out = f.result()
                if out:
                    aliases[futs[f]] = out

    # CORPUS GROUNDING: an expansion must occur somewhere in the user's own
    # mail as a phrase. This deterministically kills common-knowledge
    # hallucinations — with weak context snippets the model can guess e.g.
    # WPC = "world peace council" for a mailbox where WPC is a property name;
    # that phrase appears zero times in the corpus, so it dies here, while
    # the true expansion survives by definition (people spell out their own
    # jargon eventually).
    grounded: dict[str, dict] = {}
    dropped_exp = 0
    for term, v in aliases.items():
        kept = [e for e in v["expansions"] if _phrase_in_corpus(conn, e, uid)]
        dropped_exp += len(v["expansions"]) - len(kept)
        if kept:
            grounded[term] = {"expansions": kept, "confidence": v["confidence"]}
    logger.info(
        "alias rebuild: corpus grounding kept %d/%d terms (dropped %d ungrounded expansions)",
        len(grounded),
        len(aliases),
        dropped_exp,
    )
    aliases = grounded

    # Replace this user's rows in one transaction — readers see old or new,
    # and an insert failure rolls back rather than leaving an empty table.
    try:
        conn.execute("DELETE FROM term_aliases WHERE user_id = %s", (uid,))
        for term, v in aliases.items():
            conn.execute(
                "INSERT INTO term_aliases (term, expansions, similarity, user_id) VALUES (%s, %s, %s, %s)",
                (term, json.dumps(v["expansions"]), v["confidence"], uid),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    logger.info("alias rebuild: wrote %d LLM-derived aliases for %s", len(aliases), uid)

    # Retire the cooc pipeline's on-disk state — the WHOLE aliases dir
    # including meta.json. Leaving meta.json behind would make a later
    # alias_backend: cooc revert resume from a stale watermark with the
    # run files gone (silent partial rebuild); with the dir removed the
    # cooc path bootstraps cold, which is correct.
    if data_dir is not None:
        import shutil

        d = Path(data_dir) / "users" / uid / "aliases"
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
            logger.info("alias rebuild: removed retired cooc state %s", d)

    return len(aliases)
