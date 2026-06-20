"""Family A — proposition (atomic-fact) extraction prototype.

A standalone, self-contained implementation of the Dense X Retrieval idea
(arXiv 2312.06648), adapted for email. Each message is decomposed into atomic,
self-contained facts ("propositions"); those facts are embedded and stored in
their own `propositions` table with a back-pointer to the source message.

This is DELIBERATELY isolated from the live search/index path: a separate
table, a separate brute-force retrieval, no wiring into `serve` or the ScaNN
index. The point is to measure proposition-recall ("DecompScore") on known
ground truth (the 5-car case) before committing to the architecture.

Pipeline: extract_propositions() -> store_propositions() -> find_facts().
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import httpx
import numpy as np

logger = logging.getLogger(__name__)

PROPOSITIONIZER_VERSION = "prop-v3"

# Email-tuned adaptation of the Dense X Retrieval Fig. 8 prompt. The extra
# rules vs the Wikipedia original: owner-aware coreference (so "my car" becomes
# the owner's car — the whole point of "find ALL MY plates"), quote/signature
# stripping, absolute-date grounding, and boilerplate suppression.
_PROP_SYSTEM = """You extract atomic, durable FACTS from one email for a PERSONAL retrieval index.
Output JSON: {"facts": ["...", ...]}. Return {"facts": []} when the email has no
durable personal facts (newsletters, marketing, OTP codes, automated notices that
say nothing specific about this person).

A good fact is:
- ATOMIC: one claim, cannot be split further.
- SELF-CONTAINED: understandable with zero context. Replace every pronoun and
  "this"/"that" with the full entity name. Resolve "I"/"me"/"my"/"we"/"our" to the
  mailbox owner - write their ACTUAL NAME from the OWNER line (e.g. write the
  person's real name like "Jane Smith"), never the word "owner" or any
  placeholder. Resolve "you"/"your" to the other named party.
- SPECIFIC and DURABLE: about this person's possessions, accounts, transactions,
  travel, health, family/relationships, commitments, or identifiers. Prefer facts
  that carry a concrete value: a name, date, amount, ID/number, address, license
  plate, or VIN.
- PLANS & DECISIONS: capture concrete arrangements, commitments, and decisions
  even when briefly stated (who will do what, who is meeting whom, agreed
  dates/places/prices) - these are durable even in a short reply.
- TIME-GROUNDED: rewrite relative dates ("next week") to absolute using EMAIL_DATE.
  Keep amounts, IDs, plates, and VINs verbatim.

DO NOT extract:
- General article/newsletter content (word definitions, news, tips, recipes).
- Marketing copy, promotions, or generic company boilerplate / contact footers.
- Meta-statements about the email itself ("this is a reply", "this is a digest",
  "an email was sent").
- Unsubscribe/legal/disclaimer text; content in quoted reply chains (lines
  starting with ">") or signatures."""


def _build_prop_user_prompt(
    *,
    owner: str,
    date: str,
    from_addr: str,
    to_addr: str,
    subject: str,
    body: str,
) -> str:
    return (
        f"OWNER: {owner}\n"
        f"EMAIL_DATE: {date}\n"
        f"FROM: {from_addr}   TO: {to_addr}   SUBJECT: {subject}\n"
        f"CONTENT:\n{body}"
    )


def owner_string() -> str:
    """`Name (email)` for the mailbox owner, read from env (no PII hardcoded).
    Set GMAIL_MCP_OAUTH_OWNER_EMAIL and GMAIL_PROP_OWNER_NAME in the environment."""
    email = os.environ.get("GMAIL_MCP_OAUTH_OWNER_EMAIL", "").strip()
    name = os.environ.get("GMAIL_PROP_OWNER_NAME", "").strip()
    if name and email:
        return f"{name} ({email})"
    return name or email or "the mailbox owner"


def ensure_table(conn) -> None:
    """Create the prototype `propositions` table + indexes if absent."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS propositions (
            id          BIGSERIAL PRIMARY KEY,
            user_id     TEXT NOT NULL,
            message_id  TEXT NOT NULL,
            thread_id   TEXT,
            text        TEXT NOT NULL,
            embedding   BYTEA,
            model       TEXT NOT NULL,
            date        TEXT,
            created_at  TIMESTAMPTZ DEFAULT now()
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_props_user ON propositions(user_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_props_msg ON propositions(message_id)")
    conn.commit()


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _parse_facts(raw: str) -> list[str]:
    """Pull the facts list out of the model's JSON, tolerating code fences."""
    if not raw:
        return []
    cleaned = _FENCE_RE.sub("", raw).strip()
    try:
        data = json.loads(cleaned)
    except ValueError:
        # Last resort: grab the first {...} blob.
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except ValueError:
            return []
    facts = data.get("facts") if isinstance(data, dict) else None
    if not isinstance(facts, list):
        return []
    return [f.strip() for f in facts if isinstance(f, str) and f.strip()]


def _clean_body(body: str, *, max_chars: int = 8000) -> str:
    """Strip quoted-reply lines and clamp length. The prompt also tells the
    model to ignore quotes/signatures; this just keeps the prompt bounded."""
    if not body:
        return ""
    kept = [ln for ln in body.splitlines() if not ln.lstrip().startswith(">")]
    out = "\n".join(kept).strip()
    return out[:max_chars]


def extract_propositions(
    client: httpx.Client,
    backend,
    *,
    owner: str,
    date: str,
    from_addr: str,
    to_addr: str,
    subject: str,
    body: str,
    max_tokens: int = 1200,
) -> list[str]:
    """One message -> list of atomic, self-contained facts (may be empty)."""
    raw = backend.chat(
        client,
        messages=[
            {"role": "system", "content": _PROP_SYSTEM},
            {
                "role": "user",
                "content": _build_prop_user_prompt(
                    owner=owner,
                    date=date,
                    from_addr=from_addr,
                    to_addr=to_addr,
                    subject=subject,
                    body=_clean_body(body),
                ),
            },
        ],
        max_tokens=max_tokens,
        json_format=True,
    )
    return _parse_facts(raw)


def store_propositions(
    conn,
    embedder,
    *,
    user_id: str,
    message_id: str,
    thread_id: str | None,
    date: str | None,
    facts: list[str],
) -> int:
    """Embed each fact and INSERT it. Returns the number stored."""
    from gmail_search.embed.client import embedding_to_blob

    if not facts:
        return 0
    vectors = embedder.embed_texts_batch(facts)
    model_tag = f"{embedder.model}+{PROPOSITIONIZER_VERSION}"
    for fact, vec in zip(facts, vectors):
        conn.execute(
            """INSERT INTO propositions (user_id, message_id, thread_id, text, embedding, model, date)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (user_id, message_id, thread_id, fact, embedding_to_blob(vec), model_tag, date),
        )
    conn.commit()
    return len(facts)


def backfill(
    conn,
    client,
    backend,
    embedder,
    *,
    user_id: str,
    bm25_query: str,
    limit: int = 60,
    owner: str | None = None,
) -> dict[str, int]:
    """Extract+store propositions for the messages matching `bm25_query`
    (ParadeDB `@@@`), newest first. Skips messages already done this run."""
    owner = owner or owner_string()
    rows = conn.execute(
        """SELECT id, thread_id, date, from_addr, to_addr, subject, body_text
           FROM messages
           WHERE user_id = %s AND id @@@ %s
           ORDER BY date DESC
           LIMIT %s""",
        (user_id, bm25_query, limit),
    ).fetchall()
    stats = {"messages": 0, "facts": 0, "errors": 0}
    for r in rows:
        stats["messages"] += 1
        try:
            facts = extract_propositions(
                client,
                backend,
                owner=owner,
                date=str(r["date"] or ""),
                from_addr=r["from_addr"] or "",
                to_addr=r["to_addr"] or "",
                subject=r["subject"] or "",
                body=r["body_text"] or "",
            )
            stats["facts"] += store_propositions(
                conn,
                embedder,
                user_id=user_id,
                message_id=r["id"],
                thread_id=r["thread_id"],
                date=str(r["date"] or ""),
                facts=facts,
            )
        except Exception:
            stats["errors"] += 1
            logger.exception("propositionize failed for message %s", r["id"])
    return stats


def ensure_processed_table(conn) -> None:
    """Marker table so the live daemon is idempotent: a message is reprocessed
    only until it has been successfully propositionized once."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS prop_processed (user_id text, message_id text, PRIMARY KEY(user_id, message_id))"
    )
    conn.commit()


def owner_string_for_user(conn, user_id: str) -> str:
    """`Name (email)` for a SPECIFIC user, resolved from the users table.

    The global owner_string() reads one env var, which is wrong in multi-tenant:
    a per-user daemon must attribute each user's "my X" facts to that user, not
    to a single hardcoded owner. Falls back to the env owner if the row is bare."""
    try:
        row = conn.execute("SELECT email, name FROM users WHERE id = %s", (user_id,)).fetchone()
    except Exception:
        conn.rollback()
        row = None
    if row:
        name = (row["name"] or "").strip()
        email = (row["email"] or "").strip()
        if name and email:
            return f"{name} ({email})"
        if name or email:
            return name or email
    return owner_string()


def propositionize_pending(
    conn,
    client,
    backend,
    embedder,
    *,
    user_id: str,
    owner: str,
    batch: int = 500,
) -> dict[str, int]:
    """One bounded pass over messages NOT yet propositionized for this user.

    Idempotent via prop_processed; per-message atomic replace (delete old facts +
    insert new + stamp marker in one commit) so find_facts never sees a gap. A
    message that fails extraction is left unstamped to retry next pass. Designed
    to run in its own daemon — it never touches the latency-sensitive ingest path."""
    rows = conn.execute(
        """SELECT m.id, m.thread_id, m.date, m.from_addr, m.to_addr, m.subject, m.body_text
           FROM messages m
           WHERE m.user_id = %s
             AND NOT EXISTS (
               SELECT 1 FROM prop_processed pp WHERE pp.user_id = m.user_id AND pp.message_id = m.id
             )
           ORDER BY m.date DESC
           LIMIT %s""",
        (user_id, batch),
    ).fetchall()
    stats = {"messages": 0, "facts": 0, "errors": 0}
    for r in rows:
        stats["messages"] += 1
        try:
            facts = extract_propositions(
                client,
                backend,
                owner=owner,
                date=str(r["date"] or ""),
                from_addr=r["from_addr"] or "",
                to_addr=r["to_addr"] or "",
                subject=r["subject"] or "",
                body=r["body_text"] or "",
            )
            vectors = embedder.embed_texts_batch(facts) if facts else []
            # Atomic replace + marker in a single transaction.
            conn.execute("DELETE FROM propositions WHERE user_id = %s AND message_id = %s", (user_id, r["id"]))
            if facts:
                from gmail_search.embed.client import embedding_to_blob

                model_tag = f"{embedder.model}+{PROPOSITIONIZER_VERSION}"
                for fact, vec in zip(facts, vectors):
                    conn.execute(
                        """INSERT INTO propositions (user_id, message_id, thread_id, text, embedding, model, date)
                           VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                        (
                            user_id,
                            r["id"],
                            r["thread_id"],
                            fact,
                            embedding_to_blob(vec),
                            model_tag,
                            str(r["date"] or ""),
                        ),
                    )
            conn.execute(
                "INSERT INTO prop_processed (user_id, message_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (user_id, r["id"]),
            )
            conn.commit()
            stats["facts"] += len(facts)
        except Exception:
            conn.rollback()
            stats["errors"] += 1
            logger.exception("propositionize_pending failed for message %s", r["id"])
    return stats


# Stopwords for the BM25/lexical query path. This is the canonical NLTK / Snowball
# English stopword set (the standard linguistic list — not a bespoke hand-curated
# one), vendored as a constant so there is no runtime NLTK dependency. It is the
# right tool *only* here on the keyword side: it includes interrogatives and
# possessives (what/do/own/my/how/who) that broke "what cars do I own", which
# neither ParadeDB's minimal built-in list nor corpus document-frequency can catch
# (those words are rarer in declarative facts than real content words). It is
# deliberately NOT applied to the embedding query, where dropping function words
# would corrupt meaning (e.g. "to be or not to be" -> "be be").
_STOP = frozenset(
    (
        "i me my myself we our ours ourselves you you're you've you'll you'd your yours "
        "yourself yourselves he him his himself she she's her hers herself it it's its "
        "itself they them their theirs themselves what which who whom this that that'll "
        "these those am is are was were be been being have has had having do does did "
        "doing a an the and but if or because as until while of at by for with about "
        "against between into through during before after above below to from up down in "
        "out on off over under again further then once here there when where why how all "
        "any both each few more most other some such no nor not only own same so than too "
        "very s t can will just don don't should should've now d ll m o re ve y ain aren "
        "aren't couldn couldn't didn didn't doesn doesn't hadn hadn't hasn hasn't haven "
        "haven't isn isn't ma mightn mightn't mustn mustn't needn needn't shan shan't "
        "shouldn shouldn't wasn wasn't weren weren't won won't wouldn wouldn't"
    ).split()
)


def _singularize(tok: str) -> str | None:
    """Best-effort rule-based singular of a query token, or None when no distinct
    singular applies. Stdlib-only (the project formatter strips unused imports, so
    no NLTK/inflect dependency). Conservative: guards short tokens and "...ss"
    (so "address" is never mangled to "addres")."""
    if len(tok) <= 3 or tok.endswith("ss"):
        return None
    if tok.endswith("ies"):
        return tok[:-3] + "y"
    if tok.endswith(("ches", "shes", "xes", "ses", "zes")):
        return tok[:-2]
    if tok.endswith("s"):
        return tok[:-1]
    return None


def _query_terms(query: str) -> list[str]:
    """Discriminative lowercase tokens for the BM25 path: alphanumerics minus
    stopwords. The single source of truth for query tokenization."""
    return [t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 1 and t not in _STOP]


def ensure_bm25_index(conn) -> None:
    """ParadeDB BM25 index over proposition text — the keyword half of hybrid.
    This is what rescues bare alphanumeric facts (plates/VINs/codes) that
    embeddings rank near-zero."""
    conn.execute(
        "CREATE INDEX IF NOT EXISTS props_bm25_idx ON propositions USING bm25 (id, text) WITH (key_field='id')"
    )
    conn.commit()


def _bm25_ids(conn, user_id: str, query: str, limit: int) -> list[int]:
    """IDs of propositions whose text matches the query terms, BM25-ranked.
    A bare `@@@ 'license plate'` parses against the key_field (id), so we
    field-qualify each token against `text` (the same gotcha as messages)."""
    toks = _query_terms(query)
    if not toks:
        return []
    # Match each content token and its singular variant, so "cars" also recalls
    # facts phrased with "car". Bare alphanumerics (plates/VINs) get no variant.
    groups = []
    for t in toks:
        sing = _singularize(t)
        if sing and sing != t and sing not in _STOP:
            groups.append(f"(text:{t} OR text:{sing})")
        else:
            groups.append(f"text:{t}")
    pq = " OR ".join(groups)
    try:
        rows = conn.execute(
            """SELECT id FROM propositions
               WHERE user_id = %s AND id @@@ %s
               ORDER BY paradedb.score(id) DESC LIMIT %s""",
            (user_id, pq, limit),
        ).fetchall()
    except Exception:
        conn.rollback()
        logger.exception("bm25 proposition search failed (is props_bm25_idx built?)")
        return []
    return [r["id"] for r in rows]


def cluster_duplicates(conn, user_id: str, *, dims: int, threshold: float = 0.92) -> list[list[dict]]:
    """Group near-duplicate facts (same fact restated across many emails) by
    embedding cosine >= threshold, via connected components. Returns a list of
    clusters; each cluster is a list of {id, text, message_id} rows, the first
    being the canonical (longest = most complete). Singletons are length-1.

    This is the dedup half of entity resolution: it collapses "Scott is a
    Dartmouth alum" x5 into one. Brute-force O(n^2) — fine for the prototype's
    scale; production would block via the ScaNN index first."""
    rows = conn.execute(
        "SELECT id, text, message_id, embedding FROM propositions WHERE user_id = %s",
        (user_id,),
    ).fetchall()
    n = len(rows)
    if n == 0:
        return []
    mat = np.zeros((n, dims), dtype=np.float32)
    for i, r in enumerate(rows):
        mat[i] = np.frombuffer(r["embedding"], dtype=np.float32)[:dims]
    mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
    sims = mat @ mat.T

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in np.argwhere(np.triu(sims >= threshold, k=1)):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    clusters: list[list[dict]] = []
    for members in groups.values():
        members.sort(key=lambda i: len(rows[i]["text"]), reverse=True)  # canonical first
        clusters.append(
            [{"id": rows[i]["id"], "text": rows[i]["text"], "message_id": rows[i]["message_id"]} for i in members]
        )
    clusters.sort(key=len, reverse=True)
    return clusters


def find_facts(
    conn,
    embedder,
    *,
    user_id: str,
    query: str,
    exhaustive: bool = True,
    floor: float = 0.5,
    cap: int = 500,
    hybrid: bool = True,
    rrf_k: int = 60,
    max_load: int = 8000,
    owner_boost: float = 0.02,
) -> list[dict[str, Any]]:
    """Hybrid (semantic ∪ keyword) retrieval over this user's propositions.

    Semantic gives fuzzy recall; BM25 over fact text gives exact-term / ID
    recall (plates, VINs, codes — which embeddings can't rank). In exhaustive
    mode the candidate set is {cosine >= floor} ∪ {BM25 matches}, fused for
    ordering by Reciprocal Rank Fusion. `hybrid=False` is the pure-semantic
    baseline (for A/B).

    Memory-bounded for scale: when the user has more than `max_load` facts, we
    load embeddings only for the BM25 candidate pool (the semantic re-rank set)
    instead of the whole table — so per-call memory is O(candidates), not O(all
    facts), and it scales to the full mailbox. Below `max_load` we still
    brute-force the whole table for maximum recall (no regression at small
    scale). The cost at scale: a pure-semantic match with no keyword overlap
    won't be in the candidate pool — acceptable for enumerate-style queries
    where the entity term is in the fact; if that recall matters, a dedicated
    ScaNN index over propositions is the next step."""
    dims = embedder.dimensions
    bm_ids_full = _bm25_ids(conn, user_id, query, cap * 4) if hybrid else []
    total = conn.execute("SELECT count(*) FROM propositions WHERE user_id = %s", (user_id,)).fetchone()[0]
    if total <= max_load:
        rows = conn.execute(
            "SELECT id, message_id, thread_id, text, embedding FROM propositions WHERE user_id = %s",
            (user_id,),
        ).fetchall()
    elif bm_ids_full:
        # Bounded: re-rank only the BM25 candidate pool.
        rows = conn.execute(
            "SELECT id, message_id, thread_id, text, embedding FROM propositions WHERE user_id = %s AND id = ANY(%s)",
            (user_id, bm_ids_full),
        ).fetchall()
    else:
        # Large table + no keyword anchor: can't bound-load without an ANN index.
        logger.warning("find_facts: %d facts, no BM25 anchor for %r — needs a ScaNN props index", total, query)
        return []
    if not rows:
        return []
    ids = [r["id"] for r in rows]
    by_id = {r["id"]: r for r in rows}
    mat = np.zeros((len(rows), dims), dtype=np.float32)
    for i, r in enumerate(rows):
        mat[i] = np.frombuffer(r["embedding"], dtype=np.float32)[:dims]
    # NB: embed the FULL natural-language query here on purpose — stopword
    # stripping is only for the BM25 lexical path (_query_terms). The embedding
    # benefits from the full phrasing; do not "fix" this to use _query_terms.
    qv = np.asarray(embedder.embed_query(query), dtype=np.float32)[:dims]
    mat_n = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
    qn = qv / (np.linalg.norm(qv) + 1e-8)
    cos = mat_n @ qn

    sem_order = np.argsort(-cos)
    sem_rank = {ids[int(idx)]: rank for rank, idx in enumerate(sem_order, 1)}
    sem_score = {ids[int(idx)]: float(cos[int(idx)]) for idx in sem_order}

    bm_ids = bm_ids_full
    bm_rank = {pid: rank for rank, pid in enumerate(bm_ids, 1)}

    # Owner-scoping: boost facts that are ABOUT the mailbox owner so "my X"
    # ranks the owner's facts above third parties (e.g. a neighbor's plate).
    # Soft boost, not a filter — owner-relevant facts that don't name the owner
    # aren't dropped. Owner identity from owner_string() (env-configured).
    owner_terms: list[str] = []
    if owner_boost > 0:
        os_ = owner_string().lower()
        full = re.sub(r"\s*\(.*\)", "", os_).strip()  # "scott silver"
        m = re.search(r"\(([^@()]+)@", os_)  # email local part
        local = m.group(1) if m else ""
        owner_terms = [t for t in (full, local) if len(t) >= 4]
    owner_pids = set()
    if owner_terms:
        for pid in set(ids) | set(bm_ids):
            r = by_id.get(pid)
            if r and any(t in r["text"].lower() for t in owner_terms):
                owner_pids.add(pid)

    if exhaustive:
        candidates = {pid for pid in ids if sem_score.get(pid, 0.0) >= floor} | set(bm_ids)
    else:
        top_sem = [ids[int(i)] for i in sem_order[:20]]
        candidates = set(top_sem) | set(bm_ids[:20])

    def _rrf(pid: int) -> float:
        s = 0.0
        if pid in sem_rank:
            s += 1.0 / (rrf_k + sem_rank[pid])
        if pid in bm_rank:
            s += 1.0 / (rrf_k + bm_rank[pid])
        if pid in owner_pids:
            s += owner_boost
        return s

    out: list[dict[str, Any]] = []
    seen_text: set[str] = set()
    for pid in sorted(candidates, key=_rrf, reverse=True):
        r = by_id[pid]
        if r["text"] in seen_text:
            continue
        seen_text.add(r["text"])
        out.append(
            {
                "fact": r["text"],
                "message_id": r["message_id"],
                "thread_id": r["thread_id"],
                "cosine": round(sem_score.get(pid, 0.0), 4),
                "bm25": pid in bm_rank,
                "owner": pid in owner_pids,
            }
        )
        if len(out) >= cap:
            logger.warning("find_facts hit cap=%d — results may be incomplete", cap)
            break
    return out
