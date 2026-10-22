"""Cross-session retrieval memory ("pointers, not prose") for deep mode.

Extracts winning retrieval traces (`cited_thread_ids`, `cited_attachment_ids`,
`read_thread_ids`, `winning_queries`) from completed deep-mode turns,
gates eligibility via structural citation checks + JEV advisory verification
+ conversation superseding + battle/thumb-down exclusion, stores the result
in `agent_sessions.plan` (JSONB), and retrieves relevant prior leads for
subsequent turns.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from gmail_search.gateway.jev import JevConfig, answers_from, request_body

logger = logging.getLogger(__name__)

_REF_RE = re.compile(r"\[ref:([A-Za-z0-9_-]+)\]")
_ATT_RE = re.compile(r"\[att:(\d+)\]")
_WORD_RE = re.compile(r"[a-z0-9][a-z0-9._'-]{1,}")

_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "also",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "ask",
        "asked",
        "at",
        "back",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "check",
        "could",
        "date",
        "dates",
        "day",
        "days",
        "did",
        "do",
        "does",
        "doing",
        "done",
        "down",
        "during",
        "each",
        "email",
        "emails",
        "ever",
        "every",
        "find",
        "first",
        "for",
        "from",
        "further",
        "get",
        "give",
        "go",
        "got",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "know",
        "last",
        "latest",
        "let",
        "like",
        "list",
        "look",
        "looking",
        "made",
        "mail",
        "mails",
        "make",
        "many",
        "may",
        "me",
        "message",
        "messages",
        "might",
        "more",
        "most",
        "much",
        "must",
        "my",
        "myself",
        "need",
        "new",
        "next",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "one",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "please",
        "recent",
        "recently",
        "received",
        "same",
        "say",
        "said",
        "search",
        "see",
        "send",
        "sent",
        "she",
        "should",
        "show",
        "so",
        "some",
        "such",
        "Summary",
        "summary",
        "take",
        "tell",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "thread",
        "threads",
        "through",
        "time",
        "times",
        "to",
        "too",
        "total",
        "under",
        "until",
        "up",
        "us",
        "use",
        "used",
        "using",
        "very",
        "want",
        "was",
        "way",
        "we",
        "well",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "within",
        "without",
        "would",
        "year",
        "years",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
    }
)

_CORRECTION_RE = re.compile(
    r"^\s*(?:no\b[,\s]|that'?s\s+(?:not\s+right|wrong|incorrect)|that\s+is\s+(?:not\s+right|wrong|incorrect)|"
    r"wrong\b|incorrect\b|not\s+right\b|you\s+missed\b|actually[,\s]+that)",
    re.IGNORECASE,
)


def clean_question_text(question: str | None) -> str:
    """Strip multi-turn conversation preambles so we store and match on
    the turn's actual user question."""
    if not question:
        return ""
    marker = "# Latest user question (answer this)"
    if marker in question:
        question = question.split(marker, 1)[-1]
    battle_marker = "\nCurrent question:\n"
    if battle_marker in question:
        question = question.split(battle_marker, 1)[-1]
    return question.strip()


def is_correction_followup(question: str | None) -> bool:
    """Return True if a follow-up question appears to correct or reject
    the immediately preceding assistant turn."""
    cleaned = clean_question_text(question)
    if not cleaned:
        return False
    return bool(_CORRECTION_RE.search(cleaned))


def _coerce_plan_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {}


def normalize_tool_records(records: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Convert either `mcp_tool_call_full` event payloads (`{name, args, response}`)
    or `side_calls` (`[{name, args}, {name, response}]`) into a unified list of
    `{"name": str, "args": dict, "response": dict}` records."""
    if not records:
        return []
    out: list[dict[str, Any]] = []
    pending_args: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        name = str(rec.get("name") or "").strip()
        if not name:
            continue
        # Strip optional `gmail_` prefix if present.
        if name.startswith("gmail_"):
            name = name[len("gmail_") :]
        has_args = "args" in rec
        has_resp = "response" in rec
        if has_args and has_resp:
            out.append(
                {
                    "name": name,
                    "args": rec.get("args") if isinstance(rec.get("args"), dict) else {},
                    "response": rec.get("response") if isinstance(rec.get("response"), dict) else {},
                }
            )
        elif has_args and not has_resp:
            args = rec.get("args") if isinstance(rec.get("args"), dict) else {}
            pending_args.setdefault(name, []).append(args)
        elif has_resp and not has_args:
            resp = rec.get("response") if isinstance(rec.get("response"), dict) else {}
            queue = pending_args.get(name) or []
            args = queue.pop(0) if queue else {}
            out.append({"name": name, "args": args, "response": resp})
    for name, queue in pending_args.items():
        for args in queue:
            out.append({"name": name, "args": args, "response": {}})
    return out


def _threads_in_result_block(block: Any) -> set[str]:
    """Extract all `cite_ref` and `thread_id` strings from a tool result sub-tree."""
    tids: set[str] = set()
    if isinstance(block, dict):
        for k in ("cite_ref", "thread_id"):
            val = block.get(k)
            if isinstance(val, str) and val.strip():
                tids.add(val.strip())
        for v in block.values():
            if isinstance(v, (dict, list)):
                tids.update(_threads_in_result_block(v))
    elif isinstance(block, list):
        for item in block:
            if isinstance(item, (dict, list)):
                tids.update(_threads_in_result_block(item))
    return tids


def _compact_filter(flt: dict[str, Any]) -> dict[str, Any]:
    keep = ("from_addr", "to_addr", "subject_contains", "date_from", "date_to", "has_attachment", "label")
    return {k: flt[k] for k in keep if flt.get(k) not in (None, "", [])}


def extract_session_leads(
    question: str,
    final_answer: str | None,
    tool_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Extract structured retrieval leads from a completed session's
    `final_answer` and tool call records."""
    cleaned_q = clean_question_text(question)
    text = (final_answer or "").strip()

    cited_raw: list[str] = []
    for m in _REF_RE.finditer(text):
        ref = m.group(1).strip()
        if ref and ref not in cited_raw:
            cited_raw.append(ref)

    explicit_att_ids: list[int] = []
    for m in _ATT_RE.finditer(text):
        try:
            aid = int(m.group(1))
            if aid not in explicit_att_ids:
                explicit_att_ids.append(aid)
        except ValueError:
            pass

    normalized = normalize_tool_records(tool_records)
    seen_thread_ids: set[str] = set()
    read_thread_ids: list[str] = []
    attachment_to_thread: dict[int, str] = {}
    read_attachment_ids: list[int] = []

    for rec in normalized:
        name = rec["name"]
        args = rec["args"]
        resp = rec["response"]
        seen_thread_ids.update(_threads_in_result_block(resp))

        if name in ("get_thread_batch", "get_thread"):
            for tid in args.get("thread_ids") or []:
                if isinstance(tid, str) and tid.strip():
                    seen_thread_ids.add(tid.strip())
                    if tid.strip() not in read_thread_ids:
                        read_thread_ids.append(tid.strip())
            tid_single = args.get("thread_id")
            if isinstance(tid_single, str) and tid_single.strip():
                seen_thread_ids.add(tid_single.strip())
                if tid_single.strip() not in read_thread_ids:
                    read_thread_ids.append(tid_single.strip())

            results_list = resp.get("results") if isinstance(resp.get("results"), list) else [resp]
            for item in results_list:
                if not isinstance(item, dict):
                    continue
                res_obj = item.get("result") if isinstance(item.get("result"), dict) else item
                tid = res_obj.get("cite_ref") or item.get("thread_id") or res_obj.get("thread_id")
                if isinstance(tid, str) and tid.strip():
                    tid = tid.strip()
                    seen_thread_ids.add(tid)
                    if tid not in read_thread_ids:
                        read_thread_ids.append(tid)
                    for msg in res_obj.get("messages") or []:
                        if isinstance(msg, dict):
                            for att in msg.get("attachments") or []:
                                if isinstance(att, dict) and isinstance(att.get("id"), int):
                                    attachment_to_thread[int(att["id"])] = tid

        elif name in ("get_attachment_batch", "get_attachment"):
            items = args.get("items") if isinstance(args.get("items"), list) else [args]
            for it in items:
                if isinstance(it, dict):
                    raw_id = it.get("attachment_id")
                    if isinstance(raw_id, int) and not isinstance(raw_id, bool):
                        if raw_id not in read_attachment_ids:
                            read_attachment_ids.append(raw_id)
            for it in resp.get("results") or []:
                if isinstance(it, dict):
                    r_obj = it.get("result") if isinstance(it.get("result"), dict) else it
                    raw_id = r_obj.get("attachment_id")
                    if isinstance(raw_id, int) and not isinstance(raw_id, bool):
                        if raw_id not in read_attachment_ids:
                            read_attachment_ids.append(raw_id)

    if seen_thread_ids:
        verified_cited = [tid for tid in cited_raw if tid in seen_thread_ids]
    else:
        verified_cited = list(cited_raw)

    target_threads = set(verified_cited) or set(read_thread_ids)

    cited_attachment_ids: list[int] = list(explicit_att_ids)
    for aid in read_attachment_ids:
        if aid in cited_attachment_ids:
            continue
        owner_tid = attachment_to_thread.get(aid)
        if owner_tid is None or owner_tid in target_threads:
            cited_attachment_ids.append(aid)

    winning_queries: list[dict[str, Any]] = []
    fallback_queries: list[dict[str, Any]] = []
    seen_q_sigs: set[str] = set()

    def _add_query(bucket: list[dict[str, Any]], entry: dict[str, Any]) -> None:
        sig = json.dumps(entry, sort_keys=True)
        if sig not in seen_q_sigs and len(bucket) < 6:
            seen_q_sigs.add(sig)
            bucket.append(entry)

    for rec in normalized:
        name = rec["name"]
        args = rec["args"]
        resp = rec["response"]
        if name == "search_emails_batch":
            items = resp.get("results") if isinstance(resp.get("results"), list) else []
            if items:
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    inp = item.get("input") if isinstance(item.get("input"), dict) else {}
                    q_str = str(inp.get("query") or "").strip()
                    if not q_str:
                        continue
                    entry: dict[str, Any] = {"tool": "search_emails", "query": q_str}
                    for dk in ("date_from", "date_to"):
                        if inp.get(dk):
                            entry[dk] = inp[dk]
                    item_threads = _threads_in_result_block(item.get("result"))
                    if target_threads and (item_threads & target_threads):
                        _add_query(winning_queries, entry)
                    else:
                        _add_query(fallback_queries, entry)
            else:
                for s in args.get("searches") or []:
                    if isinstance(s, dict) and str(s.get("query") or "").strip():
                        _add_query(fallback_queries, {"tool": "search_emails", "query": str(s["query"]).strip()})
        elif name == "search_emails":
            q_str = str(args.get("query") or "").strip()
            if q_str:
                entry = {"tool": "search_emails", "query": q_str}
                if target_threads and (_threads_in_result_block(resp) & target_threads):
                    _add_query(winning_queries, entry)
                else:
                    _add_query(fallback_queries, entry)
        elif name == "query_emails_batch":
            items = resp.get("results") if isinstance(resp.get("results"), list) else []
            if items:
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    inp = item.get("input") if isinstance(item.get("input"), dict) else {}
                    c_flt = _compact_filter(inp)
                    if not c_flt:
                        continue
                    entry = {"tool": "query_emails", "filter": c_flt}
                    item_threads = _threads_in_result_block(item.get("result"))
                    if target_threads and (item_threads & target_threads):
                        _add_query(winning_queries, entry)
                    else:
                        _add_query(fallback_queries, entry)
            else:
                for flt in args.get("filters") or []:
                    if isinstance(flt, dict) and _compact_filter(flt):
                        _add_query(fallback_queries, {"tool": "query_emails", "filter": _compact_filter(flt)})
        elif name == "query_emails":
            c_flt = _compact_filter(args)
            if c_flt:
                entry = {"tool": "query_emails", "filter": c_flt}
                if target_threads and (_threads_in_result_block(resp) & target_threads):
                    _add_query(winning_queries, entry)
                else:
                    _add_query(fallback_queries, entry)
        elif name == "find_facts":
            q_str = str(args.get("query") or "").strip()
            if q_str:
                entry = {"tool": "find_facts", "query": q_str}
                if target_threads and (_threads_in_result_block(resp) & target_threads):
                    _add_query(winning_queries, entry)
                else:
                    _add_query(fallback_queries, entry)

    if not winning_queries and fallback_queries and verified_cited:
        winning_queries = fallback_queries[:3]

    is_error_or_stub = (
        not text
        or text.startswith("⚠️ Deep analysis failed")
        or text.startswith("(stub)")
        or text.startswith("_Deep-mode")
    )
    eligible = bool(verified_cited) and not is_error_or_stub

    return {
        "question": cleaned_q[:500],
        "cited_thread_ids": verified_cited[:15],
        "cited_attachment_ids": cited_attachment_ids[:15],
        "read_thread_ids": read_thread_ids[:20],
        "winning_queries": winning_queries[:6],
        "eligible": eligible,
    }


def verify_answer_with_jev(
    question: str,
    final_answer: str,
    jev_config: JevConfig | None = None,
) -> tuple[bool, float | None]:
    """Advisory JEV check: verifies that `final_answer` substantively answers
    `question` with specific findings rather than reporting that nothing was
    found or failing. Returns `(is_substantive, answered_probability)`."""
    if jev_config is None:
        return True, None
    cleaned_q = clean_question_text(question)
    if not cleaned_q or not final_answer or not final_answer.strip():
        return False, 0.0
    try:
        body = request_body(
            {"question": cleaned_q[:1000], "answer": final_answer[:3000]},
            {
                "quality": {
                    "question": (
                        "Does the answer substantively answer the user's question with specific "
                        "findings from the mailbox, rather than reporting that nothing was found "
                        "or failing to answer?"
                    ),
                    "options": {
                        "answered": "Substantively answers the question with specific findings",
                        "unanswered": "Reports nothing found, refuses, or fails to answer",
                    },
                }
            },
        )
        with httpx.Client(trust_env=False, follow_redirects=False, timeout=2.0) as client:
            response = client.post(jev_config.url, headers=jev_config.headers, json=body)
            response.raise_for_status()
            answers = answers_from(response.json())
        probs = (answers.get("quality") or {}).get("probabilities") or {}
        ans_prob = float(probs.get("answered", 1.0))
        return ans_prob >= 0.5, round(ans_prob, 3)
    except Exception as exc:  # noqa: BLE001
        logger.debug("jev answer quality check skipped (%s)", type(exc).__name__)
        return True, None


def _update_session_plan(conn, session_id: str, plan: dict[str, Any]) -> None:
    conn.execute(
        "UPDATE agent_sessions SET plan = %s::jsonb WHERE id = %s",
        (json.dumps(plan), session_id),
    )


def record_session_leads(
    conn,
    *,
    session_id: str,
    conversation_id: str | None = None,
    question: str,
    final_answer: str | None,
    side_calls: list[dict[str, Any]] | None = None,
    jev_config: JevConfig | None = None,
    check_jev: bool = True,
) -> dict[str, Any]:
    """Extract leads for `session_id`, run optional JEV quality gate,
    persist into `agent_sessions.plan`, and mark earlier turns in the
    same `conversation_id` as superseded when appropriate."""
    leads = extract_session_leads(question, final_answer, side_calls)
    if leads["eligible"] and check_jev and jev_config is not None and final_answer:
        jev_ok, jev_prob = verify_answer_with_jev(question, final_answer, jev_config)
        leads["jev_answered_prob"] = jev_prob
        if not jev_ok:
            leads["eligible"] = False

    cur = conn.execute(
        "SELECT plan, conversation_id FROM agent_sessions WHERE id = %s",
        (session_id,),
    )
    row = cur.fetchone() if hasattr(cur, "fetchone") else None
    if row is None:
        return leads
    existing_plan = _coerce_plan_dict(row["plan"])
    resolved_conv_id = conversation_id or row["conversation_id"]

    # Respect any explicit thumb-down or battle-loser flag already on the row.
    if existing_plan.get("feedback") == "down" or existing_plan.get("battle_loser"):
        leads["eligible"] = False

    existing_plan["leads"] = leads
    _update_session_plan(conn, session_id, existing_plan)

    if resolved_conv_id and is_correction_followup(question):
        prior_rows = conn.execute(
            "SELECT id, plan FROM agent_sessions WHERE conversation_id = %s AND id != %s",
            (resolved_conv_id, session_id),
        ).fetchall()
        for pr in prior_rows:
            p_plan = _coerce_plan_dict(pr["plan"])
            if p_plan.get("superseded_by") != session_id:
                p_plan["superseded_by"] = session_id
                _update_session_plan(conn, pr["id"], p_plan)

    conn.commit()
    return leads


def set_session_feedback(
    conn,
    *,
    session_id: str,
    user_id: str,
    feedback: str | None,
) -> bool:
    """Record user feedback (`'up'`, `'down'`, or `None`) on an owned
    `agent_sessions` row inside `agent_sessions.plan`. Returns False if
    the session does not exist or is not owned by `user_id`."""
    if feedback not in ("up", "down", None):
        raise ValueError("feedback must be 'up', 'down', or None")
    row = conn.execute(
        "SELECT id, plan FROM agent_sessions WHERE id = %s AND user_id = %s",
        (session_id, user_id),
    ).fetchone()
    if row is None:
        return False
    plan = _coerce_plan_dict(row["plan"])
    plan["feedback"] = feedback
    leads = plan.get("leads")
    if isinstance(leads, dict):
        if feedback == "down":
            leads["eligible"] = False
        elif feedback == "up" and leads.get("cited_thread_ids"):
            leads["eligible"] = True
    _update_session_plan(conn, session_id, plan)
    conn.commit()
    return True


def get_session_feedback(conn, *, session_id: str, user_id: str) -> str | None:
    """Return `'up'`, `'down'`, or `None` for an owned session."""
    row = conn.execute(
        "SELECT plan FROM agent_sessions WHERE id = %s AND user_id = %s",
        (session_id, user_id),
    ).fetchone()
    if row is None:
        return None
    plan = _coerce_plan_dict(row["plan"])
    fb = plan.get("feedback")
    return fb if fb in ("up", "down") else None


def apply_battle_outcome_to_sessions(
    conn,
    *,
    user_id: str,
    winner: str,
    session_a: str | None,
    session_b: str | None,
) -> None:
    """Mark losing / `both_bad` battle sessions in `agent_sessions.plan`
    so their leads are never reused in future turns."""
    outcomes: list[tuple[str | None, bool, str | None]] = []
    if winner == "a":
        outcomes = [(session_a, False, "up"), (session_b, True, "down")]
    elif winner == "b":
        outcomes = [(session_a, True, "down"), (session_b, False, "up")]
    elif winner == "both_bad":
        outcomes = [(session_a, True, "down"), (session_b, True, "down")]
    for sid, is_loser, fb in outcomes:
        if not sid:
            continue
        row = conn.execute(
            "SELECT id, plan FROM agent_sessions WHERE id = %s AND user_id = %s",
            (sid, user_id),
        ).fetchone()
        if row is None:
            continue
        plan = _coerce_plan_dict(row["plan"])
        if is_loser:
            plan["battle_loser"] = True
        if fb:
            plan["feedback"] = fb
        if is_loser and isinstance(plan.get("leads"), dict):
            plan["leads"]["eligible"] = False
        _update_session_plan(conn, sid, plan)
    conn.commit()


def _tokenize(text: str) -> set[str]:
    tokens: set[str] = set()
    for m in _WORD_RE.finditer(text.lower()):
        tok = m.group(0).strip("._'-")
        if len(tok) >= 2 and tok not in _STOPWORDS:
            tokens.add(tok)
    return tokens


def _lead_tokens(question: str, leads: dict[str, Any]) -> set[str]:
    parts = [question]
    for wq in leads.get("winning_queries") or []:
        if isinstance(wq, dict):
            if wq.get("query"):
                parts.append(str(wq["query"]))
            flt = wq.get("filter")
            if isinstance(flt, dict):
                for v in flt.values():
                    if isinstance(v, str):
                        parts.append(v)
    return _tokenize(" ".join(parts))


def find_prior_leads(
    conn,
    *,
    user_id: str | None,
    question: str,
    exclude_session_id: str | None = None,
    exclude_conversation_id: str | None = None,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Find up to `limit` relevant prior verified session leads for `user_id`."""
    if not user_id:
        return []
    cleaned_q = clean_question_text(question)
    q_tokens = _tokenize(cleaned_q)
    if not q_tokens:
        return []

    try:
        cur = conn.execute(
            """SELECT id, conversation_id, question, plan, started_at, finished_at
               FROM agent_sessions
               WHERE user_id = %s
                 AND status = 'done'
                 AND plan IS NOT NULL
               ORDER BY started_at DESC
               LIMIT 250""",
            (user_id,),
        )
        if not hasattr(cur, "fetchall"):
            return []
        rows = cur.fetchall()
    except Exception as exc:  # noqa: BLE001
        logger.warning("find_prior_leads query failed: %s", exc)
        return []

    scored: list[tuple[float, frozenset[str], dict[str, Any]]] = []

    for r in rows:
        sid = r["id"]
        if exclude_session_id and sid == exclude_session_id:
            continue
        conv_id = r["conversation_id"]
        if exclude_conversation_id and conv_id and conv_id == exclude_conversation_id:
            continue
        plan = _coerce_plan_dict(r["plan"])
        if plan.get("feedback") == "down" or plan.get("battle_loser") or plan.get("superseded_by"):
            continue
        leads = plan.get("leads")
        if not isinstance(leads, dict) or not leads.get("eligible"):
            continue
        cited_tids = [str(t) for t in (leads.get("cited_thread_ids") or []) if t]
        if not cited_tids:
            continue

        cand_q = clean_question_text(leads.get("question") or r["question"] or "")
        c_tokens = _lead_tokens(cand_q, leads)
        if not c_tokens:
            continue

        shared = q_tokens & c_tokens
        if not shared:
            continue
        # Require either >= 2 shared tokens, or 1 shared token when the query
        # has <= 2 meaningful tokens (e.g. "91wpc drywall" or "Costello appraisal").
        if len(shared) < 2 and len(q_tokens) > 2:
            # Allow a single shared token only if it's a specific identifier
            # (contains digits or >= 6 chars) and overlap ratio >= 0.34.
            tok = next(iter(shared))
            has_digit_or_long = any(ch.isdigit() for ch in tok) or len(tok) >= 6
            if not has_digit_or_long or (len(shared) / len(q_tokens)) < 0.34:
                continue

        overlap = len(shared) / max(1, len(q_tokens))
        jaccard = len(shared) / max(1, len(q_tokens | c_tokens))
        score = 0.65 * overlap + 0.35 * jaccard + 0.05 * min(len(shared), 5)
        if plan.get("feedback") == "up":
            score += 0.15

        t_set = frozenset(cited_tids)
        as_of_raw = str(r["finished_at"] or r["started_at"] or "")
        as_of_date = as_of_raw[:10] if len(as_of_raw) >= 10 else "unknown"
        scored.append(
            (
                score,
                t_set,
                {
                    "session_id": sid,
                    "question": cand_q,
                    "as_of": as_of_date,
                    "cited_thread_ids": cited_tids[:8],
                    "cited_attachment_ids": [int(a) for a in (leads.get("cited_attachment_ids") or [])[:6]],
                    "winning_queries": (leads.get("winning_queries") or [])[:4],
                    "feedback": plan.get("feedback"),
                    "score": round(score, 3),
                },
            )
        )

    scored.sort(key=lambda item: item[0], reverse=True)
    deduped: list[dict[str, Any]] = []
    seen_thread_sets: set[frozenset[str]] = set()
    for _score, t_set, payload in scored:
        if t_set in seen_thread_sets:
            continue
        seen_thread_sets.add(t_set)
        deduped.append(payload)
        if len(deduped) >= limit:
            break
    return deduped


def _format_winning_query(wq: dict[str, Any]) -> str:
    tool = wq.get("tool")
    if tool == "search_emails" and wq.get("query"):
        return f'search_emails("{wq["query"]}")'
    if tool == "find_facts" and wq.get("query"):
        return f'find_facts("{wq["query"]}")'
    if tool == "query_emails" and isinstance(wq.get("filter"), dict):
        parts = [f"{k}={v!r}" for k, v in wq["filter"].items()]
        return f"query_emails({', '.join(parts)})"
    return ""


def format_prior_leads_guidance(leads: list[dict[str, Any]]) -> str | None:
    """Format matching prior leads into a compact system-prompt block."""
    if not leads:
        return None
    lines = [
        "# Prior Investigation Leads",
        "Earlier verified chats for this user found relevant sources for related questions. "
        "Treat these as high-value starting leads (not a completeness boundary):",
    ]
    for item in leads:
        q = (item.get("question") or "").replace("\n", " ").strip()
        as_of = item.get("as_of") or "unknown"
        tids = ", ".join(f"`{t}`" for t in (item.get("cited_thread_ids") or []))
        lines.append(f'- Prior question (as of {as_of}): "{q}"')
        if tids:
            lines.append(f"  Cited threads: {tids}")
        aids = item.get("cited_attachment_ids") or []
        if aids:
            lines.append(f"  Cited attachments: {', '.join(f'`{a}`' for a in aids)}")
        wqs = [s for s in (_format_winning_query(w) for w in (item.get("winning_queries") or [])) if s]
        if wqs:
            lines.append(f"  Winning queries: {', '.join(f'`{w}`' for w in wqs)}")
    lines.append(
        "Verify relevant cited threads directly with `gmail_get_thread_batch` "
        "(and `gmail_get_attachment_batch` for listed attachments) in your first turn, "
        "and run a targeted search for newer mail since the prior chat's date when recency "
        "or completeness matters."
    )
    return "\n".join(lines)


def backfill_session_leads(
    conn,
    *,
    user_id: str | None = None,
    check_jev: bool = False,
    jev_config: JevConfig | None = None,
) -> dict[str, int]:
    """Backfill `agent_sessions.plan` for existing completed sessions from
    their `agent_events` rows, apply conversation superseding, and apply
    any recorded `model_battles` outcomes."""
    params: tuple[Any, ...] = (user_id,) if user_id else ()
    where_user = "AND user_id = %s" if user_id else ""
    rows = conn.execute(
        f"""SELECT id, conversation_id, question, final_answer, user_id, plan
            FROM agent_sessions
            WHERE status = 'done' {where_user}
            ORDER BY started_at ASC""",
        params,
    ).fetchall()

    processed = 0
    eligible_count = 0
    for r in rows:
        sid = r["id"]
        existing = _coerce_plan_dict(r["plan"])
        if "leads" in existing:
            if existing["leads"].get("eligible") and not existing.get("superseded_by") and not existing.get("battle_loser"):
                eligible_count += 1
            continue
        ev_rows = conn.execute(
            """SELECT kind, payload FROM agent_events
               WHERE session_id = %s AND kind IN ('mcp_tool_call_full', 'tool_call', 'evidence')
               ORDER BY seq ASC""",
            (sid,),
        ).fetchall()
        tool_records: list[dict[str, Any]] = []
        evidence_refs: list[str] = []
        for ev in ev_rows:
            payload = ev["payload"]
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except json.JSONDecodeError:
                    payload = {}
            if not isinstance(payload, dict):
                continue
            if ev["kind"] in ("mcp_tool_call_full", "tool_call"):
                tool_records.append(payload)
            elif ev["kind"] == "evidence":
                for cr in payload.get("cite_refs") or []:
                    if isinstance(cr, str):
                        evidence_refs.append(cr)
        if not tool_records and evidence_refs:
            tool_records.append(
                {
                    "name": "search_emails",
                    "args": {},
                    "response": {"results": [{"cite_ref": cr} for cr in evidence_refs]},
                }
            )
        leads = record_session_leads(
            conn,
            session_id=sid,
            conversation_id=r["conversation_id"],
            question=r["question"] or "",
            final_answer=r["final_answer"],
            side_calls=tool_records,
            jev_config=jev_config,
            check_jev=check_jev,
        )
        processed += 1
        if leads.get("eligible"):
            eligible_count += 1

    # Apply any model_battles outcomes if the caller has SELECT on model_battles.
    try:
        b_rows = conn.execute(
            f"SELECT user_id, winner, request_id_a, request_id_b FROM model_battles WHERE 1=1 {where_user}",
            params,
        ).fetchall()
        for b in b_rows:
            if b["user_id"]:
                apply_battle_outcome_to_sessions(
                    conn,
                    user_id=b["user_id"],
                    winner=b["winner"],
                    session_a=b["request_id_a"],
                    session_b=b["request_id_b"],
                )
    except Exception as exc:  # noqa: BLE001
        logger.debug("model_battles sync during backfill skipped: %s", exc)

    return {"processed": processed, "eligible": eligible_count}
