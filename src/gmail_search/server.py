from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from gmail_search.search.engine import SearchEngine
from gmail_search.store.cost import check_budget, get_total_spend
from gmail_search.store.db import get_connection
from gmail_search.store.queries import get_attachments_for_message, get_message


def _build_query_filters(
    sender: str | None,
    subject_contains: str | None,
    date_from: str | None,
    date_to: str | None,
    label: str | None,
) -> tuple[list[str], list]:
    clauses: list[str] = []
    params: list = []
    if sender:
        clauses.append("m.from_addr LIKE ?")
        params.append(f"%{sender}%")
    if subject_contains:
        clauses.append("m.subject LIKE ?")
        params.append(f"%{subject_contains}%")
    if date_from:
        clauses.append("m.date >= ?")
        params.append(date_from)
    if date_to:
        clauses.append("m.date <= ?")
        params.append(f"{date_to}T23:59:59+00:00")
    if label:
        clauses.append("m.labels LIKE ?")
        params.append(f'%"{label}"%')
    return clauses, params


def _thread_ids_matching_filters(
    conn,
    clauses: list[str],
    params: list,
    has_attachment: bool | None,
    order_by: str,
    limit: int,
) -> list[str]:
    where = " AND ".join(clauses) if clauses else "1=1"
    join = ""
    if has_attachment is True:
        join = "INNER JOIN attachments a ON a.message_id = m.id"
    elif has_attachment is False:
        where += " AND m.id NOT IN (SELECT message_id FROM attachments)"
    order = "MAX(m.date) DESC" if order_by == "date_desc" else "MAX(m.date) ASC"
    sql = f"""SELECT m.thread_id, MAX(m.date) as last_date
             FROM messages m {join}
             WHERE {where}
             GROUP BY m.thread_id
             ORDER BY {order}
             LIMIT ?"""
    rows = conn.execute(sql, [*params, limit]).fetchall()
    return [r["thread_id"] for r in rows]


def _load_thread_summaries(conn, thread_ids: list[str]) -> list[dict]:
    if not thread_ids:
        return []
    import json as _json

    placeholders = ",".join("?" * len(thread_ids))
    rows = conn.execute(
        f"""SELECT thread_id, subject, participants, message_count,
            date_first, date_last
            FROM thread_summary WHERE thread_id IN ({placeholders})""",
        thread_ids,
    ).fetchall()
    by_id = {r["thread_id"]: r for r in rows}
    latest_snippets = _latest_snippet_per_thread(conn, thread_ids)
    out = []
    for tid in thread_ids:
        r = by_id.get(tid)
        if r is None:
            continue
        out.append(
            {
                "thread_id": r["thread_id"],
                "subject": r["subject"],
                "participants": _json.loads(r["participants"]),
                "message_count": r["message_count"],
                "date_first": r["date_first"],
                "date_last": r["date_last"],
                "snippet": latest_snippets.get(tid, ""),
            }
        )
    return out


def _latest_snippet_per_thread(conn, thread_ids: list[str]) -> dict[str, str]:
    if not thread_ids:
        return {}
    placeholders = ",".join("?" * len(thread_ids))
    rows = conn.execute(
        f"""SELECT m.thread_id, m.body_text FROM messages m
            INNER JOIN (
                SELECT thread_id, MAX(date) as max_date
                FROM messages WHERE thread_id IN ({placeholders})
                GROUP BY thread_id
            ) latest ON m.thread_id = latest.thread_id AND m.date = latest.max_date""",
        thread_ids,
    ).fetchall()
    return {r["thread_id"]: (r["body_text"] or "")[:500] for r in rows}


def _run_structured_query(
    db_path: Path,
    sender: str | None,
    subject_contains: str | None,
    date_from: str | None,
    date_to: str | None,
    label: str | None,
    has_attachment: bool | None,
    order_by: str,
    limit: int,
) -> list[dict]:
    conn = get_connection(db_path)
    try:
        clauses, params = _build_query_filters(sender, subject_contains, date_from, date_to, label)
        thread_ids = _thread_ids_matching_filters(conn, clauses, params, has_attachment, order_by, limit)
        return _load_thread_summaries(conn, thread_ids)
    finally:
        conn.close()


def create_app(
    db_path: Path,
    data_dir: Path,
    config: dict[str, Any],
) -> FastAPI:
    app = FastAPI(title="Gmail Search")

    templates_dir = Path(__file__).parent.parent.parent / "templates"
    index_dir = data_dir / "scann_index"

    def _format_thread_result(r):
        return {
            "thread_id": r.thread_id,
            "score": r.score,
            "similarity": r.similarity,
            "subject": r.subject,
            "participants": r.participants,
            "message_count": r.message_count,
            "date_first": r.date_first,
            "date_last": r.date_last,
            "user_replied": r.user_replied,
            "matches": [
                {
                    "message_id": m.message_id,
                    "score": m.score,
                    "from_addr": m.from_addr,
                    "date": m.date,
                    "snippet": m.snippet,
                    "match_type": m.match_type,
                    "attachment_filename": m.attachment_filename,
                }
                for m in r.matches
            ],
        }

    def _collect_result_message_ids(results):
        ids = set()
        for r in results:
            for m in r.matches:
                ids.add(m.message_id)
        return ids

    def _compute_topic_facets(results, msg_topics):
        """Count how many result threads fall into each leaf topic.

        A thread belongs to a topic if any of its matching messages do.
        Counts threads (not messages) so facet counts match the result list.
        """
        from collections import Counter

        topic_thread_counts: Counter = Counter()
        topic_labels: dict[str, str] = {}

        for r in results:
            thread_topics = set()
            for m in r.matches:
                for tid in msg_topics.get(m.message_id, []):
                    thread_topics.add(tid)
            for tid in thread_topics:
                topic_thread_counts[tid] += 1

        # Get labels for the topics we found
        if topic_thread_counts:
            conn_f = get_connection(db_path)
            placeholders = ",".join("?" * len(topic_thread_counts))
            rows = conn_f.execute(
                f"SELECT topic_id, label FROM topics WHERE topic_id IN ({placeholders})",
                list(topic_thread_counts.keys()),
            ).fetchall()
            conn_f.close()
            topic_labels = {r["topic_id"]: r["label"] for r in rows}

        return sorted(
            [
                {"topic_id": tid, "label": topic_labels.get(tid, tid), "count": count}
                for tid, count in topic_thread_counts.items()
            ],
            key=lambda f: f["count"],
            reverse=True,
        )

    _engine: SearchEngine | None = None

    def get_engine() -> SearchEngine:
        nonlocal _engine
        if _engine is None:
            _engine = SearchEngine(db_path, index_dir, config)
        return _engine

    @app.get("/", response_class=HTMLResponse)
    async def home():
        html_file = templates_dir / "index.html"
        return html_file.read_text()

    def _lookup_message_topics(msg_ids):
        """Map message IDs to their leaf topic IDs for client-side filtering."""
        if not msg_ids:
            return {}
        conn_t = get_connection(db_path)
        placeholders = ",".join("?" * len(msg_ids))
        rows = conn_t.execute(
            f"""SELECT mt.message_id, mt.topic_id FROM message_topics mt
                JOIN topics t ON mt.topic_id = t.topic_id
                WHERE mt.message_id IN ({placeholders})
                AND t.topic_id NOT IN (SELECT DISTINCT parent_id FROM topics WHERE parent_id IS NOT NULL)""",
            list(msg_ids),
        ).fetchall()
        conn_t.close()
        result = {}
        for r in rows:
            result.setdefault(r["message_id"], []).append(r["topic_id"])
        return result

    @app.get("/api/search")
    async def api_search(
        q: str = Query(...),
        k: int = Query(20, le=100),
        sort: str = Query("relevance"),
        filter: bool = Query(True, alias="filter"),
    ):
        engine = get_engine()
        results = engine.search_threads(q, top_k=k, sort=sort, filter_offtopic=filter)

        # Look up topic IDs for all result messages (for client-side filtering)
        all_msg_ids = _collect_result_message_ids(results)
        msg_topics = _lookup_message_topics(all_msg_ids)

        facets = _compute_topic_facets(results, msg_topics)

        # Tag each result with its topic IDs
        formatted = []
        for r in results:
            fr = _format_thread_result(r)
            topics = set()
            for m in r.matches:
                topics.update(msg_topics.get(m.message_id, []))
            fr["topic_ids"] = list(topics)
            formatted.append(fr)

        return {
            "results": formatted,
            "facets": facets,
        }

    @app.get("/api/query")
    async def api_query(
        sender: str | None = Query(None, description="Substring match on from_addr"),
        subject_contains: str | None = Query(None),
        date_from: str | None = Query(None, description="ISO date (YYYY-MM-DD)"),
        date_to: str | None = Query(None, description="ISO date (YYYY-MM-DD)"),
        label: str | None = Query(None, description="Gmail label, e.g. INBOX, IMPORTANT"),
        has_attachment: bool | None = Query(None),
        order_by: str = Query("date_desc", pattern="^(date_desc|date_asc)$"),
        limit: int = Query(20, le=100),
    ):
        threads = _run_structured_query(
            db_path,
            sender=sender,
            subject_contains=subject_contains,
            date_from=date_from,
            date_to=date_to,
            label=label,
            has_attachment=has_attachment,
            order_by=order_by,
            limit=limit,
        )
        return {"results": threads}

    @app.get("/api/thread/{thread_id}")
    async def api_thread(thread_id: str):
        conn = get_connection(db_path)
        rows = conn.execute(
            "SELECT id FROM messages WHERE thread_id = ? ORDER BY date",
            (thread_id,),
        ).fetchall()
        messages = []
        for row in rows:
            msg = get_message(conn, row["id"])
            if msg is None:
                continue
            attachments = get_attachments_for_message(conn, msg.id)
            messages.append(
                {
                    "id": msg.id,
                    "from_addr": msg.from_addr,
                    "to_addr": msg.to_addr,
                    "subject": msg.subject,
                    "body_text": msg.body_text,
                    "date": msg.date.isoformat(),
                    "labels": msg.labels,
                    "attachments": [
                        {
                            "id": a.id,
                            "filename": a.filename,
                            "mime_type": a.mime_type,
                            "size_bytes": a.size_bytes,
                        }
                        for a in attachments
                    ],
                }
            )
        conn.close()
        return {"thread_id": thread_id, "messages": messages}

    @app.get("/api/topics")
    async def api_topics():
        conn = get_connection(db_path)
        rows = conn.execute(
            "SELECT topic_id, parent_id, label, depth, message_count, top_senders FROM topics ORDER BY depth, message_count DESC"
        ).fetchall()
        conn.close()
        import json as _json

        return [
            {
                "topic_id": r["topic_id"],
                "parent_id": r["parent_id"],
                "label": r["label"],
                "depth": r["depth"],
                "message_count": r["message_count"],
                "top_senders": _json.loads(r["top_senders"]),
            }
            for r in rows
        ]

    @app.get("/api/message/{message_id}")
    async def api_message(message_id: str):
        conn = get_connection(db_path)
        msg = get_message(conn, message_id)
        if msg is None:
            conn.close()
            return JSONResponse({"error": "Message not found"}, status_code=404)
        attachments = get_attachments_for_message(conn, message_id)
        conn.close()
        return {
            "id": msg.id,
            "thread_id": msg.thread_id,
            "from_addr": msg.from_addr,
            "to_addr": msg.to_addr,
            "subject": msg.subject,
            "body_text": msg.body_text,
            "date": msg.date.isoformat(),
            "labels": msg.labels,
            "attachments": [
                {
                    "id": a.id,
                    "filename": a.filename,
                    "mime_type": a.mime_type,
                    "size_bytes": a.size_bytes,
                }
                for a in attachments
            ],
        }

    @app.get("/api/thread_lookup")
    async def api_thread_lookup(cite_ref: str = Query(..., min_length=4, max_length=20)):
        """Resolve a cite_ref (4-20 char prefix) to a real thread_id.

        Returns the thread + subject when exactly one thread starts with
        the prefix. 404 if zero matches; 409 if ambiguous.
        """
        prefix = cite_ref.strip().lower()
        if not prefix or any(c not in "0123456789abcdef" for c in prefix):
            return JSONResponse({"error": "cite_ref must be hex"}, status_code=400)
        conn = get_connection(db_path)
        rows = conn.execute(
            "SELECT thread_id, subject FROM thread_summary WHERE thread_id LIKE ? LIMIT 5",
            (f"{prefix}%",),
        ).fetchall()
        conn.close()
        if not rows:
            return JSONResponse({"error": f"no thread starts with {prefix!r}"}, status_code=404)
        if len(rows) > 1:
            return JSONResponse(
                {
                    "error": f"{len(rows)} threads start with {prefix!r} — be more specific",
                    "candidates": [{"thread_id": r["thread_id"], "subject": r["subject"]} for r in rows],
                },
                status_code=409,
            )
        return {"thread_id": rows[0]["thread_id"], "subject": rows[0]["subject"]}

    @app.get("/api/attachment/{attachment_id}/text")
    async def api_attachment_text(attachment_id: int):
        conn = get_connection(db_path)
        row = conn.execute(
            "SELECT filename, mime_type, extracted_text FROM attachments WHERE id = ?",
            (attachment_id,),
        ).fetchone()
        conn.close()
        if row is None:
            return JSONResponse({"error": "Attachment not found"}, status_code=404)
        return {
            "attachment_id": attachment_id,
            "filename": row["filename"],
            "mime_type": row["mime_type"],
            "extracted_text": row["extracted_text"] or "",
        }

    @app.get("/api/attachment/{attachment_id}")
    async def api_attachment(attachment_id: int):
        conn = get_connection(db_path)
        row = conn.execute(
            "SELECT raw_path, mime_type, filename FROM attachments WHERE id = ?",
            (attachment_id,),
        ).fetchone()
        conn.close()
        if row is None or not row["raw_path"]:
            return JSONResponse({"error": "Attachment not found"}, status_code=404)
        # Validate path is under data_dir to prevent path traversal
        resolved = Path(row["raw_path"]).resolve()
        if not str(resolved).startswith(str(data_dir.resolve())):
            return JSONResponse({"error": "Invalid attachment path"}, status_code=403)
        if not resolved.exists():
            return JSONResponse({"error": "Attachment file missing"}, status_code=404)
        return FileResponse(
            str(resolved),
            media_type=row["mime_type"],
            filename=row["filename"],
        )

    @app.get("/api/progress")
    async def api_progress():
        from gmail_search.store.db import JobProgress

        return JobProgress.get(db_path) or []

    @app.get("/api/status")
    async def api_status():
        from gmail_search.store.db import JobProgress

        conn = get_connection(db_path)
        msg_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        emb_count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        # All dates are now UTC so string sort works correctly
        dates = conn.execute("SELECT MIN(date) as oldest, MAX(date) as newest FROM messages").fetchone()
        total_cost = get_total_spend(conn)
        ok, spent, remaining = check_budget(conn, config["budget"]["max_usd"])
        conn.close()
        jobs = JobProgress.get(db_path) or []
        running = [j for j in jobs if j["status"] == "running"]
        return {
            "messages": msg_count,
            "embeddings": emb_count,
            "date_oldest": dates["oldest"],
            "date_newest": dates["newest"],
            "total_cost_usd": round(total_cost, 4),
            "budget_remaining_usd": round(remaining, 4),
            "running_job": running[0] if running else None,
        }

    return app
