from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from gmail_search.search.engine import SearchEngine
from gmail_search.store.cost import check_budget, get_total_spend
from gmail_search.store.db import get_connection
from gmail_search.store.queries import get_attachments_for_message, get_message


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

    @app.get("/api/status")
    async def api_status():
        conn = get_connection(db_path)
        msg_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        emb_count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        dates = conn.execute("SELECT MIN(date) as oldest, MAX(date) as newest FROM messages").fetchone()
        total_cost = get_total_spend(conn)
        ok, spent, remaining = check_budget(conn, config["budget"]["max_usd"])
        conn.close()
        return {
            "messages": msg_count,
            "embeddings": emb_count,
            "date_oldest": dates["oldest"],
            "date_newest": dates["newest"],
            "total_cost_usd": round(total_cost, 4),
            "budget_remaining_usd": round(remaining, 4),
        }

    return app
