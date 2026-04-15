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

    @app.get("/api/search")
    async def api_search(
        q: str = Query(...),
        k: int = Query(20, le=100),
        sort: str = Query("relevance"),
        filter: bool = Query(True, alias="filter"),
        topic: int = Query(None),
    ):
        engine = get_engine()
        results = engine.search_threads(q, top_k=k, sort=sort, filter_offtopic=filter)
        # Filter by topic if requested
        if topic is not None:
            conn_t = get_connection(db_path)
            topic_msg_ids = {
                r["message_id"]
                for r in conn_t.execute("SELECT message_id FROM message_topics WHERE topic_id = ?", (topic,)).fetchall()
            }
            conn_t.close()
            results = [r for r in results if any(m.message_id in topic_msg_ids for m in r.matches)]
        return [
            {
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
            for r in results
        ]

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
            "SELECT topic_id, label, message_count, top_senders FROM topics ORDER BY message_count DESC"
        ).fetchall()
        conn.close()
        import json as _json

        return [
            {
                "topic_id": r["topic_id"],
                "label": r["label"],
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
