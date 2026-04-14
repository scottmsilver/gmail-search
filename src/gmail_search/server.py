from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, HTMLResponse

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
    async def api_search(q: str = Query(...), k: int = Query(20)):
        engine = get_engine()
        results = engine.search(q, top_k=k)
        return [
            {
                "score": r.score,
                "message_id": r.message_id,
                "subject": r.subject,
                "from_addr": r.from_addr,
                "date": r.date,
                "snippet": r.snippet,
                "match_type": r.match_type,
                "attachment_filename": r.attachment_filename,
            }
            for r in results
        ]

    @app.get("/api/message/{message_id}")
    async def api_message(message_id: str):
        conn = get_connection(db_path)
        msg = get_message(conn, message_id)
        if msg is None:
            conn.close()
            return {"error": "Message not found"}
        attachments = get_attachments_for_message(conn, message_id)
        conn.close()
        return {
            "id": msg.id,
            "thread_id": msg.thread_id,
            "from_addr": msg.from_addr,
            "to_addr": msg.to_addr,
            "subject": msg.subject,
            "body_text": msg.body_text,
            "body_html": msg.body_html,
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
            return {"error": "Attachment not found"}
        return FileResponse(
            row["raw_path"],
            media_type=row["mime_type"],
            filename=row["filename"],
        )

    @app.get("/api/status")
    async def api_status():
        conn = get_connection(db_path)
        msg_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        emb_count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        total_cost = get_total_spend(conn)
        ok, spent, remaining = check_budget(conn, config["budget"]["max_usd"])
        conn.close()
        return {
            "messages": msg_count,
            "embeddings": emb_count,
            "total_cost_usd": round(total_cost, 4),
            "budget_remaining_usd": round(remaining, 4),
        }

    return app
