import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gmail_search.embed.client import GeminiEmbedder
from gmail_search.index.searcher import ScannSearcher
from gmail_search.store.db import get_connection

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    score: float
    message_id: str
    subject: str
    from_addr: str
    date: str
    snippet: str
    match_type: str
    attachment_filename: str | None = None


class SearchEngine:
    def __init__(self, db_path: Path, index_dir: Path, config: dict[str, Any], embedder: GeminiEmbedder | Any = None):
        self.db_path = db_path
        self.config = config
        dims = config["embedding"]["dimensions"]
        self.searcher = ScannSearcher(index_dir, dimensions=dims)
        self.embedder = embedder or GeminiEmbedder(config)

    def search(self, query: str, top_k: int = 20) -> list[SearchResult]:
        query_vector = np.array(self.embedder.embed_query(query), dtype=np.float32)

        fetch_k = min(top_k * 3, len(self.searcher.embedding_ids))
        embedding_ids, scores = self.searcher.search(query_vector, top_k=fetch_k)

        if not embedding_ids:
            return []

        conn = get_connection(self.db_path)

        placeholders = ",".join("?" * len(embedding_ids))
        rows = conn.execute(
            f"""SELECT e.id, e.message_id, e.attachment_id, e.chunk_type, e.chunk_text,
                       m.subject, m.from_addr, m.date,
                       a.filename as att_filename
                FROM embeddings e
                JOIN messages m ON e.message_id = m.id
                LEFT JOIN attachments a ON e.attachment_id = a.id
                WHERE e.id IN ({placeholders})""",
            embedding_ids,
        ).fetchall()
        conn.close()

        row_map = {r["id"]: r for r in rows}

        seen_messages: dict[str, SearchResult] = {}

        for emb_id, score in zip(embedding_ids, scores):
            row = row_map.get(emb_id)
            if row is None:
                continue

            msg_id = row["message_id"]
            if msg_id in seen_messages:
                if score > seen_messages[msg_id].score:
                    seen_messages[msg_id].score = score
                    seen_messages[msg_id].match_type = row["chunk_type"]
                    seen_messages[msg_id].snippet = (row["chunk_text"] or "")[:200]
                continue

            seen_messages[msg_id] = SearchResult(
                score=score,
                message_id=msg_id,
                subject=row["subject"],
                from_addr=row["from_addr"],
                date=row["date"],
                snippet=(row["chunk_text"] or "")[:200],
                match_type=row["chunk_type"],
                attachment_filename=row["att_filename"],
            )

        results = sorted(seen_messages.values(), key=lambda r: r.score, reverse=True)
        return results[:top_k]
