from dataclasses import dataclass
from datetime import datetime


@dataclass
class Message:
    id: str
    thread_id: str
    from_addr: str
    to_addr: str
    subject: str
    body_text: str
    body_html: str
    date: datetime
    labels: list[str]
    history_id: int
    raw_json: str


@dataclass
class Attachment:
    id: int | None
    message_id: str
    filename: str
    mime_type: str
    size_bytes: int
    extracted_text: str | None = None
    image_path: str | None = None
    raw_path: str | None = None
    # Gmail fetch outcome: "ok" (bytes on disk at raw_path),
    # "skipped_too_large", or "fetch_failed". Non-ok rows keep the
    # Gmail-declared size_bytes and have raw_path NULL.
    fetch_status: str = "ok"
    # Image-embedding outcome: None (not failed) or "failed_permanent".
    embed_status: str | None = None


@dataclass
class EmbeddingRecord:
    id: int | None
    message_id: str
    attachment_id: int | None
    chunk_type: str  # "message", "attachment_text", "attachment_image"
    chunk_text: str | None
    embedding: bytes  # raw float32 blob
    model: str
