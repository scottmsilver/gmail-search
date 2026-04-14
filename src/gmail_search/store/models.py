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


@dataclass
class EmbeddingRecord:
    id: int | None
    message_id: str
    attachment_id: int | None
    chunk_type: str  # "message", "attachment_text", "attachment_image"
    chunk_text: str | None
    embedding: bytes  # raw float32 blob
    model: str


@dataclass
class CostRecord:
    id: int | None
    timestamp: datetime
    operation: str  # "embed_text", "embed_image"
    model: str
    input_tokens: int
    image_count: int
    estimated_cost_usd: float
    message_id: str
