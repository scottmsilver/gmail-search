import logging
import struct
from pathlib import Path
from typing import Any

from google import genai

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    return len(text) // 4


def truncate_to_token_limit(text: str, max_tokens: int) -> str:
    estimated = estimate_tokens(text)
    if estimated <= max_tokens:
        return text
    max_chars = max_tokens * 4
    return text[:max_chars]


def format_message_text(from_addr: str, to_addr: str, date: str, subject: str, body: str) -> str:
    return f"From: {from_addr} | To: {to_addr} | Date: {date} | Subject: {subject} | {body}"


def format_attachment_text(filename: str, subject: str, extracted_text: str) -> str:
    return f"Attachment: {filename} | From email: {subject} | {extracted_text}"


def embedding_to_blob(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def blob_to_embedding(blob: bytes, dimensions: int) -> list[float]:
    return list(struct.unpack(f"{dimensions}f", blob))


class GeminiEmbedder:
    def __init__(self, config: dict[str, Any]):
        self.model = config["embedding"]["model"]
        self.dimensions = config["embedding"]["dimensions"]
        self.task_type_document = config["embedding"]["task_type_document"]
        self.task_type_query = config["embedding"]["task_type_query"]
        self.client = genai.Client()

    def embed_text(self, text: str, task_type: str | None = None) -> list[float]:
        if task_type is None:
            task_type = self.task_type_document
        result = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config={
                "task_type": task_type,
                "output_dimensionality": self.dimensions,
            },
        )
        return result.embeddings[0].values

    def embed_texts_batch(self, texts: list[str], task_type: str | None = None) -> list[list[float]]:
        if task_type is None:
            task_type = self.task_type_document
        result = self.client.models.embed_content(
            model=self.model,
            contents=texts,
            config={
                "task_type": task_type,
                "output_dimensionality": self.dimensions,
            },
        )
        return [e.values for e in result.embeddings]

    def embed_image(self, image_path: Path, task_type: str | None = None) -> list[float]:
        if task_type is None:
            task_type = self.task_type_document
        image_bytes = image_path.read_bytes()
        suffix = image_path.suffix.lower()
        mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif"}
        mime_type = mime_map.get(suffix, "image/png")
        result = self.client.models.embed_content(
            model=self.model,
            contents=genai.types.Content(
                parts=[genai.types.Part(inline_data=genai.types.Blob(mime_type=mime_type, data=image_bytes))]
            ),
            config={
                "task_type": task_type,
                "output_dimensionality": self.dimensions,
            },
        )
        return result.embeddings[0].values

    def embed_query(self, query: str) -> list[float]:
        return self.embed_text(query, task_type=self.task_type_query)
