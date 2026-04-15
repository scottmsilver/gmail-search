import logging
import struct
from pathlib import Path
from typing import Any

from google import genai  # noqa: E402

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    return len(text) // 4


def truncate_to_token_limit(text: str, max_tokens: int) -> str:
    estimated = estimate_tokens(text)
    if estimated <= max_tokens:
        return text
    max_chars = max_tokens * 4
    return text[:max_chars]


def strip_quoted_replies(body: str) -> str:
    """Remove quoted reply chains from email body to improve embedding quality."""
    import re

    lines = body.split("\n")
    cleaned: list[str] = []
    skip_rest = False

    for line in lines:
        stripped = line.strip()
        # Stop at reply headers like "On Mon, Jan 5, 2026... wrote:"
        if re.match(r"^On .{10,80} wrote:\s*$", stripped):
            break
        # Stop at "From: ... Sent: ... To: ..." Outlook-style headers
        if re.match(r"^-{3,}\s*Original Message\s*-{3,}", stripped, re.IGNORECASE):
            break
        # Skip quoted lines
        if stripped.startswith(">"):
            continue
        cleaned.append(line)

    result = "\n".join(cleaned).strip()
    # If stripping removed almost everything, keep the original
    return result if len(result) > 50 else body


def format_message_text(from_addr: str, to_addr: str, date: str, subject: str, body: str) -> str:
    clean_body = strip_quoted_replies(body)
    return f"From: {from_addr} | To: {to_addr} | Date: {date} | Subject: {subject} | {clean_body}"


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
        import os

        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()

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


class BatchGeminiEmbedder(GeminiEmbedder):
    """Drop-in replacement that uses the Batch API (50% cheaper, higher rate limits).

    Batches up to BATCH_JOB_SIZE texts, submits as an async batch job, polls for
    completion, and returns results. Same interface as GeminiEmbedder so the pipeline
    code doesn't change.
    """

    BATCH_JOB_SIZE = 1000  # max inline requests per batch job
    POLL_INTERVAL = 5  # seconds between status checks

    COMPLETED_STATES = {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    }

    def embed_texts_batch(self, texts: list[str], task_type: str | None = None) -> list[list[float]]:
        """Submit texts as a batch embedding job and poll until done."""
        import time as _time

        if task_type is None:
            task_type = self.task_type_document

        if not texts:
            return []

        # For very small batches, just use the sync API (not worth the overhead)
        if len(texts) <= 5:
            return super().embed_texts_batch(texts, task_type)

        all_vectors: list[list[float]] = []

        # Process in chunks of BATCH_JOB_SIZE
        for chunk_start in range(0, len(texts), self.BATCH_JOB_SIZE):
            chunk = texts[chunk_start : chunk_start + self.BATCH_JOB_SIZE]

            logger.info(f"Submitting batch embedding job: {len(chunk)} texts")
            batch_job = self.client.batches.create_embeddings(
                model=self.model,
                src=genai.types.EmbeddingsBatchJobSource(
                    inlined_requests=genai.types.EmbedContentBatch(
                        contents=chunk,
                        config=genai.types.EmbedContentConfig(
                            task_type=task_type,
                            output_dimensionality=self.dimensions,
                        ),
                    ),
                ),
                config={"display_name": f"gmail-search-embed-{chunk_start}"},
            )

            job_name = batch_job.name
            logger.info(f"Batch job created: {job_name}")

            # Poll until complete
            while batch_job.state.name not in self.COMPLETED_STATES:
                _time.sleep(self.POLL_INTERVAL)
                batch_job = self.client.batches.get(name=job_name)
                logger.info(f"  Batch status: {batch_job.state.name}")

            if batch_job.state.name != "JOB_STATE_SUCCEEDED":
                raise RuntimeError(f"Batch job failed: {batch_job.state.name}")

            # Extract embedding vectors from results
            for resp in batch_job.dest.inlined_embed_content_responses:
                if resp.error:
                    logger.warning(f"Batch item error: {resp.error}")
                    all_vectors.append([0.0] * self.dimensions)
                elif resp.response and resp.response.embedding:
                    all_vectors.append(resp.response.embedding.values)
                else:
                    all_vectors.append([0.0] * self.dimensions)

            logger.info(f"Batch job complete: {len(chunk)} embeddings received")

        return all_vectors

    def embed_image(self, image_path: Path, task_type: str | None = None) -> list[float]:
        # Batch API doesn't support multimodal yet — fall back to sync
        return super().embed_image(image_path, task_type)
