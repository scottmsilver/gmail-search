import logging
import os
import struct
import threading
from pathlib import Path
from typing import Any

from google import genai  # noqa: E402
from PIL import Image

logger = logging.getLogger(__name__)

# The regular (non-Batch-API) embedder must call embed_content once per text
# (the SDK averages a list into one vector). Sequential calls capped the
# backfill at ~150 texts/min — pure per-call latency, no rate-limit headroom
# used. Fire them through a bounded thread pool instead. Override via env if
# Gemini starts returning 429s (the pipeline's _retry_api_call still backs off).
_EMBED_CONCURRENCY = max(1, int(os.environ.get("GMS_EMBED_CONCURRENCY", "16")))

# Image preprocessing for embed_image (issue #12). The embedding model
# accepts only PNG and JPEG; anything else is re-encoded as PNG. Images
# above MAX_IMAGE_PIXELS are downscaled first — our own PDF rasterizer
# once produced a 27,115 x 18,154 page (492 Mpx) that Gemini rejected
# with a 400 on every pass. MAX_DECODE_PIXELS is the hard ceiling we
# are willing to decode at all (RGB bytes = 3x this); above it the file
# is treated as permanently unembeddable rather than risking the box.
MAX_IMAGE_PIXELS = 16_000_000
MAX_DECODE_PIXELS = 600_000_000  # ~1.8 GB as RGB; decodes are serialized by _DECODE_LOCK
_NATIVE_MIME = {"PNG": "image/png", "JPEG": "image/jpeg"}
_GRAYSCALE_MODES = {"1", "L", "LA", "I", "F"}
_SIXTEEN_BIT_GRAY_MODES = {"I;16", "I;16B", "I;16L", "I;16N"}
_ALPHA_MODES = {"RGBA", "LA", "PA", "P"}  # P may carry a transparency index
# One decode at a time: bounds peak memory and makes the temporary
# Pillow-global change in _lifted_decompression_bomb_limit safe.
_DECODE_LOCK = threading.Lock()


class InvalidImage(ValueError):
    """The file cannot be turned into something the embedding model
    accepts. Raised BEFORE any API call; the pipeline records it as a
    permanent failure so the file is never selected again."""


def estimate_tokens(text: str) -> int:
    return len(text) // 4


def truncate_to_token_limit(text: str, max_tokens: int) -> str:
    estimated = estimate_tokens(text)
    if estimated <= max_tokens:
        return text
    max_chars = max_tokens * 4
    return text[:max_chars]


def chunk_long_text(text: str, max_chunk_tokens: int = 500, overlap_tokens: int = 50) -> list[str]:
    """Split long text into overlapping chunks for better embedding coverage.

    Short texts (under max_chunk_tokens) are returned as-is.
    Long texts are split at paragraph boundaries, with overlap to
    preserve context at chunk edges.
    """
    if estimate_tokens(text) <= max_chunk_tokens:
        return [text]

    # Split on double newlines (paragraphs) first
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return [text]

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = estimate_tokens(para)

        # If a single paragraph exceeds max, split it by sentences
        if para_tokens > max_chunk_tokens:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_tokens = 0
            # Split by sentences
            sentences = para.replace(". ", ".\n").split("\n")
            for sent in sentences:
                sent_tokens = estimate_tokens(sent)
                if current_tokens + sent_tokens > max_chunk_tokens and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    # Keep last sentence for overlap
                    current_chunk = current_chunk[-1:] if current_chunk else []
                    current_tokens = estimate_tokens(" ".join(current_chunk))
                current_chunk.append(sent)
                current_tokens += sent_tokens
        elif current_tokens + para_tokens > max_chunk_tokens and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            # Keep last paragraph for overlap
            current_chunk = current_chunk[-1:] if current_chunk else []
            current_tokens = estimate_tokens("\n\n".join(current_chunk))
            current_chunk.append(para)
            current_tokens += para_tokens
        else:
            current_chunk.append(para)
            current_tokens += para_tokens

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks if chunks else [text]


def strip_quoted_replies(body: str) -> str:
    """Remove quoted reply chains from email body to improve embedding quality."""
    import re

    lines = body.split("\n")
    cleaned: list[str] = []

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


def prepare_image_for_embedding(image_path: Path) -> tuple[bytes, str]:
    """Return (bytes, mime_type) the embedding model will accept, or raise
    InvalidImage. Valid PNG/JPEG within the pixel cap are passed through
    untouched; everything else is decoded, first frame taken, downscaled
    if needed, and re-encoded as PNG."""
    im = _open_verified_image(image_path)
    fmt = im.format
    if fmt in _NATIVE_MIME and im.width * im.height <= MAX_IMAGE_PIXELS:
        return image_path.read_bytes(), _NATIVE_MIME[fmt]
    return _reencode_as_png(im), "image/png"


def _open_verified_image(image_path: Path) -> Image.Image:
    """Open, verify() and fully load() the first frame. Any decode
    problem — not an image, truncated, corrupt — becomes InvalidImage."""
    try:
        with _DECODE_LOCK, _lifted_decompression_bomb_limit():
            with Image.open(image_path) as probe:
                _reject_if_too_large_to_decode(probe, image_path)
                probe.verify()
            im = Image.open(image_path)
            im.load()
    except InvalidImage:
        raise
    except Exception as e:  # UnidentifiedImageError, OSError("truncated"), ...
        raise InvalidImage(f"{image_path.name}: {type(e).__name__}: {e}") from e
    return im


def _reject_if_too_large_to_decode(im: Image.Image, image_path: Path) -> None:
    pixels = im.width * im.height
    if pixels > MAX_DECODE_PIXELS:
        raise InvalidImage(f"{image_path.name}: {im.width}x{im.height} ({pixels} px) exceeds decode ceiling")


def _reencode_as_png(im: Image.Image) -> bytes:
    import io

    im = _to_8bit_rgb_or_gray(im)
    im = _downscale_to_cap(im)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _to_8bit_rgb_or_gray(im: Image.Image) -> Image.Image:
    """Normalize to 'L' or 'RGB' without the silent damage a bare
    convert() does: 16-bit grays are scaled (not clipped at 255) and
    transparent pixels are composited onto white (not dropped)."""
    if im.mode in _SIXTEEN_BIT_GRAY_MODES:
        return im.convert("I").point(lambda v: v * (1 / 256)).convert("L")
    if im.mode in _ALPHA_MODES:
        im = _flatten_alpha_onto_white(im)
    target_mode = "L" if im.mode in _GRAYSCALE_MODES else "RGB"
    return im if im.mode == target_mode else im.convert(target_mode)


def _flatten_alpha_onto_white(im: Image.Image) -> Image.Image:
    rgba = im.convert("RGBA")
    background = Image.new("RGB", rgba.size, (255, 255, 255))
    background.paste(rgba, mask=rgba.getchannel("A"))
    return background


def _downscale_to_cap(im: Image.Image) -> Image.Image:
    pixels = im.width * im.height
    if pixels <= MAX_IMAGE_PIXELS:
        return im
    scale = (MAX_IMAGE_PIXELS / pixels) ** 0.5
    new_size = (max(1, int(im.width * scale)), max(1, int(im.height * scale)))
    return im.resize(new_size, Image.Resampling.LANCZOS)


class _lifted_decompression_bomb_limit:
    """Pillow refuses to open() anything over 2x Image.MAX_IMAGE_PIXELS.
    We enforce our own MAX_DECODE_PIXELS instead, so lift Pillow's limit
    for the duration of the open and restore it afterwards."""

    def __enter__(self):
        import warnings

        self._saved = Image.MAX_IMAGE_PIXELS
        self._warnings = warnings.catch_warnings()
        self._warnings.__enter__()
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        Image.MAX_IMAGE_PIXELS = None

    def __exit__(self, *exc):
        Image.MAX_IMAGE_PIXELS = self._saved
        self._warnings.__exit__(*exc)
        return False


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

        # The google-genai SDK's `embed_content` with `contents=<list>`
        # treats the list as a multi-part *single* content and returns
        # ONE averaged embedding — not N. Diagnosed when 50 texts in
        # produced 1 vector out, breaking `zip(chunk_owners, vectors)`
        # and quietly dropping 49/50 inserts. So we must call per-text —
        # but CONCURRENTLY (a bounded pool), not sequentially: sequential
        # was the backfill's hard ceiling (~150 texts/min, pure latency).
        # executor.map preserves input order, so callers' zip(owners,
        # vectors) stays correct. The Batch API path in BatchGeminiEmbedder
        # is still the cheapest way to do truly large bulk jobs.
        def _one(t: str) -> list[float]:
            return (
                self.client.models.embed_content(
                    model=self.model,
                    contents=t,
                    config={
                        "task_type": task_type,
                        "output_dimensionality": self.dimensions,
                    },
                )
                .embeddings[0]
                .values
            )

        if len(texts) <= 1:
            return [_one(texts[0])] if texts else []
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(_EMBED_CONCURRENCY, len(texts))) as ex:
            return list(ex.map(_one, texts))

    def embed_image(self, image_path: Path, task_type: str | None = None) -> list[float]:
        if task_type is None:
            task_type = self.task_type_document
        image_bytes, mime_type = prepare_image_for_embedding(image_path)
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
