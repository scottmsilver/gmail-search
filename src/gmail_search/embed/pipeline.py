import logging
from pathlib import Path
from typing import Any

from tqdm import tqdm

from gmail_search.embed.client import (
    GeminiEmbedder,
    embedding_to_blob,
    estimate_tokens,
    format_attachment_text,
    format_message_text,
    truncate_to_token_limit,
)
from gmail_search.store.cost import check_budget, estimate_cost, record_cost
from gmail_search.store.db import get_connection
from gmail_search.store.models import EmbeddingRecord
from gmail_search.store.queries import (
    embedding_exists,
    get_attachments_for_message,
    get_messages_without_embeddings,
    insert_embedding,
)

logger = logging.getLogger(__name__)

TEXT_BATCH_SIZE = 100
MAX_INPUT_TOKENS = 8192


def _retry_api_call(fn, *args, max_retries=5, **kwargs):
    """Retry an API call with exponential backoff on transient errors."""
    import time as _time

    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            err = str(e)
            if any(code in err for code in ["503", "429", "UNAVAILABLE", "rateLimitExceeded", "500"]):
                wait = 2 ** (attempt + 1)
                logger.warning(f"API error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait}s...")
                _time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"API call failed after {max_retries} retries")


def run_embedding_pipeline(
    db_path: Path,
    config: dict[str, Any],
    embedder: GeminiEmbedder | Any = None,
) -> int:
    conn = get_connection(db_path)
    model = config["embedding"]["model"]
    max_budget = config["budget"]["max_usd"]
    att_config = config.get("attachments", {})

    if embedder is None:
        embedder = GeminiEmbedder(config)

    # Use larger batches for Batch API (designed for bulk, small jobs waste queue time)
    from gmail_search.embed.client import BatchGeminiEmbedder

    batch_size = 1000 if isinstance(embedder, BatchGeminiEmbedder) else TEXT_BATCH_SIZE

    # Phase 1: Embed messages
    messages = get_messages_without_embeddings(conn, model=model)
    logger.info(f"{len(messages)} messages to embed (batch_size={batch_size})")

    total_embedded = 0

    for i in range(0, len(messages), batch_size):
        ok, spent, remaining = check_budget(conn, max_budget)
        if not ok:
            logger.warning(f"Budget limit ${max_budget:.2f} reached (${spent:.2f} spent). Stopping.")
            break

        batch = messages[i : i + batch_size]
        texts = []
        batch_msgs = []

        for msg in batch:
            text = format_message_text(
                from_addr=msg.from_addr,
                to_addr=msg.to_addr,
                date=msg.date.strftime("%Y-%m-%d"),
                subject=msg.subject,
                body=msg.body_text,
            )
            text = truncate_to_token_limit(text, MAX_INPUT_TOKENS)
            texts.append(text)
            batch_msgs.append(msg)

        vectors = _retry_api_call(embedder.embed_texts_batch, texts)

        total_tokens = sum(estimate_tokens(t) for t in texts)
        cost = estimate_cost(input_tokens=total_tokens)
        record_cost(
            conn,
            operation="embed_text",
            model=model,
            input_tokens=total_tokens,
            image_count=0,
            estimated_cost_usd=cost,
            message_id=f"batch_{i}",
        )

        for msg, text, vector in zip(batch_msgs, texts, vectors):
            insert_embedding(
                conn,
                EmbeddingRecord(
                    id=None,
                    message_id=msg.id,
                    attachment_id=None,
                    chunk_type="message",
                    chunk_text=text[:500],
                    embedding=embedding_to_blob(vector),
                    model=model,
                ),
            )
            total_embedded += 1

    # Phase 2: Embed attachments
    all_messages = conn.execute("SELECT id, subject FROM messages").fetchall()
    max_images_per_msg = att_config.get("max_images_per_message", 10)

    for row in tqdm(all_messages, desc="Processing attachments"):
        msg_id = row["id"]
        msg_subject = row["subject"]
        attachments = get_attachments_for_message(conn, msg_id)

        for att in attachments:
            if att.extracted_text and not embedding_exists(conn, msg_id, att.id, "attachment_text", model):
                ok, spent, remaining = check_budget(conn, max_budget)
                if not ok:
                    logger.warning("Budget limit reached. Stopping.")
                    conn.close()
                    return total_embedded

                text = format_attachment_text(att.filename, msg_subject, att.extracted_text)
                text = truncate_to_token_limit(text, MAX_INPUT_TOKENS)
                vector = _retry_api_call(embedder.embed_texts_batch, [text])[0]

                tokens = estimate_tokens(text)
                cost = estimate_cost(input_tokens=tokens)
                record_cost(
                    conn,
                    operation="embed_text",
                    model=model,
                    input_tokens=tokens,
                    image_count=0,
                    estimated_cost_usd=cost,
                    message_id=msg_id,
                )

                insert_embedding(
                    conn,
                    EmbeddingRecord(
                        id=None,
                        message_id=msg_id,
                        attachment_id=att.id,
                        chunk_type="attachment_text",
                        chunk_text=text[:500],
                        embedding=embedding_to_blob(vector),
                        model=model,
                    ),
                )
                total_embedded += 1

            if att.image_path:
                image_path = Path(att.image_path)
                if image_path.is_dir():
                    image_files = sorted(image_path.glob("*.png"))[:max_images_per_msg]
                elif image_path.is_file():
                    image_files = [image_path]
                else:
                    image_files = []

                for img_idx, img_file in enumerate(image_files):
                    chunk_key = f"attachment_image_{img_idx}"
                    # Check both new per-image key and legacy single key
                    if embedding_exists(conn, msg_id, att.id, chunk_key, model):
                        continue
                    if img_idx == 0 and embedding_exists(conn, msg_id, att.id, "attachment_image", model):
                        continue
                    ok, spent, remaining = check_budget(conn, max_budget)
                    if not ok:
                        logger.warning("Budget limit reached. Stopping.")
                        conn.close()
                        return total_embedded

                    try:
                        vector = _retry_api_call(embedder.embed_image, img_file)
                    except Exception as e:
                        logger.warning(f"Failed to embed image {img_file}: {e}")
                        continue

                    cost = estimate_cost(image_count=1)
                    record_cost(
                        conn,
                        operation="embed_image",
                        model=model,
                        input_tokens=0,
                        image_count=1,
                        estimated_cost_usd=cost,
                        message_id=msg_id,
                    )

                    insert_embedding(
                        conn,
                        EmbeddingRecord(
                            id=None,
                            message_id=msg_id,
                            attachment_id=att.id,
                            chunk_type=chunk_key,
                            chunk_text=f"[Image: {img_file.name} from {att.filename}]",
                            embedding=embedding_to_blob(vector),
                            model=model,
                        ),
                    )
                    total_embedded += 1

    conn.close()
    logger.info(f"Embedded {total_embedded} chunks total")
    return total_embedded
