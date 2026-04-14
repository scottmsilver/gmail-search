import logging
import time
from pathlib import Path
from typing import Any

from googleapiclient.discovery import Resource
from tqdm import tqdm

from gmail_search.gmail.parser import parse_message
from gmail_search.store.db import get_connection
from gmail_search.store.models import Attachment
from gmail_search.store.queries import get_sync_state, set_sync_state, upsert_attachment, upsert_message

logger = logging.getLogger(__name__)


def _download_attachment_data(service: Resource, message_id: str, attachment_id: str) -> bytes:
    import base64

    result = service.users().messages().attachments().get(userId="me", messageId=message_id, id=attachment_id).execute()
    data = result.get("data", "")
    padded = data + "=" * (4 - len(data) % 4)
    return base64.urlsafe_b64decode(padded)


def download_messages(
    service: Resource,
    db_path: Path,
    data_dir: Path,
    batch_size: int = 100,
    max_messages: int | None = None,
    max_attachment_size: int = 10 * 1024 * 1024,
) -> int:
    conn = get_connection(db_path)
    attachments_dir = data_dir / "attachments"
    attachments_dir.mkdir(parents=True, exist_ok=True)

    message_ids: list[str] = []
    page_token = None
    logger.info("Fetching message IDs...")

    while True:
        kwargs: dict[str, Any] = {"userId": "me", "maxResults": 500}
        if page_token:
            kwargs["pageToken"] = page_token
        result = service.users().messages().list(**kwargs).execute()
        messages = result.get("messages", [])
        message_ids.extend(m["id"] for m in messages)

        if max_messages and len(message_ids) >= max_messages:
            message_ids = message_ids[:max_messages]
            break

        page_token = result.get("nextPageToken")
        if not page_token:
            break

    logger.info(f"Found {len(message_ids)} messages to download")

    existing = set()
    for row in conn.execute("SELECT id FROM messages").fetchall():
        existing.add(row["id"])
    to_download = [mid for mid in message_ids if mid not in existing]
    logger.info(f"{len(existing)} already downloaded, {len(to_download)} remaining")

    if not to_download:
        conn.close()
        return 0

    max_history_id = 0
    downloaded = 0
    progress = tqdm(total=len(to_download), desc="Downloading messages")

    for i in range(0, len(to_download), batch_size):
        batch_ids = to_download[i : i + batch_size]
        batch_results: list[dict] = []

        def _callback(request_id, response, exception):
            if exception:
                logger.warning(f"Error fetching {request_id}: {exception}")
            else:
                batch_results.append(response)

        batch = service.new_batch_http_request(callback=_callback)
        for mid in batch_ids:
            batch.add(
                service.users().messages().get(userId="me", id=mid, format="full"),
                request_id=mid,
            )

        retries = 0
        while retries < 5:
            try:
                batch.execute()
                break
            except Exception as e:
                if "429" in str(e) or "rateLimitExceeded" in str(e):
                    wait = 2**retries
                    logger.warning(f"Rate limited, retrying in {wait}s...")
                    time.sleep(wait)
                    retries += 1
                else:
                    raise

        for raw in batch_results:
            msg, att_metas = parse_message(raw)
            upsert_message(conn, msg)

            if msg.history_id > max_history_id:
                max_history_id = msg.history_id

            for att_meta in att_metas:
                if att_meta["size"] > max_attachment_size:
                    logger.warning(
                        f"Skipping attachment {att_meta['filename']} "
                        f"({att_meta['size']} bytes) — exceeds size limit"
                    )
                    continue

                msg_att_dir = attachments_dir / msg.id
                msg_att_dir.mkdir(exist_ok=True)
                raw_path = msg_att_dir / att_meta["filename"]

                try:
                    data = _download_attachment_data(service, msg.id, att_meta["attachment_id"])
                    raw_path.write_bytes(data)
                except Exception as e:
                    logger.warning(f"Failed to download attachment: {e}")
                    continue

                att = Attachment(
                    id=None,
                    message_id=msg.id,
                    filename=att_meta["filename"],
                    mime_type=att_meta["mime_type"],
                    size_bytes=len(data),
                    raw_path=str(raw_path),
                )
                upsert_attachment(conn, att)

            downloaded += 1

        progress.update(len(batch_results))

    progress.close()

    if max_history_id > 0:
        set_sync_state(conn, "last_history_id", str(max_history_id))

    conn.close()
    return downloaded


def sync_new_messages(
    service: Resource,
    db_path: Path,
    data_dir: Path,
    max_attachment_size: int = 10 * 1024 * 1024,
) -> int:
    conn = get_connection(db_path)
    last_history_id = get_sync_state(conn, "last_history_id")
    conn.close()

    if not last_history_id:
        logger.warning("No last_history_id found. Run full download first.")
        return 0

    new_message_ids: list[str] = []
    page_token = None

    try:
        while True:
            kwargs: dict[str, Any] = {
                "userId": "me",
                "startHistoryId": last_history_id,
            }
            if page_token:
                kwargs["pageToken"] = page_token
            result = service.users().history().list(**kwargs).execute()

            for record in result.get("history", []):
                for added in record.get("messagesAdded", []):
                    new_message_ids.append(added["message"]["id"])

            page_token = result.get("nextPageToken")
            if not page_token:
                break
    except Exception as e:
        if "404" in str(e):
            logger.warning("History expired. Run full download to re-sync.")
            return 0
        raise

    if not new_message_ids:
        logger.info("No new messages since last sync")
        return 0

    new_message_ids = list(set(new_message_ids))
    logger.info(f"Found {len(new_message_ids)} new messages")

    conn = get_connection(db_path)
    attachments_dir = data_dir / "attachments"
    attachments_dir.mkdir(parents=True, exist_ok=True)
    max_history_id = int(last_history_id)
    count = 0

    for mid in tqdm(new_message_ids, desc="Syncing new messages"):
        try:
            raw = service.users().messages().get(userId="me", id=mid, format="full").execute()
        except Exception as e:
            logger.warning(f"Failed to fetch {mid}: {e}")
            continue

        msg, att_metas = parse_message(raw)
        upsert_message(conn, msg)

        if msg.history_id > max_history_id:
            max_history_id = msg.history_id

        for att_meta in att_metas:
            if att_meta["size"] > max_attachment_size:
                continue
            msg_att_dir = attachments_dir / msg.id
            msg_att_dir.mkdir(exist_ok=True)
            raw_path = msg_att_dir / att_meta["filename"]
            try:
                data = _download_attachment_data(service, msg.id, att_meta["attachment_id"])
                raw_path.write_bytes(data)
            except Exception as e:
                logger.warning(f"Failed to download attachment: {e}")
                continue
            att = Attachment(
                id=None,
                message_id=msg.id,
                filename=att_meta["filename"],
                mime_type=att_meta["mime_type"],
                size_bytes=len(data),
                raw_path=str(raw_path),
            )
            upsert_attachment(conn, att)

        count += 1

    set_sync_state(conn, "last_history_id", str(max_history_id))
    conn.close()
    return count
