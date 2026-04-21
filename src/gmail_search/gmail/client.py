import logging
import time
from pathlib import Path
from typing import Any

from googleapiclient.discovery import Resource
from tqdm import tqdm

from gmail_search.gmail.drive import drive_mime_for_kind, extract_drive_ids
from gmail_search.gmail.parser import parse_message
from gmail_search.store.db import get_connection
from gmail_search.store.models import Attachment
from gmail_search.store.queries import (
    get_sync_state,
    set_sync_state,
    upsert_attachment,
    upsert_drive_stub,
    upsert_message,
)

logger = logging.getLogger(__name__)


def _sanitize_filename(filename: str) -> str:
    """Strip path components to prevent directory traversal."""
    # Take only the final component, strip any path separators
    name = Path(filename).name
    # Remove any remaining dangerous characters
    name = name.replace("\x00", "").strip(". ")
    return name or "unnamed_attachment"


def _download_attachment_data(service: Resource, message_id: str, attachment_id: str) -> bytes:
    import base64

    result = service.users().messages().attachments().get(userId="me", messageId=message_id, id=attachment_id).execute()
    data = result.get("data", "")
    # Pad only when length is not already a multiple of 4. The previous
    # `4 - len(data) % 4` formula added 4 bogus '=' chars on already-aligned
    # input — Python tolerated it but the math was wrong.
    padded = data + "=" * (-len(data) % 4)
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

        for attempt in range(5):
            try:
                result = service.users().messages().list(**kwargs).execute()
                break
            except Exception as e:
                if "429" in str(e) or "rateLimitExceeded" in str(e):
                    wait = 2 ** (attempt + 1)
                    logger.warning(f"Rate limited on list, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise
        else:
            logger.error("Failed to list messages after 5 retries")
            break

        messages = result.get("messages", [])
        message_ids.extend(m["id"] for m in messages)
        logger.info(f"  ...fetched {len(message_ids)} IDs so far")

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

    # Gmail API allows 250 quota units/sec, messages.get = 5 units each = 50 msg/sec
    # Batch of 25 with 0.5s sleep = ~50 msg/sec theoretical
    effective_batch_size = min(batch_size, 25)

    for i in range(0, len(to_download), effective_batch_size):
        batch_ids = to_download[i : i + effective_batch_size]
        batch_results: list[dict] = []
        failed_ids: list[str] = []

        def _callback(request_id, response, exception):
            if exception:
                if "429" in str(exception) or "rateLimitExceeded" in str(exception):
                    failed_ids.append(request_id)
                else:
                    logger.warning(f"Error fetching {request_id}: {exception}")
            else:
                batch_results.append(response)

        for attempt in range(5):
            batch = service.new_batch_http_request(callback=_callback)
            ids_to_try = list(failed_ids) if (attempt > 0 and failed_ids) else list(batch_ids)
            failed_ids.clear()  # clear in-place so callback closure sees same list

            for mid in ids_to_try:
                batch.add(
                    service.users().messages().get(userId="me", id=mid, format="full"),
                    request_id=mid,
                )

            try:
                batch.execute()
            except Exception as e:
                if "429" in str(e) or "rateLimitExceeded" in str(e):
                    wait = 2 ** (attempt + 1)
                    logger.warning(f"Rate limited on batch, retrying in {wait}s...")
                    time.sleep(wait)
                    failed_ids.extend(ids_to_try)  # retry all
                    continue
                else:
                    raise

            if not failed_ids:
                break
            else:
                wait = 2 ** (attempt + 1)
                logger.warning(f"{len(failed_ids)} requests rate limited, retrying in {wait}s...")
                time.sleep(wait)

        # Throttle: 25 msgs * 5 units = 125 units per batch, limit is 250/sec
        time.sleep(0.5)

        for raw in batch_results:
            msg, att_metas = parse_message(raw)
            upsert_message(conn, msg)

            if msg.history_id > max_history_id:
                max_history_id = msg.history_id

            # Any Drive URL in the body becomes a stub attachment row.
            # The `extract` step fetches the real content via the
            # Drive API and fills in extracted_text, so the content
            # flows through embed → search → summarize just like any
            # local attachment. No separate enrichment job needed.
            # Layering: regex lives in gmail/drive (API client code),
            # the INSERT lives in store/queries (schema owner).
            for drive_id, kind in extract_drive_ids(msg.body_text or ""):
                upsert_drive_stub(conn, message_id=msg.id, drive_id=drive_id, mime_type=drive_mime_for_kind(kind))

            # Any other URL in the body becomes a URL stub — same
            # shape as Drive stubs. The `crawl-urls` command fetches
            # the page text and fills extracted_text so summaries /
            # search see the crawled content.
            from gmail_search.gmail.url_extract import extract_crawlable_urls as _extract_urls
            from gmail_search.store.queries import upsert_url_stub as _upsert_url_stub

            for url in _extract_urls(msg.body_text or "", labels=msg.labels):
                _upsert_url_stub(conn, message_id=msg.id, url=url)

            for att_meta in att_metas:
                if att_meta["size"] > max_attachment_size:
                    logger.warning(
                        f"Skipping attachment {att_meta['filename']} "
                        f"({att_meta['size']} bytes) — exceeds size limit"
                    )
                    continue

                msg_att_dir = attachments_dir / msg.id
                msg_att_dir.mkdir(exist_ok=True)
                raw_path = msg_att_dir / _sanitize_filename(att_meta["filename"])

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
    history_expired = False

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
            # Gmail drops history entries after ~7 days. Once our
            # stored `last_history_id` falls out of that window every
            # subsequent sync-new call returns 404 forever, which
            # looks like "frontfill is broken" from the outside — new
            # mail just never gets pulled in.
            #
            # Recovery: scan the last 14 days via `messages.list`
            # (idempotent upsert downstream), then bump
            # `last_history_id` to the account's current value so the
            # NEXT sync starts fresh from the incremental API again.
            history_expired = True
            logger.warning("History expired — falling back to messages.list sweep of the last 14 days.")
        else:
            raise

    if history_expired:
        try:
            next_page: str | None = None
            recovered: list[str] = []
            while True:
                resp = (
                    service.users()
                    .messages()
                    .list(userId="me", q="newer_than:14d", maxResults=500, pageToken=next_page)
                    .execute()
                )
                for m in resp.get("messages", []):
                    recovered.append(m["id"])
                next_page = resp.get("nextPageToken")
                if not next_page:
                    break
            logger.info(f"history-expired recovery found {len(recovered)} message IDs in the last 14 days")
            new_message_ids.extend(recovered)
        except Exception as e2:
            logger.warning(f"messages.list fallback failed: {e2}")

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

        # Drive stubs (same rationale as download_messages above).
        for drive_id, kind in extract_drive_ids(msg.body_text or ""):
            upsert_drive_stub(conn, message_id=msg.id, drive_id=drive_id, mime_type=drive_mime_for_kind(kind))

        # URL stubs — plain URLs filled later by the crawl-urls command.
        from gmail_search.gmail.url_extract import extract_crawlable_urls as _extract_urls
        from gmail_search.store.queries import upsert_url_stub as _upsert_url_stub

        for url in _extract_urls(msg.body_text or "", labels=msg.labels):
            _upsert_url_stub(conn, message_id=msg.id, url=url)

        for att_meta in att_metas:
            if att_meta["size"] > max_attachment_size:
                continue
            msg_att_dir = attachments_dir / msg.id
            msg_att_dir.mkdir(exist_ok=True)
            safe_name = _sanitize_filename(att_meta["filename"])
            raw_path = msg_att_dir / safe_name
            try:
                data = _download_attachment_data(service, msg.id, att_meta["attachment_id"])
                raw_path.write_bytes(data)
            except Exception as e:
                logger.warning(f"Failed to download attachment: {e}")
                continue
            att = Attachment(
                id=None,
                message_id=msg.id,
                filename=safe_name,
                mime_type=att_meta["mime_type"],
                size_bytes=len(data),
                raw_path=str(raw_path),
            )
            upsert_attachment(conn, att)

        count += 1

    # If we recovered from a 404, also pull the account's CURRENT
    # historyId so the next cycle starts from a fresh checkpoint —
    # otherwise we'd only advance to the newest message's history_id,
    # which might still be old if the inbox was quiet when we
    # recovered. Belt-and-suspenders: also guard against max_history_id
    # being somehow < the prior checkpoint.
    if history_expired:
        try:
            profile = service.users().getProfile(userId="me").execute()
            current = int(profile.get("historyId", 0))
            if current > max_history_id:
                max_history_id = current
        except Exception as e:
            logger.warning(f"couldn't fetch current historyId after recovery: {e}")

    if max_history_id > int(last_history_id or 0):
        set_sync_state(conn, "last_history_id", str(max_history_id))
    conn.close()
    return count
