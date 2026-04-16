import base64
import json
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any

from gmail_search.store.models import Message


def _get_header(headers: list[dict], name: str) -> str:
    for h in headers:
        if h["name"].lower() == name.lower():
            return h["value"]
    return ""


def _decode_body(data: str) -> str:
    padded = data + "=" * (4 - len(data) % 4)
    return base64.urlsafe_b64decode(padded).decode("utf-8", errors="replace")


def _extract_parts(payload: dict) -> tuple[str, str, list[dict]]:
    """Recursively extract text, html, and attachment metadata from payload."""
    text_parts: list[str] = []
    html_parts: list[str] = []
    attachments: list[dict] = []

    mime_type = payload.get("mimeType", "")

    if "parts" not in payload:
        body = payload.get("body", {})
        if body.get("attachmentId"):
            attachments.append(
                {
                    "filename": payload.get("filename", ""),
                    "mime_type": mime_type,
                    "attachment_id": body["attachmentId"],
                    "size": body.get("size", 0),
                }
            )
        elif "data" in body:
            decoded = _decode_body(body["data"])
            if mime_type == "text/plain":
                text_parts.append(decoded)
            elif mime_type == "text/html":
                html_parts.append(decoded)
        return "\n".join(text_parts), "\n".join(html_parts), attachments

    for part in payload.get("parts", []):
        t, h, a = _extract_parts(part)
        if t:
            text_parts.append(t)
        if h:
            html_parts.append(h)
        attachments.extend(a)

    return "\n".join(text_parts), "\n".join(html_parts), attachments


def parse_message(raw: dict[str, Any]) -> tuple[Message, list[dict]]:
    payload = raw["payload"]
    headers = payload.get("headers", [])

    date_str = _get_header(headers, "Date")
    try:
        from datetime import timezone

        date = parsedate_to_datetime(date_str)
        if date.tzinfo:
            date = date.astimezone(timezone.utc)
        else:
            date = date.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        from datetime import timezone

        date = datetime(1970, 1, 1, tzinfo=timezone.utc)

    body_text, body_html, att_metas = _extract_parts(payload)

    msg = Message(
        id=raw["id"],
        thread_id=raw.get("threadId", raw["id"]),
        from_addr=_get_header(headers, "From"),
        to_addr=_get_header(headers, "To"),
        subject=_get_header(headers, "Subject"),
        body_text=body_text,
        body_html=body_html,
        date=date,
        labels=raw.get("labelIds", []),
        history_id=int(raw.get("historyId", 0)),
        raw_json=json.dumps(raw),
    )

    return msg, att_metas
