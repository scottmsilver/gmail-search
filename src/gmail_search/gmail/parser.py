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
    # Pad only when length is not already a multiple of 4 (see client.py).
    padded = data + "=" * (-len(data) % 4)
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


def _parse_rfc2822_to_utc(s: str):
    """Parse an RFC 2822 date string to a UTC-aware datetime, or
    return None if unparseable.
    """
    from datetime import timezone

    try:
        d = parsedate_to_datetime(s)
    except (ValueError, TypeError):
        return None
    if d.tzinfo:
        return d.astimezone(timezone.utc)
    return d.replace(tzinfo=timezone.utc)


def _date_from_received_headers(headers: list[dict]):
    """Extract a UTC datetime from the first parseable `Received:`
    header. Gmail (and every conformant MTA) stamps a receipt time
    after the last semicolon:

        Received: by <host> with SMTP id <id>; Sat, 4 Apr 2026 16:31:03 -0700

    Headers appear in prepend order (newest first), so iterating in
    order gives us Gmail's own stamp first. Skip hops that have no
    semicolon (some intermediate MTAs skip the date).
    """
    for h in headers:
        if h.get("name", "").lower() != "received":
            continue
        val = h.get("value", "")
        if ";" not in val:
            continue
        # Everything after the LAST semicolon is the RFC2822 date.
        date_part = val.rsplit(";", 1)[1].strip()
        parsed = _parse_rfc2822_to_utc(date_part)
        if parsed is not None:
            return parsed
    return None


def _parse_message_date(raw: dict[str, Any], headers: list[dict]) -> datetime:
    """Determine the message timestamp. Gmail gives us three sources,
    each authoritative in a different failure mode:

    1. **`internalDate`** — Gmail's received-time in ms. Usually
       bulletproof, but we've seen it come back as `"0"` on spoofed
       phishing mail (Gmail API oddity; the actual receipt time is
       still known, just not exposed through that field).
    2. **`Received:` headers** — every MTA, Gmail included, prepends
       a `Received:` header with a semicolon-separated RFC 2822 date.
       The first one in the list is Gmail's own stamp.
    3. **`Date:` header** — sender-supplied; can be malformed or
       spoofed (e.g. "04-03-2026" — not RFC 2822).

    Preference: internalDate → Received → Date. Epoch fallback only
    if all three fail.
    """
    from datetime import timezone

    # 1. Gmail's own received-time field.
    internal = raw.get("internalDate")
    if internal is not None:
        try:
            ms = int(internal)
            if ms > 0:
                return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
        except (TypeError, ValueError):
            pass

    # 2. Gmail's (or any MTA's) own `Received:` stamp — authoritative
    #    even when `internalDate` is 0.
    recvd = _date_from_received_headers(headers)
    if recvd is not None:
        return recvd

    # 3. Sender-supplied Date.
    date_str = _get_header(headers, "Date")
    if date_str:
        parsed = _parse_rfc2822_to_utc(date_str)
        if parsed is not None:
            return parsed

    # 4. All three broken. Epoch so the UI can render distinctly.
    return datetime(1970, 1, 1, tzinfo=timezone.utc)


def _strip_nul(s: str) -> str:
    """PG TEXT columns reject NUL (0x00). Gmail occasionally ships
    attachments / mis-encoded bodies that decode to strings containing
    NUL bytes — without this, the first such message crashes the
    update / summarize / watch daemons with `psycopg.DataError:
    PostgreSQL text fields cannot contain NUL (0x00) bytes`. Strip at
    the parser boundary so every downstream consumer is safe.
    """
    if not s:
        return s
    return s.replace("\x00", "") if "\x00" in s else s


def parse_message(raw: dict[str, Any]) -> tuple[Message, list[dict]]:
    payload = raw["payload"]
    headers = payload.get("headers", [])

    date = _parse_message_date(raw, headers)

    body_text, body_html, att_metas = _extract_parts(payload)

    msg = Message(
        id=raw["id"],
        thread_id=raw.get("threadId", raw["id"]),
        from_addr=_strip_nul(_get_header(headers, "From")),
        to_addr=_strip_nul(_get_header(headers, "To")),
        subject=_strip_nul(_get_header(headers, "Subject")),
        body_text=_strip_nul(body_text),
        body_html=_strip_nul(body_html),
        date=date,
        labels=raw.get("labelIds", []),
        history_id=int(raw.get("historyId", 0)),
        raw_json=_strip_nul(json.dumps(raw)),
    )

    return msg, att_metas
