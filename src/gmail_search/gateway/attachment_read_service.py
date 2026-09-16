"""Run-authorized attachment metadata, stored text, and thread manifests.

No raw bytes, filesystem, parser, extraction writes or provider calls. Public
projection names each field explicitly, omitting the internal owner binding.
"""

import asyncio
import json

from . import attachment_reader as ar
from .registry import AccessDenied
from .service import _drain

MAX_RESPONSE_BYTES = 4 * 1024 * 1024
_PUBLIC_METADATA = (
    "attachment_id",
    "message_id",
    "thread_id",
    "filename",
    "mime_type",
    "size_bytes",
    "fetch_status",
    "text_chars",
    "stored_text_state",
    "extraction_complete",
)


def _metadata(value, owner, *, attachment_id=None, thread_id=None):
    if type(value) is not ar.AttachmentMetadata:
        raise ar.AttachmentReadUnavailable()
    # Reuse the reader's bounded pure decoder rather than accepting an arbitrary
    # object with a matching owner attribute. State/completion are also canonical.
    decoded = ar._metadata(
        tuple(getattr(value, key) for key in ar.METADATA_COLUMNS), owner
    )
    if (
        decoded != value
        or (attachment_id is not None and value.attachment_id != attachment_id)
        or (thread_id is not None and value.thread_id != thread_id)
    ):
        raise ar.AttachmentReadUnavailable()
    result = {key: getattr(value, key) for key in _PUBLIC_METADATA}
    result["cite_ref"] = value.thread_id
    return result


def _text_projection(page, owner, attachment_id, offset, limit):
    if type(page) is not ar.AttachmentTextPage:
        raise ar.AttachmentReadUnavailable()
    result = _metadata(page.attachment, owner, attachment_id=attachment_id)
    if (
        type(page.offset) is not int
        or page.offset != offset
        or type(page.limit) is not int
        or page.limit != limit
        or page.total_chars != page.attachment.text_chars
        or any(
            type(getattr(page, key)) is not bool
            for key in ("page_complete", "stored_text_complete", "pagination_limited")
        )
    ):
        raise ar.AttachmentReadUnavailable()
    total = page.attachment.text_chars
    if total is None:
        expected = (None, False, False, False)
        if page.text is not None or page.total_chars is not None:
            raise ar.AttachmentReadUnavailable()
    else:
        if (
            type(page.total_chars) is not int
            or type(page.text) is not str
            or len(page.text) != min(limit, max(0, total - offset))
        ):
            raise ar.AttachmentReadUnavailable()
        end = min(offset + limit, total)
        more = end < total
        limited = more and end > ar._MAX_OFFSET
        expected = (
            end if more and not limited else None,
            True,
            offset == 0 and not more,
            limited,
        )
    if (page.next_offset is not None and type(page.next_offset) is not int) or (
        page.next_offset,
        page.page_complete,
        page.stored_text_complete,
        page.pagination_limited,
    ) != expected:
        raise ar.AttachmentReadUnavailable()
    result.update(
        extracted_text=page.text,
        offset=offset,
        limit=limit,
        total_chars=page.total_chars,
        next_offset=page.next_offset,
        page_complete=page.page_complete,
        stored_text_complete=page.stored_text_complete,
        pagination_limited=page.pagination_limited,
    )
    return result


def _list_projection(page, owner, thread_id, after, limit):
    if (
        type(page) is not ar.AttachmentMetadataPage
        or type(page.items) is not tuple
        or len(page.items) > limit
        or type(page.after_attachment_id) is not int
        or page.after_attachment_id != after
        or type(page.limit) is not int
        or page.limit != limit
        or any(
            type(getattr(page, key)) is not bool
            for key in ("source_complete", "complete", "pagination_limited")
        )
    ):
        raise ar.AttachmentReadUnavailable()
    items = []
    previous = after
    for item in page.items:
        public = _metadata(item, owner, thread_id=thread_id)
        if item.attachment_id <= previous:
            raise ar.AttachmentReadUnavailable()
        previous = item.attachment_id
        public["id"] = item.attachment_id  # Existing thread attachment consumers.
        items.append(public)
    limited = not page.source_complete and (not items or previous == ar._MAX_ID)
    next_id = previous if not page.source_complete and not limited else None
    if (
        page.complete != (page.source_complete and after == 0)
        or page.pagination_limited != limited
        or page.next_attachment_id != next_id
        or (
            page.next_attachment_id is not None
            and type(page.next_attachment_id) is not int
        )
    ):
        raise ar.AttachmentReadUnavailable()
    return dict(
        thread_id=thread_id,
        cite_ref=thread_id,
        attachments=items,
        after_attachment_id=after,
        limit=limit,
        source_complete=page.source_complete,
        complete=page.complete,
        next_attachment_id=next_id,
        pagination_limited=limited,
    )


def _bounded(value):
    try:
        if (
            len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"))
            > MAX_RESPONSE_BYTES
        ):
            raise ar.AttachmentReadUnavailable()
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise ar.AttachmentReadUnavailable() from None
    return value


class RunAttachmentReadService:
    def __init__(self, capabilities, reader):
        self.capabilities, self.reader = capabilities, reader

    async def authorize(self, token, operation):
        if type(operation) is not str or operation not in ("meta", "text"):
            raise ValueError("Invalid attachment read operation")
        return await _drain(
            asyncio.to_thread(
                self.capabilities.authorize,
                token,
                audience="attachment",
                operation=operation,
            )
        )

    async def _run(self, token, operation, read, project):
        deadline = asyncio.get_running_loop().time() + 30
        task = None

        async def check():
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("Attachment read deadline exceeded")
            current = await self.authorize(token, operation)
            if current.owner_id != lease.owner_id or current.run_id != lease.run_id:
                raise AccessDenied()
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("Attachment read deadline exceeded")
            return True

        try:
            try:
                async with asyncio.timeout_at(deadline):
                    lease = await self.authorize(token, operation)
                    task = asyncio.create_task(read(lease.owner_id, deadline, check))
                    value = await asyncio.shield(task)
            finally:
                if task is not None:
                    if not task.done() and not task.cancelling():
                        task.cancel()
                    await _drain(asyncio.gather(task, return_exceptions=True))
            # Validate and bound the projected object before the final fresh
            # authorization. No owned cleanup or output processing follows it.
            response = _bounded(project(value, lease.owner_id))
        except (AccessDenied, TimeoutError, asyncio.CancelledError):
            raise
        except Exception:
            raise ar.AttachmentReadUnavailable() from None
        await check()
        return response

    async def describe(self, token, *, attachment_id):
        ar._id(attachment_id)

        async def read(owner, deadline, check):
            return await self.reader.describe(
                owner, attachment_id, deadline=deadline, check_active=check
            )

        return await self._run(
            token,
            "meta",
            read,
            lambda value, owner: _metadata(value, owner, attachment_id=attachment_id),
        )

    async def text_page(self, token, *, attachment_id, offset=0, limit=20000):
        ar._id(attachment_id)
        if (
            type(offset) is not int
            or not 0 <= offset <= ar._MAX_OFFSET
            or type(limit) is not int
            or not 1 <= limit <= 100000
        ):
            raise ValueError("Invalid attachment text page")

        async def read(owner, deadline, check):
            return await self.reader.text_page(
                owner,
                attachment_id,
                offset=offset,
                limit=limit,
                deadline=deadline,
                check_active=check,
            )

        return await self._run(
            token,
            "text",
            read,
            lambda value, owner: _text_projection(
                value, owner, attachment_id, offset, limit
            ),
        )

    async def list_for_thread(
        self, token, *, thread_id, after_attachment_id=0, limit=100
    ):
        if type(thread_id) is not str or not ar._ID.fullmatch(thread_id):
            raise ValueError("Invalid attachment thread identifier")
        ar._id(after_attachment_id, zero=True)
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("Invalid attachment list limit")

        async def read(owner, deadline, check):
            return await self.reader.list_for_thread(
                owner,
                thread_id,
                after_attachment_id=after_attachment_id,
                limit=limit,
                deadline=deadline,
                check_active=check,
            )

        return await self._run(
            token,
            "meta",
            read,
            lambda value, owner: _list_projection(
                value, owner, thread_id, after_attachment_id, limit
            ),
        )
