"""Bounded opaque reads from owner-specific attachment directories.

The async locator uses the existing fixed-owner analytical reader and derives
the new-layout storage location from bounded metadata. It is a trusted internal
dependency, never a request callback. No raw_path privilege is needed.
Legacy shared attachment paths deliberately require a separate migration or
reviewed mapping; this loader never falls back to the shared directory.
"""
import asyncio
from dataclasses import dataclass, field
import hashlib
import inspect
import os
import re
from pathlib import Path
import stat

from .attachment_sandbox import AttachmentDenied, AttachmentInput, INPUT_BYTES, MIMES


@dataclass(frozen=True)
class AttachmentLocator:
    owner_id: str
    attachment_id: int
    mime_type: str
    path: str = field(repr=False)
    expected_size: int | None = None


@dataclass(frozen=True)
class RawAttachmentInput:
    owner_id: str
    attachment_id: int
    mime_type: str
    data: bytes = field(repr=False)


def _raw_mime(value):
    return (type(value) is str and len(value) <= 255
            and re.fullmatch(r"[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+", value) is not None)


class QueryAttachmentLocator:
    """Locate new-layout bytes without selecting or trusting a stored host path."""
    _columns = ('id', 'message_id', 'filename', 'mime_type', 'size_bytes', 'fetch_status', 'user_id')

    def __init__(self, gateway, root):
        self.gateway, self.root = gateway, Path(root)
        if not self.root.is_absolute() or '..' in self.root.parts:
            raise AttachmentDenied()

    @staticmethod
    def _component(value):
        return (type(value) is str and 0 < len(value.encode()) <= 255
                and value not in ('.', '..')
                and not any(ord(c) < 32 or ord(c) == 127 or c in '/\\' for c in value))

    async def locate(self, owner_id, attachment_id):
        return await self._locate(owner_id, attachment_id, raw=False)

    async def locate_raw(self, owner_id, attachment_id):
        return await self._locate(owner_id, attachment_id, raw=True)

    async def _locate(self, owner_id, attachment_id, *, raw):
        if (type(owner_id) is not str or not owner_id or len(owner_id) > 2048
                or type(attachment_id) is not int or not 0 < attachment_id <= 9223372036854775807):
            raise AttachmentDenied()
        result = await self.gateway.query(owner_id,
            f"SELECT {', '.join(self._columns)} FROM attachments WHERE id = {attachment_id} LIMIT 2")
        if not result.complete or result.columns != self._columns or len(result.rows) != 1:
            raise AttachmentDenied()
        row = dict(zip(result.columns, result.rows[0], strict=True))
        try:
            mime = row['mime_type']
            if raw and mime is None:
                mime = 'application/octet-stream'
            if (type(row['id']) is not int or row['id'] != attachment_id or row['user_id'] != owner_id
                    or not self._component(row['message_id']) or not self._component(row['filename'])
                    or not (_raw_mime(mime) if raw else mime in MIMES) or row['fetch_status'] != 'ok'
                    or type(row['size_bytes']) is not int or not (0 if raw else 1) <= row['size_bytes'] <= INPUT_BYTES):
                raise AttachmentDenied()
            path = self.root / 'owners' / hashlib.sha256(owner_id.encode()).hexdigest() / row['message_id'] / row['filename']
            return AttachmentLocator(owner_id, attachment_id, mime, str(path), row['size_bytes'])
        except (UnicodeError, TypeError):
            raise AttachmentDenied() from None


class OwnerAttachmentSource:
    def __init__(self, root, *, locate):
        self.root = Path(root)
        if not self.root.is_absolute() or '..' in self.root.parts or not inspect.iscoroutinefunction(locate):
            raise AttachmentDenied()
        self.locate = locate

    async def load(self, owner_id, attachment_id):
        return await self._load(owner_id, attachment_id, raw=False)

    async def load_raw(self, owner_id, attachment_id):
        return await self._load(owner_id, attachment_id, raw=True)

    async def _load(self, owner_id, attachment_id, *, raw):
        if (type(owner_id) is not str or not owner_id or len(owner_id) > 2048
                or type(attachment_id) is not int or not 0 < attachment_id <= 9223372036854775807):
            raise AttachmentDenied()
        descriptors = []
        try:
            record = await self.locate(owner_id, attachment_id)
            if (type(record) is not AttachmentLocator or record.owner_id != owner_id
                    or record.attachment_id != attachment_id
                    or not (_raw_mime(record.mime_type) if raw else record.mime_type in MIMES)
                    or type(record.path) is not str or len(record.path) > 4096):
                raise AttachmentDenied()
            if ((raw or record.expected_size is not None)
                    and (type(record.expected_size) is not int or not (0 if raw else 1) <= record.expected_size <= INPUT_BYTES)):
                raise AttachmentDenied()
            path = Path(record.path)
            if not path.is_absolute() or '..' in path.parts or str(path) != record.path:
                raise AttachmentDenied()
            parts = path.relative_to(self.root).parts
            prefix = ('owners', hashlib.sha256(owner_id.encode()).hexdigest())
            if len(parts) < 4 or len(parts) > 16 or parts[:2] != prefix:
                raise AttachmentDenied()
            # Only the configured root is trusted. Every DB-supplied path segment
            # is opened relative to the preceding held descriptor, without links.
            directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
            directory = os.open(self.root, directory_flags)
            descriptors.append(directory)
            for part in parts[:-1]:
                directory = os.open(part, directory_flags, dir_fd=directory)
                descriptors.append(directory)
            fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=directory)
            descriptors.append(fd)
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or not (0 if raw else 1) <= before.st_size <= INPUT_BYTES:
                raise AttachmentDenied()
            if record.expected_size is not None and before.st_size != record.expected_size:
                raise AttachmentDenied()
            data = bytearray()
            while len(data) <= INPUT_BYTES:
                await asyncio.sleep(0)  # cancellation checkpoint between bounded reads
                chunk = os.read(fd, min(65536, INPUT_BYTES + 1 - len(data)))
                if not chunk:
                    break
                data.extend(chunk)
            after = os.fstat(fd)
            if (len(data) > INPUT_BYTES or len(data) != before.st_size
                    or (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                    != (after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
                raise AttachmentDenied()
            output_type = RawAttachmentInput if raw else AttachmentInput
            return output_type(owner_id, attachment_id, record.mime_type, bytes(data))
        except (OSError, ValueError, UnicodeError):
            raise AttachmentDenied() from None
        finally:
            for fd in reversed(descriptors):
                os.close(fd)
