"""Owner-authorized opaque attachment jobs; never decode documents on the host.

The injected trusted backend MUST run one job in a fresh resource-bounded VM,
stop/reap that VM on every exit, and have an independent watchdog. There is no
in-process fallback. ``load`` and ``authorized`` are trusted controller callbacks;
owner_id comes from the authenticated session, never a guest request field.
This interface is staged and not mounted in public routes.
"""
import base64
from dataclasses import dataclass
import json
import struct

INPUT_BYTES = 10 * 1024**2
OUTPUT_BYTES = 8 * 1024**2
MAX_PIXELS = 4_000_000
MIMES = frozenset({'application/pdf', 'application/zip', 'image/png', 'image/jpeg',
                   'image/gif', 'image/webp', 'image/tiff'})


class AttachmentDenied(RuntimeError):
    def __init__(self):
        super().__init__('Attachment sandbox job unavailable')


@dataclass(frozen=True)
class AttachmentInput:
    owner_id: str
    attachment_id: int
    mime_type: str
    data: bytes


@dataclass(frozen=True)
class AttachmentPage:
    number: int
    width: int
    height: int
    png: bytes


@dataclass(frozen=True)
class AttachmentResult:
    text: str
    pages: tuple[AttachmentPage, ...]
    truncated: bool


def validate_request(attachment_id, *, dpi=100, pages=()):
    """Validate the fixed parser request before loading private bytes."""
    if (type(attachment_id) is not int or attachment_id <= 0
            or type(dpi) is not int or not 72 <= dpi <= 200
            or type(pages) is not tuple or len(pages) > 8
            or any(type(page) is not int or not 1 <= page <= 100000 for page in pages)
            or len(set(pages)) != len(pages)):
        raise AttachmentDenied()


def validate_result(raw):
    try:
        if type(raw) is not bytes or len(raw) > OUTPUT_BYTES:
            raise AttachmentDenied()
        result = json.loads(raw)
        if (type(result) is not dict or set(result) != {'text', 'pages', 'truncated'}
                or type(result['text']) is not str or len(result['text'].encode()) > 200_000
                or type(result['pages']) is not list or len(result['pages']) > 8
                or type(result['truncated']) is not bool):
            raise AttachmentDenied()
        pages = []
        for page in result['pages']:
            if (type(page) is not dict or set(page) != {'number', 'width', 'height', 'png'}
                    or any(type(page[key]) is not int for key in ('number', 'width', 'height'))
                    or not 1 <= page['number'] <= 100000
                    or not 1 <= page['width'] <= 4000 or not 1 <= page['height'] <= 4000
                    or page['width'] * page['height'] > MAX_PIXELS or type(page['png']) is not str):
                raise AttachmentDenied()
            png = base64.b64decode(page['png'], validate=True)
            # Inspect only the fixed PNG envelope; no host image decompression.
            if (len(png) < 33 or png[:16] != b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'
                    or struct.unpack('!II', png[16:24]) != (page['width'], page['height'])):
                raise AttachmentDenied()
            pages.append(AttachmentPage(page['number'], page['width'], page['height'], png))
        return AttachmentResult(result['text'], tuple(pages), result['truncated'])
    except (ValueError, TypeError, UnicodeError, RecursionError, KeyError):
        raise AttachmentDenied() from None


class AttachmentSandbox:
    def __init__(self, backend, *, load, authorized):
        self.backend, self.load, self.authorized = backend, load, authorized

    def parse(self, owner_id, attachment_id, *, dpi=100, pages=()):
        if type(owner_id) is not str or not owner_id:
            raise AttachmentDenied()
        validate_request(attachment_id, dpi=dpi, pages=pages)
        if self.authorized(owner_id, attachment_id) is not True:
            raise AttachmentDenied()
        source = self.load(owner_id, attachment_id)
        if (type(source) is not AttachmentInput or source.owner_id != owner_id or source.attachment_id != attachment_id
                or source.mime_type not in MIMES or type(source.data) is not bytes
                or not 0 < len(source.data) <= INPUT_BYTES):
            raise AttachmentDenied()
        if self.authorized(owner_id, attachment_id) is not True:
            raise AttachmentDenied()
        try:
            raw = self.backend.parse(source.data, source.mime_type, {'dpi': dpi, 'pages': list(pages)})
            result = validate_result(raw)
        except AttachmentDenied:
            raise
        except Exception:
            raise AttachmentDenied() from None
        if self.authorized(owner_id, attachment_id) is not True:
            raise AttachmentDenied()
        return result
