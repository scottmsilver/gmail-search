"""Fixed browser citation reads through the existing owner-specific SQL gateway."""
import asyncio
from dataclasses import dataclass
import re
from urllib.parse import quote

from .registry import AccessDenied
from .retrieval import _ID, _COLUMNS, _body_response, _add_manifest


class AmbiguousThread(ValueError):
    pass


@dataclass(frozen=True)
class BrowserAttachment:
    """One attachment's bytes plus the metadata needed to hand them over.

    Frozen and owner-stamped so a response cannot be assembled from one owner's
    bytes and another's metadata.
    """

    owner_id: str
    attachment_id: int
    filename: str
    mime_type: str
    size_bytes: int
    message_id: str
    thread_id: str
    data: bytes

    @property
    def download_headers(self):
        """Always `application/octet-stream`, never the declared `mime_type`.

        A mailbox attachment is untrusted bytes from a third party. Serving them
        under their claimed type invites the browser to render them in the app's
        origin — an HTML or SVG attachment becomes stored XSS against the very
        session that is allowed to read the mailbox. The sandbox CSP and
        `nosniff` are the same argument twice over.
        """
        return {
            'Content-Type': 'application/octet-stream',
            'Content-Disposition': "attachment; filename*=UTF-8''" + quote(self.filename, safe=''),
            'Cache-Control': 'private, no-store',
            'X-Content-Type-Options': 'nosniff',
            'Content-Security-Policy': "sandbox; default-src 'none'",
        }


class BrowserMail:
    def __init__(self, gateway, *, attachment_reader=None, attachment_source=None):
        if attachment_reader is not None and attachment_reader.gateway is not gateway:
            raise ValueError('Shared owner query gateway required')
        if attachment_source is not None and attachment_reader is None:
            raise ValueError('Attachment bytes require the owner-scoped reader')
        self.gateway, self.attachment_reader = gateway, attachment_reader
        self.attachment_source = attachment_source

    async def lookup(self, owner, prefix):
        if type(prefix) is not str or not re.fullmatch('[a-f0-9]{4,20}', prefix):
            raise ValueError('Invalid citation')
        result = await self.gateway.query(owner,
            f"SELECT DISTINCT thread_id FROM messages WHERE thread_id LIKE '{prefix}%' LIMIT 2")
        if not result.complete or len(result.rows)>1:
            raise AmbiguousThread()
        if not result.rows:
            raise AccessDenied()
        return {'thread_id':result.rows[0][0]}

    async def thread(self, owner, thread_id, *, message_offset=0, body_offset=0):
        if (type(thread_id) is not str or not _ID.fullmatch(thread_id)
                or type(message_offset) is not int or not 0<=message_offset<=10000
                or type(body_offset) is not int or not 0<=body_offset<=2147483646):
            raise ValueError('Invalid thread page')
        result = await self.gateway.query(owner,
            f"SELECT {_COLUMNS}, pg_catalog.substr(body_text, {body_offset+1}, 20000) AS body_text, "
            f"length(body_text) AS body_total_chars FROM messages WHERE thread_id = '{thread_id}' "
            f'ORDER BY date, id LIMIT 21 OFFSET {message_offset}')
        response = _body_response(result,thread_id,message_offset,20,body_offset,20000)
        if not response['messages']:
            raise AccessDenied()
        for message in response['messages']:
            for field in ('from_addr','to_addr','subject','date'):
                if message[field] is None:
                    message[field]=''
        if self.attachment_reader is not None:
            async def active():
                self.gateway.registry.credential(owner)
                return True
            page = await self.attachment_reader.list_for_thread(owner,thread_id,
                after_attachment_id=0,limit=100,deadline=asyncio.get_running_loop().time()+10,check_active=active)
            _add_manifest(response,page,owner,thread_id,0,100)
        else:
            for message in response['messages']:
                message.update(attachments=[],attachments_complete=False)
        return response

    def _described(self, owner, attachment_id, *, deadline, check_active):
        if self.attachment_reader is None:
            raise AccessDenied()
        return self.attachment_reader.describe(owner, attachment_id,
                                               deadline=deadline, check_active=check_active)

    async def attachment_metadata(self, owner, attachment_id, *, deadline, check_active):
        """What the browser may know about one attachment, scoped to its owner.

        Deliberately a narrow projection rather than the reader's record: the
        caller gets what it needs to render a download link, not the stored text
        state or extraction bookkeeping.
        """
        record = await self._described(owner, attachment_id, deadline=deadline, check_active=check_active)
        if record.owner_id != owner or record.attachment_id != attachment_id:
            raise AccessDenied()
        return {
            'attachment_id': record.attachment_id,
            'filename': record.filename,
            'mime_type': record.mime_type,
            'size_bytes': record.size_bytes,
            'message_id': record.message_id,
            'thread_id': record.thread_id,
        }

    async def attachment_download(self, owner, attachment_id, *, deadline, check_active):
        """The attachment's bytes, bound to the owner the metadata came from.

        Metadata is read first and through the owner-scoped reader, so the name
        and type shipped in the response headers come from the same owner's row
        as the bytes. Loading bytes first and labelling them afterwards is how a
        response ends up with one owner's file under another owner's name.

        The source's own cleanup is allowed to finish if this is cancelled
        mid-read — `load_raw` holds open descriptors, and abandoning it would
        leak them.
        """
        if self.attachment_source is None:
            raise AccessDenied()
        record = await self._described(owner, attachment_id, deadline=deadline, check_active=check_active)
        if record.owner_id != owner or record.attachment_id != attachment_id:
            raise AccessDenied()
        loaded = await self.attachment_source.load_raw(owner, attachment_id)
        if loaded.owner_id != owner or loaded.attachment_id != attachment_id:
            raise AccessDenied()
        return BrowserAttachment(owner, record.attachment_id, record.filename, record.mime_type,
                                 len(loaded.data), record.message_id, record.thread_id, loaded.data)
