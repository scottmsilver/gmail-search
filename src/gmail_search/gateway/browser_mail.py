"""Fixed browser citation reads through the existing owner-specific SQL gateway."""
import asyncio
import re

from .registry import AccessDenied
from .retrieval import _ID, _COLUMNS, _body_response, _add_manifest


class AmbiguousThread(ValueError):
    pass


class BrowserMail:
    def __init__(self, gateway, *, attachment_reader=None):
        if attachment_reader is not None and attachment_reader.gateway is not gateway:
            raise ValueError('Shared owner query gateway required')
        self.gateway, self.attachment_reader = gateway, attachment_reader

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
