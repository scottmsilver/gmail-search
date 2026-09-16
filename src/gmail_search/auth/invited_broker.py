"""Fixed-registration HTTP adapter for the staged account-bound Gmail broker.

Instantiate only in the trusted app controller. Never pass its bearer or returned
access tokens to browsers/agents. An injectable httpx client permits offline tests.
"""
from urllib.parse import parse_qs, urlsplit
import re
import json

import httpx

from .identity_store import CredentialCleanup
from .gmail_consent import ConsentState


class BrokerUnavailable(RuntimeError):
    def __init__(self):
        super().__init__('Account-bound Gmail broker is unavailable.')


def _origin(value):
    try:
        parsed = urlsplit(value)
        if (parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password
            or value != f'https://{parsed.netloc}' or parsed.port not in (None, 443)
            or any(c.isspace() for c in value) or '\\' in value):
            raise ValueError()
    except (ValueError, TypeError):
        raise BrokerUnavailable() from None
    return value


class BoundGmailBroker:
    def __init__(self, origin, *, bearer, signing_secret, client: httpx.Client):
        self.origin = _origin(origin)
        if (client.trust_env is not False or bearer == signing_secret
            or any(not isinstance(secret, str) or len(secret) < 32
                   or any(ord(c) < 33 or ord(c) > 126 for c in secret)
                   for secret in (bearer, signing_secret))):
            raise BrokerUnavailable()
        self._bearer, self.signing_secret, self.client = bearer, signing_secret, client

    def _post(self, path, body):
        response = None
        try:
            request = httpx.Request('POST', self.origin + path,
                content=json.dumps(body).encode('utf-8'),
                headers={'Authorization': 'Bearer ' + self._bearer,
                         'Content-Type': 'application/json', 'Accept-Encoding': 'identity'},
                extensions={'timeout': dict(connect=10, read=10, write=10, pool=10)})
            response = self.client.send(request, auth=None, follow_redirects=False, stream=True)
            if response.status_code != 200 or response.headers.get('content-encoding', 'identity').lower() != 'identity':
                raise BrokerUnavailable()
            payload = bytearray()
            for chunk in response.iter_raw(chunk_size=8192):
                if len(payload) + len(chunk) > 16384:
                    raise BrokerUnavailable()
                payload.extend(chunk)
            result = json.loads(payload)
            if not isinstance(result, dict):
                raise BrokerUnavailable()
            return result
        except (httpx.HTTPError, ValueError, UnicodeError):
            raise BrokerUnavailable() from None
        finally:
            if response is not None:
                response.close()

    def start_consent(self, state: ConsentState):
        result = self._post('/v1/gmail/consents', dict(owner_id=state.owner_id, subject=state.google_subject,
            email=state.email, invitation_generation=state.generation,
            credential_generation=state.credential_generation, state=state.secret))
        if set(result) != {'url'} or not isinstance(result['url'], str):
            raise BrokerUnavailable()
        try:
            target = urlsplit(result['url'])
            query = parse_qs(target.query, keep_blank_values=True)
        except ValueError:
            raise BrokerUnavailable() from None
        if (target.scheme + '://' + target.netloc != self.origin or target.path != '/v1/gmail/start'
            or target.fragment or set(query) != {'ticket'} or len(query['ticket']) != 1
            or not re.fullmatch('[a-f0-9]{64}', query['ticket'][0])
            or result['url'] != self.origin + '/v1/gmail/start?ticket=' + query['ticket'][0]):
            raise BrokerUnavailable()
        return result['url']

    def disconnect(self, cleanup: CredentialCleanup):
        result = self._post('/v1/gmail/disconnect', dict(owner_id=cleanup.owner_id, subject=cleanup.google_subject,
            email=cleanup.email, invitation_generation=cleanup.invitation_generation,
            credential_generation=cleanup.credential_generation))
        if result != {'disconnected': True}:
            raise BrokerUnavailable()

    def drain_cleanup(self, consent):
        """Trusted startup/worker hook, never a browser-controlled owner sweep.

        Failed requests leave the current and remaining durable tasks untouched.
        The broker treats cleanup of an already newer generation as success.
        """
        completed = 0
        for cleanup in consent.pending_cleanup():
            self.disconnect(cleanup)
            consent.complete_cleanup(cleanup)
            completed += 1
        return completed
