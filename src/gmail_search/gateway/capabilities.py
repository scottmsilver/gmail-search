"""Opaque, revocable run capabilities. Only token hashes enter durable storage."""
from dataclasses import dataclass, field
import hashlib
import json
import re
import secrets

from .registry import AccessDenied, Registry, RunLease, _ttl

_NAME = re.compile(r'[a-z][a-z0-9_.:-]{0,63}\Z')


@dataclass(frozen=True)
class Capability:
    secret: str = field(repr=False)
    expires_at: float


def _hash(token):
    if not isinstance(token, str) or len(token) != 64 or not re.fullmatch(r'[a-f0-9]{64}', token):
        raise AccessDenied()
    return hashlib.sha256(token.encode('ascii')).hexdigest()


class Capabilities:
    def __init__(self, registry: Registry):
        self.registry = registry

    def issue(self, run_id, *, audience, operations, ttl=60):
        _ttl(ttl)
        if not isinstance(audience, str) or not _NAME.fullmatch(audience) or not isinstance(operations, (set, frozenset, list, tuple)) or not 1 <= len(operations) <= 32 or any(not isinstance(op, str) or not _NAME.fullmatch(op) for op in operations):
            raise AccessDenied()
        with self.registry._transaction() as db:
            run = self.registry._active(db, run_id)
            secret = secrets.token_hex(32)
            expires = min(self.registry.clock() + ttl, run['deadline'])
            db.execute('INSERT INTO capabilities(token_hash,run_id,audience,operations,expires_at) VALUES(?,?,?,?,?)', (_hash(secret), run_id, audience, json.dumps(sorted(set(operations))), expires))
            return Capability(secret, expires)

    def _authorize(self, db, token, audience, operation):
        capability = db.execute('SELECT * FROM capabilities WHERE token_hash=?', (_hash(token),)).fetchone()
        if not capability or capability['revoked'] or capability['expires_at'] <= self.registry.clock() or capability['audience'] != audience or operation not in json.loads(capability['operations']):
            raise AccessDenied()
        run = self.registry._active(db, capability['run_id'])
        if operation in ('artifact.commit', 'workspace.commit'):
            self.registry._fence(db, run, writer=operation == 'workspace.commit')
        return run

    def authorize(self, token, *, audience, operation) -> RunLease:
        """Derive identity from token; accepts no caller owner/conversation/run IDs.

        Long operations must call again periodically and before publishing bytes.
        artifact.commit checks the current fence; storage must honor that fence.
        """
        with self.registry._transaction(read_only=True) as db:
            return self.registry._lease(self._authorize(db, token, audience, operation))

    def revoke(self, token):
        with self.registry._transaction() as db:
            db.execute('UPDATE capabilities SET revoked=1 WHERE token_hash=?', (_hash(token),))

    def commit_workspace(self, token):
        """Atomically advance metadata and finish the writer; bytes live elsewhere."""
        with self.registry._transaction() as db:
            run = self._authorize(db, token, 'artifact', 'workspace.commit')
            db.execute('UPDATE conversations SET version=version+1 WHERE owner_id=? AND conversation_id=?', (run['owner_id'], run['conversation_id']))
            self.registry._finish(db, run['run_id'], 'completed')
            return run['workspace_version'] + 1
