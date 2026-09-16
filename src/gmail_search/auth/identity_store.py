"""Staged, durable identity metadata; this module never opens the mailbox database.

Only a trusted controller may invite or provision accounts. ``admit`` consumes
claims already authenticated by the broker adapter, never browser JSON. A Google
subject must be the provider's stable subject, not an email or arbitrary broker
account identifier. Existing public authentication does not use this store yet.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import math
import os
from pathlib import Path
import re
import secrets
import sqlite3
import stat
import time
import uuid


class IdentityDenied(PermissionError):
    def __init__(self):
        super().__init__('Identity admission or lifecycle operation is unavailable.')


@dataclass(frozen=True)
class VerifiedGoogleIdentity:
    email: str
    subject: str = field(repr=False)
    email_verified: bool


@dataclass(frozen=True)
class Account:
    owner_id: str
    email: str
    generation: int


@dataclass(frozen=True)
class Revocation:
    owner_id: str
    generation: int


@dataclass(frozen=True)
class CredentialCleanup:
    owner_id: str
    email: str
    google_subject: str = field(repr=False)
    invitation_generation: int
    credential_generation: int


_SCHEMA = '''
CREATE TABLE IF NOT EXISTS identities (
 owner_id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL,
 google_subject TEXT UNIQUE, invited INTEGER NOT NULL DEFAULT 1,
 provisioned INTEGER NOT NULL DEFAULT 0, generation INTEGER NOT NULL DEFAULT 1);
CREATE TABLE IF NOT EXISTS browser_sessions (
 token_hash TEXT PRIMARY KEY, owner_id TEXT NOT NULL REFERENCES identities(owner_id),
 generation INTEGER NOT NULL, expires_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS browser_sessions_owner ON browser_sessions(owner_id);
CREATE TABLE IF NOT EXISTS gmail_connections (
 owner_id TEXT PRIMARY KEY REFERENCES identities(owner_id), credential_generation INTEGER NOT NULL,
 invitation_generation INTEGER NOT NULL, connected INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS gmail_credential_cleanup (
 owner_id TEXT NOT NULL REFERENCES identities(owner_id), email TEXT NOT NULL, google_subject TEXT NOT NULL,
 invitation_generation INTEGER NOT NULL, credential_generation INTEGER NOT NULL,
 PRIMARY KEY(owner_id,credential_generation));
CREATE TABLE IF NOT EXISTS identity_revocations (
 owner_id TEXT PRIMARY KEY REFERENCES identities(owner_id), generation INTEGER NOT NULL);
'''


def _email(value):
    if not isinstance(value, str):
        raise IdentityDenied()
    value = value.strip().lower()
    if not 3 <= len(value) <= 254 or value.count('@') != 1 or any(c.isspace() or ord(c) < 33 for c in value):
        raise IdentityDenied()
    local, domain = value.split('@')
    if not local or not domain:
        raise IdentityDenied()
    # Do not collapse dots, plus tags, domains, or Google aliases.
    return value


def _token_hash(token):
    if not isinstance(token, str) or not re.fullmatch('[a-f0-9]{64}', token):
        return None
    return hashlib.sha256(token.encode('ascii')).hexdigest()


class IdentityStore:
    def __init__(self, path, *, clock=time.time):
        self.path, self.clock = Path(path), clock
        parent = self.path.parent.lstat()
        if not stat.S_ISDIR(parent.st_mode) or stat.S_IMODE(parent.st_mode) != 0o700 or parent.st_uid != os.getuid():
            raise IdentityDenied()
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600 or info.st_uid != os.getuid():
                raise IdentityDenied()
        finally:
            os.close(fd)
        with self._transaction() as db:
            for statement in _SCHEMA.split(';'):
                if statement.strip():
                    db.execute(statement)

    @contextmanager
    def _transaction(self):
        db = None
        try:
            db = sqlite3.connect(self.path, timeout=5, isolation_level=None)
            db.row_factory = sqlite3.Row
            db.execute('PRAGMA foreign_keys=ON')
            db.execute('BEGIN IMMEDIATE')
            yield db
            db.commit()
        except sqlite3.Error:
            raise IdentityDenied() from None
        finally:
            if db is not None:
                db.close()

    @staticmethod
    def _account(row):
        return Account(row['owner_id'], row['email'], row['generation'])

    @staticmethod
    def _ready(db, row):
        return bool(row and row['invited'] and row['provisioned'] and not db.execute(
            'SELECT 1 FROM identity_revocations WHERE owner_id=?', (row['owner_id'],)
        ).fetchone())

    def invite(self, email):
        """Trusted administrator operation; owner IDs are always generated here."""
        email = _email(email)
        with self._transaction() as db:
            row = db.execute('SELECT * FROM identities WHERE email=?', (email,)).fetchone()
            if row:
                if db.execute('SELECT 1 FROM identity_revocations WHERE owner_id=?', (row['owner_id'],)).fetchone():
                    raise IdentityDenied()
                db.execute('UPDATE identities SET invited=1 WHERE owner_id=?', (row['owner_id'],))
            else:
                db.execute('INSERT INTO identities(owner_id,email) VALUES(?,?)', (uuid.uuid4().hex, email))
            return self._account(db.execute('SELECT * FROM identities WHERE email=?', (email,)).fetchone())

    @staticmethod
    def _verified(identity):
        if not isinstance(identity, VerifiedGoogleIdentity) or identity.email_verified is not True:
            raise IdentityDenied()
        if not isinstance(identity.subject, str) or not 1 <= len(identity.subject) <= 255 or any(c.isspace() or ord(c) < 33 for c in identity.subject):
            raise IdentityDenied()
        return _email(identity.email)

    def prepare_admission(self, identity: VerifiedGoogleIdentity):
        """Bind a verified invited identity before trusted mailbox provisioning.

        This grants no session and never resolves a mailbox owner by email.
        """
        email = self._verified(identity)
        with self._transaction() as db:
            row = db.execute('SELECT * FROM identities WHERE email=?', (email,)).fetchone()
            if not row or not row['invited'] or row['google_subject'] not in (None, identity.subject) or db.execute(
                'SELECT 1 FROM identity_revocations WHERE owner_id=?', (row['owner_id'],)
            ).fetchone():
                raise IdentityDenied()
            db.execute('UPDATE identities SET google_subject=? WHERE owner_id=?', (identity.subject, row['owner_id']))
            return self._account(row)

    def import_existing_account(self, *, owner_id: str, identity: VerifiedGoogleIdentity):
        """Offline trusted import after verifying the old server-owned user ID.

        No browser route calls this. The operator must verify the old users.id,
        exact email and Google subject independently; a nullable legacy
        google_sub is not proof. Conflicts require an explicit migration, never
        automatic ID reassignment. Import alone neither invites nor provisions.
        """
        email = self._verified(identity)
        if not isinstance(owner_id, str) or not 1 <= len(owner_id) <= 512 or any(c.isspace() or ord(c) < 33 for c in owner_id):
            raise IdentityDenied()
        with self._transaction() as db:
            rows = db.execute('SELECT * FROM identities WHERE owner_id=? OR email=? OR google_subject=?',
                              (owner_id, email, identity.subject)).fetchall()
            if rows:
                if len(rows) != 1 or (rows[0]['owner_id'], rows[0]['email'], rows[0]['google_subject']) != (owner_id, email, identity.subject):
                    raise IdentityDenied()
                return self._account(rows[0])
            db.execute('INSERT INTO identities(owner_id,email,google_subject,invited,provisioned) VALUES(?,?,?,0,0)',
                       (owner_id, email, identity.subject))
            return self._account(db.execute('SELECT * FROM identities WHERE owner_id=?', (owner_id,)).fetchone())

    def mark_provisioned(self, owner_id, *, generation=None):
        """Trusted provisioning completion, after fixed owner access is verified.

        Creating an invitation alone never grants admission. Provisioning must
        create only this owner's empty account and owner-bound database readers.
        """
        with self._transaction() as db:
            row = db.execute('SELECT * FROM identities WHERE owner_id=?', (owner_id,)).fetchone()
            if not row or not row['invited'] or (generation is not None and row['generation'] != generation) or db.execute('SELECT 1 FROM identity_revocations WHERE owner_id=?', (owner_id,)).fetchone():
                raise IdentityDenied()
            db.execute('UPDATE identities SET provisioned=1 WHERE owner_id=?', (owner_id,))

    def admit(self, claims: VerifiedGoogleIdentity, *, ttl=3600):
        if not isinstance(claims, VerifiedGoogleIdentity) or claims.email_verified is not True:
            raise IdentityDenied()
        if not isinstance(claims.subject, str) or not 1 <= len(claims.subject) <= 255 or any(c.isspace() or ord(c) < 33 for c in claims.subject):
            raise IdentityDenied()
        if isinstance(ttl, bool) or not isinstance(ttl, (int, float)) or not math.isfinite(ttl) or not 0 < ttl <= 3600:
            raise IdentityDenied()
        email = _email(claims.email)
        with self._transaction() as db:
            row = db.execute('SELECT * FROM identities WHERE email=?', (email,)).fetchone()
            if not self._ready(db, row) or row['google_subject'] not in (None, claims.subject):
                raise IdentityDenied()
            # The unique subject constraint also rejects moving one Google
            # identity to another invited email. No automatic mailbox transfer.
            db.execute('UPDATE identities SET google_subject=? WHERE owner_id=?', (claims.subject, row['owner_id']))
            now = self.clock()
            db.execute('DELETE FROM browser_sessions WHERE expires_at<=?', (now,))
            if db.execute('SELECT count(*) FROM browser_sessions WHERE owner_id=?', (row['owner_id'],)).fetchone()[0] >= 100:
                raise IdentityDenied()
            token = secrets.token_hex(32)
            db.execute('INSERT INTO browser_sessions(token_hash,owner_id,generation,expires_at) VALUES(?,?,?,?)',
                       (_token_hash(token), row['owner_id'], row['generation'], now + ttl))
            return token

    def read_session(self, token):
        hashed = _token_hash(token)
        if hashed is None:
            return None
        with self._transaction() as db:
            row = db.execute('''SELECT i.* FROM identities i JOIN browser_sessions s ON s.owner_id=i.owner_id
                WHERE s.token_hash=? AND s.generation=i.generation AND s.expires_at>?''', (hashed, self.clock())).fetchone()
            return self._account(row) if self._ready(db, row) and row['google_subject'] else None

    def revoke_session(self, token):
        with self._transaction() as db:
            db.execute('DELETE FROM browser_sessions WHERE token_hash=?', (_token_hash(token),))

    def is_active(self, owner_id):
        """Online registry/credential gate; errors propagate and callers fail closed."""
        with self._transaction() as db:
            row = db.execute('SELECT * FROM identities WHERE owner_id=?', (owner_id,)).fetchone()
            return bool(self._ready(db, row) and row['google_subject'])

    @staticmethod
    def _advance_credentials(db, owner, *, invitation_generation=None):
        """Fence old/pending credentials and queue broker cleanup in this transaction."""
        prior = db.execute('SELECT * FROM gmail_connections WHERE owner_id=?', (owner['owner_id'],)).fetchone()
        generation = prior['credential_generation'] + 1 if prior else 1
        cleanup = None
        if prior and owner['google_subject']:
            cleanup = CredentialCleanup(owner['owner_id'], owner['email'], owner['google_subject'],
                                        prior['invitation_generation'], prior['credential_generation'])
            db.execute("""INSERT OR IGNORE INTO gmail_credential_cleanup
                (owner_id,email,google_subject,invitation_generation,credential_generation) VALUES(?,?,?,?,?)""",
                (cleanup.owner_id, cleanup.email, cleanup.google_subject, cleanup.invitation_generation, cleanup.credential_generation))
        db.execute("""INSERT INTO gmail_connections(owner_id,credential_generation,invitation_generation,connected) VALUES(?,?,?,0)
            ON CONFLICT(owner_id) DO UPDATE SET credential_generation=excluded.credential_generation,
            invitation_generation=excluded.invitation_generation,connected=0""",
            (owner['owner_id'], generation, invitation_generation if invitation_generation is not None else owner['generation']))
        return generation, cleanup

    def revoke(self, email):
        """Deny access atomically before attempting downstream cancellation."""
        with self._transaction() as db:
            row = db.execute('SELECT * FROM identities WHERE email=?', (_email(email),)).fetchone()
            if not row:
                raise IdentityDenied()
            owner = row['owner_id']
            self._advance_credentials(db, row, invitation_generation=row['generation'] + 1)
            db.execute('UPDATE identities SET invited=0,provisioned=0,generation=generation+1 WHERE owner_id=?', (owner,))
            db.execute('DELETE FROM browser_sessions WHERE owner_id=?', (owner,))
            generation = row['generation'] + 1
            db.execute('INSERT INTO identity_revocations(owner_id,generation) VALUES(?,?) ON CONFLICT(owner_id) DO UPDATE SET generation=excluded.generation', (owner, generation))
            return Revocation(owner, generation)

    def pending_revocations(self):
        with self._transaction() as db:
            return tuple(Revocation(row['owner_id'], row['generation']) for row in db.execute('SELECT owner_id,generation FROM identity_revocations ORDER BY owner_id'))

    def complete_revocation(self, revocation: Revocation):
        """Trusted completion only after all prior run leases are cancelled."""
        with self._transaction() as db:
            db.execute('DELETE FROM identity_revocations WHERE owner_id=? AND generation=?', (revocation.owner_id, revocation.generation))
