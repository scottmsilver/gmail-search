"""One-use Gmail consent state for a trusted broker/credential-store adapter.

Opaque 256-bit state is authenticated by its server-side hash and the initiating
browser session. Claims must come from a verified account-bound broker result or
Google OIDC response. No code, access token, or refresh token is stored here.
This module is staged: existing public routes and broker linking are unchanged.
"""
from dataclasses import dataclass, field
import math
import hmac
from urllib.parse import urlsplit

import jwt
import secrets

from .identity_store import CredentialCleanup, IdentityDenied, IdentityStore, VerifiedGoogleIdentity, _email, _token_hash


@dataclass(frozen=True)
class ConsentState:
    secret: str = field(repr=False)
    expires_at: float
    owner_id: str
    email: str
    google_subject: str = field(repr=False)
    generation: int
    credential_generation: int


@dataclass(frozen=True)
class ConsentGrant:
    """In-process trusted result, not a transferable bearer authorization.

    The credential adapter must revalidate immediately before storing/using a
    credential and bind its storage key to owner, subject, and generation. It
    must not accept serialized grants from requests or arbitrary worker code.
    """
    owner_id: str
    email: str
    google_subject: str = field(repr=False)
    generation: int  # Invitation generation; broker field is invitation_generation.
    credential_generation: int
    expires_at: float


class GmailConsent:
    def __init__(self, identities: IdentityStore):
        self.identities = identities
        with identities._transaction() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS gmail_consent_states (
                state_hash TEXT PRIMARY KEY, session_hash TEXT NOT NULL,
                owner_id TEXT NOT NULL REFERENCES identities(owner_id), email TEXT NOT NULL,
                google_subject TEXT NOT NULL, generation INTEGER NOT NULL, credential_generation INTEGER NOT NULL, expires_at REAL NOT NULL)''')

    def _session(self, db, token):
        row = db.execute('''SELECT i.* FROM identities i JOIN browser_sessions s ON s.owner_id=i.owner_id
            WHERE s.token_hash=? AND s.generation=i.generation AND s.expires_at>?''',
            (_token_hash(token), self.identities.clock())).fetchone()
        return row if self.identities._ready(db, row) and row['google_subject'] else None

    def begin(self, session, *, ttl=600):
        """Start from the authenticated browser; never accepts a target owner."""
        if isinstance(ttl, bool) or not isinstance(ttl, (int, float)) or not math.isfinite(ttl) or not 0 < ttl <= 600:
            raise IdentityDenied()
        with self.identities._transaction() as db:
            owner = self._session(db, session)
            if not owner:
                raise IdentityDenied()
            now = self.identities.clock()
            # Durable rolling attempt quota is independent of one-use pending
            # state, which is replaced whenever a new generation is requested.
            db.execute('CREATE TABLE IF NOT EXISTS gmail_consent_attempts (owner_id TEXT NOT NULL, attempted_at REAL NOT NULL)')
            db.execute('CREATE INDEX IF NOT EXISTS gmail_consent_attempt_owner ON gmail_consent_attempts(owner_id,attempted_at)')
            db.execute('DELETE FROM gmail_consent_attempts WHERE attempted_at<=?', (now - 600,))
            attempts = db.execute('SELECT count(*) FROM gmail_consent_attempts WHERE owner_id=?', (owner['owner_id'],)).fetchone()[0]
            backlog = db.execute('SELECT count(*) FROM gmail_credential_cleanup WHERE owner_id=?', (owner['owner_id'],)).fetchone()[0]
            if attempts >= 10 or backlog >= 100:
                raise IdentityDenied()
            db.execute('INSERT INTO gmail_consent_attempts VALUES(?,?)', (owner['owner_id'], now))
            db.execute('DELETE FROM gmail_consent_states WHERE expires_at<=?', (now,))
            credential_generation, _ = self.identities._advance_credentials(db, owner)
            db.execute('DELETE FROM gmail_consent_states WHERE owner_id=?', (owner['owner_id'],))
            secret = secrets.token_hex(32)
            expires = now + ttl
            db.execute('''INSERT INTO gmail_consent_states
                (state_hash,session_hash,owner_id,email,google_subject,generation,credential_generation,expires_at)
                VALUES(?,?,?,?,?,?,?,?)''', (_token_hash(secret), _token_hash(session), owner['owner_id'],
                owner['email'], owner['google_subject'], owner['generation'], credential_generation, expires))
            return ConsentState(secret, expires, owner['owner_id'], owner['email'], owner['google_subject'], owner['generation'], credential_generation)

    def consume(self, state, *, session, claims: VerifiedGoogleIdentity, broker_binding=None):
        """Burn one callback attempt atomically, returning only the matched owner.

        A wrong account also consumes state. Restart consent instead of retrying
        an intercepted/incorrect callback. Credential storage follows this call
        only on success; broker-side persistence needs the same pre-write gate.
        """
        grant = None
        with self.identities._transaction() as db:
            pending = db.execute('SELECT * FROM gmail_consent_states WHERE state_hash=?', (_token_hash(state),)).fetchone()
            if pending:
                db.execute('DELETE FROM gmail_consent_states WHERE state_hash=?', (_token_hash(state),))
                owner = self._session(db, session)
                connection = db.execute('SELECT * FROM gmail_connections WHERE owner_id=?', (pending['owner_id'],)).fetchone()
                valid_claims = isinstance(claims, VerifiedGoogleIdentity) and claims.email_verified is True
                try:
                    email = _email(claims.email) if valid_claims else None
                except IdentityDenied:
                    email = None
                if (owner and connection and connection['credential_generation'] == pending['credential_generation']
                    and (broker_binding is None or broker_binding == (owner['owner_id'], owner['generation'], connection['credential_generation']))
                    and pending['expires_at'] > self.identities.clock()
                    and pending['session_hash'] == _token_hash(session)
                    and (pending['owner_id'], pending['generation'], pending['email'], pending['google_subject'])
                    == (owner['owner_id'], owner['generation'], email, claims.subject if valid_claims else None)
                    and owner['email'] == email and owner['google_subject'] == claims.subject):
                    db.execute('UPDATE gmail_connections SET connected=1 WHERE owner_id=?', (owner['owner_id'],))
                    grant = ConsentGrant(owner['owner_id'], owner['email'], owner['google_subject'],
                                         owner['generation'], connection['credential_generation'], self.identities.clock() + 60)
        # Reject outside the transaction so even rejected callbacks consume state.
        if grant is None:
            raise IdentityDenied()
        return grant

    def consume_broker_handoff(self, state, *, session, token, broker_origin, signing_secret):
        """Verify the new broker's registered Gmail result before issuing a grant.

        Origin and secret are trusted deployment configuration. Legacy broker
        handoffs and identity-only handoffs deliberately fail this audience.
        """
        try:
            origin = urlsplit(broker_origin)
            if origin.scheme != "https" or not origin.netloc or origin.username or origin.password or broker_origin != f"https://{origin.netloc}":
                raise IdentityDenied()
            if not isinstance(signing_secret, str) or len(signing_secret.encode()) < 32 or not isinstance(token, str) or len(token) > 16384:
                raise IdentityDenied()
            claims = jwt.decode(token, signing_secret, algorithms=["HS256"], issuer=broker_origin,
                audience="gmail-search:gmail-consent", options={"strict_aud": True,
                    "verify_exp": False, "verify_iat": False,
                    "require": ["iss", "aud", "sub", "email", "email_verified", "owner_id", "invitation_generation", "credential_generation", "nonce", "iat", "exp", "jti"]})
            if any(type(claims[k]) is not str or not claims[k] or len(claims[k]) > 2048 for k in ("sub", "email", "owner_id", "nonce", "jti")):
                raise IdentityDenied()
            if type(claims["iat"]) is not int or type(claims["exp"]) is not int or type(claims["invitation_generation"]) is not int or type(claims["credential_generation"]) is not int:
                raise IdentityDenied()
            now = self.identities.clock()
            if not (claims["iat"] <= now < claims["exp"] and 0 < claims["exp"] - claims["iat"] <= 60):
                raise IdentityDenied()
            if not isinstance(state, str) or not hmac.compare_digest(claims["nonce"], state):
                raise IdentityDenied()
            verified = VerifiedGoogleIdentity(claims["email"], claims["sub"], claims["email_verified"])
            return self.consume(state, session=session, claims=verified,
                broker_binding=(claims["owner_id"], claims["invitation_generation"], claims["credential_generation"]))
        except (jwt.PyJWTError, ValueError, TypeError):
            raise IdentityDenied() from None

    def validate_grant(self, grant: ConsentGrant):
        """Recheck before credential persistence/use; never resurrect old grants.

        This is an online check, not a distributed transaction with the broker.
        The credential adapter must enforce the generation at commit and on each
        later credential use, and process revocation through its own durable path.
        """
        if not isinstance(grant, ConsentGrant) or grant.expires_at <= self.identities.clock():
            raise IdentityDenied()
        with self.identities._transaction() as db:
            owner = db.execute('SELECT * FROM identities WHERE owner_id=?', (grant.owner_id,)).fetchone()
            connection = db.execute('SELECT * FROM gmail_connections WHERE owner_id=?', (grant.owner_id,)).fetchone()
            if not connection or not connection['connected'] or connection['credential_generation'] != grant.credential_generation:
                raise IdentityDenied()
            if not self.identities._ready(db, owner) or (owner['email'], owner['google_subject'], owner['generation']) != (grant.email, grant.google_subject, grant.generation):
                raise IdentityDenied()

    def disconnect(self, session):
        """Disable credential use immediately while keeping the browser signed in."""
        with self.identities._transaction() as db:
            owner = self._session(db, session)
            if not owner:
                raise IdentityDenied()
            _, cleanup = self.identities._advance_credentials(db, owner)
            db.execute('DELETE FROM gmail_consent_states WHERE owner_id=?', (owner['owner_id'],))
            return cleanup

    def pending_cleanup(self):
        """Durable broker disconnect requests; keep until confirmed or superseded."""
        with self.identities._transaction() as db:
            return tuple(CredentialCleanup(**dict(row)) for row in db.execute(
                'SELECT * FROM gmail_credential_cleanup ORDER BY owner_id,credential_generation'))

    def complete_cleanup(self, cleanup: CredentialCleanup):
        with self.identities._transaction() as db:
            db.execute('DELETE FROM gmail_credential_cleanup WHERE owner_id=? AND credential_generation=? AND invitation_generation=?',
                       (cleanup.owner_id, cleanup.credential_generation, cleanup.invitation_generation))

    def credential_is_active(self, owner_id, invitation_generation, credential_generation):
        """Trusted credential gateway must call for every fetch/use, never cache.

        Owner comes from the authenticated session/run; both generations come
        from the stored verified credential binding, not browser-supplied values.
        """
        with self.identities._transaction() as db:
            owner = db.execute('SELECT * FROM identities WHERE owner_id=?', (owner_id,)).fetchone()
            connection = db.execute('SELECT * FROM gmail_connections WHERE owner_id=?', (owner_id,)).fetchone()
            return bool(self.identities._ready(db, owner) and owner['google_subject']
                        and connection and connection['connected']
                        and owner['generation'] == invitation_generation
                        and connection['invitation_generation'] == invitation_generation
                        and connection['credential_generation'] == credential_generation)

    def connection_status(self, session):
        """Browser-safe status: no tokens, subject or internal generations."""
        with self.identities._transaction() as db:
            owner = self._session(db, session)
            if not owner:
                raise IdentityDenied()
            connection = db.execute('SELECT * FROM gmail_connections WHERE owner_id=?', (owner['owner_id'],)).fetchone()
            return {"connected": bool(connection and connection['connected']
                                      and connection['invitation_generation'] == owner['generation'])}
