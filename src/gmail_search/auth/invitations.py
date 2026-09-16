"""Trusted invitation management, staged independently of public admission routes.

The actor is a server-authenticated owner ID, never a request-selected value.
Administrators manage invitations only; they receive no mailbox access override.
Wire IdentityStore.is_active into Registry and every credential-use gateway.
"""
from .identity_store import IdentityDenied, IdentityStore
from gmail_search.gateway.registry import Registry


class Invitations:
    def __init__(self, identities: IdentityStore, registry: Registry, *, administrators):
        self.identities, self.registry = identities, registry
        self.administrators = frozenset(administrators)

    def _administrator(self, actor):
        if not isinstance(actor, str) or actor not in self.administrators:
            raise IdentityDenied()

    def invite(self, actor, email):
        self._administrator(actor)
        return self.identities.invite(email)

    def revoke(self, actor, email):
        self._administrator(actor)
        revocation = self.identities.revoke(email)
        return self._cancel(revocation)

    def _cancel(self, revocation):
        try:
            runs = self.registry.cancel_owner(revocation.owner_id)
            self.identities.complete_revocation(revocation)
            return runs
        except Exception:
            # Admission remains denied and the durable queue remains available
            # to a trusted restart/sweeper path. Never roll back revocation.
            raise IdentityDenied() from None

    def drain_revocations(self):
        """Trusted startup/sweeper hook. Returned IDs must reach worker cancellation.

        Worker reconciliation must also terminate any nonactive registry run;
        this handles controller crashes after lease cancellation but before the
        returned IDs reach the worker. No physical worker control lives here.
        """
        runs = []
        for revocation in self.identities.pending_revocations():
            runs.extend(self._cancel(revocation))
        return tuple(runs)
