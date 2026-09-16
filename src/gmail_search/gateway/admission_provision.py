"""Offline administrator admission service, never a public worker dependency.

Expose only through a trusted controller RPC that supplies an already verified
invitation identity. The credential installer must atomically store both logins
in the trusted credential vault, never SQLite. Existing mail owners require the
separate explicit bind_existing_subject operation before admission.
"""
import secrets

import psycopg
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from .database import ReaderCredential
from .partition_profiles import require_profile
from .partitions import provision_owner_partitions
from .provision import provision_reader
from .provision_writer import provision_application_writer
from .writer import WriterCredential


def _verified(identity):
    if (identity.email_verified is not True or not isinstance(identity.subject, str)
        or not identity.subject or not isinstance(identity.email, str)
        or not identity.email or identity.email != identity.email.strip().lower()):
        raise ValueError('Verified canonical identity required')


def bind_existing_subject(conn, owner_id, identity):
    """Explicit audited administrator import; owner_id must come from trusted records."""
    _verified(identity)
    row = conn.execute('SELECT email,google_sub FROM public.users WHERE id=%s FOR UPDATE', (owner_id,)).fetchone()
    if row is None or row[0] != identity.email or row[1] not in (None, identity.subject):
        raise ValueError('Canonical identity binding rejected')
    conn.execute('UPDATE public.users SET google_sub=%s WHERE id=%s', (identity.subject, owner_id))


class AdmissionProvisioner:
    """Admits one account: partitions, roles and credentials, in one commit.

    `partition_profile` has no default on purpose. It decides the message key
    and index definition every partition this service creates is built with, and
    a caller that forgot to pass it used to get the NUMERIC shape silently —
    while the live database is TEXT. Admission is the one place that choice must
    be stated out loud, because a partition provisioned in the wrong shape is
    not something a later reader can work around.
    """

    def __init__(self, administrator_connection, *, runtime_dsn, install_credentials,
                 partition_profile):
        self._connect = administrator_connection
        self._install = install_credentials
        self._partition_profile = require_profile(partition_profile)
        try:
            self._runtime = conninfo_to_dict(runtime_dsn)
        except Exception:
            raise ValueError('Invalid runtime endpoint') from None
        if any(key in self._runtime for key in ('user', 'password', 'passfile', 'service', 'options')):
            raise ValueError('Runtime endpoint cannot include a login or settings')
        if not self._runtime.get('dbname'):
            raise ValueError('Explicit runtime database required')

    def __call__(self, account, identity):
        _verified(identity)
        if account.email != identity.email:
            raise ValueError('Canonical identity binding rejected')
        lock_key = 'gms-account-admission:' + account.owner_id
        try:
            with self._connect() as conn:
                conn.autocommit = True
                qualified = conn.execute('SELECT session_user=current_user AND rolsuper,current_database() FROM pg_roles WHERE rolname=current_user').fetchone()
                if not qualified or qualified != (True, self._runtime['dbname']):
                    raise ValueError('Qualified administrator connection required')
                # Keep rotation and vault installation serialized across commit.
                conn.execute('SELECT pg_advisory_lock(hashtextextended(%s,0))', (lock_key,))
                try:
                    with conn.transaction():
                        row = conn.execute('SELECT email,google_sub FROM public.users WHERE id=%s FOR UPDATE', (account.owner_id,)).fetchone()
                        if row is None:
                            conn.execute('INSERT INTO public.users(id,email,google_sub) VALUES(%s,%s,%s)',
                                (account.owner_id, identity.email, identity.subject))
                        elif row != (identity.email, identity.subject):
                            raise ValueError('Canonical identity binding rejected')
                        # Account, all searchable partitions, and role rotation commit
                        # together before any credentials become available to runtime.
                        provision_owner_partitions(conn, account.owner_id, profile=self._partition_profile)
                        reader_password, writer_password = secrets.token_urlsafe(48), secrets.token_urlsafe(48)
                        reader = provision_reader(conn, account.owner_id, reader_password)
                        writer = provision_application_writer(conn, account.owner_id, writer_password)
                        credentials = (
                            ReaderCredential(account.owner_id, make_conninfo(**self._runtime, user=reader, password=reader_password)),
                            WriterCredential(account.owner_id, make_conninfo(**self._runtime, user=writer, password=writer_password)))
                    # A sink failure denies admission; retry rotates both credentials.
                    self._install(*credentials)
                    return True
                finally:
                    conn.execute('SELECT pg_advisory_unlock(hashtextextended(%s,0))', (lock_key,))
        except psycopg.Error:
            raise ValueError('Account provisioning failed') from None
