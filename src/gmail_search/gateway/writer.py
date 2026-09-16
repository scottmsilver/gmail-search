"""Direct fixed-owner application writer logins for trusted controllers only.

Mailbox ingestion and user-controlled SQL are deliberately outside this role.
There is no bootstrap lookup, administrative login, or SET ROLE fallback.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
from types import MappingProxyType

import psycopg
from psycopg.conninfo import conninfo_to_dict


def application_writer_role(owner_id):
    if not isinstance(owner_id,str) or not owner_id or '\x00' in owner_id or len(owner_id)>2048:
        raise ValueError('A stable owner ID is required')
    return 'gms_app_writer_'+hashlib.sha256(owner_id.encode()).hexdigest()[:40]


def writer_binding(owner_id):
    return 'gmail-search application writer v1 owner='+owner_id


@dataclass(frozen=True)
class WriterCredential:
    owner_id: str
    dsn: str = field(repr=False)

    def __post_init__(self):
        try:
            parsed=conninfo_to_dict(self.dsn)
        except Exception:
            raise ValueError('Invalid writer connection configuration') from None
        if parsed.get('user')!=application_writer_role(self.owner_id) or parsed.get('service') or parsed.get('options'):
            raise ValueError('Writer login must match its immutable owner')


class WriterRegistry:
    def __init__(self,credentials,*,is_active):
        if any(owner!=credential.owner_id for owner,credential in credentials.items()):
            raise ValueError('Writer registry owner mismatch')
        self._credentials=MappingProxyType(dict(credentials))
        self._is_active=is_active

    def credential(self,owner_id):
        credential=self._credentials.get(owner_id)
        try:
            active=self._is_active(owner_id)
        except Exception:
            raise PermissionError('Application write access unavailable') from None
        if credential is None or active is not True:
            raise PermissionError('Application write access unavailable')
        return credential

    @contextmanager
    def connection(self,owner_id):
        """One bounded trusted transaction; callers must not commit inside it.

        Arbitrary agent SQL never receives this connection. Revalidate before
        commit; a revoked owner rolls back and no returned data is published.
        """
        credential=self.credential(owner_id)
        try:
            with psycopg.connect(credential.dsn,connect_timeout=3,application_name='gms-app-writer') as conn:
                row=conn.execute('''SELECT session_user,current_user,
                    rolsuper OR rolcreatedb OR rolcreaterole OR rolreplication OR rolbypassrls,
                    EXISTS(SELECT 1 FROM pg_auth_members WHERE member=r.oid OR roleid=r.oid),
                    shobj_description(r.oid,'pg_authid'),current_setting('temp_file_limit')
                    FROM pg_roles r WHERE rolname=current_user''').fetchone()
                expected=application_writer_role(owner_id)
                if not row or row[:2]!=(expected,expected) or row[2] or row[3] or row[4]!=writer_binding(owner_id) or row[5]!='64MB':
                    raise PermissionError('Writer identity is not qualified')
                yield conn
                self.credential(owner_id)
        except psycopg.Error:
            raise RuntimeError('Application database operation failed') from None
