"""Contract tests for trusted owner -> direct database login selection."""
import pytest

from gmail_search.gateway.database import ReaderCredential, ReaderRegistry, QueryLimits, reader_role


def test_roles_are_stable_distinct_and_not_sql_identifiers_from_users():
    assert reader_role('alice') == reader_role('alice')
    assert reader_role('alice') != reader_role('bob')
    assert len(reader_role('a' * 500)) < 64
    assert reader_role("x'; SET ROLE postgres; --").isalnum() is False
    assert reader_role("x'; SET ROLE postgres; --").replace('_', '').isalnum()
    for invalid in ('', '\x00', None):
        with pytest.raises(ValueError):
            reader_role(invalid)


def test_credential_rejects_any_other_login_and_does_not_repr_password():
    secret = 'never-print-this-password'
    cred = ReaderCredential('alice', f'user={reader_role("alice")} password={secret} dbname=test')
    assert secret not in repr(cred)
    for dsn in ('user=postgres dbname=test', 'dbname=test', 'user=wrong password=secret'):
        with pytest.raises(ValueError):
            ReaderCredential('alice', dsn)


def test_registry_has_no_unknown_owner_fallback_and_checks_revocation():
    active = {'alice'}
    credential = ReaderCredential('alice', f'user={reader_role("alice")} dbname=test')
    source = {'alice': credential}
    registry = ReaderRegistry(source, is_active=lambda owner: owner in active)
    source.clear()
    assert registry.credential('alice') is credential
    with pytest.raises(PermissionError):
        registry.credential('bob')
    active.clear()
    with pytest.raises(PermissionError):
        registry.credential('alice')


def test_registry_rejects_mismatched_mapping():
    credential = ReaderCredential('alice', f'user={reader_role("alice")} dbname=test')
    with pytest.raises(ValueError):
        ReaderRegistry({'bob': credential}, is_active=lambda _: True)


@pytest.mark.parametrize('value', [True, '1', float('nan'), float('inf'), None])
def test_malformed_deadline_rejected(value):
    with pytest.raises(ValueError):
        QueryLimits(deadline_seconds=value)
