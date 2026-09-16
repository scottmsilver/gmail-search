"""Independent cross-profile rejection tests; synthetic databases only."""
import asyncio
import secrets

import psycopg
import pytest

from gmail_search.gateway import partitions
from gmail_search.gateway.data_admission import DataAdmission
from gmail_search.gateway.partition_profiles import NUMERIC_OWNER_PARTITIONS_V1 as NUMERIC
from gmail_search.gateway.partition_profiles import TEXT_OWNER_PARTITIONS_V1 as TEXT
from gmail_search.gateway.provision_search_reader import provision_search_reader
from gmail_search.gateway.search_reader import SearchCredential, SearchProfile, SearchReader, SearchRegistry, search_role
from test_gateway_search_reader import search_database as _search_database

search_database = _search_database


@pytest.mark.asyncio
@pytest.mark.parametrize('credential_profile,reader_profile', [(NUMERIC, TEXT), (TEXT, NUMERIC)])
async def test_mismatched_profiles_never_connect(monkeypatch, credential_profile, reader_profile):
    credential = SearchCredential('alice', 'user=' + search_role('alice', profile=credential_profile),
                                  schema_profile=credential_profile)
    admission = DataAdmission()
    reader = SearchReader(SearchRegistry({'alice': credential}, is_active=lambda _: True),
                          profile=SearchProfile('fixture', 'fixture+v1', 2, schema_profile=reader_profile),
                          admission=admission)

    async def forbidden(*args, **kwargs):
        pytest.fail('Profile mismatch reached connection acquisition')

    monkeypatch.setattr(psycopg.AsyncConnection, 'connect', forbidden)
    with pytest.raises(PermissionError):
        async with reader.session('alice', deadline=asyncio.get_running_loop().time() + 5,
                                  check_active=lambda: None):
            pytest.fail('Profile mismatch opened a session')
    assert not admission.active


def test_text_administration_refuses_numeric_database_without_mutation(search_database):
    db = search_database
    owner = db.owners[0]
    before = db.conn.execute("SELECT oid,relfilenode FROM pg_class WHERE relnamespace='gms_mail_partitions'::regnamespace ORDER BY oid").fetchall()
    for operation in (partitions.verify_owner_partitions, partitions.provision_owner_partitions,
                      partitions.remove_empty_owner_partitions):
        with pytest.raises(ValueError):
            operation(db.conn, owner, profile=TEXT)
    with pytest.raises(ValueError):
        provision_search_reader(db.conn, owner, secrets.token_urlsafe(40), profile=TEXT)
    assert db.conn.execute('SELECT 1 FROM pg_roles WHERE rolname=%s', (search_role(owner, profile=TEXT),)).fetchone() is None
    assert db.conn.execute("SELECT oid,relfilenode FROM pg_class WHERE relnamespace='gms_mail_partitions'::regnamespace ORDER BY oid").fetchall() == before
