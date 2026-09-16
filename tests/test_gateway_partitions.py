"""Partition naming is bounded and independent of user-controlled SQL syntax."""
import hashlib
import importlib

import pytest


def module():
    spec = importlib.util.find_spec('gmail_search.gateway.partitions')
    assert spec is not None, 'owner partition module is not implemented'
    return importlib.import_module('gmail_search.gateway.partitions')


def test_partition_names_are_fixed_bounded_and_owner_specific():
    partitions = module()
    owner = "owner'; DROP TABLE messages; --"
    digest = hashlib.sha256(owner.encode()).hexdigest()[:48]
    assert partitions.PARTITION_SCHEMA == 'gms_mail_partitions'
    for table, prefix in [('messages','m'),('attachments','a'),('propositions','p')]:
        name = partitions.partition_name(table, owner)
        assert name == f'gms_{prefix}_{digest}'
        assert len(name) <= 63
        assert name != partitions.partition_name(table, 'other')


@pytest.mark.parametrize('owner', ['', None, 7, '\x00', 'x'*2049, '\ud800'])
def test_partition_names_reject_invalid_owner(owner):
    with pytest.raises(ValueError):
        module().partition_name('messages', owner)


def test_partition_names_reject_arbitrary_tables():
    with pytest.raises(ValueError):
        module().partition_name('users', 'owner')
