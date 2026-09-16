"""Closed trusted schema choices; never inferred from mail or model inputs."""
from enum import Enum
from types import MappingProxyType


class PartitionSchemaProfile(Enum):
    NUMERIC_OWNER_PARTITIONS_V1 = 'numeric-owner-partitions-v1'
    TEXT_OWNER_PARTITIONS_V1 = 'text-owner-partitions-v1'

    @property
    def message_key(self):
        return 'search_id' if self is NUMERIC_OWNER_PARTITIONS_V1 else 'id'

    @property
    def indexes(self):
        return _INDEXES[self]

    @property
    def keys(self):
        return _KEYS[self]


NUMERIC_OWNER_PARTITIONS_V1 = PartitionSchemaProfile.NUMERIC_OWNER_PARTITIONS_V1
TEXT_OWNER_PARTITIONS_V1 = PartitionSchemaProfile.TEXT_OWNER_PARTITIONS_V1


def require_profile(profile):
    if type(profile) is not PartitionSchemaProfile:
        raise ValueError('Unsupported trusted partition schema profile')
    return profile


_INDEXES = MappingProxyType({profile: MappingProxyType({
    'messages': ('messages_bm25_idx',
        (('search_id',) if profile is NUMERIC_OWNER_PARTITIONS_V1 else ()) + ('id','subject','body_text','from_addr','to_addr'),
        profile.message_key),
    'attachments': ('attachments_bm25_idx', ('id','filename','extracted_text'), 'id'),
    'propositions': ('props_bm25_idx', ('id','text'), 'id'),
}) for profile in PartitionSchemaProfile})
_KEYS = MappingProxyType({profile: MappingProxyType({
    'messages': frozenset({('p', ('user_id','id'))} | ({('u', ('user_id','search_id'))} if profile is NUMERIC_OWNER_PARTITIONS_V1 else set())),
    'attachments': frozenset({('p', ('user_id','id')), ('u', ('user_id','message_id','filename')), ('u', ('user_id','message_id','id'))}),
    'propositions': frozenset({('p', ('user_id','id'))}),
}) for profile in PartitionSchemaProfile})
