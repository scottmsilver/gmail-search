"""Closed analytical schema; also the reader provisioning column allowlist.

Types are PostgreSQL pg_catalog names, not caller-selected type expressions.
Schema migrations must keep this map and the reader's catalog verification aligned.
"""
from types import MappingProxyType


def _columns(text: str, *, int8: str = '', float8: str = '', timestamptz: str = ''):
    values = {name: 'text' for name in (text + ' user_id').split()}
    for kind, names in [('int8', int8), ('float8', float8), ('timestamptz', timestamptz)]:
        values.update({name: kind for name in names.split()})
    return MappingProxyType(values)


ANALYTICAL_SCHEMA = MappingProxyType({
    'messages': _columns('id thread_id from_addr to_addr subject body_text date labels'),
    'attachments': _columns('message_id filename mime_type extracted_text fetch_status', int8='id size_bytes'),
    'thread_summary': _columns('thread_id subject participants all_from_addrs all_labels date_first date_last', int8='message_count'),
    'message_summaries': _columns('message_id summary model created_at'),
    'topics': _columns('topic_id parent_id label top_senders sample_subjects', int8='depth message_count'),
    'message_topics': _columns('message_id topic_id'),
    'contact_frequency': _columns('email', int8='message_count', float8='score'),
    'propositions': _columns('message_id thread_id text model date', int8='id', timestamptz='created_at'),
    'term_aliases': _columns('term expansions', float8='similarity'),
})


# Read-only catalogues created by the extensions bundled in the ParadeDB image
# (PostGIS/tiger/topology, pg_search, pg_ivm). PUBLIC can read them, so every
# role inherits them and a per-owner reader cannot be provisioned without either
# revoking that access or accepting it. They hold geometry, geocoder and index
# metadata -- no mail, no queries over mail.
#
# `deploy/public/provision_database.py` reached this judgement first, for the
# public login. `verify_reader_access` did not share it and rejected the same
# relations, so the two provisioning paths disagreed and no per-owner reader
# could be created against the live database at all. This is now the one copy;
# a test asserts the standalone script still matches it.
#
# Deliberately absent: `pgivm.pg_ivm_immv`, whose `viewdef` column would expose
# the SQL of any incremental materialised view built over mail. PUBLIC access to
# that one is revoked rather than allowlisted.
EXTENSION_METADATA = frozenset({
    ('pdb', 'index_layer_info'), ('paradedb', 'index_layer_info'),
    ('public', 'geography_columns'), ('public', 'geometry_columns'), ('public', 'spatial_ref_sys'),
    ('topology', 'topology'), ('topology', 'layer'),
    *(('tiger', name) for name in ('geocode_settings', 'geocode_settings_default',
        'loader_platform', 'loader_variables', 'loader_lookuptables', 'pagc_gaz', 'pagc_lex', 'pagc_rules')),
})
