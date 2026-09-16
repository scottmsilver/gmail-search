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
