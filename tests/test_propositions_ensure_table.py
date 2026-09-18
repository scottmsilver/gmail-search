"""ensure_table must not need CREATE when the table already exists.

`/api/find_facts` calls `propositions.ensure_table()` on every request. That ran
`CREATE TABLE IF NOT EXISTS propositions` unconditionally, and PostgreSQL checks
CREATE privilege on the schema *before* noticing the table exists. The public
API connects as a role that deliberately has no CREATE on schema `public`, so
facts search failed on every call there with "permission denied for schema
public" -- and always had. The private app only avoided it by connecting as a
superuser.
"""
from __future__ import annotations

from gmail_search import propositions


class _Recorder:
    def __init__(self, exists: bool):
        self.exists, self.statements = exists, []

    def execute(self, sql, params=None):
        self.statements.append(" ".join(str(sql).split()))
        return self

    def fetchone(self):
        return ("public.propositions",) if self.exists else (None,)

    def commit(self):
        pass


def test_an_existing_table_issues_no_ddl():
    conn = _Recorder(exists=True)
    propositions.ensure_table(conn)
    assert not any(s.upper().startswith("CREATE") for s in conn.statements), conn.statements


def test_a_missing_table_is_still_created():
    """Standalone runs before init_db still get their table."""
    conn = _Recorder(exists=False)
    propositions.ensure_table(conn)
    assert any("CREATE TABLE IF NOT EXISTS propositions" in s for s in conn.statements)


def test_an_existing_bm25_index_issues_no_ddl():
    """The same shape one call later: `CREATE INDEX IF NOT EXISTS` needs table
    ownership before it checks existence, so the public API failed with
    "must be owner of table propositions" once ensure_table stopped failing."""
    conn = _Recorder(exists=True)
    propositions.ensure_bm25_index(conn)
    assert not any(s.upper().startswith("CREATE") for s in conn.statements), conn.statements


def test_a_missing_bm25_index_is_still_created():
    conn = _Recorder(exists=False)
    propositions.ensure_bm25_index(conn)
    assert any("props_bm25_idx" in s and s.upper().startswith("CREATE") for s in conn.statements)
