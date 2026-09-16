"""Compound owner writes need actual rollback through the application adapter."""
import pytest
from gmail_search.store.db import get_connection


def test_transaction_wrapper_rolls_back_inner_changes_without_losing_outer_write(db_backend):
    conn=get_connection(db_backend['db_path'])
    try:
        conn.execute("INSERT INTO sync_state(key,value) VALUES('outer','kept')")
        with pytest.raises(ValueError):
            with conn.transaction():
                conn.execute("UPDATE sync_state SET value='changed' WHERE key='outer'")
                conn.execute("INSERT INTO sync_state(key,value) VALUES('inner','discarded')")
                raise ValueError('simulated derived writer failure')
        conn.commit()
        assert conn.execute("SELECT key,value FROM sync_state WHERE key IN ('inner','outer')").fetchall()[0]['value']=='kept'
        assert conn.execute("SELECT count(*) FROM sync_state WHERE key='inner'").fetchone()[0]==0
    finally:
        conn.close()
