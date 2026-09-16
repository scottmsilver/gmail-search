"""Fail-closed containment for untrusted SQL until tenant identity is immutable."""


def raw_sql_disabled() -> dict:
    """Never offer an environment switch back to the unsafe SQL connection."""
    return {
        "error": "Arbitrary SQL is disabled. Use search_emails_batch, query_emails_batch, "
                 "find_facts and get_thread_batch to retrieve authorized mail.",
        "code": "raw_sql_disabled",
        "status": 403,
        "rows": [],
    }
