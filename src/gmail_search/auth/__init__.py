"""Multi-tenant identity package — silver-oauth broker pattern.

Phase 1 of PER_USER_LOGIN_2026-04-27.md: schema + CLI + a session-cookie
FastAPI dependency that's a no-op when `GMAIL_MULTI_TENANT != "1"`.
Pivoted from the NextAuth/Bearer-JWT model after recognizing the shared
silver-oauth broker (../silver-oauth) already solves the per-app
Google OAuth client problem we were re-inventing.
"""

from gmail_search.auth.session import (
    AuthError,
    User,
    broker_url,
    env_allowed_emails,
    is_email_allowed,
    is_multi_tenant_enabled,
    issue_handoff_jwt_for_test,
    require_user,
    set_session_cookie,
    verify_handoff_jwt,
)

__all__ = [
    "AuthError",
    "User",
    "broker_url",
    "env_allowed_emails",
    "is_email_allowed",
    "is_multi_tenant_enabled",
    "issue_handoff_jwt_for_test",
    "require_user",
    "set_session_cookie",
    "verify_handoff_jwt",
]
