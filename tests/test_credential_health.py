"""Classification of Gmail sync failures into credential-health reasons.

Regression for the silent 3-day outage: an expired/scope-stripped token made
`watch` log "no new messages" forever. classify_credential_error turns those
into an actionable, surfaced state.
"""

from googleapiclient.errors import HttpError

from gmail_search.gmail.auth import classify_credential_error


class _Resp:
    def __init__(self, status):
        self.status = status
        self.reason = "Forbidden"


def test_insufficient_scope_403_is_scope_missing():
    exc = HttpError(_Resp(403), b'{"error":{"message":"Request had insufficient authentication scopes."}}')
    assert classify_credential_error(exc) == "gmail_scope_missing"


def test_permission_error_is_scope_missing():
    assert (
        classify_credential_error(PermissionError("broker says scope 'gmail.readonly' not granted"))
        == "gmail_scope_missing"
    )


def test_invalid_grant_is_revoked():
    exc = Exception("broker /token returned 500: invalid_grant: Token has been expired or revoked.")
    assert classify_credential_error(exc) == "credentials_revoked"


def test_no_credentials_is_not_connected():
    exc = RuntimeError(
        "No Gmail credentials for x@y.com from the broker ... connect Gmail via /api/auth/connect-gmail."
    )
    assert classify_credential_error(exc) == "not_connected"


def test_unrelated_errors_return_none():
    assert classify_credential_error(ValueError("some parsing bug")) is None
    assert classify_credential_error(ConnectionError("connection reset by peer")) is None
    assert classify_credential_error(Exception("ScaNN index missing")) is None
