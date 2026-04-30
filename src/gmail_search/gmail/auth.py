"""Gmail credentials — broker-only path for multi-tenant.

Pivoted from `InstalledAppFlow.run_local_server` (which opens a browser
on the host — useless for a multi-user web app) to the silver-oauth
broker. Each user's Google OAuth refresh token lives broker-side; we
fetch a fresh access token via `broker /token?email=&scope=...` on
each call. No tokens stored locally, no per-app OAuth client, no
encryption-at-rest concern.

Legacy `data/token.json` path is preserved as a fallback when:
  - `SILVER_OAUTH_BROKER_URL` is unset (legacy single-user dev)
  - the broker request 404s (user hasn't connected Gmail yet)

So an existing single-pool install with a stored `token.json` keeps
working unchanged. New invitees go through the broker via the
`/api/auth/connect-gmail` endpoint.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

logger = logging.getLogger(__name__)

# Gmail is the core read scope. `drive.readonly` is required for the
# Drive enrichment path (fetching linked Google Docs/Sheets/Slides by
# body-scanning for drive.google.com URLs).
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]

# Broker scope used for the "is this user connected" probe and for
# fetching access tokens at sync time. We ask for just `gmail.readonly`
# — the core requirement. `drive.readonly` (used by the URL-crawler /
# Drive-doc enrichment path) is requested at /api/auth/connect-gmail
# time so most users grant it, but if Google's consent flow only
# returns gmail.readonly we don't want the whole connection to be
# considered "broken." The drive path is optional; gmail is not.
_BROKER_SCOPE = "gmail.readonly"


def _broker_url() -> Optional[str]:
    return os.environ.get("SILVER_OAUTH_BROKER_URL")


def _broker_bearer() -> Optional[str]:
    return os.environ.get("SILVER_OAUTH_BEARER")


def _broker_credentials_for(email: str) -> Optional[Credentials]:
    """Fetch a fresh Gmail access token via the broker's `/token`
    endpoint. Returns None when the broker isn't configured or doesn't
    have tokens for this email yet."""
    base = _broker_url()
    bearer = _broker_bearer()
    if not base or not bearer:
        return None
    try:
        r = requests.get(
            f"{base.rstrip('/')}/token",
            params={"email": email, "scope": _BROKER_SCOPE},
            headers={"Authorization": f"Bearer {bearer.strip()}"},
            timeout=15,
        )
    except requests.RequestException as exc:
        logger.warning("broker /token request failed for %s: %s", email, exc)
        return None
    if r.status_code == 404:
        # User hasn't connected Gmail (yet). Caller decides whether to
        # fall back to legacy token.json.
        return None
    if r.status_code == 403:
        raise PermissionError(f"broker says scope {_BROKER_SCOPE!r} not granted for {email}")
    if r.status_code != 200:
        logger.error("broker /token returned %d for %s: %s", r.status_code, email, r.text[:300])
        return None
    payload = r.json()
    token = payload.get("access_token")
    if not token:
        return None
    # Broker returns short-lived access tokens; we don't ask for the
    # refresh token (broker keeps it). Each call fetches fresh.
    return Credentials(token=token)


def get_credentials(data_dir: Path, *, email: Optional[str] = None) -> Credentials:
    """Resolve Google API credentials for `email`. Multi-tenant flow:
        1. Try the broker — `broker /token?email=&scope=…`.
        2. If the broker isn't configured OR returns 404, fall through
           to the legacy `data/token.json` path so single-pool dev
           installs keep working.

    `email` defaults to the `GMS_BOOTSTRAP_EMAIL` env var so daemon
    callers (`gmail-search update --loop` etc.) can switch which
    user's mail they're syncing without code changes — set the env
    var, run the daemon, done. Same env var the write path uses, so
    DB writes land under that user's user_id.

    Raises `FileNotFoundError` if neither path produces credentials —
    same behaviour as before so callers get a stable error to retry on.
    """
    if email is None:
        email = os.environ.get("GMS_BOOTSTRAP_EMAIL")
    if email:
        creds = _broker_credentials_for(email)
        if creds is not None:
            return creds
        logger.info("broker has no Gmail tokens for %s — falling back to legacy token.json", email)

    # Legacy single-pool fallback. Reads `data/token.json` written by
    # the original `InstalledAppFlow` flow. New per-user installs
    # should never hit this path — the broker IS the credential store.
    creds_path = data_dir / "credentials.json"
    token_path = data_dir / "token.json"

    if not token_path.exists():
        raise FileNotFoundError(
            f"No credentials available for {email or '<unknown>'}. "
            "Either connect Gmail via /api/auth/connect-gmail (broker path), "
            f"or place a legacy token.json at {token_path}."
        )

    creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if not creds.valid:
        if creds.expired and creds.refresh_token and creds_path.exists():
            creds.refresh(Request())
            import stat as _stat

            token_path.write_text(creds.to_json())
            token_path.chmod(_stat.S_IRUSR | _stat.S_IWUSR)
        else:
            raise FileNotFoundError(
                f"Stored token at {token_path} is expired and cannot be refreshed "
                "(no credentials.json, or no refresh_token). Re-auth via the broker."
            )

    return creds


def build_gmail_service(data_dir: Path, *, email: Optional[str] = None):
    creds = get_credentials(data_dir, email=email)
    return build("gmail", "v1", credentials=creds)
