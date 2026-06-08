"""Gmail credentials — broker-only.

Pivoted from `InstalledAppFlow.run_local_server` (which opens a browser
on the host — useless for a multi-user web app) to the silver-oauth
broker. Each user's Google OAuth refresh token lives broker-side; we
fetch a fresh access token via `broker /token?email=&scope=...` on
each call. No tokens stored locally, no per-app OAuth client, no
encryption-at-rest concern.

The legacy `data/token.json` / `InstalledAppFlow` path has been
removed: the broker is the sole credential store. Connect a Gmail
account via `/api/auth/connect-gmail`.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import requests
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
    """Resolve Google API credentials for `email` via the silver-oauth
    broker — the sole credential store. Fetches a fresh short-lived
    access token per call; the refresh token never leaves the broker.

    `email` defaults to the `GMS_BOOTSTRAP_EMAIL` env var so daemon
    callers (`gmail-search update --loop` etc.) can switch which user's
    mail they sync without code changes — set the env var, run the
    daemon, done. Same env var the write path uses, so DB writes land
    under that user's user_id.

    `data_dir` is retained for call-site compatibility (the removed
    local-token path used it) but is no longer read.

    Raises `RuntimeError` when no email can be resolved or the broker
    has no credentials for it (broker unconfigured, or the user hasn't
    connected Gmail). May raise `PermissionError` when the broker
    reports a missing scope. Callers' retry loops treat these the same
    way they did the previous `FileNotFoundError`.
    """
    if email is None:
        email = os.environ.get("GMS_BOOTSTRAP_EMAIL")
    if not email:
        raise RuntimeError(
            "No email to resolve Gmail credentials for — pass --email or set "
            "GMS_BOOTSTRAP_EMAIL. Credentials come from the silver-oauth broker."
        )
    creds = _broker_credentials_for(email)
    if creds is None:
        raise RuntimeError(
            f"No Gmail credentials for {email} from the broker. Either the broker "
            "is not configured (SILVER_OAUTH_BROKER_URL / SILVER_OAUTH_BEARER), or "
            "the user hasn't connected Gmail via /api/auth/connect-gmail."
        )
    return creds


def build_gmail_service(data_dir: Path, *, email: Optional[str] = None):
    creds = get_credentials(data_dir, email=email)
    return build("gmail", "v1", credentials=creds)
