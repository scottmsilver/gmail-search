from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


def get_credentials(data_dir: Path) -> Credentials:
    creds_path = data_dir / "credentials.json"
    token_path = data_dir / "token.json"

    if not creds_path.exists():
        raise FileNotFoundError(
            f"Missing {creds_path}. Download OAuth client credentials from "
            "Google Cloud Console and save as data/credentials.json"
        )

    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), SCOPES)
            creds = flow.run_local_server(port=0)
        token_path.write_text(creds.to_json())

    return creds


def build_gmail_service(data_dir: Path):
    creds = get_credentials(data_dir)
    return build("gmail", "v1", credentials=creds)
