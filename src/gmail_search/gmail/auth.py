from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Gmail is the core read scope. `drive.readonly` is required for the
# Drive enrichment path (fetching linked Google Docs/Sheets/Slides by
# body-scanning for drive.google.com URLs). Adding the scope here
# means stored tokens issued before this change will miss it: the
# user must delete `data/token.json` and re-auth once to unlock
# Drive features. Gmail-only flows keep working with the old token.
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]


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
        # Write token with restricted permissions (contains refresh token)
        import stat

        token_path.write_text(creds.to_json())
        token_path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600

    return creds


def build_gmail_service(data_dir: Path):
    creds = get_credentials(data_dir)
    return build("gmail", "v1", credentials=creds)
