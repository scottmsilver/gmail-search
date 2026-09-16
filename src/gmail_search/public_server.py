"""Public process launcher: no schema migrations, CLI jobs or shared runtime."""
import os
from pathlib import Path

import uvicorn

from gmail_search.auth.public import validate_public_auth_config
from gmail_search.config import load_config
from gmail_search.log_config import setup_logging, uvicorn_log_config
from gmail_search.server import create_app


def main():
    config = validate_public_auth_config()
    if config is None or len(config.emails) != 1:
        raise RuntimeError('The initial public deployment requires exactly one owner')
    if os.environ.get('GMAIL_SEARCH_API_URL') != 'http://127.0.0.1:8091':
        raise RuntimeError('Public retrieval must use its dedicated loopback backend')
    if not os.environ.get('DB_DSN') or not os.environ.get('GMAIL_MCP_ADMIN_TOKEN'):
        raise RuntimeError('Dedicated database and internal service credentials are required')
    from gmail_search.store.db import get_connection

    conn = get_connection(None)
    try:
        role = conn.execute("SELECT current_user AS role, rolsuper, rolbypassrls, rolcreaterole, rolcreatedb FROM pg_roles WHERE rolname = current_user").fetchone()
        if role["role"] != "gmail_search_public" or any(role[key] for key in ("rolsuper", "rolbypassrls", "rolcreaterole", "rolcreatedb")):
            raise RuntimeError("Public backend requires its restricted database login")
        visible = conn.execute("SELECT email FROM users").fetchall()
        if {row["email"].lower() for row in visible} != config.emails:
            raise RuntimeError("Public database owner does not match the admission allowlist")
    finally:
        conn.close()
    os.environ.setdefault('GMS_DEFAULT_STATEMENT_TIMEOUT_MS', '30000')
    setup_logging()
    data = Path(os.environ.get('GMS_DATA_DIR', 'data')).resolve()
    app = create_app(data / 'gmail_search.db', data, load_config(data_dir=data))
    uvicorn.run(app, host='127.0.0.1', port=8091, workers=1, log_config=uvicorn_log_config(), access_log=False)


if __name__ == '__main__':
    main()
