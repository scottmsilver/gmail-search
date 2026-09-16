# Public deployment

Live: https://gms.oursilverfamily.com — owner-only broker login and retrieval-only Gemini chat.

See [security boundaries, verification and rollback](../../docs/public-deployment-security.md)
and [restricted database setup](DATABASE.md). The LAN app remains separate.

The three `.service` templates run the dedicated API8091, web3001 and exact-host
Cloudflare tunnel. Secrets belong in mode0600 files outside the repository.
The public API launcher skips schema migrations and validates its restricted
DB login; use one worker. Restart revokes public sessions.

The running release is `~/.local/share/gmail-search/public-current`, targeting
`public-releases/security-20260914-final`. Both web assets and Python sources
are copied there. Build future releases outside the serving directory, test the
candidate, switch the symlink, and restart the public units. Do not point this
hostname at the private app or enable its shell/container runtimes.

The `.env.example` describes the contract; use newly generated independent
secrets rather than placeholders. Initial public admission must contain exactly
one preexisting mailbox owner matching the database role's fixed row policies.
