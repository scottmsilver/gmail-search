#!/bin/bash
# Install the gms Caddy route on this host's silver-oauth Caddy.
# Idempotent — re-run to update.
#
# Prereq: silver-oauth Caddy is set up (see /home/ssilver/development/silver-oauth/README.md).
# This script does NOT use sudo: the install user must be in the `caddy`
# group, and /etc/caddy/routes.d/ must be group-writable + setgid (the
# silver-oauth installer sets it up that way). If your current shell
# doesn't have `caddy` in its supplementary groups, run this script
# under `sg caddy -c '...'` or open a fresh login shell.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/gms.caddy"
DEST="/etc/caddy/routes.d/gms.caddy"

if [ ! -d /etc/caddy/routes.d ]; then
	echo "error: /etc/caddy/routes.d does not exist — set up silver-oauth Caddy first" >&2
	exit 1
fi

cp "$SRC" "$DEST"
chmod 0664 "$DEST"

# Caddy admin API on localhost:2019 — no privileges needed for reload.
caddy reload --config /etc/caddy/Caddyfile 2>&1 | sed 's/^/  caddy: /'

echo "installed: https://gms.i.oursilverfamily.com → 127.0.0.1:3000 (Next.js, which proxies /api/* to Python on 8090)"
