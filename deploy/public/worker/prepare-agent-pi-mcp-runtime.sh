#!/usr/bin/env bash
# Separate public-only Pi MCP image from verified, root-owned public inputs.
set -euo pipefail
umask 077
export PATH=/usr/sbin:/usr/bin:/sbin:/bin
[[ $EUID == 0 ]] || { printf 'Root-owned trusted staging and execution required\n' >&2; exit 1; }
runtime_inputs=${1:?Usage: prepare-agent-pi-mcp-runtime.sh PUBLIC_RUNTIME_INPUTS LOCKED_PI_PACKAGES}
pi_packages=${2:?Locked public Pi dependency directory required}
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
build_dir=$(mktemp -d /tmp/agent-pi-mcp-runtime.XXXXXXXX)
trap 'rm -rf -- "$build_dir"' ERR
/usr/bin/python3 -I "$script_dir/verify_pi_mcp_runtime_inputs.py" "$runtime_inputs" "$pi_packages" "$build_dir/image"
cmp "$build_dir/image/pi-pkgs/package-lock.json" "$script_dir/../../pi/pi-pkgs/package-lock.json"
cmp "$build_dir/image/pi-pkgs/package.json" "$script_dir/../../pi/pi-pkgs/package.json"
for name in guest-vsock-bridge.py guest_run_bootstrap.py guest_tool_config.py guest_mail_tools.py guest_mail_tool_cli.py guest_mail_mcp.py guest-pi-mail-mcp-smoke.py guest-pi-mail-mcp.ts; do
  cp "$script_dir/$name" "$build_dir/image/"
done
chmod -R a+rX "$build_dir/image"
mksquashfs "$build_dir/image" "$build_dir/agent-pi-mcp.squashfs" -all-root -noappend -comp zstd -processors 2 -mem 256M -no-progress
sha256sum "$build_dir/agent-pi-mcp.squashfs" > "$build_dir/SHA256SUMS"
printf 'Prepared immutable synthetic Pi MCP runtime: %s\n' "$build_dir"
