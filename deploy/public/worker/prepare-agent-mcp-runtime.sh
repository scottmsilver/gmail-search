#!/usr/bin/env bash
# Build a separate public-only immutable MCP qualification image.
set -euo pipefail
umask 077
runtime_inputs=${1:?Usage: prepare-agent-mcp-runtime.sh PUBLIC_RUNTIME_INPUTS}
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
[[ $(sha256sum "$runtime_inputs/bin/node" | cut -d' ' -f1) == 89af8424dd53e560b1933f87ba650d8bf57c83ca5a04600eefb31f416aabbae7 ]]
[[ $(sha256sum "$runtime_inputs/bin/claude" | cut -d' ' -f1) == d81396a668eb76fbddb49a2a5841f1b5d7af96b4c1f6500ced92f2c988f5bcd4 ]]
build_dir=$(mktemp -d /home/ssilver/development/gmail-search/worktrees/full-agent-assets-20260915/agent-mcp-runtime.XXXXXXXX)
mkdir "$build_dir/image"
cp -a "$runtime_inputs/bin" "$runtime_inputs/lib" "$build_dir/image/"
for name in guest-vsock-bridge.py guest_run_bootstrap.py guest_tool_config.py guest_mail_tools.py guest_mail_tool_cli.py guest_mail_mcp.py guest-mail-mcp-smoke.py; do
  cp "$script_dir/$name" "$build_dir/image/"
done
chmod -R a+rX "$build_dir/image"
mksquashfs "$build_dir/image" "$build_dir/agent-mcp.squashfs" -all-root -noappend -comp zstd -processors 2 -mem 256M -no-progress
sha256sum "$build_dir/agent-mcp.squashfs" > "$build_dir/SHA256SUMS"
printf 'Prepared immutable synthetic MCP runtime: %s\n' "$build_dir"
