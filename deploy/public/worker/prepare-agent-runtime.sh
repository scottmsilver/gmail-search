#!/usr/bin/env bash
# Build only public runtime files and two fixed synthetic guest scripts. Never
# unpack/mount a guest-modified filesystem. Output is an immutable SquashFS.
set -euo pipefail
umask 077
runtime_inputs=${1:?Usage: prepare-agent-runtime.sh PUBLIC_RUNTIME_INPUTS}
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
[[ $(sha256sum "$runtime_inputs/bin/node" | cut -d' ' -f1) == 89af8424dd53e560b1933f87ba650d8bf57c83ca5a04600eefb31f416aabbae7 ]]
[[ $(sha256sum "$runtime_inputs/bin/claude" | cut -d' ' -f1) == d81396a668eb76fbddb49a2a5841f1b5d7af96b4c1f6500ced92f2c988f5bcd4 ]]
build_dir=$(mktemp -d /tmp/gmail-agent-runtime.XXXXXXXX)
mkdir "$build_dir/image"
cp -a "$runtime_inputs/bin" "$runtime_inputs/lib" "$build_dir/image/"
cp "$script_dir/guest-vsock-bridge.py" "$script_dir/guest-agent-smoke.py" "$build_dir/image/"
chmod -R a+rX "$build_dir/image"
mksquashfs "$build_dir/image" "$build_dir/runtime.squashfs" -all-root -noappend -comp zstd -processors 2 -mem 256M -no-progress
sha256sum "$build_dir/runtime.squashfs" > "$build_dir/SHA256SUMS"
printf 'Prepared immutable synthetic runtime: %s\n' "$build_dir"
