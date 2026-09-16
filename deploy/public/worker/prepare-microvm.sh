#!/usr/bin/env bash
# Stage public, checksum-pinned synthetic inputs only; never boots a VM.
set -euo pipefail
umask 077
worker_dir=$(realpath -- "${1:?Usage: prepare-microvm.sh /tmp/gmail-worker-spike.XXXXXXXX}")
[[ "$worker_dir" == /tmp/gmail-worker-spike.* && -f "$worker_dir/.synthetic-worker" ]]
mkdir "$worker_dir/microvm"
cd "$worker_dir/microvm"
base=https://s3.amazonaws.com/spec.ccfc.min/firecracker-ci/v1.14/x86_64
curl -fsSL --proto '=https' --max-time 180 "$base/vmlinux-6.1.155" -o vmlinux
curl -fsSL --proto '=https' --max-time 180 "$base/ubuntu-24.04.squashfs" -o rootfs.squashfs
cp /usr/local/bin/firecracker /usr/local/bin/jailer .
cat > SHA256SUMS <<'SUMS'
34237ad1a6fcec150a85786488a64acc3b8f419057136113a9858c8bea68bf88  firecracker
a8683b88775f4d95802fb6f4db8ee5fba22aca31e0e9cfa99560608c560cdfef  jailer
e41c7048bd2475e7e788153823fcb9166a7e0b78c4c443bd6446d015fa735f53  vmlinux
f4fbb71a581c2f4cd204900ceaf280b71b031c58479250cb0430e0b29774ef5c  rootfs.squashfs
SUMS
sha256sum --check SHA256SUMS
printf 'Prepared pinned synthetic microVM inputs: %s/microvm\n' "$worker_dir"
