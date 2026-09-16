#!/usr/bin/env bash
# Foreground, bounded boot. No TAP/bridge, daemon, service or filesystem shares.
set -euo pipefail
worker_dir=$(realpath -- "${1:?Usage: boot.sh /tmp/gmail-worker-spike.XXXXXXXX}")
[[ ( "$worker_dir" == /tmp/gmail-worker-spike.* || "$worker_dir" == /home/ssilver/development/gmail-search/worktrees/full-agent-assets-20260915/worker-spike ) && -f "$worker_dir/.synthetic-worker" ]]
[[ $(stat -c %u "$worker_dir") == $(id -u) && $(stat -c %a "$worker_dir") == 700 ]]
[[ -r /dev/kvm && -w /dev/kvm ]]
cd "$worker_dir"
sha512sum --check --status verified.sha512
exec timeout --signal=TERM --kill-after=10s 900s qemu-system-x86_64 \
  -name gmail-synthetic-worker -nodefaults -no-user-config \
  -machine q35,accel=kvm -cpu host -smp 4 -m 4096 \
  -drive file="$worker_dir/worker.qcow2",if=virtio,format=qcow2 \
  -drive file="$worker_dir/seed.iso",if=virtio,format=raw,readonly=on \
  -netdev user,id=worker,restrict=on,hostfwd=tcp:127.0.0.1:22092-:22 \
  -device virtio-net-pci,netdev=worker \
  -display none -serial stdio -monitor none
