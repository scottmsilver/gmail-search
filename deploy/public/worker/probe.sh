#!/usr/bin/env bash
# Execute via SSH stdin in the synthetic worker; contains no source/mail/secrets.
set -euo pipefail
uname -a
printf 'nested parameter: '
cat /sys/module/kvm_intel/parameters/nested 2>/dev/null || true
sudo modprobe kvm_intel
sudo python3 - <<'PY'
import fcntl, os
fd = os.open('/dev/kvm', os.O_RDWR | os.O_CLOEXEC)
try:
    version = fcntl.ioctl(fd, 0xAE00, 0)
    assert version == 12, version
    vm = fcntl.ioctl(fd, 0xAE01, 0)
    os.close(vm)
    print('PASS: nested KVM API 12 and KVM_CREATE_VM')
finally:
    os.close(fd)
PY
for tool in firecracker jailer; do
  if command -v "$tool" >/dev/null; then "$tool" --version; else printf 'MISSING: %s\n' "$tool"; fi
done
