#!/usr/bin/env bash
# Build only a disposable synthetic worker. Does not start QEMU.
set -euo pipefail
umask 077
for tool in curl sha512sum qemu-img xorriso ssh-keygen; do
  command -v "$tool" >/dev/null
done
worker_dir=$(mktemp -d /tmp/gmail-worker-spike.XXXXXXXX)
printf '%s\n' "$worker_dir"
touch "$worker_dir/.synthetic-worker"
base=https://cloud.debian.org/images/cloud/bookworm/latest
image=debian-12-genericcloud-amd64.qcow2
curl --fail --location --proto '=https' --tlsv1.2 --max-time 60 \
  "$base/SHA512SUMS" -o "$worker_dir/SHA512SUMS"
curl --fail --location --proto '=https' --tlsv1.2 --max-time 600 \
  "$base/$image" -o "$worker_dir/$image"
(
  cd "$worker_dir"
  # Validate only the exact image name, never arbitrary paths in the manifest.
  python3 - "$image" <<'PY'
import hashlib, pathlib, re, sys
name = sys.argv[1]
matches = []
for line in pathlib.Path('SHA512SUMS').read_text().splitlines():
    parts = line.split()
    if len(parts) == 2 and parts[1].lstrip('*').removeprefix('./') == name:
        matches.append(parts[0])
if len(matches) != 1 or not re.fullmatch('[0-9a-fA-F]{128}', matches[0]):
    raise SystemExit('Missing or ambiguous SHA512 image checksum')
with open(name, 'rb') as image:
    actual = hashlib.file_digest(image, 'sha512').hexdigest()
if actual.lower() != matches[0].lower():
    raise SystemExit('Image checksum mismatch')
pathlib.Path('verified.sha512').write_text(f'{actual}  {name}\n')
print('Verified image SHA512:', actual)
PY
)
chmod 400 "$worker_dir/$image"
qemu-img create -f qcow2 -F qcow2 -b "$worker_dir/$image" "$worker_dir/worker.qcow2" 12G
ssh-keygen -q -t ed25519 -N '' -C synthetic-worker-spike -f "$worker_dir/id_ed25519"
worker_public_key=$(cat "$worker_dir/id_ed25519.pub")
cat > "$worker_dir/user-data" <<EOF
#cloud-config
hostname: synthetic-execution-worker
disable_root: true
ssh_pwauth: false
users:
  - name: worker
    groups: [sudo]
    sudo: ['ALL=(ALL) NOPASSWD:ALL']
    shell: /bin/bash
    lock_passwd: true
    ssh_authorized_keys:
      - $worker_public_key
EOF
printf 'instance-id: synthetic-worker-spike\nlocal-hostname: synthetic-execution-worker\n' > "$worker_dir/meta-data"
xorriso -as mkisofs -quiet -output "$worker_dir/seed.iso" -volid cidata -joliet -rock \
  "$worker_dir/user-data" "$worker_dir/meta-data"
printf 'Prepared synthetic worker: %s\n' "$worker_dir"
