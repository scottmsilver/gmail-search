#!/usr/bin/env bash
# Trusted immutable image build from two pinned public wheels, no host packages.
set -euo pipefail
umask 077
wheel_dir=${1:?Usage: prepare-attachment-runtime.sh PUBLIC_WHEEL_DIR}
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
build_dir=$(mktemp -d /tmp/gmail-attachment-runtime.XXXXXXXX)
mkdir -p "$build_dir/image/site-packages"
python3 - "$wheel_dir" "$build_dir/image/site-packages" <<'PY'
import hashlib, pathlib, sys, zipfile
pins = {
 'pillow-12.3.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl': '78cb2c6865a35ab8ff8b75fd122f6033b92a62c82801110e48ddd6c936a45d91',
 'pymupdf-1.28.2-cp310-abi3-manylinux_2_28_x86_64.whl': '397d6715c1f0df7548a92d0afd8ce370fc48fa47aeefac16be2bc04a16a8227f',
}
for name, digest in pins.items():
    wheel=pathlib.Path(sys.argv[1])/name
    if hashlib.sha256(wheel.read_bytes()).hexdigest()!=digest:
        raise SystemExit('Public wheel digest mismatch')
    with zipfile.ZipFile(wheel) as source:
        source.extractall(sys.argv[2])
PY
cp "$script_dir/guest-attachment-parser.py" "$build_dir/image/"
chmod -R a+rX "$build_dir/image"
mksquashfs "$build_dir/image" "$build_dir/attachment.squashfs" -all-root -noappend -comp zstd -processors 2 -mem 256M -no-progress
sha256sum "$build_dir/attachment.squashfs" > "$build_dir/SHA256SUMS"
printf 'Prepared parser-only runtime: %s\n' "$build_dir"
