#!/usr/bin/env bash
# Precompile the guest's Pi extensions into IMAGE_DIR/jiti-cache.
# jiti keys its cache by absolute path, so IMAGE_DIR is mounted at the guest
# path (/tmp/runtime) in a private mount namespace; nothing changes on the host.
# Works as root (image build) or unprivileged (user namespace).
set -euo pipefail
image_dir=$(cd -- "${1:?Usage: warm-jiti-cache.sh IMAGE_DIR}" && pwd)
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
namespace=(unshare --mount --propagation private)
[[ $EUID == 0 ]] || namespace+=(--map-root-user)
"${namespace[@]}" /bin/bash -euo pipefail -c '
  image_dir=$1; script_dir=$2
  # Stage both trees on a private tmpfs, then give /tmp a fresh tmpfs holding
  # only /tmp/runtime (the image) so the build host'"'"'s /tmp cannot leak in.
  mount -t tmpfs stage /mnt
  mkdir /mnt/image /mnt/scripts
  mount --bind "$image_dir" /mnt/image
  mount --bind "$script_dir" /mnt/scripts
  mount -t tmpfs guest-tmp /tmp
  mkdir /tmp/runtime
  mount --bind /mnt/image /tmp/runtime
  /usr/bin/python3 -I /mnt/scripts/warm_jiti_cache.py
' warm "$image_dir" "$script_dir"
