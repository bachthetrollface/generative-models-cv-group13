#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root=${DATA_ROOT:-data}
base_url=https://huggingface.co/datasets/eugenesiow/Urban100/resolve/57309de03db02ac12ee09ad6127e7b815c211328/data
mkdir -p "$root"
for name in Urban100_HR Urban100_LR_x4; do
  archive="$root/$name.tar.gz"
  if [[ ! -f "$archive" ]]; then
    if ! gzip -t "$archive.part" 2>/dev/null; then
      curl --fail --location --retry 3 --continue-at - \
        "$base_url/$name.tar.gz" -o "$archive.part"
      gzip -t "$archive.part"
    fi
    mv "$archive.part" "$archive"
  fi
  tar -xzf "$archive" -C "$root"
done
