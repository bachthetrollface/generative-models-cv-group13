#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root=${DATA_ROOT:-data}
mkdir -p "$root"
archive="$root/df2k-ost-v1.zip"
if [[ ! -f "$archive" ]]; then
  curl --fail --location --retry 3 --continue-at - \
    'https://www.kaggle.com/api/v1/datasets/download/thaihoa1476050/df2k-ost?datasetVersionNumber=1' \
    -o "$archive.part"
  unzip -tq "$archive.part" > /dev/null
  mv "$archive.part" "$archive"
fi
unzip -qo "$archive" -d "$root"
