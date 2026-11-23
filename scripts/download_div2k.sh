#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root=${DATA_ROOT:-data}
mkdir -p "$root"
for name in DIV2K_train_HR DIV2K_train_LR_bicubic_X4; do
  zip="$root/$name.zip"
  if [[ ! -f "$zip" ]]; then
    curl --fail --location --retry 3 --continue-at - \
      "https://data.vision.ee.ethz.ch/cvl/DIV2K/$name.zip" -o "$zip.part"
    unzip -tq "$zip.part" > /dev/null
    mv "$zip.part" "$zip"
  fi
  unzip -qo "$zip" -d "$root"
done
