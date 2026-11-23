#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root=${DATA_ROOT:-data}
splits=(train valid)
if [[ ${1:-} == --valid-only ]]; then
  splits=(valid)
  shift
fi
if [[ $# -eq 0 ]]; then set -- bicubic; fi

tracks=()
for subset in "$@"; do
  case "$subset" in
    bicubic|unknown) tracks+=("${subset}_X4") ;;
    realistic-mild|realistic-difficult|realistic-wild) tracks+=("${subset#realistic-}") ;;
    *)
      printf '%s\n' "Usage: $0 [--valid-only] [bicubic|unknown|realistic-mild|realistic-difficult|realistic-wild] ..." >&2
      exit 2
      ;;
  esac
done

archives=()
for split in "${splits[@]}"; do
  archives+=("DIV2K_${split}_HR")
  for track in "${tracks[@]}"; do
    archives+=("DIV2K_${split}_LR_${track}")
  done
done
mkdir -p "$root"
for name in "${archives[@]}"; do
  zip="$root/$name.zip"
  if [[ ! -f "$zip" ]]; then
    curl --fail --location --retry 3 --continue-at - \
      "https://data.vision.ee.ethz.ch/cvl/DIV2K/$name.zip" -o "$zip.part"
    unzip -tq "$zip.part" > /dev/null
    mv "$zip.part" "$zip"
  fi
  unzip -qo "$zip" -d "$root"
done
