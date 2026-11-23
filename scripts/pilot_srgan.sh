#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
default_steps=50
source scripts/pilot_common.sh
if [[ -n ${RESUME:-} ]]; then
  start=(--resume "$RESUME")
elif [[ -n ${GENERATOR:-} ]]; then
  start=(--init-generator "$GENERATOR")
else
  pretrain_dir="$run_root/srgan-pretrain"
  pretrain_args=(--stage srresnet --out "$pretrain_dir" --steps "${PRETRAIN_STEPS:-200}")
  if [[ -f "$pretrain_dir/latest.pt" ]]; then
    pretrain_args+=(--resume "$pretrain_dir/latest.pt")
  fi
  for arg in "$@"; do
    case "$arg" in --lpips|--no-lpips|--data-parallel) pretrain_args+=("$arg");; esac
  done
  "$python_bin" train.py "${common[@]}" "${pretrain_args[@]}"
  start=(--init-generator "$pretrain_dir/latest.pt")
fi
exec "$python_bin" train.py --stage srgan --out "$run_root/srgan" \
  "${common[@]}" "${duration[@]}" "${start[@]}" "$@"
