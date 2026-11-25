#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
default_steps=1000000
source scripts/msr_common.sh
exec "$python_bin" train.py "${pretrain_args[@]}" --out "$run_root/msrresnet" \
  "${common[@]}" "${duration[@]}" "$@"
