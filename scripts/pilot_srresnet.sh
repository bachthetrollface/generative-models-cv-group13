#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
default_steps=250
source scripts/pilot_common.sh
exec "$python_bin" train.py --stage srresnet --out "$run_root/srresnet" \
  "${common[@]}" "${duration[@]}" "$@"
