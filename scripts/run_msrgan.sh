#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
default_steps=400000
source scripts/msr_common.sh
run_gan msrgan 1000000 "$@"
