#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
default_steps=50
source scripts/pilot_common.sh
pretrain_args=(--stage srresnet)
gan_args=(--stage srgan)
run_gan srgan 200 "$@"
