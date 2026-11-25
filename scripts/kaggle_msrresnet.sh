#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/kaggle_common.sh
exec bash scripts/run_msrresnet.sh --test-kaggle "$@"
