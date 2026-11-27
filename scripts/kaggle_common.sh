# Sourced by the Kaggle launchers. DATA_ROOT can point at a custom mount.
if [[ -z ${DATA_ROOT:-} ]]; then
  DATA_ROOT=/kaggle/input/datasets/thaihoa1476050/df2k-ost
  if [[ ! -d "$DATA_ROOT/train" ]]; then DATA_ROOT=/kaggle/input/df2k-ost; fi
fi
export DATA_ROOT
if [[ -z ${URBAN100_ROOT:-} ]]; then
  for mount in /kaggle/input/datasets/harshraone/urban100 /kaggle/input/urban100; do
    if [[ -d "$mount" ]]; then
      export URBAN100_ROOT="$mount"
      break
    fi
  done
fi
if [[ -z ${URBAN100_ROOT:-} ]]; then
  echo "Skipping Urban100 evaluation. Attach harshraone/urban100 or set URBAN100_ROOT." >&2
fi
export RUN_ROOT=${RUN_ROOT:-runs}
export PYTHON=${PYTHON:-python3}

export TORCH_HOME=${TORCH_HOME:-"${XDG_CACHE_HOME:-${HOME:-/tmp}/.cache}/torch"}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-"${TMPDIR:-/tmp}/wandb/cache"}
export WANDB_DATA_DIR=${WANDB_DATA_DIR:-"$RUN_ROOT/.wandb/staging"}
export WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR:-"${TMPDIR:-/tmp}/wandb/config"}
export WANDB_MODE=${WANDB_MODE:-online}
