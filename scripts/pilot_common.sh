# Sourced by the two stage scripts after switching to the repository root.
python_bin=${PYTHON:-python3}
if [[ -z ${PYTHON:-} && -x .venv/bin/python ]]; then python_bin=.venv/bin/python; fi
run_root=${RUN_ROOT:-runs/t4-pilot}
export TORCH_HOME=${TORCH_HOME:-"$PWD/.cache/torch"}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-"$PWD/.cache/wandb/cache"}
export WANDB_DATA_DIR=${WANDB_DATA_DIR:-"$PWD/.cache/wandb/staging"}
export WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR:-"$PWD/.cache/wandb/config"}
common=(--data "${DATA_ROOT:-data}" --batch-size "${BATCH_SIZE:-16}" --workers "${WORKERS:-2}"
        --device "${DEVICE:-cuda}" --log-every "${LOG_EVERY:-10}" --save-every "${SAVE_EVERY:-50}"
        --lpips --val-every "${VAL_EVERY:-50}" --val-images "${VAL_IMAGES:-8}")
duration=(--steps "${STEPS:-$default_steps}")
if [[ -n ${EPOCHS:-} ]]; then duration=(--epochs "$EPOCHS"); fi
