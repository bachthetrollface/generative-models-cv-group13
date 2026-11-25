# Sourced by the launchers after switching to the repository root.
if [[ -n ${RESUME:-} ]]; then
  printf '%s\n' 'Use RUN_ROOT to start a fresh run. Set GENERATOR to initialize SRGAN from saved weights.' >&2
  exit 2
fi
python_bin=${PYTHON:-python3}
if [[ -z ${PYTHON:-} && -x .venv/bin/python ]]; then python_bin=.venv/bin/python; fi
run_root=${RUN_ROOT:-runs/t4-pilot}
export TORCH_HOME=${TORCH_HOME:-"$PWD/.cache/torch"}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-"$PWD/.cache/wandb/cache"}
export WANDB_DATA_DIR=${WANDB_DATA_DIR:-"$PWD/.cache/wandb/staging"}
export WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR:-"$PWD/.cache/wandb/config"}
common=(--data "${DATA_ROOT:-data}" --div2k-subset "${DIV2K_SUBSET:-bicubic}"
        --batch-size "${BATCH_SIZE:-16}" --workers "${WORKERS:-2}"
        --device "${DEVICE:-cuda}" --log-every "${LOG_EVERY:-10}" --save-every "${SAVE_EVERY:-50}"
        --lpips --val-every "${VAL_EVERY:-50}" --val-images "${VAL_IMAGES:-8}")
duration=(--steps "${STEPS:-$default_steps}")
if [[ -n ${EPOCHS:-} ]]; then duration=(--epochs "$EPOCHS"); fi

# Both GAN launchers supply pretrain_args and gan_args with their stage recipes.
run_gan() {
  local name=$1 pretrain_steps=$2 checkpoint=${GENERATOR:-}
  shift 2
  if [[ -z $checkpoint ]]; then
    local pretrain_dir="$run_root/$name-pretrain" i arg
    local -a args=("${pretrain_args[@]}" --out "$pretrain_dir" --steps "${PRETRAIN_STEPS:-$pretrain_steps}")
    # Forward runtime and data options to pretraining. Loss and LR overrides apply to fine-tuning.
    for ((i = 1; i <= $#; i++)); do
      arg=${!i}
      case "${arg%%=*}" in
        --help|-h|--lpips|--no-lpips|--data-parallel) args+=("$arg") ;;
        --arch|--div2k-subset|--data|--device|--workers|--batch-size|--seed|--val-every|--val-images|--val-crop-lr|--save-every|--log-every|--wandb-mode|--project|--entity)
          args+=("$arg")
          if [[ $arg != *=* ]] && ((i < $#)); then
            i=$((i + 1))
            args+=("${!i}")
          fi ;;
      esac
    done
    "$python_bin" train.py "${common[@]}" "${args[@]}"
    checkpoint="$pretrain_dir/latest.pt"
  fi
  exec "$python_bin" train.py "${gan_args[@]}" --out "$run_root/$name" \
    "${common[@]}" "${duration[@]}" --init-generator "$checkpoint" "$@"
}
