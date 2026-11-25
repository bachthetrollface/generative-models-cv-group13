# BasicSR ×4 recipes. Runtime, data and validation settings come from pilot_common.sh.
: "${RUN_ROOT:=runs}"
source scripts/pilot_common.sh

common+=(--arch msrresnet --adam-beta2 0.99 --ema-decay 0.999)
pretrain_args=(--stage srresnet --lr 2e-4 --lr-cosine-period 250000 --lr-min 1e-7
               --pixel-weight 1 --vgg-weight 0 --adv-weight 0)
gan_args=(--stage srgan --lr 1e-4 --lr-milestones 50000 100000 200000 300000 --lr-gamma 0.5
          --pixel-weight 0.01 --vgg-layer conv5_4 --vgg-weight 1 --adv-weight 0.005)
