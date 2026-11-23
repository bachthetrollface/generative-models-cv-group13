# SRResNet and SRGAN on DIV2K ×4

Each developer can run either pilot independently. SRResNet trains from scratch for 200 pixel-MSE updates. SRGAN runs its own 200-update MSE pretraining stage and initializes the GAN from those weights for 50 updates with pixel MSE plus adversarial loss.

## Run the pilot

On a Linux machine with an NVIDIA T4, use Python 3.11 or 3.12 and a driver compatible with the CUDA 13.0 PyTorch wheels. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); assert torch.version.cuda == "13.0"'
wandb login
bash scripts/download_div2k.sh
# Standalone SRResNet experiment
bash scripts/pilot_srresnet.sh
# Independent SRGAN experiment, including its own generator pretraining
bash scripts/pilot_srgan.sh
```

The pilot scripts default to CUDA and use one T4 per process by default. Set `PRETRAIN_STEPS` to adjust the GAN’s private pretraining budget. The launcher resumes an interrupted pretraining stage automatically. Setting `GENERATOR` uses an explicitly supplied checkpoint, and `RESUME` continues an existing GAN directly. Choose a GPU with `CUDA_VISIBLE_DEVICES=0`. The trainer records GPU name, peak PyTorch memory allocation, and update time. Batch size 16 is the paper setting. If memory requires a smaller batch, set the same `BATCH_SIZE` for both pilots and record the change. The scripts honor `PYTHON`, `DATA_ROOT`, `RUN_ROOT`, `DEVICE`, `WORKERS`, `BATCH_SIZE`, `STEPS`, `EPOCHS`, and the standard W&B project and entity variables. The first VGG-based run downloads the torchvision VGG19 weights into `.cache/torch`. [PyTorch wheels](https://pytorch.org/get-started/previous-versions/)

The download script defaults to the official bicubic ×4 track. It accepts one or more tracks from `bicubic`, `unknown`, `realistic-mild`, `realistic-difficult`, and `realistic-wild`, and downloads the corresponding LR training and validation archives with their shared HR archives. Set `DATA_ROOT` to use a different data directory. [Official DIV2K archives](https://data.vision.ee.ethz.ch/cvl/DIV2K/#data-access)

Use `--div2k-subset` to select the training track. Both pilot launchers accept it, including the SRGAN generator pretraining stage. For example, an unknown ×4 pilot uses:

```bash
bash scripts/download_div2k.sh unknown
RUN_ROOT=runs/unknown bash scripts/pilot_srgan.sh --div2k-subset unknown
```

All training stages read [the fixed DIV2K split](splits/div2k_train_val.txt) and record it in `split.json`. It assigns 750 HR image IDs to training and 50 to internal validation. Wild keeps all four LR training variants of each HR image in the same partition, for 3,000 training pairs and 200 internal validation pairs. The official validation IDs 0801–0900 are used as project test images for each track. To switch tracks from a trained generator, initialize a fresh run from its weights with `--init-generator` (or `GENERATOR` in the SRGAN launcher).

## Train a single experiment on two T4s

On a machine with two T4s, select both devices and pass `--data-parallel` to the experiment you want to run.

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/pilot_srresnet.sh --data-parallel
CUDA_VISIBLE_DEVICES=0,1 bash scripts/pilot_srgan.sh --data-parallel
```

Each command runs one process. The default global batch of 16 splits into eight images per GPU. SRGAN uses both GPUs for its private generator pretraining and adversarial stage. BatchNorm computes statistics per replica, so results can differ from a single-GPU batch of 16. Validation, optional VGG feature loss, W&B logging, and checkpoint writing use the primary GPU or host process. Checkpoint parameter names are compatible with single-GPU evaluation and resume. The logged peak GPU memory is for the primary GPU.

## What the pilot measures

Both stages use up to 16 distinct training pairs per batch, random 24×24 LR / 96×96 HR crops on the nominal ×4 grid, and Adam at 1e-4. The generator uses 16 residual blocks, BatchNorm, PReLU, two PixelShuffle stages, and a linear three-channel output in BGR order. The discriminator uses eight 3×3 convolutions and a 1024-unit dense layer. Both networks follow Figure 4. The paper reports training on 350,000 ImageNet images. This project uses DIV2K. [Ledig et al., architecture and training](https://arxiv.org/html/1609.04802v5#S2)

| Stage | Default loss | Full paper schedule |
|---|---|---|
| SRResNet-MSE | Pixel MSE on HR intensities scaled to [−1, 1] | 1,000,000 updates at 1e-4 |
| SRGAN-MSE | Pixel MSE + 0.001 × non-saturating adversarial generator loss | 100,000 updates at 1e-4, then 100,000 at 1e-5 |

LR inputs use [0, 1]. Each VGG feature tensor is divided by 12.75 before MSE. VGG receives RGB standardized for the torchvision ImageNet weights. The GAN alternates one discriminator and one generator update. We compute binary cross-entropy in logit space. Implementation choices include PyTorch initialization, per-channel PReLU, Adam β₂ = 0.999, mean loss reduction across batch and channels, and frozen discriminator running statistics during the generator update. The pilot ends before the paper's GAN learning-rate drop. [Losses and training details](https://arxiv.org/html/1609.04802v5#S2.SS2)

The pilot uses the SRGAN-MSE variant from section 3.4 of the paper. For a later SRGAN-VGG54 comparison, reuse the same generator checkpoint with `RUN_ROOT=runs/gan-vgg54 GENERATOR=runs/t4-pilot/srgan-pretrain/latest.pt bash scripts/pilot_srgan.sh --pixel-weight 0 --vgg-weight 1`. Use these same loss flags when resuming an existing VGG54 run. The trainer also supports `--vgg-layer vgg22|vgg54` and `--adv-weight`. A zero weight disables that component. Calling the trainer directly without a duration uses the full schedule for the selected stage. [Paper loss variants](https://arxiv.org/html/1609.04802v5#S3.SS4)

## Metrics, images, and checkpoints

Under `runs/t4-pilot/`, the independent experiments write to `srresnet/` and `srgan/`. SRGAN keeps its own generator pretraining state in `srgan-pretrain/`. Read `metrics.jsonl` for step, learning rate, epoch progress, individual active losses, and discriminator real/fake scores. SRResNet and SRGAN both validate every 50 updates. Both also validate at the final update. Set `VAL_EVERY` to override the interval. Validation logs bicubic and model PSNR/SSIM on BT.601 luminance after shaving four HR pixels per edge. SSIM uses an 11×11 Gaussian window. The pilot evaluates the same eight held-out IDs on centered 256×256 HR crops and saves native-resolution bicubic/model/HR comparisons under `previews/`. These are pilot metrics. Use `--val-images 50 --val-crop-lr 0` for full-image internal validation. Validation and full-image evaluation report LPIPS-Alex v0.1 for both bicubic and model outputs by default. The metric downloads its pretrained backbone on first use. Pass `--no-lpips` to disable it.

The T4 scripts default to online W&B logging, including images and versioned checkpoint artifacts. Set `WANDB_PROJECT` and optionally `WANDB_ENTITY` before starting. On an offline machine, set `WANDB_MODE=offline` and skip login. To upload an offline run later, use `wandb sync <offline-run-directory>` on the directory printed by W&B and keep the local artifact staging directory until sync completes. Use `WANDB_MODE=disabled` for local checkpoints, images, and JSONL only.

The trainer atomically replaces `latest.pt` at each save interval and at the final update. W&B versions contain `training.pt` and carry `latest` and `step-N` aliases. Checkpoints store both active networks (including BatchNorm buffers), Adam states, update count, sampling settings, split, RNG state, and run ID. The learning rate and next training batch are determined from the completed update. A stage initialization loads generator weights only. A resume restores the whole stage and requires the original training settings.

```bash
# Extend the pilot to 500 total SRResNet updates.
STEPS=500 RESUME=runs/t4-pilot/srresnet/latest.pt bash scripts/pilot_srresnet.sh

# Extend the existing GAN to 10 total epochs with its own optimizer and discriminator.
EPOCHS=10 RESUME=runs/t4-pilot/srgan/latest.pt bash scripts/pilot_srgan.sh
```

Each epoch uses every training pair once. With a batch size of 16, the 750-pair tracks take 47 updates and wild takes 188. Each sampled pair gets a fresh crop each epoch. Changing the SRResNet checkpoint after a GAN starts has no effect on a resumed GAN. To initialize a fresh GAN from an extended SRResNet, select a new `RUN_ROOT` and set `GENERATOR` to that checkpoint. On another machine, set `RESUME=wandb://ENTITY/PROJECT/ARTIFACT:latest` with an authenticated account and a fresh output directory. Online resumes reuse the saved W&B run ID within the same project and entity. Offline resumes write separate W&B segments.

## Evaluate full images

Run `bash scripts/download_urban100.sh` to fetch the 100 Urban100 HR and bicubic ×4 LR pairs. It uses `DATA_ROOT` (default `data`) and extracts the images into `Urban100_HR/` and `Urban100_LR_x4/`, ready for the command below. [Urban100 archives](https://huggingface.co/datasets/eugenesiow/Urban100/tree/57309de03db02ac12ee09ad6127e7b815c211328/data)

Pass one or more names to `--test-sets` to evaluate a checkpoint across datasets in a single command. Named datasets are read from `--data` (default `data`). Each set gets its own folder under `--output`, with `sr/` PNG exports, a per-image CSV, mean scores, and a comparison preview. Each PNG is four times the LR width and height and uses the HR image ID as its filename.

```bash
python evaluate.py --checkpoint runs/t4-pilot/srgan/latest.pt \
  --test-sets div2k-bicubic urban100 --output runs/eval/srgan --lpips

bash scripts/download_div2k.sh --valid-only unknown realistic-mild realistic-difficult realistic-wild
python evaluate.py --checkpoint runs/t4-pilot/srgan/latest.pt \
  --test-sets div2k-unknown div2k-realistic-mild div2k-realistic-difficult div2k-realistic-wild \
  --output runs/eval/srgan --lpips
```

For custom pairs, pass `--hr` and `--lr` in place of `--test-sets`. Custom LR filenames can use the same stem as the HR image or add `x4`. Results go directly under `--output`.

The realistic tracks can introduce spatial shifts and scaling. Training crops and evaluation scores use the project's nominal ×4 grid. The NTIRE 2018 leaderboard protocol included a translation search. [NTIRE 2018 challenge report](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w13/Timofte_NTIRE_2018_Challenge_CVPR_2018_paper.pdf)

## Unit tests

Run `python -m unittest discover -s tests -v` in the pinned environment. Tests use synthetic images and toy networks to check resume, loss settings, pairing, metrics, and launcher arguments. They do not start a training job or contact W&B.
